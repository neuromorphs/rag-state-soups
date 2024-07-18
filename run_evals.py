"""Run evaluation of the state soup mechanism for RAG on a synthetically generated RAG dataset with multiple choice answers.
"""
# !pip install transformers accelerate datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import MambaConfig, MambaForCausalLM
from tqdm import tqdm
import argparse
import numpy as np
import os
import torch
import pandas as pd
import copy
import re
import json
from dataclasses import dataclass
from typing import List, Union
import logging


logger = logging.getLogger(__name__)
logging.basicConfig(filename='run.log', level=logging.DEBUG)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="./model-folder")
    parser.add_argument("--dataset_path", type=str, default="./dataset/ragdataset-MC-QA.csv")
    parser.add_argument("--embedding_folder", type=str, default="./ssm-states")
    parser.add_argument("--results_folder", type=str, default="./results")
    parser.add_argument("--n_examples", type=int, default=None)
    return parser.parse_args()

def soup_fn(context, query, context_lambda=0.9, query_lambda=None):
  query_lambda = query_lambda if query_lambda is not None else (1. - context_lambda)
  return context * context_lambda + query * query_lambda

def format_context(ex):
  letters = ['A', 'B', 'C', 'D']
  # question = f"Answer to the following multiple-choice question precisely with ONLY the letter of correct response, no other text, only the correct letter.\nQuestion: {ex['question']}\n" + '\n'.join([f'{letter}] ' + ex[f'answer_{i}'] for i, letter in zip(range(4), letters)])
  question = f"Question: {ex['question']}\n" + '\n'.join([f'{letter}] ' + ex[f'answer_{i}'] for i, letter in zip(range(4), letters)])
  question += f"\nAnswer the above multiple-choice question."
  return question

@dataclass
class Configuration:
  """Configuration for a single run."""
  layers: List[int]
  ssm_ratio: float
  conv_ratio: float

  def __post_init__(self):
    if isinstance(self.layers, int):
      self.layers = [self.layers]

  def __str__(self):
    l = self.layers if isinstance(self.layers, int) or len(self.layers) == 1 else f"N{len(self.layers)}"
    return f"layers_{l}_ssm_{self.ssm_ratio}_conv_{self.conv_ratio}"


args = parse_args()
model_name_or_path = args.model_name_or_path
dataset_path = args.dataset_path
embedding_folder = args.embedding_folder
results_folder = args.results_folder
os.makedirs(results_folder, exist_ok=True)
write_path = os.path.join(results_folder, 'results.csv')
n_examples = args.n_examples

device = "cuda"

logging.info("Loading Mamba model from local checkpoint")
model = MambaForCausalLM.from_pretrained(model_name_or_path).cuda()

# setup tokenizer
if "state-spaces/mamba-2.8b-hf" in model_name_or_path:
  tokenizer = tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-2.8b-hf")
else:
  tokenizer = AutoTokenizer.from_pretrained("Schmadge/mamba-slim-orca")
  tokenizer.eos_token = tokenizer.pad_token = "<|endoftext|>"
  tokenizer.chat_template = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta").chat_template

# load dataset with multiple choice questions
ds = pd.read_csv(dataset_path, nrows=n_examples)

n_ssm_layers = 64
list_layers_to_apply = [int(percent*n_ssm_layers) for percent in [0.25, 0.5, 0.75]]
list_layers_to_apply.append(list(set(list_layers_to_apply)))
list_layers_to_apply.append(list(range(n_ssm_layers)))
list_ssm_ratio_to_apply = [0.5, 0.7, 0.9]
list_conv_ratio_to_apply = [0.0, 0.5]

conditions = []
conditions.append(Configuration(layers=[], ssm_ratio=0.0, conv_ratio=0.0))
for layers in list_layers_to_apply[::-1]:
  for ssm_ratio in list_ssm_ratio_to_apply:
    for conv_ratio in list_conv_ratio_to_apply:
      conditions.append(Configuration(layers=layers, ssm_ratio=ssm_ratio, conv_ratio=conv_ratio))

logging.info(f"Compiled {len(conditions)} conditions")

results = []
for qa_id in tqdm(range(n_examples)):
  # load question and correct answer from dataset
  row = ds.iloc[qa_id]
  question = format_context(row)
  correct_ans = np.where([row[f'correct_{i}'] for i in range(4)])[0][0]

  # load the context embedding as Mamba cache
  cache_path = os.path.join(embedding_folder, f"context_{qa_id}.npz")
  cache_context = np.load(cache_path, allow_pickle=True)

  # create the prompt for the question
  if "state-spaces/mamba-2.8b-hf" in model_name_or_path:
    query_input_ids = tokenizer(question, return_tensors="pt", add_generation_prompt=False).to(device)
  else:
    prompt = [{"role": "user", "content": question}]
    query_input_ids = tokenizer.apply_chat_template(prompt, return_tensors="pt", add_generation_prompt=False).to(device)
  
  # create an empty query to start the generation from
  input_ids_empty = tokenizer("<|assistant|>", return_tensors="pt")["input_ids"].to(device)

  # forward pass with query to collect hidden state
  query_out = model(input_ids=query_input_ids, max_new_tokens=1, return_dict=True)
  cache_query = query_out.cache_params

  # analyze mean abs value of SSM states
  avg_abs_val_query_ssm = [
    torch.abs(cache_query.ssm_states[k]).mean().item()
    for k in cache_query.ssm_states.keys()
  ]
  avg_abs_val_query_ssm = sum(avg_abs_val_query_ssm) / len(avg_abs_val_query_ssm)
  avg_abs_val_context_ssm = [
    torch.abs(torch.from_numpy(cache_context['ssm_states'].item()[k])).mean().item()
    for k in cache_context['ssm_states'].item().keys()
  ]
  avg_abs_val_context_ssm = sum(avg_abs_val_context_ssm) / len(avg_abs_val_context_ssm)
  logging.debug(f"[{qa_id}] Mean absolute value of SSM states: query {avg_abs_val_query_ssm:.5f}, context {avg_abs_val_context_ssm:.5f}")

  for condition in tqdm(conditions):
    logging.info(f"{qa_id} {condition}")

    # apply the souping
    cache_soup = copy.copy(cache_query)
    cache_soup.ssm_states = {
        k: soup_fn(
          torch.from_numpy(cache_context['ssm_states'].item()[k]).cuda(),
          cache_query.ssm_states[k],
          condition.ssm_ratio if (k in condition.layers) else 0.0,  # 0.0 <> no context
        )
        for k in cache_soup.ssm_states.keys()
    }
    cache_soup.conv_states = {
        k: soup_fn(
          torch.from_numpy(cache_context['conv_states'].item()[k]).cuda(),
          cache_query.conv_states[k],
          condition.conv_ratio if (k in condition.layers) else 0.0,  # 0.0 <> no context
        )
        for k in cache_soup.conv_states.keys()
    }
    # TODO: right now we're always takign the max of the two seqlen_offsets, but we could be more clever
    cache_soup.seqlen_offset = max([cache_context['seqlen_offset'], cache_query.seqlen_offset])

    out_full = model.generate(
      input_ids=input_ids_empty,
      max_new_tokens=100,
      min_length=50,
      # temperature=0.1,
      # do_sample=False,
      cache_params=copy.copy(cache_soup)
    )
    out_full_str = tokenizer.decode(out_full[0]).strip()

    logging.debug(out_full_str)
    # model_ans = out_full_str.strip()[0]
    matches = re.findall(r'[ABCD]\]', out_full_str)
    if len(matches) == 1:
      model_ans = matches[0][0]
    else:
      model_ans = "/"
    matches = "".join([e[0] for e in matches])

    letters = ['A', 'B', 'C', 'D']
    # store the results for this sample
    results.append({
      'sample_id': row['sample_id'],
      'condition': str(condition),
      'correct_answer': letters[correct_ans-1],
      'model_answer': model_ans,
      'correct': letters[correct_ans-1] == model_ans,
      'matches': matches,
      'full_answer': out_full_str.replace("\n", " *** "),
    })

    # save the results to a file
    pd.DataFrame(results).to_csv(write_path, index=True, header=True)
