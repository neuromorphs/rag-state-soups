from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import MambaConfig, MambaForCausalLM, AutoModelForCausalLM
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

device = "cuda"

# print("Loading Mamba model from local checkpoint")
# model = MambaForCausalLM.from_pretrained('./model').cuda()
# tokenizer = AutoTokenizer.from_pretrained("Schmadge/mamba-slim-orca")
# tokenizer.eos_token = tokenizer.pad_token = "<|endoftext|>"
# tokenizer.chat_template = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta").chat_template

# print("Loading mamba from hf")
# model = MambaForCausalLM.from_pretrained('state-spaces/mamba-2.8b-hf').cuda()
# tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-2.8b-hf")

print("Loading RecurrentGemma-9B-IT from HF")
tokenizer = AutoTokenizer.from_pretrained("google/recurrentgemma-9b-it")
model = AutoModelForCausalLM.from_pretrained("google/recurrentgemma-9b-it").cuda()


sample = {
  "context": "In 1867, the United States purchased Alaska from Russia for $7.2 million. This purchase, known as the Alaska Purchase, added a vast territory to the United States and eventually became a rich source of natural resources.",
  "question": "What was the cost of Alaska when the United States purchased it from Russia?",
  "options": [
    "$7.2 million",
    "$10 million",
    "$5 million",
    "$15 million"
  ],
  "answer": "$7.2 million"
}
sample = {
  "context": "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods with the help of chlorophyll, a green pigment. This process involves the intake of carbon dioxide and water, and the release of oxygen as a byproduct.",
  "question": "Which pigment is essential for the process of photosynthesis in green plants?",
  "options": [
    "Chlorophyll",
    "Carotene",
    "Xanthophyll",
    "Anthocyanin"
  ],
  "answer": "Chlorophyll"
}
sample = {
  "context": "In the mythical land of Eldoria, the ancient texts describe a powerful artifact known as the Crystal of Arinthia. This crystal, when placed in the Temple of Shadows during the lunar eclipse, grants its bearer the ability to control time. The temple, located in the heart of the Forbidden Forest, can only be accessed by those who possess the Amulet of Ydron, an heirloom passed down through generations of the royal family.",
  "question": "What must be done to gain the ability to control time in Eldoria?",
  "options": [
    "Place the Crystal of Arinthia in the Temple of Shadows during a lunar eclipse",
    "Wear the Amulet of Ydron while standing on the peak of Mount Eldor",
    "Recite the ancient incantations in the presence of the Eldorian Council",
    "Offer a sacrifice at the Altar of Eternity during the winter solstice"
  ],
  "answer": "Place the Crystal of Arinthia in the Temple of Shadows during a lunar eclipse"
}

letters = ['A', 'B', 'C', 'D']
# question = f"Answer to the following multiple-choice question precisely with ONLY the letter of correct response, no other text, only the correct letter.\nQuestion: {ex['question']}\n" + '\n'.join([f'{letter}] ' + ex[f'answer_{i}'] for i, letter in zip(range(4), letters)])
prompt_question_str = f"Question: {sample['question']}\n"
prompt_question_str += '\n'.join([f'{letter}] ' + sample["options"][i] for i, letter in zip(range(4), letters)])
prompt_question_str += f"\nAnswer the above multiple-choice question."
# prompt_question_str += f" Hint: the answer is A]."
print(prompt_question_str)
print("\n" + "*"*50 + "\n")

prompt_context_str = f'Use the following context to answer the question below: {sample["context"]}'
prompt_context = [{"role": "user", "content": prompt_context_str}]
context_input_ids = tokenizer.apply_chat_template(prompt_context, return_tensors="pt", add_generation_prompt=False).to(device)
context_out = model(input_ids=context_input_ids, max_new_tokens=0, return_dict=True)
if True:
    # RG
    rec_states_context = {}
    for layer_idx, layer in enumerate(model.model.layers):
        if hasattr(layer.temporal_block, "rg_lru"):
            rec_states_context[layer_idx] = layer.temporal_block.rg_lru.recurrent_states.detach().cpu()
# else:
#     cache_context = context_out.cache_params
#     del context_out

prompt_question = [{"role": "user", "content": prompt_question_str}]
query_input_ids = tokenizer.apply_chat_template(prompt_question, return_tensors="pt", add_generation_prompt=False).to(device)
query_out = model(input_ids=query_input_ids, max_new_tokens=1, return_dict=True)
if True:
    # RG
    rec_states_query = {}
    for layer_idx, layer in enumerate(model.model.layers):
        if hasattr(layer.temporal_block, "rg_lru"):
            rec_states_query[layer_idx] = layer.temporal_block.rg_lru.recurrent_states.detach().cpu()
# else:
#     cache_query = query_out.cache_params
#     del query_out

out_query = model.generate(
    input_ids=query_input_ids,
    max_new_tokens=100,
    min_length=50,
)
out_query_str = tokenizer.decode(out_query[0]).strip()
print("Output with question only:")
print(out_query_str)
print("\n" + "*"*50 + "\n")

prompt_context_query = [
    {"role": "user", "content": prompt_context_str + "\n" + prompt_question_str},  # context-question
    # {"role": "user", "content": prompt_question_str + "\n" + prompt_context_str},  # question-context
]
context_query_input_ids = tokenizer.apply_chat_template(prompt_context_query, return_tensors="pt", add_generation_prompt=False).to(device)
out_context_query = model.generate(
    input_ids=context_query_input_ids,
    max_new_tokens=100,
    min_length=50,
)
out_context_query_str = tokenizer.decode(out_context_query[0]).strip()
print("Output with context and query:")
print(out_context_query_str)
print("\n" + "*"*50 + "\n")

# ######## SOUPING
def soup_fn(context, query, context_lambda=0.9, query_lambda=None):
  query_lambda = query_lambda if query_lambda is not None else (1. - context_lambda)
  return context * context_lambda + query * query_lambda

soup_states = {}
for layer_idx, layer in enumerate(model.model.layers):
    if hasattr(layer.temporal_block, "rg_lru"):
        soup_states[layer_idx] = soup_fn(
            rec_states_context[layer_idx].to(device),
            rec_states_query[layer_idx].to(device),
            0.5,
        )

def set_recurrent_state(rec_states: torch.Tensor, layer_idx: int):
    def hook(module, input):
        if module.recurrent_states.data.sum() == 0.0:
            print(f"setting state for layer {layer_idx} with sum {rec_states.sum():,}")
            module.recurrent_states.data = rec_states.to("cuda") * 10_000.
    return hook

print("adding state setting hook")
hooks = []
for layer_idx, layer in enumerate(model.model.layers):
    if hasattr(layer.temporal_block, "rg_lru"):
        h = layer.temporal_block.rg_lru.register_forward_pre_hook(
            set_recurrent_state(soup_states[layer_idx], layer_idx)
        )
        hooks.append(h)

# cache_soup = copy.copy(cache_query)
# cache_soup.ssm_states = {
#     k: soup_fn(
#         torch.from_numpy(cache_context['ssm_states'].item()[k]).cuda(),
#         cache_query.ssm_states[k],
#         0.5,
#         # condition.ssm_ratio if (k in condition.layers) else 0.0,  # 0.0 <> no context
#     )
#     for k in cache_soup.ssm_states.keys()
# }
# cache_soup.conv_states = {
#     k: soup_fn(
#         torch.from_numpy(cache_context['conv_states'].item()[k]).cuda(),
#         cache_query.conv_states[k],
#         0.5
#         # condition.conv_ratio if (k in condition.layers) else 0.0,  # 0.0 <> no context
#     )
#     for k in cache_soup.conv_states.keys()
# }
# # TODO: right now we're always takign the max of the two seqlen_offsets, but we could be more clever
# cache_soup.seqlen_offset = max([cache_context['seqlen_offset'], cache_query.seqlen_offset])

# ########

prompt_empty = "<bos>"
empty_input_ids = tokenizer(prompt_empty, return_tensors="pt")["input_ids"].to(device)
out_context_query = model.generate(
    input_ids=empty_input_ids,
    max_new_tokens=100,
    min_length=50,
    # cache_params=copy.copy(cache_soup)
)
out_context_query_str = tokenizer.decode(out_context_query[0]).strip()
print("Output with soup(query, context):")
print(out_context_query_str)
print("\n" + "*"*50 + "\n")