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

print("Loading dataset")
# ds = load_dataset("zeitgeist-ai/financial-rag-nvidia-sec", "few-shot-examples")['train']
# few_shot_prompt = '\n\n'.join([f"""Question: {ds['question'][idx]}
# Context: {ds['context'][idx]}
# Answer: {ds['answer'][idx]}
# Evaluation: {ds['eval'][idx]}
# """ for idx in range(3)])

ds = load_dataset("zeitgeist-ai/financial-rag-nvidia-sec", "default")

N_examples = 5
few_shot_prompt = '\n'.join([f"""Question: {ds['train'][idx]['question']}
Context: {ds['train'][idx]['context']}
Answer: {ds['train'][idx]['answer']}""" for idx in range(N_examples)])

question = f"""Question: {ds['train'][N_examples+1]['question']}
Context: {ds['train'][N_examples+1]['context']}
Answer:"""

# print("Loading Mamba-IT from local checkpoint")
# model = MambaForCausalLM.from_pretrained('./model').cuda()
# tokenizer = AutoTokenizer.from_pretrained("Schmadge/mamba-slim-orca")
# tokenizer.eos_token = tokenizer.pad_token = "<|endoftext|>"
# tokenizer.chat_template = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta").chat_template

print("Loading mamba from hf")
model = MambaForCausalLM.from_pretrained('state-spaces/mamba-2.8b-hf').cuda()
tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-2.8b-hf")

# print("Loading RecurrentGemma-9B-IT from HF")
# tokenizer = AutoTokenizer.from_pretrained("google/recurrentgemma-9b-it")
# model = AutoModelForCausalLM.from_pretrained("google/recurrentgemma-9b-it").cuda()

print("collect state for few-shot-prompt")
few_shot_input_ids = tokenizer(few_shot_prompt, return_tensors="pt")["input_ids"].to(device)
icl_out = model(input_ids=few_shot_input_ids, max_new_tokens=0, return_dict=True)
cache_icl = icl_out.cache_params
del icl_out
print("\n" + "*"*50 + "\n")

question_input_ids = tokenizer(question, return_tensors="pt")["input_ids"].to(device)
out_context_query = model.generate(
    input_ids=question_input_ids,
    max_new_tokens=200,
    min_length=50,
    cache_params=copy.copy(cache_icl)
)
out_context_query_str = tokenizer.decode(out_context_query[0]).strip()
print("Output with cache_icl:")
print(out_context_query_str)
print("*"*20)
print(f"Correct answer: {ds['train'][N_examples+1]['answer']}")
print("\n" + "*"*50 + "\n")


out_context_query = model.generate(
    input_ids=question_input_ids,
    max_new_tokens=200,
    min_length=50,
    # cache_params=copy.copy(cache_icl)
)
out_context_query_str = tokenizer.decode(out_context_query[0]).strip()
print("Output without cache_icl:")
print(out_context_query_str)
print("*"*20)
print(f"Correct answer: {ds['train'][N_examples+1]['answer']}")
print("\n" + "*"*50 + "\n")

question_input_ids = tokenizer(few_shot_prompt + question, return_tensors="pt")["input_ids"].to(device)
out_context_query = model.generate(
    input_ids=question_input_ids,
    max_new_tokens=200,
    min_length=50,
    # cache_params=copy.copy(cache_icl)
)
out_context_query_str = tokenizer.decode(out_context_query[0]).strip()
print("Output without icl+query in prompt:")
print(out_context_query_str)
print("*"*20)
print(f"Correct answer: {ds['train'][N_examples+1]['answer']}")
print("\n" + "*"*50 + "\n")