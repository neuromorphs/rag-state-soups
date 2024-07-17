"""Script to collect Mamba states for the neural-bridge/rag-dataset-12000 dataset."""
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import MambaConfig, MambaForCausalLM
from transformers.models.mamba.convert_mamba_ssm_checkpoint_to_pytorch import convert_mamba_checkpoint_file_to_huggingface_model_file
from tqdm import tqdm
import numpy as np
import os
import torch


tmp_model_folder = "./tmp"
model_folder = "./hfmodel"
data_folder = "./data/"

MAX_CONTEXT_LENGTH = None

os.makedirs(tmp_model_folder, exist_ok=True)
os.makedirs(model_folder, exist_ok=True)
os.makedirs(data_folder, exist_ok=True)

device = "cuda"

try:
  # try to load model from checkpoint
  print("try to load Mamba model from local checkpoint")
  model = MambaForCausalLM.from_pretrained(model_folder).cuda()
except:
  print("could not find local Mamba model. loading remote model")
  from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
  # load mamba_ssm checkpoint
  model = MambaLMHeadModel.from_pretrained("Schmadge/mamba-slim-orca", device=device, dtype=torch.float16)
  model.save_pretrained(tmp_model_folder)

  print("converting remote model to local model...")
  # convert checkpoint from mamba_ssm to HF
  convert_mamba_checkpoint_file_to_huggingface_model_file(
    f'{tmp_model_folder}/pytorch_model.bin', 
    f'{tmp_model_folder}/config.json',
    model_folder,
  )
  # free up memory
  del model
  torch.cuda.empty_cache()
  # setup model
  print("loading local model...")
  model = MambaForCausalLM.from_pretrained(model_folder).cuda()

# setup tokenizer
tokenizer = AutoTokenizer.from_pretrained("Schmadge/mamba-slim-orca")
tokenizer.eos_token = tokenizer.pad_token = "<|endoftext|>"
tokenizer.chat_template = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta").chat_template

# setup dataset
ds = load_dataset("neural-bridge/rag-dataset-12000")

for sample_idx in tqdm(range(0, ds["train"].num_rows)):
  filepath = os.path.join(data_folder, f"context_{sample_idx}.npz")
  question = ds["train"]["question"][sample_idx]
  context = ds["train"]["context"][sample_idx]
  answer = ds["train"]["answer"][sample_idx]

  context = context[:MAX_CONTEXT_LENGTH]

  if not os.path.exists(filepath):
    # forward pass through the model to collect the states
    prompt = [{"role": "system", "content": context}]
    input_ids = tokenizer.apply_chat_template(prompt, return_tensors="pt", add_generation_prompt=True).to(device)
    out = model(input_ids=input_ids, return_dict=True)

    # postprocess cache (conv_states and ssm_states)
    out.cache_params.conv_states = {k: v.detach().cpu().numpy() for k, v in out.cache_params.conv_states.items()}
    out.cache_params.ssm_states = {k: v.detach().cpu().numpy() for k, v in out.cache_params.ssm_states.items()}
    cache_context = out.cache_params.__dict__

    # save the context cache
    np.savez_compressed(filepath, **cache_context)

    # # load these files again with:
    # x = np.load(f"{data_folder}/context_13.npz", allow_pickle=True)
    # print(x['seqlen_offset'], x['dtype'])
    # conv_states = x['conv_states'][()]
    # ssm_states = x['ssm_states'][()]
    # print(conv_states[0].shape, ssm_states[0].shape)