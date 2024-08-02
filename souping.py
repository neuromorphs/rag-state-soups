from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import MambaConfig, MambaForCausalLM
from transformers.models.mamba.convert_mamba_ssm_checkpoint_to_pytorch import convert_mamba_checkpoint_file_to_huggingface_model_file
from tqdm import tqdm
import numpy as np
import os
import torch


def convert_mamba_ssm_to_hf_local(
    model_folder: str,
    model_id: str = None,
    device: str = "cuda",
    tmp_model_folder: str = "./tmp",
):
    # NOTE: use this function e.g. for model_id=Schmadge/mamba-slim-orca
    try:
        print("try to load Mamba model from local checkpoint")
        model = MambaForCausalLM.from_pretrained(model_folder).cuda()
    except:
        print("could not find local Mamba model. loading remote model")
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
        model = MambaLMHeadModel.from_pretrained(model_id, device=device, dtype=torch.float16)
        model.save_pretrained(tmp_model_folder)

        print("converting remote model to local model...")
        convert_mamba_checkpoint_file_to_huggingface_model_file(
            f'{tmp_model_folder}/pytorch_model.bin', 
            f'{tmp_model_folder}/config.json',
            model_folder,
        )

        # free up memory
        del model
        torch.cuda.empty_cache()

        print("loading local model...")
        model = MambaForCausalLM.from_pretrained(model_folder).cuda()

    return model


def collect_states(
    model, tokenizer, ds, data_folder="./data/", 
    max_context_length=None, device="cuda"
):
    """Script to collect Mamba states for the neural-bridge/rag-dataset-12000 dataset."""
    for sample_idx in tqdm(range(0, ds["train"].num_rows)):
        filepath = os.path.join(data_folder, f"context_{sample_idx}.npz")
        question = ds["train"]["question"][sample_idx]
        context = ds["train"]["context"][sample_idx]
        answer = ds["train"]["answer"][sample_idx]
        context = context[:max_context_length]

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


def collect_mamba_it_states(ds):
    model_folder = "mamba-it-hf"
    tmp_model_folder = "mamba-it-tmp"
    os.makedirs(tmp_model_folder, exist_ok=True)
    os.makedirs(model_folder, exist_ok=True)
    model = convert_mamba_ssm_to_hf_local(model_folder, "Schmadge/mamba-slim-orca")
    tokenizer = AutoTokenizer.from_pretrained("Schmadge/mamba-slim-orca")
    tokenizer.eos_token = tokenizer.pad_token = "<|endoftext|>"
    tokenizer.chat_template = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta").chat_template
    collect_states(model, tokenizer, ds)
