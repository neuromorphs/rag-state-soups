from datasets import load_dataset
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, RecurrentGemmaForCausalLM

model_id = "google/recurrentgemma-9b-it"

model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir="/data/hf-cache", device_map="auto")
model = model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="/data/hf-cache")

recurrent_states = np.load("embeddings-recurrentgemma-9b-it.npz")["embeddings"]
def set_recurrent_state(dataset_idx: int):
    def hook(layer, input):
        layer.recurrent_states = torch.from_numpy(recurrent_states[dataset_idx]).unsqueeze(0).to(layer.recurrent_states.device, torch.float32)
    return hook

dataset = load_dataset("google-research-datasets/natural_questions", "dev")["validation"]

with torch.no_grad():
    for i, example in enumerate(dataset):
        # run with the RAG-state
        handle = model.model.layers[21].temporal_block.rg_lru.register_forward_pre_hook(set_recurrent_state(i))
        question = example["question"]["text"]
        chat = [
            {"role": "user", "content": question},
        ]
        inputs = tokenizer.apply_chat_template(chat, tokenize=False)
        inputs = tokenizer(inputs, add_special_tokens=False, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=150)
        str_out = tokenizer.decode(outputs[0])
        handle.remove()
        # run without the RAG-state
        outputs = model.generate(**inputs, max_new_tokens=150)
        raw_str_out = tokenizer.decode(outputs[0])
        continue

    
# ds['validation'][0]['question']['text']
# , ds['validation'][0]['long_answer_candidates']