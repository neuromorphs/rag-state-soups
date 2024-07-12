import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, RecurrentGemmaForCausalLM

model_id = "google/recurrentgemma-9b-it"

recurrent_states = {}
def get_recurrent_state(layer_idx: int):
    def hook(layer, input, output):
        recurrent_states[layer_idx] = layer.recurrent_states.detach().cpu().numpy()
    return hook

model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir="/data/hf-cache", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="/data/hf-cache")

for layer_idx, layer in enumerate(model.model.layers):
    try:
        rg_lru = getattr(layer.temporal_block, "rg_lru")
        rg_lru.register_forward_hook(get_recurrent_state(layer_idx))
    except AttributeError:
        recurrent_states[layer_idx] = None

with torch.no_grad():
    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    outputs = model(**inputs, return_dict=True)

print(recurrent_states)
