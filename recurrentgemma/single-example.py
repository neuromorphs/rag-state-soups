from datasets import load_dataset
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, RecurrentGemmaForCausalLM


model_id = "google/recurrentgemma-9b-it"
cache_dir = "./data/hf-cache"
token = "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"

tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir, device_map="auto", token=token)
model = model.eval()

context_text = "Context: Owen likes small experts."
context_text = "Context: Owen likes nuclear weapons."
query_text = "What does Owen like?"

input_ids = tokenizer(context_text, return_tensors="pt").to("cuda") #, padding=True, truncation=True)
outputs = model(**input_ids)
rec_states = {}
for layer_idx, layer in enumerate(model.model.layers):
    if hasattr(layer.temporal_block, "rg_lru"):
        rec_states[layer_idx] = layer.temporal_block.rg_lru.recurrent_states.detach().cpu()

# input_ids = tokenizer(query_text, return_tensors="pt").to("cuda") #, padding=True, truncation=True)
chat = [
    {"role": "user", "content": query_text},
]
inputs = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
print(inputs)
input_ids = tokenizer(inputs, add_special_tokens=False, return_tensors="pt").to("cuda")


print("unmodified query generate")
outputs = model.generate(**input_ids, max_new_tokens=50)
str_out = tokenizer.decode(outputs[0])
print(f'unmodified generation: "{str_out}"')



# did_set_state_once = False
layer_set = {layer_idx: False for layer_idx in rec_states.keys()}
def set_recurrent_state(rec_states: torch.Tensor, layer_idx: int):
    def hook(module, input):
        if module.recurrent_states.data.sum() == 0.0:
            module.recurrent_states.data = rec_states.to("cuda") * 100
            print("steering")
        else:
            print("not steering")
        # if not layer_set[layer_idx]:
        #     # if not did_set_state_once:
        #     # print("entered state setting hook")
        #     # print(module.recurrent_states)
        #     module.recurrent_states.data = rec_states.to("cuda") * 100
        #     # print(module.recurrent_states)
        #     # did_set_state_once = True
        #     layer_set[layer_idx] = True
        #     if layer_idx == 0:
        #         print("steering")
        # else:
        #     if layer_idx == 0:
        #         print("not steering")
    return hook

print("adding state setting hook")
for layer_idx, layer in enumerate(model.model.layers):
    if hasattr(layer.temporal_block, "rg_lru"):
        layer.temporal_block.rg_lru.register_forward_pre_hook(set_recurrent_state(rec_states[layer_idx], layer_idx))
print("modified query generate")
outputs = model.generate(**input_ids, max_new_tokens=50)
str_out = tokenizer.decode(outputs[0])
print(f'modified generation: "{str_out}"')
