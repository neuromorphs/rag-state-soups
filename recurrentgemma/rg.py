from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM


tokenizer = AutoTokenizer.from_pretrained("google/recurrentgemma-9b-it")
model = AutoModelForCausalLM.from_pretrained("google/recurrentgemma-9b-it").cuda()

prompt_str = [
    # {"role": "system", "content": ""},  # TODO: add context here
    {"role": "user", "content": ""},  # TODO: add question here
]
prompt = tokenizer.apply_chat_template(
    prompt_str, return_tensors="pt", add_generation_prompt=False
).cuda()
model_out = model.generate(
    input_ids=prompt,
    max_new_tokens=100,
    min_length=50,
)
model_out_str = tokenizer.decode(model_out[0]).strip()
