'''This is the script that generates the rationales.
As we cannot move the data from the workstation upon which it is stored
we are limited to an RTX 4070 with 12GB VRAM. That forces the reasoning generation step to use smallest model possible
in this case a Llama 3.1. 8B with 4-bit quantization.'''
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,  
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,   
    device_map="auto",                
)
messages = [
    {
        "role": "system", 
        "content": "You are a helpful data analyst. Your task is to provide a brief, step-by-step rationale for why a time series snippet is associated with a given label. Focus on the data to justify the conclusion."
    },
    {
        "role": "user", 
        "content": "Please provide the rationale for the following: \n\n**Snippet:** [10.1, 10.3, 9.9, 10.2, 35.8, 10.1, 9.8] \n**Label:** 'Spike Anomaly'"
    }
]


input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True, 
    return_tensors="pt"
).to(model.device)


terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = model.generate(
    input_ids,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)


response = outputs[0][input_ids.shape[-1]:]
print(tokenizer.decode(response, skip_special_tokens=True))
