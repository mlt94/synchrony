from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from IPython import embed

from utils import extract_speech_turns_from_file
'''This is the script that takes as input the transcribed session and outputs some classification pr. sequence according to some inventory'''

checkpoint = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint)

source_path = "/mnt/c/users/mlut/OneDrive - ITU/DESKTOP/sync/synchrony/files/results.json"
turns = extract_speech_turns_from_file(source_path)

messages = [{"role": "system", "content": "You are an expert psychologist. Please annotate the following german psychotherapy sequences according to one of three categories, negative, neutral or positive. Please only output the annotation category."},
            {"role" : "user", "content": turns[0]["text"]}]

input_text = tokenizer.apply_chat_template(messages, tokenize=False)
inputs = tokenizer.encode(input_text, return_tensors="pt")
outputs = model.generate(inputs, max_new_tokens=50, temperature=0.2, top_p=0.9, do_sample=True, return_dict_in_generate=True)
assistant_start = outputs.rfind('<|im_start|>assistant') + len('<|im_start|>assistant\n')
assistant_end = outputs.rfind('<|im_end|>')
label = outputs[assistant_start:assistant_end].strip()


