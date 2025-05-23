from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import json
import torch

from huggingface_hub import login
login(token=)
import time
from utils import extract_speech_turns_from_file
from IPython import embed
'''This is the script that takes as input the transcribed session and outputs some classification pr. sequence according to some inventory'''

def llm():
    pipe = pipeline("text-generation", model="google/gemma-3-1b-it", device="cpu", torch_dtype=torch.bfloat16)
    source_path = "/mnt/c/users/mlut/OneDrive - ITU/DESKTOP/sync/synchrony/files/results.json"
    with open(source_path, 'r', encoding='utf-8') as f:
        turns = json.load(f)

    for idx, speech_turn in enumerate(turns):
        messages = [{"role": "system", "content": "You are an expert psychologist. Please annotate the following german psychotherapy sequences according to one of three categories, negative, neutral or positive. Please only output the annotation category."},
                {"role" : "user", "content": turns[idx]["text"]}]    
        outputs = pipe(messages, max_new_tokens=50)
        label = outputs[0]["generated_text"][-1]["content"]
        turns[idx]["label"] = label

    output_path = "/mnt/c/users/mlut/OneDrive - ITU/DESKTOP/sync/synchrony/files/results_labels.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(turns, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    start = time.time()
    llm()
    print(f"Execution time: {time.time() - start:.2f} seconds")