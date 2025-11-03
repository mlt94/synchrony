"""Test if the model can generate text at all with a simple prompt."""

import sys
import os
import torch
import numpy as np
from huggingface_hub import hf_hub_download
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "opentslm", "src")))

from transformers import AutoTokenizer, AutoModelForCausalLM
from model.encoder.CNNTokenizer import CNNTokenizer
from prompt.full_prompt import FullPrompt
from prompt.text_prompt import TextPrompt

from prompt.text_time_series_prompt import TextTimeSeriesPrompt
from time_series_datasets.psychotherapy.psychotherapyCoTQADataset import PsychotherapyCoTQADataset
from model.llm.OpenTSLMSP import OpenTSLMSP


print("=" * 60)
print("TESTING SIMPLE GENERATION")
print("=" * 60)

# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
BASE_LLM_ID = "meta-llama/Llama-3.2-3B" 

CHECKPOINT_REPO_ID = "OpenTSLM/llama-3.2-3b-har-sp"
CHECKPOINT_FILENAME = "softprompt-llama_3_2_3b-har.pt"

try:
    print(f"Initializing model architecture using base: {BASE_LLM_ID}...")
    model = OpenTSLMSP(
        device=device,
        llm_id=BASE_LLM_ID
    )
    print("Model architecture built.")
    
    # Override encoder with correct max_patches to match checkpoint
    # The checkpoint was trained with max_patches=1024
    from model.encoder.TransformerCNNEncoder import TransformerCNNEncoder
    print("Replacing encoder with max_patches=1024 to match checkpoint...")
    model.encoder = TransformerCNNEncoder(max_patches=1024).to(device)
    
    # Enable LoRA before loading checkpoint (checkpoint contains LoRA adapters)
    print("Enabling LoRA...")
    model.enable_lora()

    print(f"Downloading checkpoint from {CHECKPOINT_REPO_ID}...")
    checkpoint_path = hf_hub_download(
        repo_id=CHECKPOINT_REPO_ID,
        filename=CHECKPOINT_FILENAME
    )

    model.load_from_file(checkpoint_path)

    model.eval() 


except Exception as e:
    print(f"\n❌ An error occurred: {e}")

#own
dataset = PsychotherapyCoTQADataset(split="train", EOS_TOKEN=";", max_samples=1, feature_columns=['AU06_r'])
pre_prompt = dataset._get_pre_prompt(dataset[0])
post_prompt = dataset._get_post_prompt(dataset[0])
train_raw, _,_ = dataset._load_splits()
ts_prompt = dataset._get_text_time_series_prompt_list(train_raw[0])
prompt = FullPrompt(TextPrompt(pre_prompt), [ts_prompt[0]], TextPrompt(post_prompt))

# Test with different max_new_tokens values
test_configs = [
    {"max_new_tokens": 500, "name": "500 tokens"},
]
print("Sample keys:", dataset[0].keys())
print("Patient ID:", dataset[0].get("patient_id"))
print("Therapist ID:", dataset[0].get("therapist_id"))
print("Interview type:", dataset[0].get("interview_type"))

for config in test_configs:
    print("=" * 60)
    print(f"Testing with {config['name']}")
    print("=" * 60)
    
    with torch.no_grad():
        output = model.eval_prompt(prompt, max_new_tokens=config['max_new_tokens'])
    
    print(f"Output length: {len(output)} characters")
    print(f"Output: '{output}'")

print("=" * 60)
print("Output ended")
print("=" * 60)


prompt = prompt.to_dict()
print(prompt["pre_prompt"])
print(prompt["post_prompt"])



