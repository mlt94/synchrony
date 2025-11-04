"""Test if the model can generate text at all with a simple prompt."""

import sys
import os
import torch
import numpy as np
from huggingface_hub import hf_hub_download

# Add both the parent directory (opentslm) and src/ to path
opentslm_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_dir = os.path.join(opentslm_dir, "src")
sys.path.insert(0, opentslm_dir)
sys.path.insert(0, src_dir)

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from src.model.encoder.CNNTokenizer import CNNTokenizer
from src.prompt.full_prompt import FullPrompt
from src.prompt.text_prompt import TextPrompt
from src.prompt.text_time_series_prompt import TextTimeSeriesPrompt
from src.time_series_datasets.psychotherapy.psychotherapyCoTQADataset import PsychotherapyCoTQADataset
from src.model.llm.OpenTSLMSP import OpenTSLMSP
import matplotlib.pyplot as plt

from PIL import Image
import requests
from io import BytesIO

#own
dataset = PsychotherapyCoTQADataset(split="train", EOS_TOKEN=";", max_samples=1, feature_columns=['AU04_r', "AU15_r", "AU6_r", "AU12_r", "AU1_r", "AU7_r"])
pre_prompt = dataset._get_pre_prompt(dataset[0])
post_prompt = dataset._get_post_prompt(dataset[0])
train_raw, _,_ = dataset._load_splits()
ts_prompt = dataset._get_text_time_series_prompt_list(train_raw[0])

# Plot all 6 AUs on a single plot with legend
num_aus = 6
au_names = ['AU04_r', "AU15_r", "AU6_r", "AU12_r", "AU1_r", "AU7_r"]

# Create a single plot
fig, ax = plt.subplots(figsize=(14, 6))

colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#BC4B51']
line_styles = ['-', '-', '-', '-', '-', '-']

# Plot all therapist AUs on the same axes
for i in range(num_aus):
    ts = ts_prompt[i].get_time_series()  # Therapist AU
    ax.plot(ts, linewidth=2, color=colors[i], linestyle=line_styles[i], 
            label=f"Therapist {au_names[i]}", alpha=0.8)

ax.set_title("Therapist Facial Action Units Over Time", fontsize=14, pad=15, fontweight='bold')
ax.set_xlabel("Time (frames)", fontsize=11)
ax.set_ylabel("Activation", fontsize=11)
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
ax.tick_params(labelsize=9)

plt.tight_layout()
plt.savefig("/home/mlut/synchrony/.garbage/ts_plot.jpg", dpi=150, bbox_inches='tight')
plt.close()

print(f"Number of prompts: {len(ts_prompt)}")
print(f"Plotting all {num_aus} AUs (therapist) on single plot")
print(f"Pre-prompt: {pre_prompt}")


# --- Setup ---
# 1. Define the model ID for an instruction-tuned, multimodal version of Gemma 3.
#    The 'it' (Instruction Tuned) and '4b' (4 billion parameters) or '27b' are good options.
MODEL_ID = "google/gemma-3-27b-it"

# 2. Check for GPU availability and set dtype for performance (bfloat16 is standard for modern GPUs)
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32

# 3. Initialize the multimodal pipeline
#    The task for VLM is "image-text-to-text".
print(f"Loading model {MODEL_ID} on {device}...")
pipe = pipeline(
    "image-text-to-text",
    model=MODEL_ID,
    device=device,
    torch_dtype=torch_dtype
)

# --- Define Inputs ---
# 4. Define the image source. This can be a local file path or a public URL.
#    (For simplicity and reproducibility, a URL is used here.)
IMAGE_URL = "/home/mlut/synchrony/.garbage/ts_plot.jpg"

# 5. Define the multimodal prompt using the standard 'messages' format.
#    The content is a list containing both the image and the text.
messages = [
    {
        "role": "user",
        "content": [
            # The image object is passed using a dict with "type": "image" and a "url" or a PIL Image
            {"type": "image", "url": IMAGE_URL}, 
            # The text question is passed using a dict with "type": "text"
            {"type": "text", "text": pre_prompt}
        ]
    }
]

# --- Run Inference ---
print("\nRunning inference...")
output = pipe(
    text=messages, 
    max_new_tokens=85,  # Reduced from 200 to force brevity
    do_sample=False,  # Greedy decoding for more focused output
    temperature=0.1  # Very low temperature for deterministic, concise responses
)

# --- Print Result ---
# The pipeline returns the full conversation, so we extract the model's final response.
# The structure is output[0]["generated_text"][-1]["content"]
model_response = output[0]["generated_text"][-1]["content"]

print("\n--- Model Response ---")
print(model_response)

