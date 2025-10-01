import json
import os
from itertools import groupby
from typing import List, Dict
import yaml
import torch

def extract_speech_turns_from_file(json_path: str) -> List[Dict]:
    """
    Extracts speech turns from a JSON file, grouping consecutive words by speaker.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    speech_turns = []
    for speaker, group in groupby(results, key=lambda w: w["speaker_id"]):
        group = list(group)
        speech_turns.append({
            "speaker_id": speaker,
            "text": " ".join(w["text"].strip() for w in group),
            "start": group[0]["start"],
            "end": group[-1]["end"]
        })
    return speech_turns

def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        # Dynamically set the device if "auto" is specified
        if config.get("device", "auto") == "auto":
            config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        if config.get("llm_device", "auto") == "auto":
            config["llm_device"] = "cuda" if torch.cuda.is_available() else "cpu"
        # Resolve HF token
        hf_token = config.get("hf_token", "auto")
        if hf_token == "auto" or hf_token is None:
            hf_token = (
                os.getenv("HUGGINGFACE_HUB_TOKEN")
                or os.getenv("HF_TOKEN")
                or os.getenv("HUGGINGFACE_TOKEN")
            )
        config["hf_token"] = hf_token
        return config
