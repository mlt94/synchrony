import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import json
import time
from transformers import pipeline
from typing import List
import torch
from utils import load_config


def create_prompt(text: str) -> list:
    """
    Create prompt messages for the LLM labeling task.
    """
    return [
        {"role": "system", "content": "You are an expert psychologist. Please annotate the following german psychotherapy sequences according to one of three categories, negative, neutral or positive. Please only output the annotation category."},
        {"role": "user", "content": text}
    ]

def validate_label(label: str) -> str:
    """
    Ensure label is one of the expected categories.
    """
    valid = {"negative", "neutral", "positive"}
    label = label.strip().lower()
    for v in valid:
        if v in label:
            return v
    return "unknown"

def process_result_file(result_json: str, config: dict):
    start = time.time()
    model_name = config.get("llm_model_name", "google/gemma-3-1b-it")
    device = str(config.get("llm_device", "cpu"))
    result_path = Path(result_json)
    output_path = result_path.parent / f"labels_{result_path.stem}.json"
    try:
        try:
            device_arg = int(device)
        except ValueError:
            device_arg = 0 if device.lower().startswith("cuda") else -1
        print(f"[llm] Labeling {result_json} using {model_name} on device {device} (pipeline arg={device_arg})")
        pipe = pipeline("text-generation", model=model_name, device=device_arg)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    try:
        with open(result_json, 'r', encoding='utf-8') as f:
            turns = json.load(f)
    except Exception as e:
        print(f"Error reading input file {result_json}: {e}")
        return

    labeled_count = 0
    for idx, speech_turn in enumerate(turns):
        messages = create_prompt(speech_turn["text"])
        try:
            outputs = pipe(messages, max_new_tokens=50)
            # Try to robustly retrieve generated text across model variants
            label_raw = None
            try:
                # Typical chat pipeline output: [{"generated_text": [{"role":..., "content": "..."}, ...]}]
                if isinstance(outputs, list) and outputs:
                    first = outputs[0]
                    if isinstance(first, dict) and "generated_text" in first:
                        gen = first["generated_text"]
                        if isinstance(gen, list) and gen and isinstance(gen[-1], dict) and "content" in gen[-1]:
                            label_raw = gen[-1]["content"]
                        elif isinstance(gen, str):
                            label_raw = gen
            except Exception:
                pass
            if label_raw is None:
                label_raw = str(outputs)

            label = validate_label(label_raw)
            turns[idx]["label"] = label
            labeled_count += 1
        except Exception as e:
            print(f"Error labeling turn {idx} in {result_json}: {e}")
            turns[idx]["label"] = "error"

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(turns, f, indent=4, ensure_ascii=False)
        print(f"Labeled {labeled_count} segments in {result_json}. Output: {output_path}. Time: {time.time() - start:.2f}s")
    except Exception as e:
        print(f"Error writing output file {output_path}: {e}")



def discover_result_jsons(root: Path, pattern_prefix: str = "results_", pattern_suffix: str = ".json") -> List[Path]:
    """
    Recursively discover result JSON files in subfolders of root that match pattern results_*.json
    and are not already labeled (exclude files starting with labels_).
    """
    files = []
    if not root.exists():
        print(f"Root output directory not found: {root}")
        return files
    for p in root.rglob(f"{pattern_prefix}*{pattern_suffix}"):
        if p.is_file() and not p.name.startswith("labels_"):
            files.append(p)
    return files

def main():
    parser = argparse.ArgumentParser(description="Label psychotherapy session result JSONs using an LLM.")
    parser.add_argument("--config", type=str, default="config_language.yaml", help="Path to config file")
    parser.add_argument("--output_dir", type=str, required=True, help="Root output directory containing per-file subfolders with results_*.json")
    parser.add_argument("--result_jsons", nargs="*", help="Optional explicit list of result JSON files. If omitted, auto-discovers results_*.json recursively under --output_dir.")
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel workers")
    args = parser.parse_args()

    config = load_config(args.config) or {}

    device = config.get("llm_device", "auto")
    if str(device).lower() in ("auto", ""):
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    config["llm_device"] = device
    print(f"[llm] Selected device: {device}")

    root_output = Path(args.output_dir)
    if args.result_jsons and len(args.result_jsons) > 0:
        result_jsons = [Path(f) for f in args.result_jsons]
    else:
        result_jsons = discover_result_jsons(root_output)
        if not result_jsons:
            print(f"No result JSON files found under {root_output} matching pattern results_*.json")
            return

    print(f"Discovered {len(result_jsons)} result files to label.")

    with ProcessPoolExecutor(max_workers=args.parallel) as executor:
        futures = [executor.submit(process_result_file, str(f), config) for f in result_jsons]
        for fut in futures:
            fut.result()

if __name__ == "__main__":
    main()