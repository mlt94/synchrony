import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from transformers import pipeline

from utils import load_config


def _coerce_ms(value, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip()
    if not text:
        return default
    if text.lstrip("-").isdigit():
        return int(text)
    return default


def group_text_by_speaker_turns(turns: List[dict]) -> List[Dict]:
    """Group consecutive turns by the same speaker into speaker turns.
    
    Returns:
        List of dicts with keys: speaker_id, text, start_ms, end_ms, turn_indices
    """
    if not turns:
        return []
    
    speaker_turns = []
    current_speaker = None
    current_texts = []
    current_start_ms = None
    current_end_ms = None
    current_turn_indices = []
    
    for idx, turn in enumerate(turns):
        text = str(turn.get("text", "")).strip()
        if not text:
            continue
        
        speaker_id = str(turn.get("speaker_id", "unknown")).strip().lower()
        start_ms = max(0, _coerce_ms(turn.get("start"), 0))
        end_ms = max(_coerce_ms(turn.get("end"), start_ms), start_ms)
        
        # If speaker changes or first turn, start a new speaker turn
        if speaker_id != current_speaker:
            # Save previous speaker turn if exists
            if current_speaker is not None and current_texts:
                speaker_turns.append({
                    "speaker_id": current_speaker,
                    "text": " ".join(current_texts),
                    "start_ms": current_start_ms,
                    "end_ms": current_end_ms,
                    "turn_indices": current_turn_indices
                })
            
            # Start new speaker turn
            current_speaker = speaker_id
            current_texts = [text]
            current_start_ms = start_ms
            current_end_ms = end_ms
            current_turn_indices = [idx]
        else:
            # Same speaker, concatenate text
            current_texts.append(text)
            current_end_ms = end_ms  # Update end time to latest
            current_turn_indices.append(idx)
    
    # Don't forget the last speaker turn
    if current_speaker is not None and current_texts:
        speaker_turns.append({
            "speaker_id": current_speaker,
            "text": " ".join(current_texts),
            "start_ms": current_start_ms,
            "end_ms": current_end_ms,
            "turn_indices": current_turn_indices
        })
    
    return speaker_turns


def create_summary_prompt(speaker_id: str, start_ms: int, end_ms: int, combined_text: str) -> str:
    start_seconds = max(start_ms, 0) // 1000
    end_seconds = max(end_ms, 0) // 1000
    start_minute, start_second = divmod(start_seconds, 60)
    end_minute, end_second = divmod(end_seconds, 60)
    time_range = f"Time {start_minute:02d}:{start_second:02d}–{end_minute:02d}:{end_second:02d}"
    
    speaker_label = "Therapist" if speaker_id == "therapist" else "Client" if speaker_id == "client" else speaker_id.title()
    
    return (
        f"You are a concise psychotherapy note-taker. "
        f"Provide a short, anonymous English summary (1-2 sentences) of what the {speaker_label} said.\n"
        f"{speaker_label}'s speech excerpt:\n"
        f"{combined_text}\n"
        f"Summary:"
    )


def generate_summary(text_pipe, prompt: str, max_new_tokens: int = 120) -> str:
    outputs = text_pipe(prompt, max_new_tokens=max_new_tokens, return_full_text=False)
    if isinstance(outputs, list) and outputs:
        return str(outputs[0].get("generated_text", "")).strip()
    if isinstance(outputs, dict):
        return str(outputs.get("generated_text", "")).strip()
    return str(outputs).strip()


def process_result_file(result_json: str, config: dict, force: bool = False):
    start_time = time.time()
    model_name = config.get("llm_model_name", "google/gemma-7b-it")
    device = str(config.get("llm_device", "cpu"))
    result_path = Path(result_json)
    output_path = result_path.parent / f"summaries_speaker_turns_{result_path.stem}.json"

    # Skip if summary already exists (unless forced)
    if not force and output_path.exists():
        print(f"[llm] Skip existing summary: {output_path}")
        return

    # Convert device string to the format expected by transformers pipeline
    # transformers expects: -1 for CPU, 0+ for GPU index, or "cuda"/"cpu" string
    if device.lower() in ("cuda", "gpu"):
        device_arg = 0  # Use first GPU
    elif device.lower() == "cpu":
        device_arg = -1  # Use CPU
    else:
        # Try to parse as integer (GPU index)
        try:
            device_arg = int(device)
        except ValueError:
            print(f"[warning] Invalid device '{device}', falling back to CPU")
            device_arg = -1

    device_name = f"cuda:{device_arg}" if device_arg >= 0 else "cpu"
    print(f"[llm] Summarising {result_json} by speaker turns using {model_name} on {device_name}")
    
    text_pipe = pipeline(
        "text-generation", 
        model=model_name, 
        device=device_arg,
        torch_dtype=torch.float16 if device_arg >= 0 else torch.float32  # Use fp16 on GPU for speed
    )

    with open(result_json, "r", encoding="utf-8") as f:
        turns = json.load(f)

    speaker_turns = group_text_by_speaker_turns(turns)
    print(f"[llm] Processing {len(speaker_turns)} speaker turns from {len(turns)} original turns")
    results: List[Dict[str, int | str | List[int]]] = []

    for idx, speaker_turn in enumerate(speaker_turns):
        combined_text = speaker_turn["text"]
        if not combined_text:
            continue

        speaker_id = speaker_turn["speaker_id"]
        start_ms = speaker_turn["start_ms"]
        end_ms = speaker_turn["end_ms"]
        turn_indices = speaker_turn["turn_indices"]

        turn_start = time.time()
        prompt = create_summary_prompt(speaker_id, start_ms, end_ms, combined_text)
        summary = generate_summary(text_pipe, prompt)
        turn_elapsed = time.time() - turn_start
        
        print(f"[llm]   Turn {idx+1}/{len(speaker_turns)} ({speaker_id}): {turn_elapsed:.1f}s")

        results.append(
            {
                "turn_index": idx,
                "speaker_id": speaker_id,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "original_turn_indices": turn_indices,
                "summary": summary,
            }
        )

    # Write atomically: write to a temp file then replace the target
    tmp_path = output_path.parent / (output_path.name + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    # os.replace is atomic on most platforms and overwrites if exists
    os.replace(tmp_path, output_path)

    elapsed = time.time() - start_time
    print(f"Summarised {len(results)} speaker turns in {result_json}. Output: {output_path}. Time: {elapsed:.2f}s")


def discover_result_jsons(root: Path, pattern_prefix: str = "results_", pattern_suffix: str = ".json") -> List[Path]:
    if not root.exists():
        print(f"Root output directory not found: {root}")
        return []

    candidates = root.rglob(f"{pattern_prefix}*{pattern_suffix}")
    pending = []
    for path in candidates:
        if not path.is_file() or path.name.startswith("labels_"):
            continue
        summary_path = path.parent / f"summaries_speaker_turns_{path.stem}.json"
        if not summary_path.exists():
            pending.append(path)
    return pending


def main():
    parser = argparse.ArgumentParser(description="Summarise psychotherapy session result JSONs by speaker turns using an LLM.")
    parser.add_argument("--config", type=str, default="config_language.yaml", help="Path to config file")
    parser.add_argument("--output_dir", type=str, required=True, help="Root output directory containing per-file subfolders with results_*.json")
    parser.add_argument("--result_jsons", nargs="*", help="Optional explicit list of result JSON files. If omitted, auto-discovers results_*.json recursively under --output_dir.")
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--force", action="store_true", help="Recompute summaries even if summaries_speaker_turns_*.json already exists")
    args = parser.parse_args()

    config = load_config(args.config) or {}

    requested_device = str(config.get("llm_device", "auto")).lower()
    if requested_device in ("auto", ""):
        config["llm_device"] = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        config["llm_device"] = requested_device

    print(f"[llm] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[llm] CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"[llm] Selected device: {config['llm_device']}")
    print(f"[llm] Model: {config.get('llm_model_name', 'google/gemma-7b-it')}")

    root_output = Path(args.output_dir)
    if args.result_jsons:
        all_paths = [Path(p) for p in args.result_jsons]
        if args.force:
            result_jsons = all_paths
        else:
            result_jsons = []
            for p in all_paths:
                summary_path = p.parent / f"summaries_speaker_turns_{p.stem}.json"
                if summary_path.exists():
                    print(f"[llm] Skip (already summarised): {p} -> {summary_path}")
                else:
                    result_jsons.append(p)
    else:
        # Auto-discovery already skips existing summaries
        result_jsons = discover_result_jsons(root_output)
        if not result_jsons:
            print(f"No result JSON files found under {root_output} matching pattern results_*.json")
            return

    print(f"Discovered {len(result_jsons)} result files to summarise by speaker turns.")

    with ProcessPoolExecutor(max_workers=args.parallel) as executor:
        futures = [
            executor.submit(process_result_file, str(path), config, args.force)
            for path in result_jsons
        ]
        for future in futures:
            future.result()


if __name__ == "__main__":
    main()