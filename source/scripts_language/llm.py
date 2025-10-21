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


def _window_indices(start_ms: int, end_ms: int, window_ms: int) -> Iterable[int]:
    if window_ms <= 0:
        yield 0
        return
    last_ms = max(end_ms - 1, start_ms)
    start_idx = start_ms // window_ms
    end_idx = last_ms // window_ms
    for idx in range(start_idx, end_idx + 1):
        yield idx


def group_text_by_window(turns: List[dict], window_ms: int) -> Tuple[Dict[int, List[str]], int]:
    grouped: Dict[int, List[str]] = {}
    max_end_ms = 0

    for turn in turns:
        text = str(turn.get("text", "")).strip()
        if not text:
            continue

        start_ms = max(0, _coerce_ms(turn.get("start"), 0))
        end_ms = _coerce_ms(turn.get("end"), start_ms)
        end_ms = max(end_ms, start_ms)

        max_end_ms = max(max_end_ms, end_ms)

        for idx in _window_indices(start_ms, end_ms, window_ms):
            grouped.setdefault(idx, []).append(text)

    return grouped, max_end_ms


def create_summary_prompt(window_start_ms: int, window_end_ms: int, combined_text: str) -> str:
    start_seconds = max(window_start_ms, 0) // 1000
    end_seconds = max(window_end_ms, 0) // 1000
    start_minute, start_second = divmod(start_seconds, 60)
    end_minute, end_second = divmod(end_seconds, 60)
    time_range = f"Time {start_minute:02d}:{start_second:02d}–{end_minute:02d}:{end_second:02d}"
    return (
        "You are a concise psychotherapy note-taker."
        "Provide a short, anonymous English summary (1-2 sentences) of the conversation."
        "Conversation excerpt:\n"
        f"{combined_text}\n"
        "Summary:"
    )


def generate_summary(text_pipe, prompt: str, max_new_tokens: int = 120) -> str:
    outputs = text_pipe(prompt, max_new_tokens=max_new_tokens, return_full_text=False)
    if isinstance(outputs, list) and outputs:
        return str(outputs[0].get("generated_text", "")).strip()
    if isinstance(outputs, dict):
        return str(outputs.get("generated_text", "")).strip()
    return str(outputs).strip()


def process_result_file(result_json: str, config: dict, window_seconds: int, force: bool = False):
    start_time = time.time()
    model_name = config.get("llm_model_name", "google/gemma-3-1b-it")
    device = str(config.get("llm_device", "cpu"))
    result_path = Path(result_json)
    output_path = result_path.parent / f"summaries_{result_path.stem}.json"

    # Skip if summary already exists (unless forced)
    if not force and output_path.exists():
        print(f"[llm] Skip existing summary: {output_path}")
        return

    try:
        device_arg = int(device)
    except ValueError:
        device_arg = 0 if device.lower().startswith("cuda") else -1

    print(f"[llm] Summarising {result_json} using {model_name} on device {device}")
    text_pipe = pipeline("text-generation", model=model_name, device=device_arg)

    with open(result_json, "r", encoding="utf-8") as f:
        turns = json.load(f)

    window_ms = window_seconds * 1000
    grouped_windows, max_end_ms = group_text_by_window(turns, window_ms)
    results: List[Dict[str, int | str]] = []

    for window_index in sorted(grouped_windows):
        combined_text = " ".join(grouped_windows[window_index]).strip()
        if not combined_text:
            continue

        window_start_ms = window_index * window_ms
        default_end = (window_index + 1) * window_ms - 1
        window_end_ms = min(default_end, max_end_ms) if max_end_ms else default_end

        prompt = create_summary_prompt(window_start_ms, window_end_ms, combined_text)
        summary = generate_summary(text_pipe, prompt)

        results.append(
            {
                "window_index": window_index,
                "window_start_ms": window_start_ms,
                "window_end_ms": window_end_ms,
                "summary": summary,
            }
        )

    # Write atomically: write to a temp file then replace the target
    tmp_path = output_path.parent / (output_path.name + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    # os.replace is atomic on most platforms and overwrites if exists
    os.replace(tmp_path, output_path)

    duration_desc = f"{window_seconds}s" if window_seconds < 60 else f"{window_seconds / 60:.1f}min"
    elapsed = time.time() - start_time
    print(f"Summarised {len(results)} windows ({duration_desc}) in {result_json}. Output: {output_path}. Time: {elapsed:.2f}s")


def discover_result_jsons(root: Path, pattern_prefix: str = "results_", pattern_suffix: str = ".json") -> List[Path]:
    if not root.exists():
        print(f"Root output directory not found: {root}")
        return []

    candidates = root.rglob(f"{pattern_prefix}*{pattern_suffix}")
    pending = []
    for path in candidates:
        if not path.is_file() or path.name.startswith("labels_"):
            continue
        summary_path = path.parent / f"summaries_{path.stem}.json"
        if not summary_path.exists():
            pending.append(path)
    return pending


def main():
    parser = argparse.ArgumentParser(description="Summarise psychotherapy session result JSONs using an LLM.")
    parser.add_argument("--config", type=str, default="config_language.yaml", help="Path to config file")
    parser.add_argument("--output_dir", type=str, required=True, help="Root output directory containing per-file subfolders with results_*.json")
    parser.add_argument("--result_jsons", nargs="*", help="Optional explicit list of result JSON files. If omitted, auto-discovers results_*.json recursively under --output_dir.")
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--window_seconds", type=int, default=60, help="Length of each summarisation window in seconds")
    parser.add_argument("--force", action="store_true", help="Recompute summaries even if summaries_*.json already exists")
    args = parser.parse_args()

    config = load_config(args.config) or {}

    requested_device = str(config.get("llm_device", "auto")).lower()
    if requested_device in ("auto", ""):
        config["llm_device"] = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        config["llm_device"] = requested_device

    print(f"[llm] Selected device: {config['llm_device']}")

    root_output = Path(args.output_dir)
    if args.result_jsons:
        all_paths = [Path(p) for p in args.result_jsons]
        if args.force:
            result_jsons = all_paths
        else:
            result_jsons = []
            for p in all_paths:
                summary_path = p.parent / f"summaries_{p.stem}.json"
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

    print(f"Discovered {len(result_jsons)} result files to summarise.")

    with ProcessPoolExecutor(max_workers=args.parallel) as executor:
        futures = [
            executor.submit(process_result_file, str(path), config, args.window_seconds, args.force)
            for path in result_jsons
        ]
        for future in futures:
            future.result()


if __name__ == "__main__":
    main()