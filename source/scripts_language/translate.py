import argparse
import json
from pathlib import Path
from typing import List

import torch
from transformers import pipeline


DEFAULT_MODEL_NAME = "Helsinki-NLP/opus-mt-de-en"


def discover_result_jsons(root: Path) -> List[Path]:
    return sorted(root.rglob("results_*.json"))


def resolve_device(preferred: str | int) -> tuple[str, int]:
    device_pref = str(preferred)
    if device_pref.lower() in ("auto", ""):
        try:
            device_pref = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device_pref = "cpu"
    try:
        device_arg = int(device_pref)
        device_name = f"cuda:{device_arg}" if device_arg >= 0 else "cpu"
        return device_name, device_arg
    except ValueError:
        device_name = device_pref
        device_arg = 0 if device_pref.lower().startswith("cuda") else -1
        return device_name, device_arg


def load_translation_pipeline(device: str = "auto"):
    device_name, device_arg = resolve_device(device)
    print(f"[translate] Loading translation model '{DEFAULT_MODEL_NAME}' on {device_name} (pipeline device arg={device_arg})")

    pipe_kwargs = {
        "task": "translation",
        "model": DEFAULT_MODEL_NAME,
        "device": device_arg
    }
    if device_arg != -1 and torch.cuda.is_available():
        pipe_kwargs["torch_dtype"] = torch.float16

    text_pipe = pipeline(**pipe_kwargs)
    return text_pipe, device_name


def translate_file(
    input_path: Path,
    output_path: Path,
    text_pipe,
    max_new_tokens: int | None,
):
    with open(input_path, "r", encoding="utf-8") as handle:
        records = json.load(handle)

    if not isinstance(records, list):
        raise ValueError(f"Expected list in {input_path}, found {type(records).__name__}")

    translated_records = []
    for idx, entry in enumerate(records):
        text = (entry.get("text") or "").strip()
        if not text:
            translated_text = ""
        else:
            try:
                # Helsinki models expect plain text input and return [{'translation_text': '...'}]
                outputs = text_pipe(text, max_length=max_new_tokens or 512)
                if isinstance(outputs, list) and outputs:
                    translated_text = outputs[0].get("translation_text", text).strip()
                else:
                    translated_text = text
            except Exception as exc:
                print(f"[translate] Error translating segment {idx} in {input_path}: {exc}")
                translated_text = text

        translated_records.append(
            {
                "text": translated_text,
                "start": entry.get("start"),
                "end": entry.get("end"),
                "speaker_id": entry.get("speaker_id"),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(translated_records, handle, ensure_ascii=False, indent=4)
    print(f"[translate] Wrote {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Translate ASR results from German to English using Helsinki-NLP/opus-mt-de-en.")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing results_*.json files (recursively scanned).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to write translated JSON files. Defaults to the same location as inputs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use for translation: 'auto', 'cuda', 'cpu', or device index (default: auto).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=None,
        help="Optional cap for generated tokens per segment.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing translated files instead of skipping them.",
    )
    args = parser.parse_args()

    text_pipe, device_name = load_translation_pipeline(args.device)

    input_root = Path(args.input_dir).resolve()
    if not input_root.exists():
        raise FileNotFoundError(f"Input directory {input_root} does not exist")

    output_root = Path(args.output_dir).resolve() if args.output_dir else None

    result_files = discover_result_jsons(input_root)
    if not result_files:
        print(f"[translate] No results_*.json files found in {input_root}")
        return

    print(f"[translate] Found {len(result_files)} files to translate")

    for input_path in result_files:
        # Use the subdirectory name (parent folder name) as the ID
        file_id = input_path.parent.name
        
        # Output translate_<id>.json in the same directory as the source results_*.json
        if output_root:
            relative_parent = input_path.parent.relative_to(input_root)
            target_dir = output_root / relative_parent
        else:
            target_dir = input_path.parent

        output_path = target_dir / f"translate_{file_id}.json"

        if output_path.exists() and not args.overwrite:
            print(f"[translate] Skipping existing {output_path}")
            continue

        translate_file(
            input_path,
            output_path,
            text_pipe,
            max_new_tokens=args.max_new_tokens,
        )


if __name__ == "__main__":
    main()

