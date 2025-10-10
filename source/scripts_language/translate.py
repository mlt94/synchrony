import argparse
import json
from pathlib import Path
from typing import List

import torch
from transformers import pipeline

from utils import load_config


DEFAULT_MODEL_NAME = "google/gemma-3-1b-it"

LANG_LABELS = {
    "de": "German",
    "deu": "German",
    "deu_latn": "German",
    "ger": "German",
    "german": "German",
    "en": "English",
    "eng": "English",
    "eng_latn": "English",
    "english": "English",
}


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


def load_translation_pipeline(config: dict):
    model_name = config.get("translation_model_name") or config.get("llm_model_name") or DEFAULT_MODEL_NAME
    hf_token = config.get("hf_token")
    device_pref = config.get("translation_device", config.get("llm_device", "auto"))
    device_name, device_arg = resolve_device(device_pref)
    print(f"[translate] Loading translation model '{model_name}' on {device_name} (pipeline device arg={device_arg})")

    pipe_kwargs = {
        "task": "text-generation",
        "model": model_name,
        "device": device_arg,
    }
    if device_arg != -1 and torch.cuda.is_available():
        pipe_kwargs["torch_dtype"] = torch.float16
    if hf_token:
        pipe_kwargs["token"] = hf_token

    text_pipe = pipeline(**pipe_kwargs)
    return text_pipe, device_name


def normalise_language_label(label: str | None, fallback: str) -> str:
    if not label:
        return fallback
    lookup = label.strip().lower()
    return LANG_LABELS.get(lookup, label)


def create_translation_prompt(text: str, source_lang: str, target_lang: str) -> list:
    return [
        {
            "role": "system",
            "content": (
                "You are a professional translator. Always translate the user's text into {}. "
                "Return only the translated text without additional commentary."
            ).format(target_lang or "English"),
        },
        {
            "role": "user",
            "content": (
                "Source language: {}. Translate this text to {} and preserve the meaning faithfully:\n{}"
            ).format(source_lang or "German", target_lang or "English", text),
        },
    ]


def extract_generated_text(outputs) -> str:
    try:
        if isinstance(outputs, list) and outputs:
            first = outputs[0]
            if isinstance(first, dict):
                generated = first.get("generated_text")
                if isinstance(generated, list):
                    for message in reversed(generated):
                        if isinstance(message, dict) and "content" in message:
                            content = message["content"].strip()
                            if content:
                                return content
                elif isinstance(generated, str):
                    return generated.strip()
    except Exception:
        pass
    return str(outputs).strip()


def translate_file(
    input_path: Path,
    output_path: Path,
    text_pipe,
    source_lang_label: str,
    target_lang_label: str,
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
            messages = create_translation_prompt(text, source_lang_label, target_lang_label)
            try:
                outputs = text_pipe(messages, max_new_tokens=max_new_tokens or 256, temperature=0.1, top_p=0.9)
                translated_text = extract_generated_text(outputs)
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
    parser = argparse.ArgumentParser(description="Translate ASR results from German to English.")
    parser.add_argument(
        "--config",
        type=str,
        default="config_language.yaml",
        help="Path to config file with translation settings.",
    )
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

    config = load_config(args.config) or {}

    text_pipe, device_name = load_translation_pipeline(config)
    source_lang_label = normalise_language_label(config.get("translation_source_lang"), "German")
    target_lang_label = normalise_language_label(config.get("translation_target_lang"), "English")

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
        relative_parent = input_path.parent.relative_to(input_root)
        if output_root:
            target_dir = output_root / relative_parent
        else:
            target_dir = input_path.parent

        output_name = input_path.name
        if output_name.startswith("results_"):
            output_name = f"translated_{output_name}"
        else:
            output_name = f"translated_{output_name}"
        output_path = target_dir / output_name

        if output_path.exists() and not args.overwrite:
            print(f"[translate] Skipping existing {output_path}")
            continue

        translate_file(
            input_path,
            output_path,
            text_pipe,
            source_lang_label,
            target_lang_label,
            max_new_tokens=args.max_new_tokens,
        )


if __name__ == "__main__":
    main()

