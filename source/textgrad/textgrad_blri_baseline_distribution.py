import argparse
import json
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


INTERVIEW_TO_TIMEPOINT = {
    "bindung": "T5",
    "personal": "T3",
    "wunder": "T7",
}


@dataclass
class InterviewSample:
    patient_id: str
    interview_type: str
    snippet_text: str
    blri_score: float
    blri_bin: str


@dataclass
class BLRIBinConfig:
    poor_max: float
    reasonable_max: float
    good_max: float


class LocalHFEngine:
    system_prompt: str = "You are a helpful, concise assistant."

    def __init__(
        self,
        model_name: str,
        torch_dtype: str = "auto",
        device_map: str = "auto",
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 0.95,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        resolved_dtype = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[torch_dtype]

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=resolved_dtype,
                device_map=device_map,
                low_cpu_mem_usage=True,
                trust_remote_code=False,
            )
        except torch.cuda.OutOfMemoryError as exc:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise RuntimeError(
                f"CUDA OOM while loading model '{model_name}'. Try a smaller model or lower precision."
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"Failed to load model/tokenizer '{model_name}': {exc}") from exc

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        try:
            self.input_device = next(self.model.parameters()).device
        except StopIteration:
            self.input_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _format_messages(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        sys_prompt = system_prompt if system_prompt else self.system_prompt
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return f"System: {sys_prompt}\nUser: {prompt}\nAssistant:"

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        final_prompt = self._format_messages(prompt=prompt, system_prompt=system_prompt)
        inputs = self.tokenizer(final_prompt, return_tensors="pt").to(self.input_device)

        try:
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    do_sample=self.temperature > 0,
                    temperature=self.temperature if self.temperature > 0 else None,
                    top_p=self.top_p,
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
        except torch.cuda.OutOfMemoryError as exc:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise RuntimeError(
                "CUDA OOM during generation. Reduce max_new_tokens, use smaller model, or shorten input."
            ) from exc

        generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def is_finite_number(x) -> bool:
    try:
        v = float(x)
    except Exception:
        return False
    return math.isfinite(v)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BLRI LLM baseline (no optimization)")
    parser.add_argument("--data_model", type=Path, default=Path("data_model.yaml"))
    parser.add_argument("--combined_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("source/textgrad/outputs_baseline_llm"))
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--torch_dtype", choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--gpu_ids", type=str, default=None)
    parser.add_argument("--max_turns_per_interview", type=int, default=80)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--train_fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--baseline_mode", choices=["two_stage", "direct"], default="two_stage")
    return parser.parse_args()


def _percentile(sorted_values: List[float], percentile: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot compute percentile on empty data.")
    if len(sorted_values) == 1:
        return sorted_values[0]
    idx = (len(sorted_values) - 1) * percentile
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_values[lo]
    frac = idx - lo
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac


def infer_balanced_blri_bins(scores: List[float]) -> BLRIBinConfig:
    if not scores:
        return BLRIBinConfig(poor_max=5.0, reasonable_max=14.0, good_max=24.0)
    ordered = sorted(scores)
    q1 = _percentile(ordered, 0.25)
    q2 = _percentile(ordered, 0.50)
    q3 = _percentile(ordered, 0.75)
    return BLRIBinConfig(poor_max=q1, reasonable_max=q2, good_max=q3)


def blri_score_to_bin(score: float, bin_config: BLRIBinConfig) -> str:
    if score <= bin_config.poor_max:
        return "poor"
    if score <= bin_config.reasonable_max:
        return "reasonable"
    if score <= bin_config.good_max:
        return "good"
    return "very_good"


def format_bin_rules(bin_config: BLRIBinConfig) -> str:
    return (
        f"poor <= {bin_config.poor_max:.2f}, "
        f"reasonable <= {bin_config.reasonable_max:.2f}, "
        f"good <= {bin_config.good_max:.2f}, "
        "very_good above good threshold"
    )


def parse_predicted_score_and_bin(text: str, bin_config: BLRIBinConfig) -> Tuple[Optional[float], Optional[str]]:
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not match:
        return None, None
    score = float(match.group(0))
    return score, blri_score_to_bin(score, bin_config)


def load_data_model(yaml_path: Path) -> Dict:
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError as exc:
        raise RuntimeError(f"data_model file not found: {yaml_path}") from exc
    except OSError as exc:
        raise RuntimeError(f"Failed to read data_model file '{yaml_path}': {exc}") from exc
    except yaml.YAMLError as exc:
        raise RuntimeError(f"Invalid YAML in '{yaml_path}': {exc}") from exc


def collect_blri_scores(data_model: Dict) -> List[float]:
    scores: List[float] = []
    interviews = data_model.get("interviews", [])
    for interview in interviews:
        type_block = interview.get("types", {})
        for interview_type, cfg in type_block.items():
            if interview_type not in INTERVIEW_TO_TIMEPOINT:
                continue
            label_block = cfg.get("labels", {})
            if not isinstance(label_block, dict):
                continue

            prefix = INTERVIEW_TO_TIMEPOINT[interview_type]
            for key, value in label_block.items():
                if key.startswith(prefix) and key.endswith("BLRI_ges_Pr") and is_finite_number(value):
                    scores.append(float(value))
                    break
    return scores


def extract_blri_labels(data_model: Dict, bin_config: BLRIBinConfig) -> Dict[Tuple[str, str], Tuple[float, str]]:
    labels: Dict[Tuple[str, str], Tuple[float, str]] = {}
    interviews = data_model.get("interviews", [])
    for interview in interviews:
        patient_id = interview.get("patient", {}).get("patient_id")
        if not patient_id:
            continue

        type_block = interview.get("types", {})
        for interview_type, cfg in type_block.items():
            if interview_type not in INTERVIEW_TO_TIMEPOINT:
                continue
            label_block = cfg.get("labels", {})
            if not isinstance(label_block, dict):
                continue

            target_key = None
            prefix = INTERVIEW_TO_TIMEPOINT[interview_type]
            for key in label_block.keys():
                if key.startswith(prefix) and key.endswith("BLRI_ges_Pr"):
                    target_key = key
                    break

            if target_key is None:
                continue

            score_raw = label_block.get(target_key)
            if not is_finite_number(score_raw):
                continue

            score = float(score_raw)
            labels[(patient_id, interview_type)] = (score, blri_score_to_bin(score, bin_config))
    return labels


def build_interview_text(entries: List[Dict], max_turns: int) -> str:
    sorted_entries = sorted(entries, key=lambda x: x.get("turn_index", 10**9))
    if max_turns is not None:
        sorted_entries = sorted_entries[:max_turns]

    lines: List[str] = []
    for row in sorted_entries:
        text = (row.get("combined_description") or "").strip()
        if not text:
            continue
        speaker = row.get("speaker_id", "unknown")
        idx = row.get("turn_index", "?")
        lines.append(f"turn {idx} | {speaker}: {text}")
    return "\n".join(lines)


def load_combined_samples(combined_dir: Path, max_turns_per_interview: int) -> Dict[Tuple[str, str], str]:
    grouped: Dict[Tuple[str, str], List[Dict]] = {}
    for file in sorted(combined_dir.glob("*.json")):
        try:
            with open(file, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except OSError as exc:
            raise RuntimeError(f"Failed to read combined file '{file}': {exc}") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON in combined file '{file}': {exc}") from exc

        if not isinstance(payload, list):
            continue

        for row in payload:
            patient_id = row.get("patient_id")
            interview_type = row.get("interview_type")
            if not patient_id or not interview_type:
                continue
            grouped.setdefault((patient_id, interview_type), []).append(row)

    merged: Dict[Tuple[str, str], str] = {}
    for key, entries in grouped.items():
        text = build_interview_text(entries, max_turns=max_turns_per_interview)
        if text.strip():
            merged[key] = text
    return merged


def make_dataset(
    labels: Dict[Tuple[str, str], Tuple[float, str]],
    interview_texts: Dict[Tuple[str, str], str],
    max_samples: Optional[int],
    seed: int,
) -> List[InterviewSample]:
    keys = sorted(set(labels.keys()) & set(interview_texts.keys()))
    dataset: List[InterviewSample] = []
    for patient_id, interview_type in keys:
        score, blri_bin = labels[(patient_id, interview_type)]
        text = interview_texts[(patient_id, interview_type)]
        dataset.append(
            InterviewSample(
                patient_id=patient_id,
                interview_type=interview_type,
                snippet_text=text,
                blri_score=score,
                blri_bin=blri_bin,
            )
        )

    random.Random(seed).shuffle(dataset)
    if max_samples is not None:
        dataset = dataset[:max_samples]
    return dataset


def split_dataset(dataset: List[InterviewSample], train_fraction: float) -> Tuple[List[InterviewSample], List[InterviewSample]]:
    if len(dataset) < 2:
        return dataset, []
    split_idx = max(1, min(len(dataset) - 1, int(len(dataset) * train_fraction)))
    return dataset[:split_idx], dataset[split_idx:]


def create_summary_prompt(snippet_text: str) -> str:
    return (
        "Task: Condense the psychotherapy conversation state into exactly 3 words.\n"
        "BLRI_ges_Pr is a CLIENT self-report measure of how much empathy the client felt from the therapist during this conversation.\n"
        "Use neutral, clinically grounded wording and no punctuation list format.\n"
        "Instruction to follow:\n"
        "Read the snippets and produce a 2-3 word state summary that captures client-perceived therapist empathy cues.\n\n"
        "Conversation snippets:\n"
        f"{snippet_text}\n\n"
        "Return only the 2-3 word summary."
    )


def create_prediction_prompt(summary: str, bin_config: BLRIBinConfig) -> str:
    return (
        "Given this 2-3 word psychotherapy conversation summary, estimate BLRI_ges_Pr score.\n"
        "BLRI_ges_Pr is the client's self-report of therapist empathy as perceived by the client during the conversation.\n"
        "Output strictly in this format: SCORE=<number>; BIN=<poor|reasonable|good|very_good>\n"
        "Scoring instruction to follow:\n"
        "Estimate the BLRI_ges_Pr score from the summary as client-perceived therapist empathy, and ensure BIN matches the numeric score threshold.\n\n"
        "Summary:\n"
        f"{summary}\n"
        f"Remember bin rules: {format_bin_rules(bin_config)}."
    )


def create_direct_prediction_prompt(snippet_text: str, bin_config: BLRIBinConfig) -> str:
    return (
        "Given the psychotherapy conversation snippets, estimate BLRI_ges_Pr score directly.\n"
        "BLRI_ges_Pr is the client's self-report of therapist empathy as perceived by the client during the conversation.\n"
        "Output strictly in this format: SCORE=<number>; BIN=<poor|reasonable|good|very_good>\n"
        "Scoring instruction to follow:\n"
        "Infer client-perceived therapist empathy from the full snippets and ensure BIN matches the numeric score threshold.\n\n"
        "Conversation snippets:\n"
        f"{snippet_text}\n"
        f"Remember bin rules: {format_bin_rules(bin_config)}."
    )


def run_llm_baseline(
    engine: LocalHFEngine,
    eval_samples: List[InterviewSample],
    bin_config: BLRIBinConfig,
    baseline_mode: str,
) -> Tuple[List[Dict], Dict[str, float]]:
    rows: List[Dict] = []
    parsed = 0
    correct = 0
    abs_errors: List[float] = []

    for sample in eval_samples:
        summary_output: Optional[str] = None
        if baseline_mode == "direct":
            prediction_output = engine.generate(create_direct_prediction_prompt(sample.snippet_text, bin_config))
        else:
            summary_output = engine.generate(create_summary_prompt(sample.snippet_text))
            prediction_output = engine.generate(create_prediction_prompt(summary_output, bin_config))
        pred_score, pred_bin = parse_predicted_score_and_bin(prediction_output, bin_config)

        if pred_bin is not None:
            parsed += 1
        if pred_bin == sample.blri_bin:
            correct += 1
        if pred_score is not None:
            abs_errors.append(abs(pred_score - sample.blri_score))

        rows.append(
            {
                "patient_id": sample.patient_id,
                "interview_type": sample.interview_type,
                "target_score": sample.blri_score,
                "target_bin": sample.blri_bin,
                "baseline_mode": baseline_mode,
                "summary_output": summary_output,
                "prediction_output": prediction_output,
                "parsed_pred_score": pred_score,
                "parsed_pred_bin": pred_bin,
            }
        )

    total = len(eval_samples)
    metrics = {
        "eval_samples": total,
        "parsed_predictions": parsed,
        "bin_accuracy": (correct / total) if total > 0 else 0.0,
        "score_mae": (sum(abs_errors) / len(abs_errors)) if abs_errors else None,
    }
    return rows, metrics


def write_outputs(
    output_dir: Path,
    bin_config: BLRIBinConfig,
    dataset: List[InterviewSample],
    train_samples: List[InterviewSample],
    eval_samples: List[InterviewSample],
    eval_rows: List[Dict],
    metrics: Dict,
):
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "baseline_eval_predictions.json", "w", encoding="utf-8") as f:
            json.dump(eval_rows, f, indent=2, ensure_ascii=False)

        with open(output_dir / "baseline_metrics.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "dataset_size": len(dataset),
                    "train_size": len(train_samples),
                    "eval_size": len(eval_samples),
                    "bin_thresholds": {
                        "poor_max": bin_config.poor_max,
                        "reasonable_max": bin_config.reasonable_max,
                        "good_max": bin_config.good_max,
                    },
                    **metrics,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
    except OSError as exc:
        raise RuntimeError(f"Failed writing outputs to '{output_dir}': {exc}") from exc


def main():
    args = parse_args()

    if args.gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    data_model = load_data_model(args.data_model)
    all_scores = collect_blri_scores(data_model)
    bin_config = infer_balanced_blri_bins(all_scores)
    labels = extract_blri_labels(data_model, bin_config)

    interview_texts = load_combined_samples(
        args.combined_dir,
        max_turns_per_interview=args.max_turns_per_interview,
    )

    dataset = make_dataset(
        labels=labels,
        interview_texts=interview_texts,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    if not dataset:
        raise RuntimeError("No usable samples found. Check --combined_dir and data_model label coverage.")

    train_samples, eval_samples = split_dataset(dataset, train_fraction=args.train_fraction)

    print(f"Loaded dataset size: {len(dataset)}")
    print(f"Train samples: {len(train_samples)}")
    print(f"Eval samples: {len(eval_samples)}")
    print(f"Model: {args.model_name}")
    print(f"Device map: {args.device_map}")
    print(f"Baseline mode: {args.baseline_mode}")
    print(
        "BLRI bin thresholds: "
        f"poor<= {bin_config.poor_max:.2f}, "
        f"reasonable<= {bin_config.reasonable_max:.2f}, "
        f"good<= {bin_config.good_max:.2f}, "
        "very_good above good threshold"
    )

    engine = LocalHFEngine(
        model_name=args.model_name,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    eval_rows, metrics = run_llm_baseline(
        engine=engine,
        eval_samples=eval_samples,
        bin_config=bin_config,
        baseline_mode=args.baseline_mode,
    )

    write_outputs(
        output_dir=args.output_dir,
        bin_config=bin_config,
        dataset=dataset,
        train_samples=train_samples,
        eval_samples=eval_samples,
        eval_rows=eval_rows,
        metrics=metrics,
    )

    print(f"Baseline parsed predictions: {metrics['parsed_predictions']}/{metrics['eval_samples']}")
    print(f"Baseline bin accuracy: {metrics['bin_accuracy']:.4f}")
    if metrics["score_mae"] is not None:
        print(f"Baseline score MAE: {metrics['score_mae']:.4f}")
    print("Done.")
    print(f"Predictions saved to: {args.output_dir / 'baseline_eval_predictions.json'}")
    print(f"Metrics saved to: {args.output_dir / 'baseline_metrics.json'}")


if __name__ == "__main__":
    main()
