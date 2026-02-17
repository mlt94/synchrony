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

import textgrad as tg
from textgrad.engine.base import CachedEngine, EngineLM


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
    blri_bin: str


@dataclass
class BLRIBinConfig:
    poor_max: float
    reasonable_max: float
    good_max: float


class LocalHFEngine(EngineLM, CachedEngine):
    system_prompt: str = "You are a helpful, concise assistant."

    def __init__(
        self,
        model_name: str,
        cache_path: str,
        torch_dtype: str = "auto",
        device_map: str = "auto",
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 0.95,
    ):
        super().__init__(cache_path=cache_path)
        self.model_string = model_name
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
                f"CUDA OOM while loading model '{model_name}'. "
                "Try smaller model, lower precision, or different device_map."
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

    def generate(self, prompt, system_prompt=None, temperature=None, max_tokens=None, top_p=None):
        local_temperature = self.temperature if temperature is None else temperature
        local_max_tokens = self.max_new_tokens if max_tokens is None else max_tokens
        local_top_p = self.top_p if top_p is None else top_p

        final_prompt = self._format_messages(prompt=prompt, system_prompt=system_prompt)
        cache_key = f"{self.model_string}||{final_prompt}||{local_temperature}||{local_max_tokens}||{local_top_p}"
        cached = self._check_cache(cache_key)
        if cached is not None:
            return cached

        inputs = self.tokenizer(final_prompt, return_tensors="pt").to(self.input_device)
        try:
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    do_sample=local_temperature > 0,
                    temperature=local_temperature if local_temperature > 0 else None,
                    top_p=local_top_p,
                    max_new_tokens=local_max_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
        except torch.cuda.OutOfMemoryError as exc:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise RuntimeError(
                "CUDA OOM during generation. Reduce max_new_tokens, use smaller model, "
                "or reduce batch/input size."
            ) from exc

        generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        self._save_cache(cache_key, text)
        return text

    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)


def is_finite_number(x) -> bool:
    try:
        v = float(x)
    except Exception:
        return False
    return math.isfinite(v)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TextGrad BLRI pipeline with open-source LLM")
    parser.add_argument("--data_model", type=Path, default=Path("data_model.yaml"))
    parser.add_argument("--combined_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("source/textgrad/outputs"))
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--torch_dtype", choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--gpu_ids", type=str, default=None)
    parser.add_argument("--max_turns_per_interview", type=int, default=80)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--train_fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--verbose", type=int, default=0)
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


def parse_predicted_bin(text: str) -> Optional[str]:
    match = re.search(r"\b(poor|reasonable|good|very_good)\b", text.strip(), flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).lower()


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


def extract_blri_labels(data_model: Dict, bin_config: BLRIBinConfig) -> Dict[Tuple[str, str], str]:
    labels: Dict[Tuple[str, str], str] = {}
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
            b = blri_score_to_bin(score, bin_config)

            labels[(patient_id, interview_type)] = b
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
    labels: Dict[Tuple[str, str], str],
    interview_texts: Dict[Tuple[str, str], str],
    max_samples: Optional[int],
    seed: int,
) -> List[InterviewSample]:
    keys = sorted(set(labels.keys()) & set(interview_texts.keys()))
    dataset: List[InterviewSample] = []
    for patient_id, interview_type in keys:
        blri_bin = labels[(patient_id, interview_type)]
        text = interview_texts[(patient_id, interview_type)]
        dataset.append(
            InterviewSample(
                patient_id=patient_id,
                interview_type=interview_type,
                snippet_text=text,
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


def build_engine(args: argparse.Namespace) -> EngineLM:
    cache_dir = args.output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = str(cache_dir / "hf_engine_cache.db")
    return LocalHFEngine(
        model_name=args.model_name,
        cache_path=cache_path,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )


def create_bin_prediction_output(
    classifier: tg.BlackboxLLM,
    classifier_instruction_var: tg.Variable,
    sample: InterviewSample,
    bin_config: BLRIBinConfig,
) -> tg.Variable:
    prefix = tg.Variable(
        "Task: Predict BLRI_ges_Pr bin directly from psychotherapy conversation snippets.\n"
        "BLRI_ges_Pr is the client's self-report of therapist empathy as perceived by the client during the conversation.\n"
        "Output strictly and only one token from: poor, reasonable, good, very_good.\n"
        "Instruction to follow:\n",
        requires_grad=False,
        role_description="bin prediction prompt prefix",
    )
    mid = tg.Variable(
        "\n\nConversation snippets:\n",
        requires_grad=False,
        role_description="bin prediction prompt mid",
    )
    suffix = tg.Variable(
        f"\n\nBin rules: {format_bin_rules(bin_config)}.\nReturn only the bin label.",
        requires_grad=False,
        role_description="bin prediction prompt suffix",
    )
    snippet_var = tg.Variable(
        sample.snippet_text,
        requires_grad=False,
        role_description="conversation snippets",
    )
    prediction_input = prefix + classifier_instruction_var + mid + snippet_var + suffix
    return classifier(prediction_input)


def create_loss(
    evaluator: tg.TextLoss,
    prediction_output_var: tg.Variable,
    target_bin: str,
) -> tg.Variable:
    start = tg.Variable(
        f"Target bin: {target_bin}\n"
        "Reminder: BLRI_ges_Pr is the client's self-report of how empathic the therapist felt to the client.\n"
        "Model predicted bin:\n",
        requires_grad=False,
        role_description="loss preamble",
    )
    end = tg.Variable(
        "\nEvaluate prediction quality. Focus feedback on how to improve the bin-classification instruction so the future predicted bin matches target bin.",
        requires_grad=False,
        role_description="loss suffix",
    )
    evaluation_input = start + prediction_output_var + end
    return evaluator(evaluation_input)


def run_training(
    engine: EngineLM,
    train_samples: List[InterviewSample],
    bin_config: BLRIBinConfig,
    args: argparse.Namespace,
) -> Tuple[tg.Variable, List[Dict]]:
    tg.set_backward_engine(engine)
    classifier = tg.BlackboxLLM(engine=engine)

    evaluator = tg.TextLoss(
        eval_system_prompt=(
            "You are a strict evaluator for empathy (BLRI_ges_Pr) prediction. "
            "BLRI_ges_Pr is the client's self-report of perceived therapist empathy during the conversation. "
            "Provide concise, specific textual feedback for improving the bin-classification instruction. "
            "Emphasize errors that lead to wrong bins."
        ),
        engine=engine,
    )

    classifier_instruction_var = tg.Variable(
        value=(
            "Infer the client's perceived therapist empathy from the snippets and output exactly one bin label: poor, reasonable, good, or very_good."
        ),
        requires_grad=True,
        role_description="instruction for direct BLRI bin classification",
    )

    optimizer = tg.TextualGradientDescent(
        parameters=[classifier_instruction_var],
        engine=engine,
        constraints=[
            "Keep instruction concise.",
            "Ensure output remains strictly one label: poor, reasonable, good, or very_good.",
        ],
        verbose=args.verbose,
    )

    trace: List[Dict] = []
    for epoch in range(args.epochs):
        random.shuffle(train_samples)
        for step, sample in enumerate(train_samples):
            prediction_output = create_bin_prediction_output(
                classifier,
                classifier_instruction_var,
                sample,
                bin_config,
            )
            loss = create_loss(evaluator, prediction_output, sample.blri_bin)

            optimizer.zero_grad()
            try:
                loss.backward()
                optimizer.step()
            except torch.cuda.OutOfMemoryError as exc:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise RuntimeError(
                    "CUDA OOM during TextGrad optimization. "
                    "Try reducing max_turns_per_interview/max_samples or use a smaller model."
                ) from exc

            pred_bin = parse_predicted_bin(prediction_output.value)
            trace.append(
                {
                    "epoch": epoch,
                    "step": step,
                    "patient_id": sample.patient_id,
                    "interview_type": sample.interview_type,
                    "target_bin": sample.blri_bin,
                    "prediction_output": prediction_output.value,
                    "parsed_pred_bin": pred_bin,
                    "updated_classifier_instruction": classifier_instruction_var.value,
                }
            )

    return classifier_instruction_var, trace


def run_eval(
    engine: EngineLM,
    classifier_instruction_var: tg.Variable,
    bin_config: BLRIBinConfig,
    eval_samples: List[InterviewSample],
) -> Tuple[List[Dict], Dict[str, float]]:
    tg.set_backward_engine(engine)
    classifier = tg.BlackboxLLM(engine=engine)

    rows: List[Dict] = []
    correct = 0
    parsed = 0
    for sample in eval_samples:
        prediction_output = create_bin_prediction_output(
            classifier,
            classifier_instruction_var,
            sample,
            bin_config,
        )
        pred_bin = parse_predicted_bin(prediction_output.value)
        if pred_bin is not None:
            parsed += 1
        if pred_bin == sample.blri_bin:
            correct += 1

        rows.append(
            {
                "patient_id": sample.patient_id,
                "interview_type": sample.interview_type,
                "target_bin": sample.blri_bin,
                "prediction_output": prediction_output.value,
                "parsed_pred_bin": pred_bin,
            }
        )

    total = len(eval_samples)
    metrics = {
        "eval_samples": total,
        "parsed_predictions": parsed,
        "bin_accuracy": (correct / total) if total > 0 else 0.0,
    }
    return rows, metrics


def write_outputs(
    output_dir: Path,
    bin_config: BLRIBinConfig,
    dataset: List[InterviewSample],
    train_samples: List[InterviewSample],
    eval_samples: List[InterviewSample],
    classifier_instruction_var: tg.Variable,
    train_trace: List[Dict],
    eval_rows: List[Dict],
    eval_metrics: Dict[str, float],
):
    try:
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "optimized_classifier_instruction.txt", "w", encoding="utf-8") as f:
            f.write(classifier_instruction_var.value.strip() + "\n")

        with open(output_dir / "train_trace.jsonl", "w", encoding="utf-8") as f:
            for row in train_trace:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        with open(output_dir / "eval_predictions.json", "w", encoding="utf-8") as f:
            json.dump(eval_rows, f, indent=2, ensure_ascii=False)

        with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
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
                    **eval_metrics,
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
    print(
        "BLRI bin thresholds: "
        f"poor<= {bin_config.poor_max:.2f}, "
        f"reasonable<= {bin_config.reasonable_max:.2f}, "
        f"good<= {bin_config.good_max:.2f}, "
        "very_good above good threshold"
    )
    if args.gpu_ids:
        print(f"CUDA_VISIBLE_DEVICES={args.gpu_ids}")

    engine = build_engine(args)

    classifier_instruction_var, train_trace = run_training(
        engine=engine,
        train_samples=train_samples,
        bin_config=bin_config,
        args=args,
    )

    eval_rows, eval_metrics = run_eval(
        engine=engine,
        classifier_instruction_var=classifier_instruction_var,
        bin_config=bin_config,
        eval_samples=eval_samples,
    )

    write_outputs(
        output_dir=args.output_dir,
        bin_config=bin_config,
        dataset=dataset,
        train_samples=train_samples,
        eval_samples=eval_samples,
        classifier_instruction_var=classifier_instruction_var,
        train_trace=train_trace,
        eval_rows=eval_rows,
        eval_metrics=eval_metrics,
    )

    print("Done.")
    print(f"Optimized instruction saved to: {args.output_dir / 'optimized_classifier_instruction.txt'}")
    print(f"Metrics saved to: {args.output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
