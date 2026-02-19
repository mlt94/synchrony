"""
Hierarchical multimodal mixed-outcome trainer with LOTO-only evaluation.

Key behavior:
- Uses therapist leave-one-out (LOTO) folds only.
- Supports mixed targets discovered from `data_model.yaml`:
    - numeric -> regression
    - categorical/text -> multiclass classification
- Uses all transcript turns in each session (no therapist/client role filtering).
- Uses both patient and therapist AU descriptions per turn when available.
- Aggregates cumulative sessions by target timepoint (T3/T5/T7), and uses
    all available sessions for baseline targets.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import confusion_matrix, f1_score
from transformers import AutoTokenizer

try:
    from source.modeling.dataset import _is_nan, _resolve_path, load_au_descriptions
    from source.modeling.model import HMANClassifier, HMANRegressor, SessionEncoder, TurnAttention, TurnEncoder
    from source.modeling.multisession_eval_common import (
        PatientRecord,
        SessionFeatureRow,
        build_patient_records,
        compute_reg_metrics,
        discover_target_specs,
        required_types_for_target,
    )
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from source.modeling.dataset import _is_nan, _resolve_path, load_au_descriptions
    from source.modeling.model import HMANClassifier, HMANRegressor, SessionEncoder, TurnAttention, TurnEncoder
    from source.modeling.multisession_eval_common import (
        PatientRecord,
        SessionFeatureRow,
        build_patient_records,
        compute_reg_metrics,
        discover_target_specs,
        required_types_for_target,
    )


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stderr,
)
log = logging.getLogger("hman_mixed_loto")


INTERVIEW_PREFIX: dict[str, str] = {
    "bindung": "T5",
    "personal": "T3",
    "wunder": "T7",
}

AU_R_COLS: list[str] = [
    "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r",
    "AU09_r", "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r",
    "AU20_r", "AU23_r", "AU25_r", "AU26_r", "AU45_r",
]
N_AUS: int = len(AU_R_COLS)
N_AU_NUMERIC: int = N_AUS * 2  # patient + therapist mean AU vectors concatenated


@dataclass
class TurnInfo:
    start_s: float
    end_s: float
    speech_summarised: str
    speech_unsummarised: str
    au_text: str
    au_numbers: str
    au_numeric_vec: np.ndarray = field(default_factory=lambda: np.zeros(N_AUS * 2, dtype=np.float32))


@dataclass
class SessionData:
    split: str
    patient_id: str
    therapist_id: str
    interview_type: str
    turns: list[TurnInfo] = field(default_factory=list)
    labels: dict[str, float] = field(default_factory=dict)


@dataclass
class PatientData:
    split: str
    patient_id: str
    therapist_id: str
    sessions: dict[str, SessionData] = field(default_factory=dict)
    labels_by_type: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class Sample:
    therapist_id: str
    target_value: float | int
    turns: list[tuple[str, str, np.ndarray]]


@dataclass
class UnsTurn:
    start_s: float
    end_s: float
    text: str


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_target_mapping(records: list[PatientRecord]) -> tuple[list[str], list[str], dict[str, list[str]]]:
    specs = discover_target_specs(records)
    reg_targets = [s.name for s in specs if s.task == "regression"]
    clf_targets = [s.name for s in specs if s.task == "classification"]
    clf_classes = {s.name: s.classes for s in specs if s.task == "classification"}
    return reg_targets, clf_targets, clf_classes


def get_target_value(rec: PatientData, target_key: str) -> Any:
    if target_key.startswith("T3_"):
        return rec.labels_by_type.get("personal", {}).get(target_key)
    if target_key.startswith("T5_"):
        return rec.labels_by_type.get("bindung", {}).get(target_key)
    if target_key.startswith("T7_"):
        return rec.labels_by_type.get("wunder", {}).get(target_key)

    for d in rec.labels_by_type.values():
        if target_key in d:
            return d[target_key]
    return None


def _safe_text(x: str) -> str:
    x = (x or "").strip()
    return x if x else "[UNK]"


def _strip_known_prefixes(name: str) -> str:
    out = name
    changed = True
    while changed:
        changed = False
        for prefix in (
            "summaries_speaker_turns_",
            "summaries_",
            "speaker_turns_",
            "translate_",
            "results_",
            "unsummarised_",
            "unsummarized_",
        ):
            if out.startswith(prefix):
                out = out[len(prefix):]
                changed = True
    return out


def _session_id_candidates(stem: str) -> set[str]:
    tokens = {_strip_known_prefixes(stem), stem}
    expanded: set[str] = set(tokens)
    for token in list(tokens):
        expanded.add(_strip_known_prefixes(token))
        if token.startswith("results_"):
            expanded.add(token[len("results_"):])
        if token.startswith("translate_"):
            expanded.add(token[len("translate_"):])
    return {x for x in expanded if x}


def _parse_unsummarised_records(records: Any) -> list[UnsTurn]:
    if not isinstance(records, list):
        return []

    out: list[UnsTurn] = []
    for row in records:
        if not isinstance(row, dict):
            continue

        txt = ""
        for key in ("text", "snippet", "utterance", "content", "transcript"):
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                txt = value.strip()
                break
        if not txt:
            continue

        start_ms = row.get("start_ms", row.get("start", 0))
        end_ms = row.get("end_ms", row.get("end", start_ms))
        try:
            start_s = float(start_ms) / 1000.0
            end_s = float(end_ms) / 1000.0
        except (TypeError, ValueError):
            continue

        if end_s < start_s:
            start_s, end_s = end_s, start_s

        out.append(UnsTurn(start_s=start_s, end_s=end_s, text=txt))

    out.sort(key=lambda x: x.start_s)
    return out


def build_unsummarised_index(path: Path) -> dict[str, list[UnsTurn]]:
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted(path.rglob("*.json"))
    else:
        raise FileNotFoundError(f"Unsummarised path does not exist: {path}")

    index: dict[str, list[UnsTurn]] = {}
    loaded = 0
    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception:
            continue

        turns = _parse_unsummarised_records(data)
        if not turns:
            continue

        loaded += 1
        candidates = _session_id_candidates(file_path.stem)
        candidates.update(_session_id_candidates(file_path.parent.name))

        for key in candidates:
            previous = index.get(key)
            if previous is None or len(turns) > len(previous):
                index[key] = turns

    log.info("Unsummarised index: loaded %d files, %d session keys", loaded, len(index))
    return index


def _resolve_unsummarised_turns(transcript_path: Path, uns_index: dict[str, list[UnsTurn]]) -> list[UnsTurn]:
    candidates = _session_id_candidates(transcript_path.stem)
    candidates.update(_session_id_candidates(transcript_path.parent.name))
    for candidate in candidates:
        if candidate in uns_index:
            return uns_index[candidate]
    return []


def _compose_unsummarised_texts(turn_bounds: list[tuple[float, float]], uns_turns: list[UnsTurn]) -> list[str]:
    if not turn_bounds:
        return []
    if not uns_turns:
        return ["[UNK]"] * len(turn_bounds)

    output: list[str] = []
    for start_s, end_s in turn_bounds:
        parts = [u.text for u in uns_turns if u.end_s > start_s and u.start_s < end_s]
        output.append(_safe_text(" ".join(parts).strip()))
    return output


def load_openface_au(path: Path, confidence_threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    import pandas as pd

    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    mask = (df["success"] == 1) & (df["confidence"] >= confidence_threshold)
    aus = df.loc[mask, AU_R_COLS].values.astype(np.float32)
    timestamps = df.loc[mask, "timestamp"].values.astype(np.float64)
    return aus, timestamps


def segment_au_by_turns(aus: np.ndarray, timestamps: np.ndarray, turn_bounds: list[tuple[float, float]]) -> list[np.ndarray]:
    segments: list[np.ndarray] = []
    for start_s, end_s in turn_bounds:
        mask = (timestamps >= start_s) & (timestamps < end_s)
        seg = aus[mask]
        if seg.shape[0] == 0:
            seg = np.zeros((0, N_AUS), dtype=np.float32)
        segments.append(seg)
    return segments


def _format_au_segment(seg: np.ndarray) -> str:
    if seg.shape[0] == 0:
        return "[UNK]"
    mean_values = seg.mean(axis=0)
    parts = [f"{name}={float(val):.3f}" for name, val in zip(AU_R_COLS, mean_values)]
    return " ".join(parts)


def load_sessions(
    data_model_path: Path,
    config_path: Path,
    au_descriptions_dir: Path,
    unsummarised_text_path: Path,
    split: str,
    openface_confidence_threshold: float,
) -> list[SessionData]:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    allowed = set(config["psychotherapy_splits"][split])

    with open(data_model_path, "r", encoding="utf-8") as f:
        data_model = yaml.safe_load(f)

    au_index = load_au_descriptions(au_descriptions_dir)
    uns_index = build_unsummarised_index(unsummarised_text_path)

    sessions: list[SessionData] = []
    skipped_no_transcript = 0

    for interview in data_model["interviews"]:
        therapist_id = interview["therapist"]["therapist_id"]
        if therapist_id not in allowed:
            continue

        patient_id = interview["patient"]["patient_id"]

        for itype, idata in interview.get("types", {}).items():
            if itype not in INTERVIEW_PREFIX:
                continue

            labels_raw = idata.get("labels", {})
            labels: dict[str, float] = {}
            for key, value in labels_raw.items():
                if _is_nan(value):
                    continue
                try:
                    labels[key] = float(value)
                except (TypeError, ValueError):
                    labels[key] = value

            transcript_raw = idata.get("transcript", "")
            transcript_path = _resolve_path(transcript_raw) if transcript_raw else None
            if transcript_path is None or not transcript_path.exists():
                skipped_no_transcript += 1
                continue

            try:
                with open(transcript_path, "r", encoding="utf-8") as f:
                    turns_data = json.load(f)
            except Exception:
                skipped_no_transcript += 1
                continue

            if not isinstance(turns_data, list) or len(turns_data) == 0:
                skipped_no_transcript += 1
                continue

            turns_data = sorted(turns_data, key=lambda t: t.get("turn_index", 0))
            turn_meta: list[tuple[float, float, str, str]] = []
            for turn in turns_data:
                tidx = turn.get("turn_index")
                try:
                    tidx_i = int(tidx)
                except (TypeError, ValueError):
                    continue

                start_s = float(turn.get("start_ms", 0)) / 1000.0
                end_s = float(turn.get("end_ms", 0)) / 1000.0
                speech_summary = _safe_text(turn.get("summary", ""))

                patient_au = (au_index.get((str(patient_id), itype, tidx_i), "") or "").strip()
                therapist_au = (au_index.get((str(therapist_id), itype, tidx_i), "") or "").strip()
                if patient_au and therapist_au:
                    au_text = _safe_text(f"Patient AU: {patient_au}\nTherapist AU: {therapist_au}")
                else:
                    au_text = _safe_text(patient_au or therapist_au)

                turn_meta.append((start_s, end_s, speech_summary, au_text))

            if not turn_meta:
                continue

            turn_bounds = [(start_s, end_s) for start_s, end_s, _, _ in turn_meta]
            uns_turns = _resolve_unsummarised_turns(transcript_path, uns_index)
            speech_unsummaries = _compose_unsummarised_texts(turn_bounds, uns_turns)

            patient_segments = [np.zeros((0, N_AUS), dtype=np.float32) for _ in turn_bounds]
            therapist_segments = [np.zeros((0, N_AUS), dtype=np.float32) for _ in turn_bounds]

            patient_openface_raw = idata.get("patient_openface", "")
            if patient_openface_raw:
                patient_openface_path = _resolve_path(patient_openface_raw)
                if patient_openface_path is not None and patient_openface_path.exists():
                    try:
                        aus, timestamps = load_openface_au(
                            patient_openface_path,
                            confidence_threshold=openface_confidence_threshold,
                        )
                        patient_segments = segment_au_by_turns(aus, timestamps, turn_bounds)
                    except Exception as exc:
                        log.warning("OpenFace load failed for patient (%s): %s", patient_openface_path, exc)

            therapist_openface_raw = idata.get("therapist_openface", "")
            if therapist_openface_raw:
                therapist_openface_path = _resolve_path(therapist_openface_raw)
                if therapist_openface_path is not None and therapist_openface_path.exists():
                    try:
                        aus, timestamps = load_openface_au(
                            therapist_openface_path,
                            confidence_threshold=openface_confidence_threshold,
                        )
                        therapist_segments = segment_au_by_turns(aus, timestamps, turn_bounds)
                    except Exception as exc:
                        log.warning("OpenFace load failed for therapist (%s): %s", therapist_openface_path, exc)

            turns: list[TurnInfo] = []
            for idx, (start_s, end_s, speech_summary, au_text) in enumerate(turn_meta):
                patient_num = _format_au_segment(patient_segments[idx])
                therapist_num = _format_au_segment(therapist_segments[idx])
                if patient_num != "[UNK]" and therapist_num != "[UNK]":
                    au_numbers = _safe_text(f"Patient AU numeric: {patient_num} Therapist AU numeric: {therapist_num}")
                else:
                    au_numbers = _safe_text(patient_num if patient_num != "[UNK]" else therapist_num)

                # Numeric float vector: patient mean ++ therapist mean  (N_AU_NUMERIC = N_AUS * 2)
                p_seg = patient_segments[idx]
                t_seg = therapist_segments[idx]
                p_mean = p_seg.mean(axis=0) if p_seg.shape[0] > 0 else np.zeros(N_AUS, dtype=np.float32)
                t_mean = t_seg.mean(axis=0) if t_seg.shape[0] > 0 else np.zeros(N_AUS, dtype=np.float32)
                au_numeric_vec = np.concatenate([p_mean, t_mean]).astype(np.float32)

                turns.append(
                    TurnInfo(
                        start_s=start_s,
                        end_s=end_s,
                        speech_summarised=speech_summary,
                        speech_unsummarised=speech_unsummaries[idx],
                        au_text=au_text,
                        au_numbers=au_numbers,
                        au_numeric_vec=au_numeric_vec,
                    )
                )

            sessions.append(
                SessionData(
                    split=split,
                    patient_id=patient_id,
                    therapist_id=therapist_id,
                    interview_type=itype,
                    turns=turns,
                    labels=labels,
                )
            )

    log.info("[%s] Loaded %d sessions (skipped transcript issues: %d)", split.upper(), len(sessions), skipped_no_transcript)
    return sessions


def build_patient_data(sessions: list[SessionData]) -> list[PatientData]:
    grouped: dict[tuple[str, str, str], PatientData] = {}
    for s in sessions:
        key = (s.split, s.therapist_id, s.patient_id)
        if key not in grouped:
            grouped[key] = PatientData(
                split=s.split,
                patient_id=s.patient_id,
                therapist_id=s.therapist_id,
                sessions={},
                labels_by_type={},
            )
        rec = grouped[key]
        rec.sessions[s.interview_type] = s
        rec.labels_by_type[s.interview_type] = dict(s.labels)
    return list(grouped.values())


def to_common_records(patients: list[PatientData]) -> list[PatientRecord]:
    rows: list[SessionFeatureRow] = []
    for p in patients:
        for itype in ["personal", "bindung", "wunder"]:
            if itype not in p.sessions:
                continue
            rows.append(
                SessionFeatureRow(
                    split=p.split,
                    patient_id=p.patient_id,
                    therapist_id=p.therapist_id,
                    interview_type=itype,
                    au_feat=np.zeros((1,), dtype=np.float32),
                    text_feat=np.zeros((1,), dtype=np.float32),
                    labels=dict(p.labels_by_type.get(itype, {})),
                    baseline_labels={},
                )
            )
    return build_patient_records(rows)


def build_samples_for_target(
    patients: list[PatientData],
    target: str,
    task: str,
    class_to_idx: dict[str, int] | None,
    speech_input: str,
    au_input: str,
) -> list[Sample]:
    req_types: list[str] = required_types_for_target(target)   # ordered, deterministic
    out: list[Sample] = []

    for p in patients:
        target_value = get_target_value(p, target)
        if target_value is None:
            continue

        turns: list[tuple[str, str, np.ndarray]] = []
        for itype in req_types:
            if itype in p.sessions:
                for turn in p.sessions[itype].turns:
                    speech_val = turn.speech_summarised if speech_input == "summarised" else turn.speech_unsummarised
                    au_val = turn.au_text if au_input == "au_text" else turn.au_numbers
                    turns.append((_safe_text(speech_val), _safe_text(au_val), turn.au_numeric_vec))

        if not turns:
            continue

        if task == "regression":
            try:
                val = float(target_value)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(val):
                continue
            out.append(Sample(therapist_id=p.therapist_id, target_value=val, turns=turns))
        else:
            label = str(target_value).strip()
            if not label or class_to_idx is None or label not in class_to_idx:
                continue
            out.append(Sample(therapist_id=p.therapist_id, target_value=class_to_idx[label], turns=turns))

    return out


def build_samples_for_treatment_modality(
    sessions: list[SessionData],
    speech_input: str,
    au_input: str,
) -> tuple[list[Sample], list[str]]:
    valid_types = ["bindung", "personal", "wunder"]
    class_labels = [label for label in valid_types if any(s.interview_type == label for s in sessions)]
    class_to_idx = {label: idx for idx, label in enumerate(class_labels)}

    samples: list[Sample] = []
    for session in sessions:
        if session.interview_type not in class_to_idx:
            continue

        turns: list[tuple[str, str, np.ndarray]] = []
        for turn in session.turns:
            speech_val = turn.speech_summarised if speech_input == "summarised" else turn.speech_unsummarised
            au_val = turn.au_text if au_input == "au_text" else turn.au_numbers
            turns.append((_safe_text(speech_val), _safe_text(au_val), turn.au_numeric_vec))

        if not turns:
            continue

        samples.append(
            Sample(
                therapist_id=session.therapist_id,
                target_value=class_to_idx[session.interview_type],
                turns=turns,
            )
        )

    return samples, class_labels


def iter_batches(samples: list[Sample], batch_size: int, shuffle: bool) -> list[list[Sample]]:
    idx = np.arange(len(samples))
    if shuffle and len(idx) > 1:
        np.random.shuffle(idx)
    return [[samples[i] for i in idx[j: j + batch_size]] for j in range(0, len(idx), batch_size)]


def collate_batch(
    batch: list[Sample],
    tokenizer,
    max_token_length: int,
    device: torch.device,
    au_input_mode: str,  # "au_text" | "au_numbers"
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    speech_texts: list[str] = []
    au_texts: list[str] = []
    au_numeric_vecs: list[np.ndarray] = []
    turn_counts: list[int] = []
    y_vals: list[float | int] = []

    for s in batch:
        turn_counts.append(len(s.turns))
        y_vals.append(s.target_value)
        for speech, au_str, au_vec in s.turns:
            speech_texts.append(_safe_text(speech))
            if au_input_mode == "au_text":
                au_texts.append(_safe_text(au_str))
            else:
                au_numeric_vecs.append(au_vec)

    speech_tok = tokenizer(
        speech_texts,
        truncation=True,
        padding=True,
        max_length=max_token_length,
        return_tensors="pt",
    )

    turn_counts_t = torch.tensor(turn_counts, dtype=torch.long)
    max_turns = int(max(turn_counts)) if turn_counts else 0
    turn_mask = torch.zeros((len(batch), max_turns), dtype=torch.bool)
    for i, n in enumerate(turn_counts):
        turn_mask[i, :n] = True

    x: dict[str, torch.Tensor] = {
        "speech_input_ids": speech_tok["input_ids"].to(device),
        "speech_attention_mask": speech_tok["attention_mask"].to(device),
        "turn_counts": turn_counts_t.to(device),
        "turn_mask": turn_mask.to(device),
    }

    if au_input_mode == "au_text":
        au_tok = tokenizer(
            au_texts,
            truncation=True,
            padding=True,
            max_length=max_token_length,
            return_tensors="pt",
        )
        x["au_input_ids"] = au_tok["input_ids"].to(device)
        x["au_attention_mask"] = au_tok["attention_mask"].to(device)
    else:
        # Stack float vectors: (total_turns, N_AU_NUMERIC)
        au_array = np.stack(au_numeric_vecs, axis=0).astype(np.float32)
        x["au_numeric"] = torch.from_numpy(au_array).to(device)

    if isinstance(y_vals[0], float):
        y = torch.tensor(y_vals, dtype=torch.float32, device=device)
    else:
        y = torch.tensor(y_vals, dtype=torch.long, device=device)

    return x, y


def train_target_loto(
    all_samples: list[Sample],
    task: str,
    class_labels: list[str] | None,
    args: argparse.Namespace,
    device: torch.device,
    au_input: str,  # "au_text" | "au_numbers"
) -> tuple[list[dict[str, Any]], dict[str, list[Any]]]:
    therapist_ids = sorted({s.therapist_id for s in all_samples})
    fold_rows: list[dict[str, Any]] = []
    accum = {"y_true": [], "y_pred": []}

    n_au_numeric = N_AU_NUMERIC if au_input == "au_numbers" else None

    for held in therapist_ids:
        train_samples = [s for s in all_samples if s.therapist_id != held]
        test_samples = [s for s in all_samples if s.therapist_id == held]

        min_train = 8 if task == "classification" else 5
        if len(train_samples) < min_train or len(test_samples) == 0:
            continue

        if task == "classification":
            tr_classes = {int(s.target_value) for s in train_samples}
            if len(tr_classes) < 2:
                continue

        common_kwargs = dict(
            bert_model_name=args.bert_model,
            fusion_dim=args.fusion_dim,
            gru_hidden_dim=args.gru_hidden,
            gru_layers=args.gru_layers,
            dropout=args.dropout,
            bert_sub_batch=args.bert_sub_batch,
            n_au_numeric=n_au_numeric,
        )
        if task == "regression":
            model: HMANRegressor | HMANClassifier = HMANRegressor(**common_kwargs).to(device)
        else:
            model = HMANClassifier(n_classes=len(class_labels), **common_kwargs).to(device)

        if args.freeze_bert_epochs > 0:
            model.freeze_bert()

        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=3, min_lr=1e-7
        )

        for epoch in range(1, args.epochs + 1):
            if epoch == args.freeze_bert_epochs + 1:
                model.unfreeze_bert()

            model.train()
            batches = iter_batches(train_samples, args.batch_size, shuffle=True)
            epoch_loss = 0.0
            n_batches = 0
            for batch in batches:
                x, y = collate_batch(
                    batch, args.tokenizer, args.max_token_length, device, au_input_mode=au_input
                )
                opt.zero_grad()
                logits = model(**x)

                if task == "regression":
                    loss = nn.functional.mse_loss(logits, y)
                else:
                    loss = nn.functional.cross_entropy(logits, y)

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                opt.step()
                epoch_loss += loss.item()
                n_batches += 1

            if n_batches > 0:
                scheduler.step(epoch_loss / n_batches)

        model.eval()
        y_true: list[Any] = []
        y_pred: list[Any] = []
        with torch.no_grad():
            test_batches = iter_batches(test_samples, args.batch_size, shuffle=False)
            for batch in test_batches:
                x, y = collate_batch(
                    batch, args.tokenizer, args.max_token_length, device, au_input_mode=au_input
                )
                logits = model(**x)
                if task == "regression":
                    pred = logits.detach().cpu().numpy().tolist()
                else:
                    pred = torch.argmax(logits, dim=1).detach().cpu().numpy().tolist()
                truth = y.detach().cpu().numpy().tolist()
                y_true.extend(truth)
                y_pred.extend(pred)

        if task == "regression":
            m = compute_reg_metrics(np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float))
        else:
            labels = list(range(len(class_labels)))
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            m = {
                "accuracy": float(np.mean(np.asarray(y_true) == np.asarray(y_pred))),
                "f1_macro": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
                "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)),
                "f1_micro": float(f1_score(y_true, y_pred, average="micro", labels=labels, zero_division=0)),
                "confusion_matrix": cm.tolist(),
                "n": len(y_true),
            }

        m.update(held_out_therapist=held, n_train=len(train_samples), n_test=len(test_samples))
        fold_rows.append(m)
        accum["y_true"].extend(y_true)
        accum["y_pred"].extend(y_pred)

    return fold_rows, accum


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LOTO-only mixed-outcome training with hierarchical multimodal model")
    p.add_argument("--data_model", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--au_descriptions_dir", type=Path, required=True)
    p.add_argument("--unsummarised_text_path", type=Path, required=True)
    p.add_argument("--output_json", type=Path, default=Path("hman_mixed_loto_results.json"))
    p.add_argument("--openface_confidence_threshold", type=float, default=0.5)

    p.add_argument("--bert_model", type=str, default="distilbert-base-uncased")
    p.add_argument("--fusion_dim", type=int, default=192)
    p.add_argument("--gru_hidden", type=int, default=96)
    p.add_argument("--gru_layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--bert_sub_batch", type=int, default=64)

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--freeze_bert_epochs", type=int, default=1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--max_token_length", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Using device: %s", device)

    args.tokenizer = AutoTokenizer.from_pretrained(args.bert_model)

    datasets: dict[str, list[SessionData]] = {}
    for split in ("train", "val", "test"):
        datasets[split] = load_sessions(
            args.data_model,
            args.config,
            args.au_descriptions_dir,
            args.unsummarised_text_path,
            split,
            args.openface_confidence_threshold,
        )

    all_sessions = datasets["train"] + datasets["val"] + datasets["test"]
    patients = build_patient_data(all_sessions)
    common_records = to_common_records(patients)

    reg_targets, clf_targets, clf_classes = build_target_mapping(common_records)
    reg_targets = [t for t in reg_targets if not t.startswith(("T0_", "T1_"))]
    clf_targets = [t for t in clf_targets if not t.startswith(("T0_", "T1_"))]
    all_targets = [(t, "regression") for t in reg_targets] + [(t, "classification") for t in clf_targets]

    ablations = [
        ("Summarised x AU-text", "summarised", "au_text"),
        ("Summarised x AU-numbers", "summarised", "au_numbers"),
        ("Unsummarised x AU-text", "unsummarised", "au_text"),
        ("Unsummarised x AU-numbers", "unsummarised", "au_numbers"),
    ]
    all_results: list[dict[str, Any]] = []

    log.info("Targets discovered: regression=%d classification=%d", len(reg_targets), len(clf_targets))

    for ablation_name, text_input, au_input in ablations:
        ablation_results: list[dict[str, Any]] = []

        for target, task in all_targets:
            class_to_idx = None
            class_labels = None
            if task == "classification":
                class_labels = clf_classes[target]
                class_to_idx = {c: i for i, c in enumerate(class_labels)}

            samples = build_samples_for_target(
                patients,
                target,
                task,
                class_to_idx,
                text_input,
                au_input,
            )
            min_total = 10 if task == "classification" else 8
            if len(samples) < min_total:
                continue

            log.info("ABL=%s TARGET=%s [%s] samples=%d", ablation_name, target, task, len(samples))
            fold_rows, accum = train_target_loto(samples, task, class_labels, args, device, au_input=au_input)

            for r in fold_rows:
                r.update(
                    task=task,
                    target=target,
                    features="AU+Text",
                    model="HMAN",
                    ablation=ablation_name,
                    text_input=text_input,
                    au_input=au_input,
                )
                ablation_results.append(r)

            if len(accum["y_true"]) >= 3:
                if task == "regression":
                    agg = compute_reg_metrics(np.asarray(accum["y_true"], dtype=float), np.asarray(accum["y_pred"], dtype=float))
                else:
                    y_true = np.asarray(accum["y_true"], dtype=int)
                    y_pred = np.asarray(accum["y_pred"], dtype=int)
                    labels = list(range(len(class_labels)))
                    cm = confusion_matrix(y_true, y_pred, labels=labels)
                    agg = {
                        "accuracy": float(np.mean(y_true == y_pred)),
                        "f1_macro": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
                        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)),
                        "f1_micro": float(f1_score(y_true, y_pred, average="micro", labels=labels, zero_division=0)),
                        "confusion_matrix": cm.tolist(),
                        "n": int(len(y_true)),
                    }
                agg.update(
                    task=task,
                    target=target,
                    features="AU+Text",
                    type="loto_aggregate",
                    model="HMAN",
                    ablation=ablation_name,
                    text_input=text_input,
                    au_input=au_input,
                )
                ablation_results.append(agg)

        modality_samples, modality_labels = build_samples_for_treatment_modality(
            all_sessions,
            text_input,
            au_input,
        )
        if len(modality_samples) >= 10 and len(modality_labels) >= 2:
            log.info("ABL=%s TARGET=treatment_modality [classification] samples=%d", ablation_name, len(modality_samples))
            fold_rows, accum = train_target_loto(
                modality_samples, "classification", modality_labels, args, device, au_input=au_input
            )

            for r in fold_rows:
                r.update(
                    task="classification",
                    target="treatment_modality",
                    features="AU+Text",
                    model="HMAN",
                    ablation=ablation_name,
                    text_input=text_input,
                    au_input=au_input,
                )
                ablation_results.append(r)

            if len(accum["y_true"]) >= 3:
                y_true = np.asarray(accum["y_true"], dtype=int)
                y_pred = np.asarray(accum["y_pred"], dtype=int)
                labels = list(range(len(modality_labels)))
                cm = confusion_matrix(y_true, y_pred, labels=labels)
                agg = {
                    "accuracy": float(np.mean(y_true == y_pred)),
                    "f1_macro": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
                    "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)),
                    "f1_micro": float(f1_score(y_true, y_pred, average="micro", labels=labels, zero_division=0)),
                    "confusion_matrix": cm.tolist(),
                    "n": int(len(y_true)),
                }
                agg.update(
                    task="classification",
                    target="treatment_modality",
                    features="AU+Text",
                    type="loto_aggregate",
                    model="HMAN",
                    ablation=ablation_name,
                    text_input=text_input,
                    au_input=au_input,
                )
                ablation_results.append(agg)

        # --- Per-ablation checkpoint save ---
        abl_slug = re.sub(r"[^a-zA-Z0-9]+", "_", ablation_name.lower()).strip("_")
        partial_path = args.output_json.parent / (args.output_json.stem + f"_{abl_slug}.json")
        with open(partial_path, "w", encoding="utf-8") as f:
            json.dump(ablation_results, f, indent=2)
        log.info("Checkpoint saved %d rows for ablation '%s' → %s", len(ablation_results), ablation_name, partial_path)

        all_results.extend(ablation_results)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    log.info("Saved %d rows to %s", len(all_results), args.output_json)


if __name__ == "__main__":
    main()
