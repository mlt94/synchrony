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
    from source.modeling.model import SessionEncoder, TurnAttention, TurnEncoder
    from source.modeling.multisession_eval_common import (
        PatientRecord,
        SessionFeatureRow,
        compute_reg_metrics,
        discover_target_specs,
        required_types_for_target,
    )
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from source.modeling.dataset import _is_nan, _resolve_path, load_au_descriptions
    from source.modeling.model import SessionEncoder, TurnAttention, TurnEncoder
    from source.modeling.multisession_eval_common import (
        PatientRecord,
        SessionFeatureRow,
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


@dataclass
class TurnInfo:
    speech: str
    au: str


@dataclass
class SessionData:
    split: str
    patient_id: str
    therapist_id: str
    interview_type: str
    turns: list[TurnInfo] = field(default_factory=list)
    labels: dict[str, float] = field(default_factory=dict)
    baseline_labels: dict[str, Any] = field(default_factory=dict)


@dataclass
class PatientData:
    split: str
    patient_id: str
    therapist_id: str
    sessions: dict[str, SessionData] = field(default_factory=dict)
    labels_by_type: dict[str, dict[str, Any]] = field(default_factory=dict)
    baseline_labels: dict[str, Any] = field(default_factory=dict)


@dataclass
class Sample:
    therapist_id: str
    target_value: float | int
    turns: list[TurnInfo]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class HMANMixedTask(nn.Module):
    def __init__(
        self,
        bert_model_name: str,
        fusion_dim: int,
        gru_hidden_dim: int,
        gru_layers: int,
        dropout: float,
        n_reg_targets: int,
        clf_num_classes: list[int],
        bert_sub_batch: int = 64,
    ):
        super().__init__()
        self.fusion_dim = fusion_dim
        self.bert_sub_batch = bert_sub_batch
        self.clf_num_classes = clf_num_classes

        self.turn_encoder = TurnEncoder(bert_model_name, fusion_dim, dropout)
        self.session_encoder = SessionEncoder(fusion_dim, gru_hidden_dim, num_layers=gru_layers, dropout=dropout)

        bigru_out = gru_hidden_dim * 2
        self.attention = TurnAttention(bigru_out)

        self.shared = nn.Sequential(
            nn.LayerNorm(bigru_out),
            nn.Linear(bigru_out, gru_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.reg_head = nn.Linear(gru_hidden_dim, n_reg_targets) if n_reg_targets > 0 else None
        self.clf_heads = nn.ModuleList([nn.Linear(gru_hidden_dim, n) for n in clf_num_classes])

    def _encode_turns_batched(
        self,
        speech_ids: torch.Tensor,
        speech_mask: torch.Tensor,
        au_ids: torch.Tensor,
        au_mask: torch.Tensor,
    ) -> torch.Tensor:
        total = speech_ids.size(0)
        if self.bert_sub_batch <= 0 or total <= self.bert_sub_batch:
            return self.turn_encoder(speech_ids, speech_mask, au_ids, au_mask)

        parts: list[torch.Tensor] = []
        for start in range(0, total, self.bert_sub_batch):
            end = min(start + self.bert_sub_batch, total)
            parts.append(self.turn_encoder(speech_ids[start:end], speech_mask[start:end], au_ids[start:end], au_mask[start:end]))
        return torch.cat(parts, dim=0)

    def encode_context(
        self,
        speech_input_ids: torch.Tensor,
        speech_attention_mask: torch.Tensor,
        au_input_ids: torch.Tensor,
        au_attention_mask: torch.Tensor,
        turn_counts: torch.Tensor,
        turn_mask: torch.Tensor,
    ) -> torch.Tensor:
        turn_vecs = self._encode_turns_batched(speech_input_ids, speech_attention_mask, au_input_ids, au_attention_mask)

        batch_size = turn_counts.size(0)
        max_turns = turn_mask.size(1)
        turn_emb = turn_vecs.new_zeros(batch_size, max_turns, self.fusion_dim)

        idx = 0
        for i in range(batch_size):
            n = int(turn_counts[i].item())
            turn_emb[i, :n] = turn_vecs[idx: idx + n]
            idx += n

        session_repr = self.session_encoder(turn_emb, turn_counts)
        context, _ = self.attention(session_repr, turn_mask)
        return context

    def forward(
        self,
        speech_input_ids: torch.Tensor,
        speech_attention_mask: torch.Tensor,
        au_input_ids: torch.Tensor,
        au_attention_mask: torch.Tensor,
        turn_counts: torch.Tensor,
        turn_mask: torch.Tensor,
    ) -> tuple[torch.Tensor | None, list[torch.Tensor]]:
        context = self.encode_context(
            speech_input_ids,
            speech_attention_mask,
            au_input_ids,
            au_attention_mask,
            turn_counts,
            turn_mask,
        )
        h = self.shared(context)
        reg_out = self.reg_head(h) if self.reg_head is not None else None
        clf_out = [head(h) for head in self.clf_heads]
        return reg_out, clf_out


def build_target_mapping(records: list[PatientRecord]) -> tuple[list[str], list[str], dict[str, list[str]]]:
    specs = discover_target_specs(records)
    reg_targets = [s.name for s in specs if s.task == "regression"]
    clf_targets = [s.name for s in specs if s.task == "classification"]
    clf_classes = {s.name: s.classes for s in specs if s.task == "classification"}
    return reg_targets, clf_targets, clf_classes


def get_target_value(rec: PatientData, target_key: str) -> Any:
    if target_key in rec.baseline_labels:
        return rec.baseline_labels.get(target_key)

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


def load_sessions(data_model_path: Path, config_path: Path, au_descriptions_dir: Path, split: str) -> list[SessionData]:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    allowed = set(config["psychotherapy_splits"][split])

    with open(data_model_path, "r", encoding="utf-8") as f:
        data_model = yaml.safe_load(f)

    au_index = load_au_descriptions(au_descriptions_dir)

    sessions: list[SessionData] = []
    skipped_no_transcript = 0

    for interview in data_model["interviews"]:
        therapist_id = interview["therapist"]["therapist_id"]
        if therapist_id not in allowed:
            continue

        patient_id = interview["patient"]["patient_id"]
        baseline_labels = dict((interview.get("baseline", {}) or {}))

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
            turns: list[TurnInfo] = []
            for turn in turns_data:
                tidx = turn.get("turn_index")
                try:
                    tidx_i = int(tidx)
                except (TypeError, ValueError):
                    continue
                speech = _safe_text(turn.get("summary", ""))
                patient_au = (au_index.get((str(patient_id), itype, tidx_i), "") or "").strip()
                therapist_au = (au_index.get((str(therapist_id), itype, tidx_i), "") or "").strip()
                if patient_au and therapist_au:
                    au_text = _safe_text(f"Patient AU: {patient_au}\nTherapist AU: {therapist_au}")
                else:
                    au_text = _safe_text(patient_au or therapist_au)
                turns.append(TurnInfo(speech=speech, au=au_text))

            if not turns:
                continue

            sessions.append(
                SessionData(
                    split=split,
                    patient_id=patient_id,
                    therapist_id=therapist_id,
                    interview_type=itype,
                    turns=turns,
                    labels=labels,
                    baseline_labels=baseline_labels,
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
                baseline_labels=dict(s.baseline_labels),
            )
        rec = grouped[key]
        rec.sessions[s.interview_type] = s
        rec.labels_by_type[s.interview_type] = dict(s.labels)
        if s.baseline_labels:
            rec.baseline_labels = dict(s.baseline_labels)
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
                    baseline_labels=dict(p.baseline_labels),
                )
            )
    return build_patient_records(rows)


def required_types_for_target_name(target: str) -> set[str]:
    if target.startswith("T3_"):
        return {"personal"}
    if target.startswith("T5_"):
        return {"personal", "bindung"}
    if target.startswith("T7_"):
        return {"personal", "bindung", "wunder"}
    return {"personal", "bindung", "wunder"}


def build_samples_for_target(
    patients: list[PatientData],
    target: str,
    task: str,
    class_to_idx: dict[str, int] | None,
    feature_mode: str,
) -> list[Sample]:
    req = required_types_for_target_name(target)
    out: list[Sample] = []

    for p in patients:
        target_value = get_target_value(p, target)
        if target_value is None:
            continue

        turns: list[TurnInfo] = []
        for itype in ["personal", "bindung", "wunder"]:
            if itype in req and itype in p.sessions:
                turns.extend(p.sessions[itype].turns)

        if not turns:
            continue

        if feature_mode == "Text_only":
            turns = [TurnInfo(speech=t.speech, au="[UNK]") for t in turns]
        elif feature_mode == "AU_only":
            turns = [TurnInfo(speech="[UNK]", au=t.au) for t in turns]

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
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    speech_texts: list[str] = []
    au_texts: list[str] = []
    turn_counts: list[int] = []
    y_vals: list[float | int] = []

    for s in batch:
        turn_counts.append(len(s.turns))
        y_vals.append(s.target_value)
        for t in s.turns:
            speech_texts.append(_safe_text(t.speech))
            au_texts.append(_safe_text(t.au))

    speech_tok = tokenizer(
        speech_texts,
        truncation=True,
        padding=True,
        max_length=max_token_length,
        return_tensors="pt",
    )
    au_tok = tokenizer(
        au_texts,
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

    x = {
        "speech_input_ids": speech_tok["input_ids"].to(device),
        "speech_attention_mask": speech_tok["attention_mask"].to(device),
        "au_input_ids": au_tok["input_ids"].to(device),
        "au_attention_mask": au_tok["attention_mask"].to(device),
        "turn_counts": turn_counts_t.to(device),
        "turn_mask": turn_mask.to(device),
    }

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
) -> tuple[list[dict[str, Any]], dict[str, list[Any]]]:
    therapist_ids = sorted({s.therapist_id for s in all_samples})
    fold_rows: list[dict[str, Any]] = []
    accum = {"y_true": [], "y_pred": []}

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

        model = HMANMixedTask(
            bert_model_name=args.bert_model,
            fusion_dim=args.fusion_dim,
            gru_hidden_dim=args.gru_hidden,
            gru_layers=args.gru_layers,
            dropout=args.dropout,
            n_reg_targets=1 if task == "regression" else 0,
            clf_num_classes=[len(class_labels)] if task == "classification" else [],
            bert_sub_batch=args.bert_sub_batch,
        ).to(device)

        if args.freeze_bert_epochs > 0:
            for p in model.turn_encoder.bert.parameters():
                p.requires_grad = False

        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        for epoch in range(1, args.epochs + 1):
            if epoch == args.freeze_bert_epochs + 1:
                for p in model.turn_encoder.bert.parameters():
                    p.requires_grad = True

            model.train()
            batches = iter_batches(train_samples, args.batch_size, shuffle=True)
            for batch in batches:
                x, y = collate_batch(batch, args.tokenizer, args.max_token_length, device)
                opt.zero_grad()
                reg_out, clf_out = model(**x)

                if task == "regression":
                    pred = reg_out[:, 0]
                    loss = nn.functional.mse_loss(pred, y)
                else:
                    logits = clf_out[0]
                    loss = nn.functional.cross_entropy(logits, y)

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                opt.step()

        model.eval()
        y_true: list[Any] = []
        y_pred: list[Any] = []
        with torch.no_grad():
            test_batches = iter_batches(test_samples, args.batch_size, shuffle=False)
            for batch in test_batches:
                x, y = collate_batch(batch, args.tokenizer, args.max_token_length, device)
                reg_out, clf_out = model(**x)
                if task == "regression":
                    pred = reg_out[:, 0].detach().cpu().numpy().tolist()
                    truth = y.detach().cpu().numpy().tolist()
                else:
                    pred = torch.argmax(clf_out[0], dim=1).detach().cpu().numpy().tolist()
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
    p.add_argument("--output_json", type=Path, default=Path("hman_mixed_loto_results.json"))

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

    p.add_argument("--feature_modes", type=str, default="AU_only,Text_only,AU+Text")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Using device: %s", device)

    args.tokenizer = AutoTokenizer.from_pretrained(args.bert_model)

    datasets: dict[str, list[SessionData]] = {}
    for split in ("train", "val", "test"):
        datasets[split] = load_sessions(args.data_model, args.config, args.au_descriptions_dir, split)

    all_sessions = datasets["train"] + datasets["val"] + datasets["test"]
    patients = build_patient_data(all_sessions)
    common_records = to_common_records(patients)

    reg_targets, clf_targets, clf_classes = build_target_mapping(common_records)
    all_targets = [(t, "regression") for t in reg_targets] + [(t, "classification") for t in clf_targets]

    feature_modes = [x.strip() for x in args.feature_modes.split(",") if x.strip()]
    all_results: list[dict[str, Any]] = []

    log.info("Targets discovered: regression=%d classification=%d", len(reg_targets), len(clf_targets))

    for target, task in all_targets:
        for feat in feature_modes:
            class_to_idx = None
            class_labels = None
            if task == "classification":
                class_labels = clf_classes[target]
                class_to_idx = {c: i for i, c in enumerate(class_labels)}

            samples = build_samples_for_target(patients, target, task, class_to_idx, feat)
            min_total = 10 if task == "classification" else 8
            if len(samples) < min_total:
                continue

            log.info("TARGET=%s [%s] feat=%s samples=%d", target, task, feat, len(samples))
            fold_rows, accum = train_target_loto(samples, task, class_labels, args, device)

            for r in fold_rows:
                r.update(task=task, target=target, features=feat, model="HMAN")
                all_results.append(r)

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
                agg.update(task=task, target=target, features=feat, type="loto_aggregate", model="HMAN")
                all_results.append(agg)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    log.info("Saved %d rows to %s", len(all_results), args.output_json)


if __name__ == "__main__":
    main()
