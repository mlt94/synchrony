"""
Embedding baseline with mixed-task outcomes and cumulative-session feature rules.

This script evaluates:
- Post-treatment outcomes under interview type labels (T3/T5/T7)
- Baseline outcomes under interview['baseline'] (T0/T1 and any additional keys)

It supports mixed targets automatically:
- numeric -> regression
- categorical/text -> multiclass classification

Ablation modalities:
- AU_only   (AU descriptions as text embeddings)
- Text_only (speech summaries embeddings)
- AU+Text   (concatenation)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    from source.modeling.multisession_eval_common import (
        SessionFeatureRow,
        build_features_and_targets,
        build_patient_records,
        compute_clf_metrics,
        compute_reg_metrics,
        discover_target_specs,
        get_classifiers,
        get_regressors,
    )
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from source.modeling.multisession_eval_common import (
        SessionFeatureRow,
        build_features_and_targets,
        build_patient_records,
        compute_clf_metrics,
        compute_reg_metrics,
        discover_target_specs,
        get_classifiers,
        get_regressors,
    )

try:
    from source.modeling.dataset import load_au_descriptions, _resolve_path, _is_nan
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from source.modeling.dataset import load_au_descriptions, _resolve_path, _is_nan


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stderr,
)
log = logging.getLogger("baseline_embedding")


INTERVIEW_PREFIX: dict[str, str] = {
    "bindung": "T5",
    "personal": "T3",
    "wunder": "T7",
}

OUTCOME_SUFFIXES: list[str] = [
    "PANAS_pos_Pr", "PANAS_neg_Pr",
    "IRF_self_Pr", "IRF_other_Pr",
    "BLRI_ges_Pr",
    "PANAS_pos_In", "PANAS_neg_In",
    "IRF_self_In", "IRF_other_In",
    "BLRI_ges_In",
]

LABEL_MAP: dict[str, dict[str, str]] = {
    itype: {suffix: f"{prefix}_{suffix}" for suffix in OUTCOME_SUFFIXES}
    for itype, prefix in INTERVIEW_PREFIX.items()
}


@dataclass
class SessionTextData:
    split: str
    patient_id: str
    therapist_id: str
    interview_type: str
    speech_summaries: list[str] = field(default_factory=list)
    au_descriptions: list[str] = field(default_factory=list)
    labels: dict[str, float] = field(default_factory=dict)
    baseline_labels: dict[str, Any] = field(default_factory=dict)


def _safe_text(x: str) -> str:
    x = (x or "").strip()
    return x if x else "[UNK]"


def load_sessions(
    data_model_path: Path,
    config_path: Path,
    au_descriptions_dir: Path,
    split: str,
) -> list[SessionTextData]:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    allowed = set(config["psychotherapy_splits"][split])

    with open(data_model_path, "r", encoding="utf-8") as f:
        data_model = yaml.safe_load(f)

    au_index = load_au_descriptions(au_descriptions_dir)

    sessions: list[SessionTextData] = []
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
            for _suffix, full_key in LABEL_MAP[itype].items():
                value = labels_raw.get(full_key)
                if not _is_nan(value):
                    labels[full_key] = float(value)

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
            speech: list[str] = []
            au_text: list[str] = []

            for turn in turns_data:
                tidx = turn.get("turn_index")
                try:
                    tidx_i = int(tidx)
                except (TypeError, ValueError):
                    continue

                speech.append((turn.get("summary", "") or "").strip())
                au_text.append((au_index.get((patient_id, itype, tidx_i), "") or "").strip())

            if not speech:
                continue

            sessions.append(
                SessionTextData(
                    split=split,
                    patient_id=patient_id,
                    therapist_id=therapist_id,
                    interview_type=itype,
                    speech_summaries=speech,
                    au_descriptions=au_text,
                    labels=labels,
                    baseline_labels=baseline_labels,
                )
            )

    log.info("[%s] Loaded %d sessions (skipped %d transcript issues)", split.upper(), len(sessions), skipped_no_transcript)
    return sessions


def embed_session_texts(
    sessions: list[SessionTextData],
    field: str,
    model_name: str,
    batch_size: int,
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    if field not in {"speech", "au"}:
        raise ValueError("field must be 'speech' or 'au'")

    st_model = SentenceTransformer(model_name)
    dim = st_model.get_sentence_embedding_dimension()

    all_texts: list[str] = []
    boundaries: list[tuple[int, int]] = []
    offset = 0

    for s in sessions:
        texts = s.speech_summaries if field == "speech" else s.au_descriptions
        texts = [_safe_text(t) for t in texts]
        all_texts.extend(texts)
        boundaries.append((offset, offset + len(texts)))
        offset += len(texts)

    if offset == 0:
        return np.zeros((len(sessions), dim), dtype=np.float32)

    log.info("Encoding %d turns for %s", offset, field)
    embs = st_model.encode(all_texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)

    X = np.zeros((len(sessions), dim), dtype=np.float32)
    for i, (st, en) in enumerate(boundaries):
        if en > st:
            X[i] = embs[st:en].mean(axis=0)

    return X


def run_all_evaluations(records: list, output_path: Path) -> list[dict]:
    train_records = [r for r in records if r.split == "train"]
    val_records = [r for r in records if r.split == "val"]
    test_records = [r for r in records if r.split == "test"]

    all_results: list[dict] = []
    specs = discover_target_specs(records)

    log.info("Discovered %d targets (%d regression, %d classification)",
             len(specs),
             sum(1 for s in specs if s.task == "regression"),
             sum(1 for s in specs if s.task == "classification"))

    feature_modes = ["AU_only", "Text_only", "AU+Text"]

    for spec in specs:
        log.info("=" * 96)
        log.info("TARGET: %s [%s]", spec.name, spec.task)
        log.info("=" * 96)

        class_to_idx = {c: i for i, c in enumerate(spec.classes)} if spec.task == "classification" else None

        for feat in feature_modes:
            X_tr, y_tr, _ = build_features_and_targets(train_records, spec.name, feat, class_to_idx)
            X_va, y_va, _ = build_features_and_targets(val_records, spec.name, feat, class_to_idx)
            X_te, y_te, _ = build_features_and_targets(test_records, spec.name, feat, class_to_idx)

            log.info("  [%s] valid n: train=%d val=%d test=%d", feat, len(y_tr), len(y_va), len(y_te))

            if spec.task == "regression":
                if len(y_tr) >= 5 and len(y_va) >= 2:
                    for name, reg in get_regressors().items():
                        try:
                            reg.fit(X_tr, y_tr)
                            pred = reg.predict(X_va)
                            m = compute_reg_metrics(y_va, pred)
                            m.update(task="regression", target=spec.name, features=feat, split="train→val", model=name)
                            all_results.append(m)
                        except Exception:
                            pass

                if len(y_tr) >= 5 and len(y_te) >= 2:
                    for name, reg in get_regressors().items():
                        try:
                            reg.fit(X_tr, y_tr)
                            pred = reg.predict(X_te)
                            m = compute_reg_metrics(y_te, pred)
                            m.update(task="regression", target=spec.name, features=feat, split="train→test", model=name)
                            all_results.append(m)
                        except Exception:
                            pass

                if (len(y_tr) + len(y_va)) >= 5 and len(y_te) >= 2:
                    X_trv = np.vstack([X_tr, X_va]) if len(y_va) > 0 else X_tr
                    y_trv = np.concatenate([y_tr, y_va]) if len(y_va) > 0 else y_tr
                    for name, reg in get_regressors().items():
                        try:
                            reg.fit(X_trv, y_trv)
                            pred = reg.predict(X_te)
                            m = compute_reg_metrics(y_te, pred)
                            m.update(task="regression", target=spec.name, features=feat, split="train+val→test", model=name)
                            all_results.append(m)
                        except Exception:
                            pass

                from sklearn.linear_model import Ridge
                from sklearn.pipeline import make_pipeline
                from sklearn.preprocessing import StandardScaler

                all_true: list[float] = []
                all_pred: list[float] = []
                therapist_ids = sorted({r.therapist_id for r in records})

                X_all, y_all, _ = build_features_and_targets(records, spec.name, feat, None)
                if len(y_all) >= 10:
                    rec_all_valid = [r for r in records if build_features_and_targets([r], spec.name, feat, None)[1].shape[0] == 1]
                    for held in therapist_ids:
                        tr_idx = [i for i, r in enumerate(rec_all_valid) if r.therapist_id != held]
                        te_idx = [i for i, r in enumerate(rec_all_valid) if r.therapist_id == held]
                        if len(tr_idx) < 5 or len(te_idx) == 0:
                            continue
                        reg = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
                        reg.fit(X_all[tr_idx], y_all[tr_idx])
                        pred = reg.predict(X_all[te_idx])
                        m = compute_reg_metrics(y_all[te_idx], pred)
                        m.update(task="regression", target=spec.name, features=feat, held_out_therapist=held, n_train=len(tr_idx), n_test=len(te_idx))
                        all_results.append(m)
                        all_true.extend(y_all[te_idx].tolist())
                        all_pred.extend(pred.tolist())

                    if len(all_true) > 2:
                        agg = compute_reg_metrics(np.array(all_true), np.array(all_pred))
                        agg.update(task="regression", target=spec.name, features=feat, type="loto_aggregate")
                        all_results.append(agg)

            else:
                if len(y_tr) >= 8 and len(set(y_tr.tolist())) >= 2 and len(y_va) >= 2:
                    for name, clf in get_classifiers().items():
                        try:
                            clf.fit(X_tr, y_tr)
                            pred = clf.predict(X_va)
                            m = compute_clf_metrics(y_va, pred, spec.classes)
                            m.update(task="classification", target=spec.name, features=feat, split="train→val", model=name)
                            all_results.append(m)
                        except Exception:
                            pass

                if len(y_tr) >= 8 and len(set(y_tr.tolist())) >= 2 and len(y_te) >= 2:
                    for name, clf in get_classifiers().items():
                        try:
                            clf.fit(X_tr, y_tr)
                            pred = clf.predict(X_te)
                            m = compute_clf_metrics(y_te, pred, spec.classes)
                            m.update(task="classification", target=spec.name, features=feat, split="train→test", model=name)
                            all_results.append(m)
                        except Exception:
                            pass

                if (len(y_tr) + len(y_va)) >= 8 and len(y_te) >= 2:
                    X_trv = np.vstack([X_tr, X_va]) if len(y_va) > 0 else X_tr
                    y_trv = np.concatenate([y_tr, y_va]) if len(y_va) > 0 else y_tr
                    if len(set(y_trv.tolist())) >= 2:
                        for name, clf in get_classifiers().items():
                            try:
                                clf.fit(X_trv, y_trv)
                                pred = clf.predict(X_te)
                                m = compute_clf_metrics(y_te, pred, spec.classes)
                                m.update(task="classification", target=spec.name, features=feat, split="train+val→test", model=name)
                                all_results.append(m)
                            except Exception:
                                pass

                from sklearn.linear_model import LogisticRegression
                from sklearn.pipeline import make_pipeline
                from sklearn.preprocessing import StandardScaler

                all_true: list[int] = []
                all_pred: list[int] = []
                therapist_ids = sorted({r.therapist_id for r in records})

                X_all, y_all, _ = build_features_and_targets(records, spec.name, feat, class_to_idx)
                if len(y_all) >= 10:
                    rec_all_valid = [r for r in records if build_features_and_targets([r], spec.name, feat, class_to_idx)[1].shape[0] == 1]
                    for held in therapist_ids:
                        tr_idx = [i for i, r in enumerate(rec_all_valid) if r.therapist_id != held]
                        te_idx = [i for i, r in enumerate(rec_all_valid) if r.therapist_id == held]
                        if len(tr_idx) < 8 or len(te_idx) == 0:
                            continue
                        if len(set(y_all[tr_idx].tolist())) < 2:
                            continue
                        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=4000, random_state=42))
                        clf.fit(X_all[tr_idx], y_all[tr_idx])
                        pred = clf.predict(X_all[te_idx])
                        m = compute_clf_metrics(y_all[te_idx], pred, spec.classes)
                        m.update(task="classification", target=spec.name, features=feat, held_out_therapist=held, n_train=len(tr_idx), n_test=len(te_idx))
                        all_results.append(m)
                        all_true.extend(y_all[te_idx].tolist())
                        all_pred.extend(pred.tolist())

                    if len(all_true) > 2:
                        agg = compute_clf_metrics(np.array(all_true), np.array(all_pred), spec.classes)
                        agg.update(task="classification", target=spec.name, features=feat, type="loto_aggregate")
                        all_results.append(agg)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)

    return all_results


def main() -> None:
    args = parse_args()
    log.setLevel(getattr(logging, args.log_level.upper()))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets: dict[str, list[SessionTextData]] = {}
    for split in ("train", "val", "test"):
        datasets[split] = load_sessions(args.data_model, args.config, args.au_descriptions_dir, split)

    train_s = datasets["train"]
    val_s = datasets["val"]
    test_s = datasets["test"]
    all_s = train_s + val_s + test_s

    if not train_s:
        log.error("Train split empty.")
        sys.exit(1)

    log.info("Extracting speech-summary embeddings ...")
    X_text_all = embed_session_texts(all_s, field="speech", model_name=args.embed_model, batch_size=args.embed_batch_size)

    log.info("Extracting AU-description embeddings ...")
    X_au_all = embed_session_texts(all_s, field="au", model_name=args.embed_model, batch_size=args.embed_batch_size)

    rows: list[SessionFeatureRow] = []
    for i, s in enumerate(all_s):
        rows.append(
            SessionFeatureRow(
                split=s.split,
                patient_id=s.patient_id,
                therapist_id=s.therapist_id,
                interview_type=s.interview_type,
                labels=s.labels,
                baseline_labels=s.baseline_labels,
                au_feat=X_au_all[i],
                text_feat=X_text_all[i],
            )
        )

    patient_records = build_patient_records(rows)
    log.info("Built %d patient records.", len(patient_records))

    out_path = out_dir / "baseline_embedding_mixed_outcomes_results.json"
    results = run_all_evaluations(patient_records, out_path)
    log.info("Saved %d result rows -> %s", len(results), out_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Embedding baseline with mixed outcomes and cumulative-session rules",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--data_model", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--au_descriptions_dir", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, default=Path("results/baseline_embedding_mixed"))

    p.add_argument("--embed_model", type=str, default="all-MiniLM-L6-v2")
    p.add_argument("--embed_batch_size", type=int, default=256)

    p.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING"])

    return p.parse_args()


if __name__ == "__main__":
    main()
