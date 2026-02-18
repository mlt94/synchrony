"""
Baseline diagnostic: raw AU time-series (Conv1D) + text summaries with
mixed-task outcome evaluation.

Refactor highlights:
- Predicts all post-treatment outcomes (T3/T5/T7 labels under interview types)
  AND all baseline outcomes under interview['baseline'] (e.g., T0_*).
- Handles mixed target types automatically:
  - numeric -> regression metrics
  - categorical/text -> multiclass classification metrics
- Uses cumulative session information by target timepoint:
  - T3: personal
  - T5: personal + bindung
  - T7: personal + bindung + wunder
  - baseline (T0/T1): all available session types
- Ablation remains explicit: AU_only, Text_only, AU+Text.
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
import torch
import torch.nn as nn
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

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stderr,
)
log = logging.getLogger("baseline_au")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AU_R_COLS: list[str] = [
    "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r",
    "AU09_r", "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r",
    "AU20_r", "AU23_r", "AU25_r", "AU26_r", "AU45_r",
]
N_AUS: int = len(AU_R_COLS)

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


# ---------------------------------------------------------------------------
# Path / value helpers
# ---------------------------------------------------------------------------


def _wsl_to_windows(p: str) -> str:
    if p.startswith("/mnt/") and len(p) > 6 and p[5].isalpha() and p[6] == "/":
        drive = p[5].upper()
        rest = p[7:].replace("/", "\\")
        return f"{drive}:\\{rest}"
    return p


def _windows_to_wsl(p: str) -> str:
    if len(p) >= 3 and p[1] == ":" and p[2] in ("\\/",):
        drive = p[0].lower()
        rest = p[3:].replace("\\", "/")
        return f"/mnt/{drive}/{rest}"
    return p


def _resolve_path(raw: str) -> Path | None:
    candidates = [raw]
    if raw.startswith("/mnt/"):
        candidates.append(_wsl_to_windows(raw))
    elif len(raw) >= 3 and raw[1] == ":":
        candidates.append(_windows_to_wsl(raw))
    for c in candidates:
        p = Path(c)
        if p.exists():
            return p
    return None


def _is_nan(x: Any) -> bool:
    if x is None:
        return True
    try:
        return not np.isfinite(float(x))
    except (TypeError, ValueError):
        return True


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class TurnInfo:
    turn_index: int
    start_s: float
    end_s: float
    summary: str


@dataclass
class SessionData:
    split: str
    patient_id: str
    therapist_id: str
    interview_type: str
    turns: list[TurnInfo]
    labels: dict[str, float] = field(default_factory=dict)
    baseline_labels: dict[str, Any] = field(default_factory=dict)
    openface_path: Path | None = None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_openface_au(path: Path, confidence_threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    import pandas as pd

    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    mask = (df["success"] == 1) & (df["confidence"] >= confidence_threshold)
    aus = df.loc[mask, AU_R_COLS].values.astype(np.float32)
    timestamps = df.loc[mask, "timestamp"].values.astype(np.float64)
    return aus, timestamps


def load_sessions(data_model_path: Path, config_path: Path, split: str) -> list[SessionData]:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    allowed = set(config["psychotherapy_splits"][split])

    with open(data_model_path, "r", encoding="utf-8") as f:
        data_model = yaml.safe_load(f)

    sessions: list[SessionData] = []
    skipped_unknown_type = 0
    skipped_no_transcript = 0

    for interview in data_model["interviews"]:
        therapist_id = interview["therapist"]["therapist_id"]
        if therapist_id not in allowed:
            continue

        patient_id = interview["patient"]["patient_id"]

        baseline_raw = interview.get("baseline", {}) or {}
        baseline_labels: dict[str, Any] = {}
        for k, v in baseline_raw.items():
            if v is None:
                continue
            baseline_labels[k] = v

        for itype, idata in interview.get("types", {}).items():
            if itype not in INTERVIEW_PREFIX:
                skipped_unknown_type += 1
                continue

            labels_raw = idata.get("labels", {})
            labels: dict[str, float] = {}
            for _suffix, full_key in LABEL_MAP[itype].items():
                val = labels_raw.get(full_key)
                if not _is_nan(val):
                    labels[full_key] = float(val)

            transcript_raw = idata.get("transcript", "")
            transcript_path = _resolve_path(transcript_raw) if transcript_raw else None
            if transcript_path is None or not transcript_path.exists():
                skipped_no_transcript += 1
                continue

            try:
                with open(transcript_path, "r", encoding="utf-8") as f:
                    turns_raw = json.load(f)
            except Exception:
                skipped_no_transcript += 1
                continue

            if not isinstance(turns_raw, list) or len(turns_raw) == 0:
                skipped_no_transcript += 1
                continue

            turns_raw = sorted(turns_raw, key=lambda t: t.get("turn_index", 0))
            turns: list[TurnInfo] = []
            for t in turns_raw:
                turns.append(
                    TurnInfo(
                        turn_index=int(t.get("turn_index", 0)),
                        start_s=float(t.get("start_ms", 0)) / 1000.0,
                        end_s=float(t.get("end_ms", 0)) / 1000.0,
                        summary=(t.get("summary", "") or "").strip(),
                    )
                )

            of_raw = idata.get("patient_openface", "")
            of_path = _resolve_path(of_raw) if of_raw else None

            sessions.append(
                SessionData(
                    split=split,
                    patient_id=patient_id,
                    therapist_id=therapist_id,
                    interview_type=itype,
                    turns=turns,
                    labels=labels,
                    baseline_labels=baseline_labels,
                    openface_path=of_path,
                )
            )

    log.info(
        "[%s] Loaded %d sessions (skipped: %d unknown type, %d no transcript)",
        split.upper(), len(sessions), skipped_unknown_type, skipped_no_transcript,
    )
    return sessions


# ---------------------------------------------------------------------------
# AU feature extractor
# ---------------------------------------------------------------------------


class DepthwiseConv1DExtractor(nn.Module):
    def __init__(
        self,
        n_channels: int = N_AUS,
        n_kernels_per_channel: int = 4,
        kernel_sizes: list[int] | None = None,
        seed: int = 42,
    ):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [7, 15, 31]

        torch.manual_seed(seed)
        self.convs = nn.ModuleList()
        for ks in kernel_sizes:
            conv = nn.Conv1d(
                in_channels=n_channels,
                out_channels=n_channels * n_kernels_per_channel,
                kernel_size=ks,
                groups=n_channels,
                padding=ks // 2,
                bias=True,
            )
            conv.requires_grad_(False)
            self.convs.append(conv)

        self.output_dim = n_channels * n_kernels_per_channel * len(kernel_sizes) * 2

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chunks = []
        for conv in self.convs:
            h = conv(x)
            chunks.append(h.max(dim=-1).values)
            chunks.append(h.mean(dim=-1))
        return torch.cat(chunks, dim=-1)


MIN_FRAMES_FOR_CONV = 3


def _au_stats(seg: np.ndarray) -> np.ndarray:
    if seg.shape[0] == 0:
        return np.zeros(N_AUS * 6, dtype=np.float32)
    mean = seg.mean(axis=0)
    std = seg.std(axis=0)
    mx = seg.max(axis=0)
    mn = seg.min(axis=0)
    rng = mx - mn
    med = np.median(seg, axis=0)
    return np.concatenate([mean, std, mx, mn, rng, med]).astype(np.float32)


def segment_au_by_turns(aus: np.ndarray, timestamps: np.ndarray, turns: list[TurnInfo]) -> list[np.ndarray]:
    segments: list[np.ndarray] = []
    for turn in turns:
        mask = (timestamps >= turn.start_s) & (timestamps < turn.end_s)
        seg = aus[mask]
        if seg.shape[0] == 0:
            seg = np.zeros((0, N_AUS), dtype=np.float32)
        segments.append(seg)
    return segments


def extract_au_features(sessions: list[SessionData], au_extractor: DepthwiseConv1DExtractor) -> np.ndarray:
    conv_dim = au_extractor.output_dim
    stats_dim = N_AUS * 6
    out_dim = conv_dim + stats_dim

    X = np.zeros((len(sessions), out_dim), dtype=np.float32)
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for i, s in enumerate(sessions):
        if s.openface_path is None:
            continue

        key = str(s.openface_path)
        if key not in cache:
            try:
                cache[key] = load_openface_au(s.openface_path)
            except Exception as e:
                log.warning("Failed OpenFace load (%s): %s", s.openface_path, e)
                continue

        aus, timestamps = cache[key]
        segments = segment_au_by_turns(aus, timestamps, s.turns)

        turn_vecs: list[np.ndarray] = []
        for seg in segments:
            parts: list[np.ndarray] = []
            if seg.shape[0] >= MIN_FRAMES_FOR_CONV:
                x = torch.tensor(seg.T, dtype=torch.float32).unsqueeze(0)
                feat = au_extractor(x).squeeze(0).cpu().numpy().astype(np.float32)
            else:
                feat = np.zeros(conv_dim, dtype=np.float32)
            parts.append(feat)
            parts.append(_au_stats(seg))
            turn_vecs.append(np.concatenate(parts))

        if turn_vecs:
            X[i] = np.mean(turn_vecs, axis=0)

    return X


def extract_text_features(
    sessions: list[SessionData],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 256,
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(model_name)
    dim = st.get_sentence_embedding_dimension()

    all_texts: list[str] = []
    idx_map: list[int] = []
    for i, s in enumerate(sessions):
        for t in s.turns:
            txt = (t.summary or "").strip()
            if txt:
                all_texts.append(txt)
                idx_map.append(i)

    if not all_texts:
        return np.zeros((len(sessions), dim), dtype=np.float32)

    embs = st.encode(all_texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)

    X = np.zeros((len(sessions), dim), dtype=np.float32)
    counts = np.zeros(len(sessions), dtype=np.int32)
    for i, emb in zip(idx_map, embs):
        X[i] += emb
        counts[i] += 1

    valid = counts > 0
    X[valid] /= counts[valid, None]
    return X


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


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
        log.info("TARGET: %s  [%s]", spec.name, spec.task)
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

                # LOTO (Ridge default)
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
                # classification
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

                # LOTO (LogReg default)
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    log.setLevel(getattr(logging, args.log_level.upper()))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets: dict[str, list[SessionData]] = {}
    for split in ("train", "val", "test"):
        datasets[split] = load_sessions(args.data_model, args.config, split)

    train_s = datasets["train"]
    val_s = datasets["val"]
    test_s = datasets["test"]
    all_s = train_s + val_s + test_s

    if not train_s:
        log.error("Train split is empty.")
        sys.exit(1)

    log.info("Extracting AU features ...")
    au_extractor = DepthwiseConv1DExtractor(
        n_channels=N_AUS,
        n_kernels_per_channel=args.n_kernels,
        kernel_sizes=[int(x) for x in args.kernel_sizes.split(",")],
        seed=args.seed,
    )
    X_au_all = extract_au_features(all_s, au_extractor)

    log.info("Extracting text features ...")
    X_text_all = extract_text_features(all_s, model_name=args.embed_model, batch_size=args.embed_batch_size)

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

    results_path = out_dir / "baseline_au_mixed_outcomes_results.json"
    results = run_all_evaluations(patient_records, results_path)

    log.info("Saved %d result rows -> %s", len(results), results_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AU-conv baseline with mixed outcomes (regression+classification) and cumulative sessions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--data_model", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, default=Path("results/baseline_au_mixed"))

    p.add_argument("--n_kernels", type=int, default=4)
    p.add_argument("--kernel_sizes", type=str, default="7,15,31")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--embed_model", type=str, default="all-MiniLM-L6-v2")
    p.add_argument("--embed_batch_size", type=int, default=256)

    p.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING"])

    return p.parse_args()


if __name__ == "__main__":
    main()
