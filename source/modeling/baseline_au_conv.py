"""
Baseline diagnostic: raw AU time-series (1D Conv) + text summaries for
treatment-type classification (bindung / personal / wunder).

Purpose
-------
Tests whether raw OpenFace AU intensities carry predictive signal for
therapy treatment type, using depthwise 1D convolutions (one filter bank
per AU channel) to reduce each variable-length turn into a fixed-size
feature vector.

Two feature modalities, evaluated independently and jointly:

* **AU features** – per-turn AU intensity time-series from the patient's
  OpenFace CSV, segmented by transcript turn boundaries, reduced via
  depthwise Conv1D + global pooling.
* **Text features** – per-turn speech summaries encoded with a frozen
  sentence-transformer (``all-MiniLM-L6-v2``).

Session vectors are obtained by mean-pooling turn vectors, then fed to a
battery of sklearn classifiers (Logistic Regression, SVC, RF, GBC).

Additionally performs leave-one-therapist-out cross-validation.

Usage
-----
    python source/modeling/baseline_au_conv.py \\
        --data_model data_model.yaml \\
        --config config.yaml \\
        --output_dir results/baseline_au_classification
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

import torch
import torch.nn as nn

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
N_AUS: int = len(AU_R_COLS)  # 17

BLRI_LABEL_MAP: dict[str, tuple[str, str]] = {
    "bindung":  ("T5_BLRI_ges_Pr", "T5_BLRI_ges_In"),
    "personal": ("T3_BLRI_ges_Pr", "T3_BLRI_ges_In"),
    "wunder":   ("T7_BLRI_ges_Pr", "T7_BLRI_ges_In"),
}

# Treatment-type classification labels
CLASS_LABELS: list[str] = ["bindung", "personal", "wunder"]
CLASS_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(CLASS_LABELS)}
N_CLASSES: int = len(CLASS_LABELS)

# ---------------------------------------------------------------------------
# Path helpers  (same logic as dataset.py)
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
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TurnInfo:
    """Timing and text for a single speech turn."""
    turn_index: int
    speaker_id: str
    start_s: float          # seconds
    end_s: float            # seconds
    summary: str = ""       # may be empty if not available


@dataclass
class SessionData:
    """One therapy session with all modalities."""
    patient_id: str
    therapist_id: str
    interview_type: str
    turns: list[TurnInfo]
    blri_pr: float | None = None
    blri_in: float | None = None
    # Resolved paths (None if unavailable)
    openface_path: Path | None = None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_openface_au(
    path: Path, confidence_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Load patient OpenFace CSV → (AU intensities, timestamps).

    Returns
    -------
    aus : np.ndarray, shape (T, 17)
        AU intensity values for frames passing quality filters.
    timestamps : np.ndarray, shape (T,)
        Timestamp in seconds for each retained frame.
    """
    import pandas as pd

    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # Quality filter
    mask = (df["success"] == 1) & (df["confidence"] >= confidence_threshold)
    aus = df.loc[mask, AU_R_COLS].values.astype(np.float32)
    timestamps = df.loc[mask, "timestamp"].values.astype(np.float64)
    return aus, timestamps


def segment_au_by_turns(
    aus: np.ndarray,
    timestamps: np.ndarray,
    turns: list[TurnInfo],
) -> list[np.ndarray]:
    """Slice the AU matrix by turn boundaries.

    Returns a list of arrays, one per turn, each of shape (t_i, 17).
    If a turn has no frames (face not detected), returns an empty (0, 17) array.
    """
    segments: list[np.ndarray] = []
    for turn in turns:
        mask = (timestamps >= turn.start_s) & (timestamps < turn.end_s)
        seg = aus[mask]
        if seg.shape[0] == 0:
            seg = np.zeros((0, N_AUS), dtype=np.float32)
        segments.append(seg)
    return segments


def load_sessions(
    data_model_path: Path,
    config_path: Path,
    split: str,
) -> list[SessionData]:
    """Load sessions for a given split from data_model.yaml.

    Parameters
    ----------
    data_model_path : Path
        Path to ``data_model.yaml``.
    config_path : Path
        Path to ``config.yaml`` with therapist-based splits.
    split : str
        One of ``"train"``, ``"val"``, ``"test"``.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    allowed = set(config["psychotherapy_splits"][split])

    with open(data_model_path, "r", encoding="utf-8") as f:
        data_model = yaml.safe_load(f)

    sessions: list[SessionData] = []
    skipped_unknown_type = 0
    skipped_no_transcript = 0
    skipped_no_openface = 0

    for interview in data_model["interviews"]:
        tid = interview["therapist"]["therapist_id"]
        if tid not in allowed:
            continue
        pid = interview["patient"]["patient_id"]

        for itype, idata in interview.get("types", {}).items():
            # --- Classification target: interview type ---
            if itype not in CLASS_TO_IDX:
                skipped_unknown_type += 1
                continue

            # --- Optional BLRI labels (kept for metadata) ---
            labels = idata.get("labels", {})
            pr_key, in_key = BLRI_LABEL_MAP.get(itype, (None, None))
            blri_pr = labels.get(pr_key) if pr_key else None
            blri_in = labels.get(in_key) if in_key else None
            if _is_nan(blri_pr):
                blri_pr = None
            if _is_nan(blri_in):
                blri_in = None

            # --- Transcript (for turn boundaries + optional summaries) ---
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
                tidx = t.get("turn_index", 0)
                start_ms = t.get("start_ms", 0)
                end_ms = t.get("end_ms", 0)
                summary = t.get("summary", "").strip()
                turns.append(TurnInfo(
                    turn_index=int(tidx),
                    speaker_id=t.get("speaker_id", ""),
                    start_s=start_ms / 1000.0,
                    end_s=end_ms / 1000.0,
                    summary=summary,
                ))

            # --- Patient OpenFace ---
            of_raw = idata.get("patient_openface", "")
            of_path = _resolve_path(of_raw) if of_raw else None
            if of_path is None:
                skipped_no_openface += 1
                # Still load session; AU features will just be zeros
                # This allows text-only evaluation

            sessions.append(SessionData(
                patient_id=pid,
                therapist_id=tid,
                interview_type=itype,
                turns=turns,
                blri_pr=float(blri_pr) if blri_pr is not None else None,
                blri_in=float(blri_in) if blri_in is not None else None,
                openface_path=of_path,
            ))

    log.info(
        "[%s] Loaded %d sessions  (skipped: %d unknown type, %d no transcript, %d no openface)",
        split.upper(), len(sessions), skipped_unknown_type, skipped_no_transcript, skipped_no_openface,
    )
    return sessions


# ---------------------------------------------------------------------------
# 1D Conv AU Feature Extractor (frozen random kernels – ROCKET-inspired)
# ---------------------------------------------------------------------------


class DepthwiseConv1DExtractor(nn.Module):
    """Fixed (non-trainable) depthwise 1D convolutions for AU time-series.

    One independent filter bank per AU channel.  Uses multiple kernel sizes
    to capture patterns at different temporal scales.

    Parameters
    ----------
    n_channels : int
        Number of AU channels (17).
    n_kernels_per_channel : int
        Number of random kernels per AU per kernel size.
    kernel_sizes : list[int]
        Conv kernel widths (in frames; at 30 fps, 7≈0.23s, 15≈0.5s, 31≈1s).
    seed : int
        Random seed for reproducible kernel initialisation.

    Output dimension = n_channels * n_kernels_per_channel * len(kernel_sizes) * 2
    (×2 because we compute both global max-pool and global avg-pool per kernel).
    """

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

        self.n_channels = n_channels
        self.n_kernels = n_kernels_per_channel
        self.kernel_sizes = kernel_sizes

        torch.manual_seed(seed)
        self.convs = nn.ModuleList()
        for ks in kernel_sizes:
            conv = nn.Conv1d(
                in_channels=n_channels,
                out_channels=n_channels * n_kernels_per_channel,
                kernel_size=ks,
                groups=n_channels,   # depthwise: each AU has its own kernels
                padding=ks // 2,
                bias=True,
            )
            conv.requires_grad_(False)
            self.convs.append(conv)

        self.output_dim = n_channels * n_kernels_per_channel * len(kernel_sizes) * 2

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape (batch, n_channels, T)

        Returns
        -------
        Tensor of shape (batch, output_dim)
        """
        features: list[torch.Tensor] = []
        for conv in self.convs:
            h = conv(x)                             # (batch, C*K, T')
            h_max = h.max(dim=-1).values            # (batch, C*K)
            h_avg = h.mean(dim=-1)                   # (batch, C*K)
            features.append(h_max)
            features.append(h_avg)
        return torch.cat(features, dim=-1)           # (batch, output_dim)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

# Minimum frames for conv processing; shorter segments → stats fallback
MIN_FRAMES_FOR_CONV = 3


def _au_stats(seg: np.ndarray) -> np.ndarray:
    """Temporal statistics per AU for a single turn segment.

    Parameters
    ----------
    seg : (T, 17)

    Returns
    -------
    (17 * 6,) = (102,) vector: mean, std, max, min, range, median per AU.
    """
    if seg.shape[0] == 0:
        return np.zeros(N_AUS * 6, dtype=np.float32)
    mean = seg.mean(axis=0)
    std = seg.std(axis=0)
    mx = seg.max(axis=0)
    mn = seg.min(axis=0)
    rng = mx - mn
    med = np.median(seg, axis=0)
    return np.concatenate([mean, std, mx, mn, rng, med]).astype(np.float32)


def extract_au_features(
    sessions: list[SessionData],
    au_extractor: DepthwiseConv1DExtractor,
    include_stats: bool = True,
) -> np.ndarray:
    """Extract AU features for all sessions.

    For each session:
      1. Load the patient OpenFace CSV.
      2. Segment by turn boundaries.
      3. Apply Conv1D extractor to each turn segment.
      4. Optionally append temporal statistics.
      5. Mean-pool across turns → session AU vector.

    Returns
    -------
    np.ndarray of shape (n_sessions, au_dim)
    """
    import pandas as pd   # noqa: deferred import for HPC compatibility

    conv_dim = au_extractor.output_dim
    stats_dim = N_AUS * 6 if include_stats else 0
    au_dim = conv_dim + stats_dim

    X = np.zeros((len(sessions), au_dim), dtype=np.float32)
    loaded_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for i, session in enumerate(sessions):
        if session.openface_path is None:
            continue  # leave zeros

        # Cache OpenFace loading (same CSV may be used by different sessions?
        # No — each session has its own CSV.  But cache anyway for safety.)
        cache_key = str(session.openface_path)
        if cache_key not in loaded_cache:
            try:
                aus, timestamps = load_openface_au(session.openface_path)
                loaded_cache[cache_key] = (aus, timestamps)
            except Exception as e:
                log.warning("Failed to load %s: %s", session.openface_path, e)
                continue
        aus, timestamps = loaded_cache[cache_key]

        # Segment by turns
        segments = segment_au_by_turns(aus, timestamps, session.turns)

        # Per-turn features → list of vectors
        turn_features: list[np.ndarray] = []
        for seg in segments:
            parts: list[np.ndarray] = []

            # Conv1D features
            if seg.shape[0] >= MIN_FRAMES_FOR_CONV:
                x = torch.tensor(seg.T, dtype=torch.float32).unsqueeze(0)  # (1, 17, T)
                feat_t = au_extractor(x).squeeze(0)                        # (conv_dim,)
                feat = np.array(feat_t.tolist(), dtype=np.float32)
                parts.append(feat)
            else:
                parts.append(np.zeros(conv_dim, dtype=np.float32))

            # Stats features
            if include_stats:
                parts.append(_au_stats(seg))

            turn_features.append(np.concatenate(parts))

        if turn_features:
            X[i] = np.mean(turn_features, axis=0)

        if (i + 1) % 50 == 0 or i == len(sessions) - 1:
            log.info("  AU extraction: %d / %d sessions", i + 1, len(sessions))

    return X


def extract_text_features(
    sessions: list[SessionData],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 256,
) -> np.ndarray:
    """Encode speech summaries with a sentence-transformer, mean-pool per session.

    Sessions whose turns lack summaries get a zero vector.

    Returns
    -------
    np.ndarray of shape (n_sessions, embed_dim)
    """
    from sentence_transformers import SentenceTransformer

    log.info("Loading sentence-transformer '%s' ...", model_name)
    st_model = SentenceTransformer(model_name)
    dim = st_model.get_sentence_embedding_dimension()
    log.info("Text embedding dim = %d", dim)

    # Collect all summaries with (session_idx, turn_idx) tracking
    all_texts: list[str] = []
    index_map: list[tuple[int, int]] = []  # (session_idx, turn_idx)

    for si, session in enumerate(sessions):
        for ti, turn in enumerate(session.turns):
            if turn.summary:
                all_texts.append(turn.summary)
                index_map.append((si, ti))

    if not all_texts:
        log.warning("No speech summaries found in any session — text features will be all zeros.")
        return np.zeros((len(sessions), dim), dtype=np.float32)

    log.info("Encoding %d turn summaries ...", len(all_texts))
    embeddings = st_model.encode(
        all_texts, batch_size=batch_size, show_progress_bar=True,
        convert_to_numpy=True,
    )

    # Mean-pool per session
    X = np.zeros((len(sessions), dim), dtype=np.float32)
    counts = np.zeros(len(sessions), dtype=np.int32)
    for (si, _), emb in zip(index_map, embeddings):
        X[si] += emb
        counts[si] += 1

    nonzero = counts > 0
    X[nonzero] /= counts[nonzero, np.newaxis]

    n_with_text = int(nonzero.sum())
    log.info("Text features: %d/%d sessions have summaries", n_with_text, len(sessions))

    return X


# ---------------------------------------------------------------------------
# Targets & metrics
# ---------------------------------------------------------------------------


def extract_targets(sessions: list[SessionData]) -> np.ndarray:
    """Return integer class labels from interview_type.

    Returns
    -------
    y : np.ndarray of shape (n_sessions,), dtype int
        Class indices: bindung=0, personal=1, wunder=2.
    """
    return np.array([CLASS_TO_IDX[s.interview_type] for s in sessions], dtype=np.int64)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Classification metrics: accuracy, macro-F1, per-class F1, confusion matrix."""
    from sklearn.metrics import (
        accuracy_score, f1_score, classification_report, confusion_matrix,
    )

    acc = float(accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_per_class = f1_score(y_true, y_pred, average=None, labels=list(range(N_CLASSES)), zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        **{f"f1_{CLASS_LABELS[i]}": float(f1_per_class[i]) for i in range(N_CLASSES)},
        "confusion_matrix": cm.tolist(),
        "n": len(y_true),
    }


# ---------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------


def get_classifiers() -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        "LogReg(C=1)":     make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=42)),
        "LogReg(C=10)":    make_pipeline(StandardScaler(), LogisticRegression(C=10.0, max_iter=2000, random_state=42)),
        "LogReg(C=0.1)":   make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000, random_state=42)),
        "SVC(rbf)":        make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, random_state=42)),
        "SVC(rbf,C=10)":   make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10.0, random_state=42)),
        "RF(100)":         RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "GBC(100)":        GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42),
    }


# ---------------------------------------------------------------------------
# Evaluation routines
# ---------------------------------------------------------------------------


def run_fixed_split(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    label: str = "",
) -> list[dict]:
    results = []
    for name, clf in get_classifiers().items():
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        m = compute_metrics(y_test, pred)
        m.update(classifier=name, split=label)
        results.append(m)
        log.info(
            "  %-20s %-15s  Acc=%.4f  F1_macro=%.4f  F1_w=%.4f  (n=%d)",
            name, label, m["accuracy"], m["f1_macro"], m["f1_weighted"], m["n"],
        )
    return results


def run_loto_cv(
    all_sessions: list[SessionData], X_all: np.ndarray, feature_label: str,
) -> list[dict]:
    """Leave-one-therapist-out CV with LogisticRegression(C=1)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    therapist_ids = sorted(set(s.therapist_id for s in all_sessions))
    log.info("LOTO-CV across %d therapists: %s  [%s]", len(therapist_ids), therapist_ids, feature_label)

    y_all = extract_targets(all_sessions)

    all_true: list[int] = []
    all_pred: list[int] = []
    results: list[dict] = []

    for held_out in therapist_ids:
        tr_idx = [i for i, s in enumerate(all_sessions) if s.therapist_id != held_out]
        te_idx = [i for i, s in enumerate(all_sessions) if s.therapist_id == held_out]
        if len(tr_idx) < 5 or len(te_idx) == 0:
            continue

        # Check that training set has at least 2 classes
        n_classes_train = len(set(y_all[tr_idx]))
        if n_classes_train < 2:
            log.warning("  LOTO held=%s — only %d class(es) in train, skipping", held_out, n_classes_train)
            continue

        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=2000, random_state=42),
        )
        clf.fit(X_all[tr_idx], y_all[tr_idx])
        pred = clf.predict(X_all[te_idx])

        m = compute_metrics(y_all[te_idx], pred)
        m.update(held_out_therapist=held_out,
                 n_train=len(tr_idx), n_test=len(te_idx), features=feature_label)
        results.append(m)
        all_true.extend(y_all[te_idx].tolist())
        all_pred.extend(pred.tolist())
        log.info(
            "  LOTO held=%s  Acc=%.4f  F1_macro=%.4f  n_test=%d  [%s]",
            held_out, m["accuracy"], m["f1_macro"], len(te_idx), feature_label,
        )

    if all_true:
        agg = compute_metrics(np.array(all_true), np.array(all_pred))
        agg.update(type="loto_aggregate", features=feature_label)
        results.append(agg)
        log.info(
            "  LOTO AGG  Acc=%.4f  F1_macro=%.4f  F1_w=%.4f  (n=%d)  [%s]",
            agg["accuracy"], agg["f1_macro"], agg["f1_weighted"], agg["n"], feature_label,
        )

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    log.setLevel(getattr(logging, args.log_level.upper()))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load sessions
    # ------------------------------------------------------------------
    log.info("Loading sessions ...")
    datasets: dict[str, list[SessionData]] = {}
    for split in ("train", "val", "test"):
        datasets[split] = load_sessions(
            args.data_model, args.config, split,
        )

    train_s = datasets["train"]
    val_s = datasets["val"]
    test_s = datasets["test"]
    all_s = train_s + val_s + test_s

    if not train_s:
        log.error("Train split empty — aborting.")
        sys.exit(1)

    n_with_of = sum(1 for s in all_s if s.openface_path is not None)
    n_with_text = sum(1 for s in all_s if any(t.summary for t in s.turns))
    log.info("Total sessions: %d  (with OpenFace: %d, with text summaries: %d)", len(all_s), n_with_of, n_with_text)

    # Class distribution
    for split_name, split_sessions in [("train", train_s), ("val", val_s), ("test", test_s), ("all", all_s)]:
        from collections import Counter
        dist = Counter(s.interview_type for s in split_sessions)
        log.info("  [%s] class distribution: %s", split_name, dict(dist))

    # ------------------------------------------------------------------
    # 2. Extract AU features (Conv1D + stats)
    # ------------------------------------------------------------------
    log.info("=== EXTRACTING AU FEATURES ===")
    au_extractor = DepthwiseConv1DExtractor(
        n_channels=N_AUS,
        n_kernels_per_channel=args.n_kernels,
        kernel_sizes=[int(k) for k in args.kernel_sizes.split(",")],
        seed=args.seed,
    )
    log.info("Conv1D output dim = %d  (+ stats %d = total AU dim %d)",
             au_extractor.output_dim, N_AUS * 6, au_extractor.output_dim + N_AUS * 6)

    X_au_all = extract_au_features(all_s, au_extractor, include_stats=True)

    n_tr = len(train_s)
    n_va = len(val_s)
    X_au_train = X_au_all[:n_tr]
    X_au_val = X_au_all[n_tr:n_tr + n_va]
    X_au_test = X_au_all[n_tr + n_va:]

    # ------------------------------------------------------------------
    # 3. Extract text features
    # ------------------------------------------------------------------
    log.info("=== EXTRACTING TEXT FEATURES ===")
    X_txt_all = extract_text_features(all_s, model_name=args.embed_model, batch_size=args.embed_batch_size)

    X_txt_train = X_txt_all[:n_tr]
    X_txt_val = X_txt_all[n_tr:n_tr + n_va]
    X_txt_test = X_txt_all[n_tr + n_va:]

    # ------------------------------------------------------------------
    # 4. Build combined feature sets
    # ------------------------------------------------------------------
    feature_sets: dict[str, dict[str, np.ndarray]] = {
        "AU_only": {
            "train": X_au_train, "val": X_au_val, "test": X_au_test, "all": X_au_all,
        },
        "AU+Text": {
            "train": np.hstack([X_au_train, X_txt_train]),
            "val": np.hstack([X_au_val, X_txt_val]),
            "test": np.hstack([X_au_test, X_txt_test]),
            "all": np.hstack([X_au_all, X_txt_all]),
        },
    }

    # Only add text-only if summaries exist
    if n_with_text > 0:
        feature_sets["Text_only"] = {
            "train": X_txt_train, "val": X_txt_val, "test": X_txt_test, "all": X_txt_all,
        }

    # ------------------------------------------------------------------
    # 5. Targets (classification: treatment type)
    # ------------------------------------------------------------------
    y_train = extract_targets(train_s)
    y_val = extract_targets(val_s)
    y_test = extract_targets(test_s)

    log.info(
        "Targets — train: %d  val: %d  test: %d", len(y_train), len(y_val), len(y_test),
    )

    # ------------------------------------------------------------------
    # 6. Majority-class baseline
    # ------------------------------------------------------------------
    log.info("=== MAJORITY-CLASS BASELINE ===")
    from collections import Counter
    majority_class = Counter(y_train.tolist()).most_common(1)[0][0]
    majority_label = CLASS_LABELS[majority_class]

    for sn, y_eval in [("val", y_val), ("test", y_test)]:
        if len(y_eval) == 0:
            continue
        pred = np.full_like(y_eval, majority_class)
        m = compute_metrics(y_eval, pred)
        log.info(
            "  Majority(%s)  %-5s  Acc=%.4f  F1_macro=%.4f  (n=%d)",
            majority_label, sn, m["accuracy"], m["f1_macro"], m["n"],
        )

    # ------------------------------------------------------------------
    # 7. Fixed-split evaluation per feature set
    # ------------------------------------------------------------------
    all_results: list[dict] = []

    for feat_name, feat in feature_sets.items():
        log.info("=" * 70)
        log.info("FEATURE SET: %s  (dim=%d)", feat_name, feat["train"].shape[1])
        log.info("=" * 70)

        # train → val
        if len(y_val) > 0:
            log.info("--- train → val [%s] ---", feat_name)
            res = run_fixed_split(
                feat["train"], y_train,
                feat["val"], y_val,
                f"val({feat_name})",
            )
            all_results.extend(res)

        # train → test
        if len(y_test) > 0:
            log.info("--- train → test [%s] ---", feat_name)
            res = run_fixed_split(
                feat["train"], y_train,
                feat["test"], y_test,
                f"test({feat_name})",
            )
            all_results.extend(res)

        # train+val → test
        if len(y_test) > 0 and len(y_val) > 0:
            X_trv = np.vstack([feat["train"], feat["val"]])
            y_trv = np.concatenate([y_train, y_val])
            log.info("--- train+val → test [%s] ---", feat_name)
            res = run_fixed_split(X_trv, y_trv, feat["test"], y_test, f"test_tv({feat_name})")
            all_results.extend(res)

    # ------------------------------------------------------------------
    # 8. LOTO-CV per feature set
    # ------------------------------------------------------------------
    for feat_name, feat in feature_sets.items():
        log.info("=== LOTO-CV [%s] ===", feat_name)
        loto_res = run_loto_cv(all_s, feat["all"], feat_name)
        all_results.extend(loto_res)

    # ------------------------------------------------------------------
    # 9. Save results
    # ------------------------------------------------------------------
    results_path = output_dir / "baseline_au_classification_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info("Results saved to %s", results_path)

    # ------------------------------------------------------------------
    # 10. Summary table
    # ------------------------------------------------------------------
    log.info("")
    log.info("=" * 110)
    log.info("SUMMARY — FIXED SPLITS (Treatment-Type Classification)")
    log.info("=" * 110)
    log.info(
        "%-20s %-20s %8s %10s %10s %6s",
        "Classifier", "Split(Features)", "Acc", "F1_macro", "F1_wtd", "n",
    )
    log.info("-" * 110)
    for r in all_results:
        if "classifier" in r:
            log.info(
                "%-20s %-20s %8.4f %10.4f %10.4f %6d",
                r["classifier"], r.get("split", ""),
                r["accuracy"], r["f1_macro"], r["f1_weighted"], r["n"],
            )
    log.info("=" * 110)

    log.info("")
    log.info("=" * 110)
    log.info("SUMMARY — LOTO-CV AGGREGATES")
    log.info("=" * 110)
    for r in all_results:
        if r.get("type") == "loto_aggregate":
            log.info(
                "  %-15s  Acc=%.4f  F1_macro=%.4f  F1_w=%.4f  (n=%d)",
                r.get("features", ""),
                r["accuracy"], r["f1_macro"], r["f1_weighted"], r["n"],
            )
    log.info("=" * 110)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Baseline: raw AU (1D Conv) + text summaries for treatment-type classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data paths
    p.add_argument("--data_model", type=Path, required=True, help="Path to data_model.yaml")
    p.add_argument("--config", type=Path, required=True, help="Path to config.yaml (split definitions)")
    p.add_argument("--output_dir", type=Path, default=Path("results/baseline_au_classification"),
                   help="Where to save results")

    # Conv1D parameters
    p.add_argument("--n_kernels", type=int, default=4,
                   help="Number of random conv kernels per AU per kernel size")
    p.add_argument("--kernel_sizes", type=str, default="7,15,31",
                   help="Comma-separated kernel sizes (frames; 30fps)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for conv kernel initialisation")

    # Text embedding
    p.add_argument("--embed_model", type=str, default="all-MiniLM-L6-v2",
                   help="Sentence-transformer model for text summaries")
    p.add_argument("--embed_batch_size", type=int, default=256,
                   help="Batch size for sentence-transformer encoding")

    # Misc
    p.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING"])

    return p.parse_args()


if __name__ == "__main__":
    main()
