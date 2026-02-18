"""
Baseline diagnostic: raw AU time-series (1D Conv) + text summaries for
multi-target regression of ALL numeric T3 / T5 / T7 outcomes.

Purpose
-------
Predicts all numeric outcomes from data_model.yaml — PANAS (positive &
negative affect), IRF (interpersonal reaction: self & other), and BLRI
(global empathy) scores for both Practitioner (*_Pr*) and Instructor
(*_In*) ratings — using raw OpenFace AU intensities and/or speech
summaries.

Depthwise 1D convolutions (one filter bank per AU channel) reduce each
variable-length AU turn into a fixed-size feature vector.

Three feature modalities are evaluated in an **ablation study**:

* **AU_only**  – per-turn AU intensity time-series from the patient's
  OpenFace CSV, segmented by transcript turn boundaries, reduced via
  depthwise Conv1D + global pooling + temporal statistics.
* **Text_only** – per-turn speech summaries encoded with a frozen
  sentence-transformer (``all-MiniLM-L6-v2``).
* **AU+Text**  – concatenation of both.

Session vectors are obtained by mean-pooling turn vectors, then fed to a
battery of sklearn regressors (Ridge, Lasso, SVR, RF, GBR).

Performs fixed-split evaluation AND leave-one-therapist-out cross-
validation (LOTO-CV).  Results are saved as a comprehensive JSON and a
ranked summary of which outcomes are best predicted is printed.

Usage
-----
    python source/modeling/baseline_au_conv.py \\
        --data_model data_model.yaml \\
        --config config.yaml \\
        --output_dir results/baseline_au_regression
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
from collections import Counter, defaultdict
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

# Interview type → time-point prefix for label keys in data_model.yaml
INTERVIEW_PREFIX: dict[str, str] = {
    "bindung":  "T5",
    "personal": "T3",
    "wunder":   "T7",
}

# All outcome suffixes (after stripping the T3_/T5_/T7_ prefix)
OUTCOME_SUFFIXES: list[str] = [
    "PANAS_pos_Pr", "PANAS_neg_Pr",
    "IRF_self_Pr",  "IRF_other_Pr",
    "BLRI_ges_Pr",
    "PANAS_pos_In", "PANAS_neg_In",
    "IRF_self_In",  "IRF_other_In",
    "BLRI_ges_In",
]

# Full label-key map: {interview_type: {outcome_suffix: yaml_label_key}}
LABEL_MAP: dict[str, dict[str, str]] = {
    itype: {suffix: f"{prefix}_{suffix}" for suffix in OUTCOME_SUFFIXES}
    for itype, prefix in INTERVIEW_PREFIX.items()
}

# For backwards compatibility / reference
BLRI_LABEL_MAP: dict[str, tuple[str, str]] = {
    "bindung":  ("T5_BLRI_ges_Pr", "T5_BLRI_ges_In"),
    "personal": ("T3_BLRI_ges_Pr", "T3_BLRI_ges_In"),
    "wunder":   ("T7_BLRI_ges_Pr", "T7_BLRI_ges_In"),
}

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
    labels: dict[str, float] = field(default_factory=dict)  # full yaml key (e.g. T5_BLRI_ges_Pr) → value
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
            if itype not in INTERVIEW_PREFIX:
                skipped_unknown_type += 1
                continue

            # --- Extract ALL numeric labels for this interview type ---
            labels_raw = idata.get("labels", {})
            labels: dict[str, float] = {}
            for suffix, yaml_key in LABEL_MAP[itype].items():
                val = labels_raw.get(yaml_key)
                if not _is_nan(val):
                    labels[yaml_key] = float(val)

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
                labels=labels,
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

        # Cache OpenFace loading
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
# Targets & metrics (REGRESSION)
# ---------------------------------------------------------------------------


def extract_targets(
    sessions: list[SessionData], full_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return targets and validity mask for a specific outcome.

    Parameters
    ----------
    sessions : list of SessionData
    full_key : str
        Full yaml label key (e.g. ``"T5_BLRI_ges_Pr"``, ``"T3_PANAS_pos_In"``).
        Only sessions whose ``labels`` dict contains this key will have
        valid entries; effectively this restricts to the matching interview
        type.

    Returns
    -------
    y : np.ndarray of shape (n_sessions,)
        Target values (NaN where missing / wrong interview type).
    valid : np.ndarray of shape (n_sessions,), dtype bool
        True where a valid label exists.
    """
    y = np.full(len(sessions), np.nan, dtype=np.float64)
    for i, s in enumerate(sessions):
        v = s.labels.get(full_key)
        if v is not None:
            y[i] = v
    valid = ~np.isnan(y)
    return y, valid


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Regression metrics: MSE, MAE, R², Pearson r."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from scipy.stats import pearsonr

    n = len(y_true)
    mse = float(mean_squared_error(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred)) if n > 1 else float("nan")

    if n > 2 and np.std(y_true) > 1e-12 and np.std(y_pred) > 1e-12:
        r, p = pearsonr(y_true, y_pred)
        r, p = float(r), float(p)
    else:
        r, p = 0.0, 1.0

    return {"mse": mse, "mae": mae, "r2": r2, "pearson_r": r, "pearson_p": p, "n": n}


# ---------------------------------------------------------------------------
# Regressors
# ---------------------------------------------------------------------------


def get_regressors() -> dict[str, Any]:
    from sklearn.linear_model import Ridge, Lasso
    from sklearn.svm import SVR
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        "Ridge(a=1)":   make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Ridge(a=10)":  make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "Ridge(a=0.1)": make_pipeline(StandardScaler(), Ridge(alpha=0.1)),
        "Lasso(a=0.1)": make_pipeline(StandardScaler(), Lasso(alpha=0.1, max_iter=5000)),
        "SVR(rbf)":     make_pipeline(StandardScaler(), SVR(kernel="rbf", C=1.0)),
        "RF(100)":      RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "GBR(100)":     GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
    }


# ---------------------------------------------------------------------------
# Evaluation routines
# ---------------------------------------------------------------------------


def run_reg_fixed_split(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    outcome: str, feature_label: str, split_label: str,
) -> list[dict]:
    """Train all regressors on a fixed split and return per-regressor metrics."""
    results: list[dict] = []
    best_r2 = -np.inf
    best_name = ""

    for name, reg in get_regressors().items():
        try:
            reg.fit(X_train, y_train)
            pred = reg.predict(X_test)
            m = compute_metrics(y_test, pred)
            m.update(regressor=name, outcome=outcome, features=feature_label, split=split_label)
            results.append(m)
            if m["r2"] > best_r2:
                best_r2 = m["r2"]
                best_name = name
        except Exception as e:
            log.warning("  %s failed on %s/%s: %s", name, outcome, feature_label, e)

    if best_name:
        best_r = next((r["pearson_r"] for r in results if r["regressor"] == best_name), 0)
        log.info(
            "  %-18s %-10s %-18s best=%-14s R²=%7.4f  r=%7.4f  (n=%d)",
            outcome, feature_label, split_label, best_name, best_r2, best_r, len(y_test),
        )
    return results


def run_reg_loto_cv(
    all_sessions: list[SessionData],
    X_all: np.ndarray,
    outcome: str,
    feature_label: str,
) -> list[dict]:
    """Leave-one-therapist-out CV with Ridge(alpha=1) for one outcome."""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y_all, valid = extract_targets(all_sessions, outcome)
    n_valid = int(valid.sum())
    if n_valid < 10:
        log.info("  LOTO [%s/%s] skipped — only %d valid samples", outcome, feature_label, n_valid)
        return []

    therapist_ids = sorted(set(s.therapist_id for s in all_sessions))

    all_true: list[float] = []
    all_pred: list[float] = []
    fold_results: list[dict] = []

    for held_out in therapist_ids:
        tr_idx = [i for i, s in enumerate(all_sessions)
                  if s.therapist_id != held_out and valid[i]]
        te_idx = [i for i, s in enumerate(all_sessions)
                  if s.therapist_id == held_out and valid[i]]
        if len(tr_idx) < 5 or len(te_idx) == 0:
            continue

        reg = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        reg.fit(X_all[tr_idx], y_all[tr_idx])
        pred = reg.predict(X_all[te_idx])

        m = compute_metrics(y_all[te_idx], pred)
        m.update(
            held_out_therapist=held_out,
            n_train=len(tr_idx), n_test=len(te_idx),
            outcome=outcome, features=feature_label,
        )
        fold_results.append(m)
        all_true.extend(y_all[te_idx].tolist())
        all_pred.extend(pred.tolist())

    results: list[dict] = list(fold_results)
    if len(all_true) > 2:
        agg = compute_metrics(np.array(all_true), np.array(all_pred))
        agg.update(
            type="loto_aggregate", outcome=outcome, features=feature_label,
            n_valid=n_valid, n_folds=len(fold_results),
        )
        results.append(agg)
        log.info(
            "  LOTO AGG  %-18s %-10s  R²=%7.4f  r=%7.4f  MAE=%7.4f  (n=%d, %d folds)",
            outcome, feature_label,
            agg["r2"], agg["pearson_r"], agg["mae"], agg["n"], len(fold_results),
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
    log.info(
        "Total sessions: %d  (with OpenFace: %d, with text: %d)",
        len(all_s), n_with_of, n_with_text,
    )

    # Label availability overview (grouped by interview type)
    log.info("Label availability (per interview type):")
    for itype, prefix in INTERVIEW_PREFIX.items():
        n_type = sum(1 for s in all_s if s.interview_type == itype)
        log.info("  --- %s (n=%d) ---", itype, n_type)
        for suffix in OUTCOME_SUFFIXES:
            full_key = f"{prefix}_{suffix}"
            _, valid = extract_targets(all_s, full_key)
            log.info("    %-25s  %d / %d", full_key, int(valid.sum()), n_type)

    # Interview type distribution
    for split_name, split_sessions in [("train", train_s), ("val", val_s), ("test", test_s), ("all", all_s)]:
        dist = Counter(s.interview_type for s in split_sessions)
        log.info("  [%s] type distribution: %s", split_name, dict(dist))

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
    log.info(
        "Conv1D output dim = %d  (+ stats %d = total AU dim %d)",
        au_extractor.output_dim, N_AUS * 6, au_extractor.output_dim + N_AUS * 6,
    )

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
    X_txt_all = extract_text_features(
        all_s, model_name=args.embed_model, batch_size=args.embed_batch_size,
    )

    X_txt_train = X_txt_all[:n_tr]
    X_txt_val = X_txt_all[n_tr:n_tr + n_va]
    X_txt_test = X_txt_all[n_tr + n_va:]

    # ------------------------------------------------------------------
    # 4. Build feature sets (ablation)
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
    # 5. Multi-target regression: loop over interview types × outcomes
    #    Each full label key (e.g. T5_BLRI_ges_Pr) is only available for
    #    sessions of the matching interview type.  The validity mask from
    #    extract_targets() enforces this automatically since labels are
    #    stored with the full key.
    # ------------------------------------------------------------------
    all_results: list[dict] = []
    loto_summary: list[dict] = []  # LOTO aggregates for the final ranking

    # Flat list of all (type, suffix, full_key) triples for summary tables
    ALL_FULL_KEYS: list[tuple[str, str, str]] = [
        (itype, suffix, f"{prefix}_{suffix}")
        for itype, prefix in INTERVIEW_PREFIX.items()
        for suffix in OUTCOME_SUFFIXES
    ]

    for itype, prefix in INTERVIEW_PREFIX.items():
        log.info("")
        log.info("#" * 80)
        log.info("INTERVIEW TYPE: %s  (prefix=%s)", itype, prefix)
        log.info("#" * 80)

        for suffix in OUTCOME_SUFFIXES:
            full_key = f"{prefix}_{suffix}"
            log.info("")
            log.info("=" * 80)
            log.info("OUTCOME: %s", full_key)
            log.info("=" * 80)

            # --- Fixed-split evaluation ---
            y_train, v_train = extract_targets(train_s, full_key)
            y_val, v_val = extract_targets(val_s, full_key)
            y_test, v_test = extract_targets(test_s, full_key)

            log.info(
                "  Valid labels — train: %d/%d  val: %d/%d  test: %d/%d",
                int(v_train.sum()), len(train_s),
                int(v_val.sum()), len(val_s),
                int(v_test.sum()), len(test_s),
            )

            for feat_name, feat in feature_sets.items():
                # train → val
                if v_train.sum() >= 5 and v_val.sum() >= 2:
                    res = run_reg_fixed_split(
                        feat["train"][v_train], y_train[v_train],
                        feat["val"][v_val], y_val[v_val],
                        full_key, feat_name, "train→val",
                    )
                    all_results.extend(res)

                # train → test
                if v_train.sum() >= 5 and v_test.sum() >= 2:
                    res = run_reg_fixed_split(
                        feat["train"][v_train], y_train[v_train],
                        feat["test"][v_test], y_test[v_test],
                        full_key, feat_name, "train→test",
                    )
                    all_results.extend(res)

                # train+val → test
                if v_train.sum() + v_val.sum() >= 5 and v_test.sum() >= 2:
                    X_trv = np.vstack([feat["train"], feat["val"]])
                    y_trv = np.concatenate([y_train, y_val])
                    v_trv = np.concatenate([v_train, v_val])
                    res = run_reg_fixed_split(
                        X_trv[v_trv], y_trv[v_trv],
                        feat["test"][v_test], y_test[v_test],
                        full_key, feat_name, "train+val→test",
                    )
                    all_results.extend(res)

            # --- LOTO-CV ---
            log.info("--- LOTO-CV for %s ---", full_key)
            for feat_name, feat in feature_sets.items():
                loto_res = run_reg_loto_cv(all_s, feat["all"], full_key, feat_name)
                all_results.extend(loto_res)
                # Collect aggregates for summary
                for r in loto_res:
                    if r.get("type") == "loto_aggregate":
                        loto_summary.append(r)

    # ------------------------------------------------------------------
    # 6. Save results
    # ------------------------------------------------------------------
    results_path = output_dir / "baseline_au_regression_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info("Results saved to %s  (%d entries)", results_path, len(all_results))

    # ------------------------------------------------------------------
    # 7. Summary tables
    # ------------------------------------------------------------------

    # --- 7a. Full LOTO table sorted by R² ---
    log.info("")
    log.info("=" * 130)
    log.info("SUMMARY — LOTO-CV AGGREGATES (all outcomes × feature sets, sorted by R²)")
    log.info("=" * 130)
    log.info(
        "%-26s %-12s %8s %10s %10s %10s %6s",
        "Outcome", "Features", "R²", "Pearson_r", "Pearson_p", "MAE", "n",
    )
    log.info("-" * 130)

    loto_sorted = sorted(loto_summary, key=lambda r: r.get("r2", -999), reverse=True)
    for r in loto_sorted:
        log.info(
            "%-26s %-12s %8.4f %10.4f %10.4f %10.4f %6d",
            r["outcome"], r["features"],
            r["r2"], r["pearson_r"], r.get("pearson_p", 0), r["mae"], r["n"],
        )
    log.info("=" * 130)

    # --- 7b. Best modality per outcome ---
    log.info("")
    log.info("=" * 130)
    log.info("BEST MODALITY PER OUTCOME (by LOTO R²)")
    log.info("=" * 130)
    log.info(
        "%-26s %-12s %8s %10s %10s %6s",
        "Outcome", "Best_feat", "R²", "Pearson_r", "MAE", "n",
    )
    log.info("-" * 130)

    best_per_outcome: dict[str, dict] = {}
    for r in loto_summary:
        oc = r["outcome"]
        if oc not in best_per_outcome or r["r2"] > best_per_outcome[oc]["r2"]:
            best_per_outcome[oc] = r

    for _itype, _suffix, full_key in ALL_FULL_KEYS:
        if full_key in best_per_outcome:
            r = best_per_outcome[full_key]
            log.info(
                "%-26s %-12s %8.4f %10.4f %10.4f %6d",
                full_key, r["features"], r["r2"], r["pearson_r"], r["mae"], r["n"],
            )
        else:
            log.info("%-26s  (no results — insufficient data)", full_key)
    log.info("=" * 130)

    # --- 7c. Best outcome per modality ---
    log.info("")
    log.info("=" * 130)
    log.info("BEST OUTCOME PER MODALITY (by LOTO R²)")
    log.info("=" * 130)
    for feat_name in feature_sets:
        feat_results = [r for r in loto_summary if r["features"] == feat_name]
        if feat_results:
            best = max(feat_results, key=lambda r: r["r2"])
            log.info(
                "  %-12s → best outcome: %-26s  R²=%7.4f  r=%7.4f  MAE=%7.4f  (n=%d)",
                feat_name, best["outcome"],
                best["r2"], best["pearson_r"], best["mae"], best["n"],
            )
    log.info("=" * 130)

    # --- 7d. Ablation delta: AU+Text vs individual modalities ---
    log.info("")
    log.info("=" * 130)
    log.info("ABLATION DELTAS (AU+Text R² minus single-modality R²)")
    log.info("=" * 130)
    log.info(
        "%-26s %10s %10s %12s %12s",
        "Outcome", "AU_only", "Text_only", "AU+Text", "Δ(best→comb)",
    )
    log.info("-" * 130)

    loto_by_key: dict[tuple[str, str], dict] = {}
    for r in loto_summary:
        loto_by_key[(r["outcome"], r["features"])] = r

    for _itype, _suffix, full_key in ALL_FULL_KEYS:
        r_au = loto_by_key.get((full_key, "AU_only"))
        r_tx = loto_by_key.get((full_key, "Text_only"))
        r_at = loto_by_key.get((full_key, "AU+Text"))

        au_r2 = r_au["r2"] if r_au else float("nan")
        tx_r2 = r_tx["r2"] if r_tx else float("nan")
        at_r2 = r_at["r2"] if r_at else float("nan")

        best_single = max(
            x for x in [au_r2, tx_r2] if not (x != x)  # filter NaN
        ) if any(not (x != x) for x in [au_r2, tx_r2]) else float("nan")

        delta = at_r2 - best_single if not (at_r2 != at_r2 or best_single != best_single) else float("nan")

        log.info(
            "%-26s %10s %10s %12s %12s",
            full_key,
            f"{au_r2:8.4f}" if not (au_r2 != au_r2) else "   N/A   ",
            f"{tx_r2:8.4f}" if not (tx_r2 != tx_r2) else "   N/A   ",
            f"{at_r2:8.4f}" if not (at_r2 != at_r2) else "   N/A   ",
            f"{delta:+8.4f}" if not (delta != delta) else "   N/A   ",
        )
    log.info("=" * 130)

    log.info("\nDone.  %d total result entries saved.", len(all_results))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Baseline: raw AU (1D Conv) + text summaries — "
            "multi-target regression of all T3/T5/T7 outcomes"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data paths
    p.add_argument("--data_model", type=Path, required=True,
                   help="Path to data_model.yaml")
    p.add_argument("--config", type=Path, required=True,
                   help="Path to config.yaml (split definitions)")
    p.add_argument("--output_dir", type=Path,
                   default=Path("results/baseline_au_regression"),
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
    p.add_argument("--log_level", type=str, default="INFO",
                   choices=["DEBUG", "INFO", "WARNING"])

    return p.parse_args()


if __name__ == "__main__":
    main()
