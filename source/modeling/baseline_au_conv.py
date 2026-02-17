"""
Baseline diagnostic: raw AU time-series (1D Conv) + text summaries for BLRI.

Purpose
-------
Tests whether raw OpenFace AU intensities carry predictive signal for BLRI
scores, using depthwise 1D convolutions (one filter bank per AU channel) to
reduce each variable-length turn into a fixed-size feature vector.

Two feature modalities, evaluated independently and jointly:

* **AU features** – per-turn AU intensity time-series from the patient's
  OpenFace CSV, segmented by transcript turn boundaries, reduced via
  depthwise Conv1D + global pooling.
* **Text features** – per-turn speech summaries encoded with a frozen
  sentence-transformer (``all-MiniLM-L6-v2``).

Session vectors are obtained by mean-pooling turn vectors, then fed to a
battery of sklearn regressors (Ridge, SVR, Lasso, RF, GBR).

Additionally performs leave-one-therapist-out cross-validation.

Usage
-----
    python source/modeling/baseline_au_conv.py \\
        --data_model data_model.yaml \\
        --config config.yaml \\
        --output_dir results/baseline_au
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
    skipped_no_target = 0
    skipped_no_transcript = 0
    skipped_no_openface = 0

    for interview in data_model["interviews"]:
        tid = interview["therapist"]["therapist_id"]
        if tid not in allowed:
            continue
        pid = interview["patient"]["patient_id"]

        for itype, idata in interview.get("types", {}).items():
            # --- BLRI targets ---
            labels = idata.get("labels", {})
            pr_key, in_key = BLRI_LABEL_MAP.get(itype, (None, None))
            blri_pr = labels.get(pr_key) if pr_key else None
            blri_in = labels.get(in_key) if in_key else None
            if _is_nan(blri_pr):
                blri_pr = None
            if _is_nan(blri_in):
                blri_in = None
            if blri_pr is None and blri_in is None:
                skipped_no_target += 1
                continue

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
        "[%s] Loaded %d sessions  (skipped: %d no target, %d no transcript, %d no openface)",
        split.upper(), len(sessions), skipped_no_target, skipped_no_transcript, skipped_no_openface,
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


def extract_targets(
    sessions: list[SessionData],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (y_pr, y_in, mask_pr, mask_in)."""
    y_pr = np.array([s.blri_pr if s.blri_pr is not None else np.nan for s in sessions])
    y_in = np.array([s.blri_in if s.blri_in is not None else np.nan for s in sessions])
    return y_pr, y_in, ~np.isnan(y_pr), ~np.isnan(y_in)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 2 else float("nan")
    return {"mse": mse, "mae": mae, "r2": r2, "pearson_r": corr, "n": len(y_true)}


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
        "Ridge(a=1)":     make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Ridge(a=10)":    make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "Ridge(a=100)":   make_pipeline(StandardScaler(), Ridge(alpha=100.0)),
        "Lasso(a=1)":     make_pipeline(StandardScaler(), Lasso(alpha=1.0, max_iter=5000)),
        "SVR(rbf)":       make_pipeline(StandardScaler(), SVR(kernel="rbf", C=1.0)),
        "SVR(rbf,C=10)":  make_pipeline(StandardScaler(), SVR(kernel="rbf", C=10.0)),
        "RF(100)":        RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "GBR(100)":       GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
    }


# ---------------------------------------------------------------------------
# Evaluation routines
# ---------------------------------------------------------------------------


def run_fixed_split(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    target_name: str,
    label: str = "",
) -> list[dict]:
    results = []
    for name, reg in get_regressors().items():
        reg.fit(X_train, y_train)
        pred = reg.predict(X_test)
        m = compute_metrics(y_test, pred)
        m.update(regressor=name, target=target_name, split=label)
        results.append(m)
        log.info(
            "  %-20s %-6s %-15s  R2=%+.4f  MSE=%7.2f  MAE=%5.2f  r=%+.4f  (n=%d)",
            name, label, target_name, m["r2"], m["mse"], m["mae"], m["pearson_r"], m["n"],
        )
    return results


def run_loto_cv(
    all_sessions: list[SessionData], X_all: np.ndarray, feature_label: str,
) -> list[dict]:
    """Leave-one-therapist-out CV with Ridge(alpha=10)."""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    therapist_ids = sorted(set(s.therapist_id for s in all_sessions))
    log.info("LOTO-CV across %d therapists: %s  [%s]", len(therapist_ids), therapist_ids, feature_label)

    y_pr, y_in, m_pr, m_in = extract_targets(all_sessions)

    results: list[dict] = []
    for target_name, y_all, mask_all in [("BLRI_Pr", y_pr, m_pr), ("BLRI_In", y_in, m_in)]:
        all_true: list[float] = []
        all_pred: list[float] = []

        for held_out in therapist_ids:
            tr_idx = [i for i, s in enumerate(all_sessions) if s.therapist_id != held_out and mask_all[i]]
            te_idx = [i for i, s in enumerate(all_sessions) if s.therapist_id == held_out and mask_all[i]]
            if len(tr_idx) < 5 or len(te_idx) == 0:
                continue

            reg = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
            reg.fit(X_all[tr_idx], y_all[tr_idx])
            pred = reg.predict(X_all[te_idx])

            m = compute_metrics(y_all[te_idx], pred)
            m.update(held_out_therapist=held_out, target=target_name,
                     n_train=len(tr_idx), n_test=len(te_idx), features=feature_label)
            results.append(m)
            all_true.extend(y_all[te_idx].tolist())
            all_pred.extend(pred.tolist())
            log.info(
                "  LOTO held=%s  %s  R2=%+.4f  MSE=%7.2f  n_test=%d  [%s]",
                held_out, target_name, m["r2"], m["mse"], len(te_idx), feature_label,
            )

        if all_true:
            agg = compute_metrics(np.array(all_true), np.array(all_pred))
            agg.update(target=target_name, type="loto_aggregate", features=feature_label)
            results.append(agg)
            log.info(
                "  LOTO AGG  %s  R2=%+.4f  MSE=%7.2f  MAE=%5.2f  r=%+.4f  (n=%d)  [%s]",
                target_name, agg["r2"], agg["mse"], agg["mae"], agg["pearson_r"], agg["n"], feature_label,
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
    # 5. Targets
    # ------------------------------------------------------------------
    y_pr_tr, y_in_tr, m_pr_tr, m_in_tr = extract_targets(train_s)
    y_pr_va, y_in_va, m_pr_va, m_in_va = extract_targets(val_s)
    y_pr_te, y_in_te, m_pr_te, m_in_te = extract_targets(test_s)

    log.info(
        "Valid targets — train: Pr=%d In=%d  val: Pr=%d In=%d  test: Pr=%d In=%d",
        m_pr_tr.sum(), m_in_tr.sum(), m_pr_va.sum(), m_in_va.sum(), m_pr_te.sum(), m_in_te.sum(),
    )

    # ------------------------------------------------------------------
    # 6. Mean-predictor baseline
    # ------------------------------------------------------------------
    log.info("=== MEAN-PREDICTOR BASELINE ===")
    for tgt, y_tr, mtr, y_te, mte, sn in [
        ("BLRI_Pr", y_pr_tr, m_pr_tr, y_pr_te, m_pr_te, "test"),
        ("BLRI_In", y_in_tr, m_in_tr, y_in_te, m_in_te, "test"),
        ("BLRI_Pr", y_pr_tr, m_pr_tr, y_pr_va, m_pr_va, "val"),
        ("BLRI_In", y_in_tr, m_in_tr, y_in_va, m_in_va, "val"),
    ]:
        if mtr.sum() == 0 or mte.sum() == 0:
            continue
        train_mean = y_tr[mtr].mean()
        m = compute_metrics(y_te[mte], np.full(mte.sum(), train_mean))
        log.info(
            "  Mean-pred  %-5s %-8s  R2=%+.4f  MSE=%7.2f  MAE=%5.2f  (train_mean=%.2f)",
            sn, tgt, m["r2"], m["mse"], m["mae"], train_mean,
        )

    # ------------------------------------------------------------------
    # 7. Fixed-split evaluation per feature set
    # ------------------------------------------------------------------
    all_results: list[dict] = []

    for feat_name, feat in feature_sets.items():
        log.info("=" * 70)
        log.info("FEATURE SET: %s  (dim=%d)", feat_name, feat["train"].shape[1])
        log.info("=" * 70)

        for tgt, y_tr, mtr, y_te, mte, sn in [
            ("BLRI_Pr", y_pr_tr, m_pr_tr, y_pr_va, m_pr_va, "val"),
            ("BLRI_In", y_in_tr, m_in_tr, y_in_va, m_in_va, "val"),
            ("BLRI_Pr", y_pr_tr, m_pr_tr, y_pr_te, m_pr_te, "test"),
            ("BLRI_In", y_in_tr, m_in_tr, y_in_te, m_in_te, "test"),
        ]:
            if mtr.sum() < 3 or mte.sum() == 0:
                continue
            log.info("--- %s -> %s [%s] ---", tgt, sn, feat_name)
            X_tr = feat["train"][mtr] if sn != "test" else feat["train"][mtr]
            X_te = feat[sn][mte]
            res = run_fixed_split(X_tr, y_tr[mtr], X_te, y_te[mte], tgt, f"{sn}({feat_name})")
            all_results.extend(res)

        # train+val → test
        for tgt, y_tr, mtr, y_va, mva, y_te, mte in [
            ("BLRI_Pr", y_pr_tr, m_pr_tr, y_pr_va, m_pr_va, y_pr_te, m_pr_te),
            ("BLRI_In", y_in_tr, m_in_tr, y_in_va, m_in_va, y_in_te, m_in_te),
        ]:
            if mte.sum() == 0:
                continue
            X_trv = np.vstack([feat["train"][mtr], feat["val"][mva]]) if mva.sum() > 0 else feat["train"][mtr]
            y_trv = np.concatenate([y_tr[mtr], y_va[mva]]) if mva.sum() > 0 else y_tr[mtr]
            log.info("--- %s -> test(tr+val) [%s] ---", tgt, feat_name)
            res = run_fixed_split(X_trv, y_trv, feat["test"][mte], y_te[mte], tgt, f"test_tv({feat_name})")
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
    results_path = output_dir / "baseline_au_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info("Results saved to %s", results_path)

    # ------------------------------------------------------------------
    # 10. Summary table
    # ------------------------------------------------------------------
    log.info("")
    log.info("=" * 100)
    log.info("SUMMARY — FIXED SPLITS")
    log.info("=" * 100)
    log.info(
        "%-25s %-10s %-20s %8s %8s %8s %8s",
        "Regressor", "Target", "Split(Features)", "R2", "MSE", "MAE", "r",
    )
    log.info("-" * 100)
    for r in all_results:
        if "regressor" in r:
            log.info(
                "%-25s %-10s %-20s %+8.4f %8.2f %8.2f %+8.4f",
                r["regressor"], r["target"], r.get("split", ""),
                r["r2"], r["mse"], r["mae"], r.get("pearson_r", float("nan")),
            )
    log.info("=" * 100)

    log.info("")
    log.info("=" * 100)
    log.info("SUMMARY — LOTO-CV AGGREGATES")
    log.info("=" * 100)
    for r in all_results:
        if r.get("type") == "loto_aggregate":
            log.info(
                "  %-15s %-10s  R2=%+.4f  MSE=%7.2f  MAE=%5.2f  r=%+.4f  (n=%d)",
                r.get("features", ""), r["target"],
                r["r2"], r["mse"], r["mae"], r.get("pearson_r", float("nan")), r["n"],
            )
    log.info("=" * 100)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Baseline: raw AU (1D Conv) + text summaries for BLRI prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data paths
    p.add_argument("--data_model", type=Path, required=True, help="Path to data_model.yaml")
    p.add_argument("--config", type=Path, required=True, help="Path to config.yaml (split definitions)")
    p.add_argument("--output_dir", type=Path, default=Path("results/baseline_au"),
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
