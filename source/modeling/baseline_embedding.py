"""
Embedding-baseline diagnostic for BLRI prediction.

Purpose
-------
Before investing in complex architectures (HMAN), this script tests whether
the text features (speech summaries + AU descriptions) carry *any* signal
predictive of BLRI scores.  It does so by:

1. Encoding every turn's speech summary and AU description with a frozen
   sentence-transformer (``all-MiniLM-L6-v2`` by default).
2. Mean-pooling across turns to get a fixed-size session vector.
3. Fitting simple regressors (Ridge, SVR, Lasso, RF) on the training split
   and evaluating on val/test.

If even a Ridge regression on mean-pooled sentence embeddings scores R² < 0,
then the text features themselves are insufficient and architectural
complexity won't help.

Additionally performs leave-one-therapist-out cross-validation as a
robustness check.

Usage
-----
    # Run as module (recommended):
    python -m source.modeling.baseline_embedding \
        --data_model data_model.yaml \
        --config config.yaml \
        --au_descriptions_dir <path_to_au_jsons>

    # Or run directly:
    python source/modeling/baseline_embedding.py \
        --data_model data_model.yaml \
        --config config.yaml \
        --au_descriptions_dir <path_to_au_jsons>

    # On HPC (paths are WSL-style in data_model.yaml):
    python source/modeling/baseline_embedding.py \
        --data_model data_model.yaml \
        --config config.yaml \
        --au_descriptions_dir /home/mlut/PsyTSLM/au_descriptions \
        --output_dir results/baseline
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

# Reuse the data-loading machinery from the HMAN dataset module.
# This keeps the exact same data pipeline and avoids bugs from
# re-implementing path resolution, BLRI label mapping, modality
# completeness checks, etc.
try:
    from source.modeling.dataset import (
        PsychotherapyDataset,
        SessionSample,
        BLRI_LABEL_MAP,
        load_au_descriptions,
        _resolve_path,
        _is_nan,
        _load_json,
    )
except ModuleNotFoundError:
    # Script mode fallback (e.g., `python source/modeling/baseline_embedding.py`)
    # requires project root on sys.path so `source.*` can be resolved.
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from source.modeling.dataset import (
        PsychotherapyDataset,
        SessionSample,
        BLRI_LABEL_MAP,
        load_au_descriptions,
        _resolve_path,
        _is_nan,
        _load_json,
    )

import yaml

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stderr,
)
log = logging.getLogger("baseline")


# ---------------------------------------------------------------------------
# Embedding utilities
# ---------------------------------------------------------------------------


def embed_sessions(
    sessions: list[SessionSample],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 256,
    combine: str = "concat",
) -> np.ndarray:
    """Embed all sessions into fixed-size vectors.

    Parameters
    ----------
    sessions : list[SessionSample]
        Sessions to embed.
    model_name : str
        Sentence-transformer model name.
    batch_size : int
        Encoding batch size for the sentence transformer.
    combine : str
        How to combine the two modality embeddings per turn.
        ``"concat"`` → [speech; au] (2*dim), ``"mean"`` → mean (dim).

    Returns
    -------
    np.ndarray of shape ``(len(sessions), embed_dim)``
    """
    from sentence_transformers import SentenceTransformer

    log.info(
        "Loading sentence-transformer '%s' …", model_name
    )
    st_model = SentenceTransformer(model_name)
    dim = st_model.get_sentence_embedding_dimension()
    log.info("Embedding dim = %d", dim)

    # Collect ALL texts across all sessions so we can batch-encode once.
    all_speech: list[str] = []
    all_au: list[str] = []
    boundaries: list[tuple[int, int]] = []  # (start, end) per session

    offset = 0
    for s in sessions:
        n = s.num_turns
        all_speech.extend(s.speech_summaries)
        all_au.extend(s.au_descriptions)
        boundaries.append((offset, offset + n))
        offset += n

    total_turns = offset
    log.info(
        "Encoding %d turns (%d sessions) …", total_turns, len(sessions)
    )

    t0 = time.time()
    speech_embs = st_model.encode(
        all_speech, batch_size=batch_size, show_progress_bar=True,
        convert_to_numpy=True,
    )  # (total_turns, dim)
    au_embs = st_model.encode(
        all_au, batch_size=batch_size, show_progress_bar=True,
        convert_to_numpy=True,
    )  # (total_turns, dim)
    elapsed = time.time() - t0
    log.info("Encoding done in %.1f s", elapsed)

    # Per-session mean-pool → session vector
    if combine == "concat":
        session_dim = 2 * dim
    else:
        session_dim = dim

    X = np.zeros((len(sessions), session_dim), dtype=np.float32)

    for i, (start, end) in enumerate(boundaries):
        sp = speech_embs[start:end].mean(axis=0)   # (dim,)
        au = au_embs[start:end].mean(axis=0)        # (dim,)
        if combine == "concat":
            X[i] = np.concatenate([sp, au])
        else:
            X[i] = (sp + au) / 2.0

    return X


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    """Compute MSE, MAE, R² for a single target."""
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
    """Return a dict of name → sklearn regressor (unfitted)."""
    from sklearn.linear_model import Ridge, Lasso
    from sklearn.svm import SVR
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        "Ridge(a=1)": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Ridge(a=10)": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "Ridge(a=100)": make_pipeline(StandardScaler(), Ridge(alpha=100.0)),
        "Lasso(a=1)": make_pipeline(StandardScaler(), Lasso(alpha=1.0, max_iter=5000)),
        "SVR(rbf)": make_pipeline(StandardScaler(), SVR(kernel="rbf", C=1.0)),
        "SVR(rbf,C=10)": make_pipeline(StandardScaler(), SVR(kernel="rbf", C=10.0)),
        "RF(100)": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "GBR(100)": GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
    }


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def extract_targets(sessions: list[SessionSample]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract BLRI target arrays and valid masks.

    Returns (y_pr, y_in, mask_pr, mask_in).
    """
    y_pr = np.array([s.blri_pr if s.blri_pr is not None else np.nan for s in sessions])
    y_in = np.array([s.blri_in if s.blri_in is not None else np.nan for s in sessions])
    mask_pr = ~np.isnan(y_pr)
    mask_in = ~np.isnan(y_in)
    return y_pr, y_in, mask_pr, mask_in


def run_fixed_split(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    target_name: str,
    label: str = "",
) -> list[dict]:
    """Fit all regressors on (X_train, y_train), evaluate on (X_test, y_test)."""
    results = []
    regressors = get_regressors()

    for name, reg in regressors.items():
        reg.fit(X_train, y_train)
        pred = reg.predict(X_test)
        m = compute_metrics(y_test, pred)
        m["regressor"] = name
        m["target"] = target_name
        m["split"] = label
        results.append(m)
        log.info(
            "  %-20s %s %-8s  R2=%+.4f  MSE=%7.2f  MAE=%5.2f  r=%+.4f  (n=%d)",
            name, label, target_name, m["r2"], m["mse"], m["mae"], m["pearson_r"], m["n"],
        )

    return results


# ---------------------------------------------------------------------------
# Leave-one-therapist-out CV
# ---------------------------------------------------------------------------


def run_loto_cv(
    all_sessions: list[SessionSample],
    X_all: np.ndarray,
    model_name: str,
) -> list[dict]:
    """Leave-one-therapist-out cross-validation across all data."""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    # Group sessions by therapist
    therapist_ids = sorted(set(s.therapist_id for s in all_sessions))
    log.info("LOTO-CV across %d therapists: %s", len(therapist_ids), therapist_ids)

    y_pr_all, y_in_all, mask_pr_all, mask_in_all = extract_targets(all_sessions)

    results = []

    for target_name, y_all, mask_all in [("BLRI_Pr", y_pr_all, mask_pr_all), ("BLRI_In", y_in_all, mask_in_all)]:
        all_true = []
        all_pred = []
        per_therapist_results = []

        for held_out in therapist_ids:
            train_idx = [i for i, s in enumerate(all_sessions) if s.therapist_id != held_out and mask_all[i]]
            test_idx = [i for i, s in enumerate(all_sessions) if s.therapist_id == held_out and mask_all[i]]

            if len(train_idx) < 5 or len(test_idx) == 0:
                continue

            X_tr = X_all[train_idx]
            y_tr = y_all[train_idx]
            X_te = X_all[test_idx]
            y_te = y_all[test_idx]

            # Use Ridge(alpha=10) as a reasonable default for CV
            reg = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
            reg.fit(X_tr, y_tr)
            pred = reg.predict(X_te)

            m = compute_metrics(y_te, pred)
            m["held_out_therapist"] = held_out
            m["target"] = target_name
            m["n_train"] = len(train_idx)
            m["n_test"] = len(test_idx)
            per_therapist_results.append(m)

            all_true.extend(y_te.tolist())
            all_pred.extend(pred.tolist())

            log.info(
                "  LOTO held_out=%s  %s  R2=%+.4f  MSE=%7.2f  MAE=%5.2f  n_train=%d  n_test=%d",
                held_out, target_name, m["r2"], m["mse"], m["mae"], len(train_idx), len(test_idx),
            )

        # Aggregate across all folds
        if all_true:
            agg = compute_metrics(np.array(all_true), np.array(all_pred))
            agg["target"] = target_name
            agg["type"] = "loto_aggregate"
            results.append(agg)
            log.info(
                "  LOTO AGGREGATE  %s  R2=%+.4f  MSE=%7.2f  MAE=%5.2f  r=%+.4f  (n=%d)",
                target_name, agg["r2"], agg["mse"], agg["mae"], agg["pearson_r"], agg["n"],
            )

        results.extend(per_therapist_results)

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
    # 1. Load data using the same pipeline as HMAN
    # ------------------------------------------------------------------
    log.info("Loading datasets …")
    datasets: dict[str, PsychotherapyDataset] = {}
    for split in ("train", "val", "test"):
        ds = PsychotherapyDataset(
            data_model_path=args.data_model,
            au_descriptions_dir=args.au_descriptions_dir,
            split=split,
            config_path=args.config,
        )
        datasets[split] = ds
        log.info("  %s: %d sessions", split, len(ds))

    train_sessions = datasets["train"].sessions
    val_sessions = datasets["val"].sessions
    test_sessions = datasets["test"].sessions
    all_sessions = train_sessions + val_sessions + test_sessions

    if len(train_sessions) == 0:
        log.error("Train split is empty — aborting.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Embed all sessions
    # ------------------------------------------------------------------
    log.info("=== EMBEDDING (model=%s, combine=%s) ===", args.embed_model, args.combine)
    X_all = embed_sessions(
        all_sessions,
        model_name=args.embed_model,
        batch_size=args.embed_batch_size,
        combine=args.combine,
    )

    n_train = len(train_sessions)
    n_val = len(val_sessions)
    n_test = len(test_sessions)

    X_train = X_all[:n_train]
    X_val = X_all[n_train:n_train + n_val]
    X_test = X_all[n_train + n_val:]

    log.info("Embedding shapes: train=%s  val=%s  test=%s", X_train.shape, X_val.shape, X_test.shape)

    # ------------------------------------------------------------------
    # 3. Extract targets
    # ------------------------------------------------------------------
    y_pr_train, y_in_train, m_pr_train, m_in_train = extract_targets(train_sessions)
    y_pr_val, y_in_val, m_pr_val, m_in_val = extract_targets(val_sessions)
    y_pr_test, y_in_test, m_pr_test, m_in_test = extract_targets(test_sessions)

    log.info(
        "Valid targets — train: Pr=%d In=%d  val: Pr=%d In=%d  test: Pr=%d In=%d",
        m_pr_train.sum(), m_in_train.sum(),
        m_pr_val.sum(), m_in_val.sum(),
        m_pr_test.sum(), m_in_test.sum(),
    )

    # ------------------------------------------------------------------
    # 4. Mean-predictor baseline (for reference)
    # ------------------------------------------------------------------
    log.info("=== MEAN-PREDICTOR BASELINE ===")
    for tgt, y_tr, mask_tr, y_te, mask_te, split_name in [
        ("BLRI_Pr", y_pr_train, m_pr_train, y_pr_test, m_pr_test, "test"),
        ("BLRI_In", y_in_train, m_in_train, y_in_test, m_in_test, "test"),
        ("BLRI_Pr", y_pr_train, m_pr_train, y_pr_val, m_pr_val, "val"),
        ("BLRI_In", y_in_train, m_in_train, y_in_val, m_in_val, "val"),
    ]:
        if mask_tr.sum() == 0 or mask_te.sum() == 0:
            continue
        train_mean = y_tr[mask_tr].mean()
        pred_mean = np.full(mask_te.sum(), train_mean)
        m = compute_metrics(y_te[mask_te], pred_mean)
        log.info(
            "  Mean-predictor  %-5s %-8s  R2=%+.4f  MSE=%7.2f  MAE=%5.2f  (train_mean=%.2f, n=%d)",
            split_name, tgt, m["r2"], m["mse"], m["mae"], train_mean, m["n"],
        )

    # ------------------------------------------------------------------
    # 5. Fixed-split evaluation (train→val, train→test)
    # ------------------------------------------------------------------
    all_results = []

    for tgt, y_tr, mask_tr, y_te, mask_te, split_name in [
        ("BLRI_Pr", y_pr_train, m_pr_train, y_pr_val, m_pr_val, "val"),
        ("BLRI_In", y_in_train, m_in_train, y_in_val, m_in_val, "val"),
        ("BLRI_Pr", y_pr_train, m_pr_train, y_pr_test, m_pr_test, "test"),
        ("BLRI_In", y_in_train, m_in_train, y_in_test, m_in_test, "test"),
    ]:
        if mask_tr.sum() < 3 or mask_te.sum() == 0:
            log.warning("Skipping %s/%s — not enough valid targets", tgt, split_name)
            continue

        log.info("=== %s → %s ===", tgt, split_name)
        res = run_fixed_split(
            X_train[mask_tr], y_tr[mask_tr],
            X_te=X_test[mask_te] if split_name == "test" else X_val[mask_te],
            y_test=y_te[mask_te],
            target_name=tgt,
            label=split_name,
        )
        all_results.extend(res)

    # Also try train+val → test
    if m_pr_test.sum() > 0:
        log.info("=== BLRI_Pr → test (train+val) ===")
        X_trainval = np.vstack([X_train[m_pr_train], X_val[m_pr_val]]) if m_pr_val.sum() > 0 else X_train[m_pr_train]
        y_trainval = np.concatenate([y_pr_train[m_pr_train], y_pr_val[m_pr_val]]) if m_pr_val.sum() > 0 else y_pr_train[m_pr_train]
        res = run_fixed_split(
            X_trainval, y_trainval,
            X_test[m_pr_test], y_pr_test[m_pr_test],
            target_name="BLRI_Pr", label="test(tr+val)",
        )
        all_results.extend(res)

    if m_in_test.sum() > 0:
        log.info("=== BLRI_In → test (train+val) ===")
        X_trainval = np.vstack([X_train[m_in_train], X_val[m_in_val]]) if m_in_val.sum() > 0 else X_train[m_in_train]
        y_trainval = np.concatenate([y_in_train[m_in_train], y_in_val[m_in_val]]) if m_in_val.sum() > 0 else y_in_train[m_in_train]
        res = run_fixed_split(
            X_trainval, y_trainval,
            X_test[m_in_test], y_in_test[m_in_test],
            target_name="BLRI_In", label="test(tr+val)",
        )
        all_results.extend(res)

    # ------------------------------------------------------------------
    # 6. Leave-one-therapist-out CV
    # ------------------------------------------------------------------
    log.info("=== LEAVE-ONE-THERAPIST-OUT CV ===")
    loto_results = run_loto_cv(all_sessions, X_all, args.embed_model)
    all_results.extend(loto_results)

    # ------------------------------------------------------------------
    # 7. Save results
    # ------------------------------------------------------------------
    results_path = output_dir / "baseline_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info("Results saved to %s", results_path)

    # ------------------------------------------------------------------
    # 8. Summary table
    # ------------------------------------------------------------------
    log.info("")
    log.info("=" * 90)
    log.info("SUMMARY")
    log.info("=" * 90)
    log.info(
        "%-25s %-10s %-15s %8s %8s %8s %8s",
        "Regressor", "Target", "Split", "R2", "MSE", "MAE", "r",
    )
    log.info("-" * 90)
    for r in all_results:
        if "regressor" in r:
            log.info(
                "%-25s %-10s %-15s %+8.4f %8.2f %8.2f %+8.4f",
                r["regressor"], r["target"], r.get("split", ""),
                r["r2"], r["mse"], r["mae"], r.get("pearson_r", float("nan")),
            )
        elif r.get("type") == "loto_aggregate":
            log.info(
                "%-25s %-10s %-15s %+8.4f %8.2f %8.2f %+8.4f",
                "LOTO-Ridge(a=10)", r["target"], "aggregate",
                r["r2"], r["mse"], r["mae"], r.get("pearson_r", float("nan")),
            )
    log.info("=" * 90)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Embedding-baseline diagnostic for BLRI prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data paths
    p.add_argument("--data_model", type=Path, required=True, help="Path to data_model.yaml")
    p.add_argument("--config", type=Path, required=True, help="Path to config.yaml (split definitions)")
    p.add_argument("--au_descriptions_dir", type=Path, required=True, help="Directory with AU-description JSONs")
    p.add_argument("--output_dir", type=Path, default=Path("results/baseline"), help="Where to save results")

    # Embedding
    p.add_argument(
        "--embed_model", type=str, default="all-MiniLM-L6-v2",
        help="Sentence-transformer model for encoding turns",
    )
    p.add_argument(
        "--embed_batch_size", type=int, default=256,
        help="Batch size for sentence-transformer encoding",
    )
    p.add_argument(
        "--combine", type=str, default="concat", choices=["concat", "mean"],
        help="How to combine speech and AU embeddings per turn: 'concat' (2*dim) or 'mean' (dim)",
    )

    # Misc
    p.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING"])

    return p.parse_args()


if __name__ == "__main__":
    main()
