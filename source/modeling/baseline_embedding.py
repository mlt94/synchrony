"""
Embedding baseline: multi-target regression for all numeric T3/T5/T7 outcomes
with modality ablation (AU-text only, speech-text only, combined).

Compared to baseline_au_conv.py, this script differs only in AU modality
representation: AU is treated as text (generated per-turn AU descriptions),
encoded with a sentence-transformer.

Evaluated feature sets:
- AU_only   : per-turn AU descriptions (text) -> session embedding
- Text_only : per-turn speech summaries (text) -> session embedding
- AU+Text   : concatenation of AU_only and Text_only session embeddings

Targets:
All numeric outcomes for each interview type prefix (T5/T3/T7):
PANAS_pos_Pr, PANAS_neg_Pr, IRF_self_Pr, IRF_other_Pr, BLRI_ges_Pr,
PANAS_pos_In, PANAS_neg_In, IRF_self_In, IRF_other_In, BLRI_ges_In.
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

# Reuse robust path/data helpers from dataset module
try:
    from source.modeling.dataset import load_au_descriptions, _resolve_path, _is_nan
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from source.modeling.dataset import load_au_descriptions, _resolve_path, _is_nan


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stderr,
)
log = logging.getLogger("baseline_embedding")


# ---------------------------------------------------------------------------
# Label mapping (same target space as baseline_au_conv.py)
# ---------------------------------------------------------------------------

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
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class SessionTextData:
    patient_id: str
    therapist_id: str
    interview_type: str
    speech_summaries: list[str] = field(default_factory=list)
    au_descriptions: list[str] = field(default_factory=list)
    labels: dict[str, float] = field(default_factory=dict)  # full key -> value


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_sessions(
    data_model_path: Path,
    config_path: Path,
    au_descriptions_dir: Path,
    split: str,
) -> list[SessionTextData]:
    """Load sessions with per-turn speech + AU-description texts and full labels."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    allowed = set(config["psychotherapy_splits"][split])

    with open(data_model_path, "r", encoding="utf-8") as f:
        data_model = yaml.safe_load(f)

    au_index = load_au_descriptions(au_descriptions_dir)

    sessions: list[SessionTextData] = []
    skipped_unknown_type = 0
    skipped_no_transcript = 0
    skipped_no_turns = 0

    for interview in data_model["interviews"]:
        therapist_id = interview["therapist"]["therapist_id"]
        if therapist_id not in allowed:
            continue

        patient_id = interview["patient"]["patient_id"]

        for itype, idata in interview.get("types", {}).items():
            if itype not in INTERVIEW_PREFIX:
                skipped_unknown_type += 1
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
                skipped_no_turns += 1
                continue

            turns_data = sorted(turns_data, key=lambda t: t.get("turn_index", 0))

            speech_summaries: list[str] = []
            au_descriptions: list[str] = []

            for turn in turns_data:
                turn_index = turn.get("turn_index")
                summary = (turn.get("summary", "") or "").strip()

                try:
                    turn_index_int = int(turn_index)
                except (TypeError, ValueError):
                    continue

                au_desc = (au_index.get((patient_id, itype, turn_index_int), "") or "").strip()

                # Keep alignment between modalities; if missing text, keep empty string.
                speech_summaries.append(summary)
                au_descriptions.append(au_desc)

            # Keep sessions with at least one turn.
            if len(speech_summaries) == 0:
                skipped_no_turns += 1
                continue

            sessions.append(
                SessionTextData(
                    patient_id=patient_id,
                    therapist_id=therapist_id,
                    interview_type=itype,
                    speech_summaries=speech_summaries,
                    au_descriptions=au_descriptions,
                    labels=labels,
                )
            )

    log.info(
        "[%s] Loaded %d sessions (skipped: %d unknown type, %d no transcript, %d no turns)",
        split.upper(), len(sessions), skipped_unknown_type, skipped_no_transcript, skipped_no_turns,
    )
    return sessions


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------


def _safe_text(x: str) -> str:
    x = (x or "").strip()
    return x if x else "[UNK]"


def embed_session_texts(
    sessions: list[SessionTextData],
    field: str,
    model_name: str,
    batch_size: int,
) -> np.ndarray:
    """Embed one modality ('speech' or 'au') and mean-pool per session."""
    from sentence_transformers import SentenceTransformer

    if field not in {"speech", "au"}:
        raise ValueError("field must be 'speech' or 'au'")

    log.info("Loading sentence-transformer '%s' for %s texts ...", model_name, field)
    st_model = SentenceTransformer(model_name)
    dim = st_model.get_sentence_embedding_dimension()

    all_texts: list[str] = []
    boundaries: list[tuple[int, int]] = []
    offset = 0

    for s in sessions:
        if field == "speech":
            texts = [_safe_text(t) for t in s.speech_summaries]
        else:
            texts = [_safe_text(t) for t in s.au_descriptions]

        all_texts.extend(texts)
        boundaries.append((offset, offset + len(texts)))
        offset += len(texts)

    if offset == 0:
        return np.zeros((len(sessions), dim), dtype=np.float32)

    log.info("Encoding %d %s turns ...", offset, field)
    emb = st_model.encode(
        all_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    X = np.zeros((len(sessions), dim), dtype=np.float32)
    for i, (start, end) in enumerate(boundaries):
        if end > start:
            X[i] = emb[start:end].mean(axis=0)

    return X


# ---------------------------------------------------------------------------
# Targets & metrics
# ---------------------------------------------------------------------------


def extract_targets(
    sessions: list[SessionTextData],
    full_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.full(len(sessions), np.nan, dtype=np.float64)
    for i, s in enumerate(sessions):
        v = s.labels.get(full_key)
        if v is not None:
            y[i] = v
    valid = ~np.isnan(y)
    return y, valid


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
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

    return {
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "pearson_r": r,
        "pearson_p": p,
        "n": n,
    }


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
        "Ridge(a=1)": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Ridge(a=10)": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "Ridge(a=0.1)": make_pipeline(StandardScaler(), Ridge(alpha=0.1)),
        "Lasso(a=0.1)": make_pipeline(StandardScaler(), Lasso(alpha=0.1, max_iter=5000)),
        "SVR(rbf)": make_pipeline(StandardScaler(), SVR(kernel="rbf", C=1.0)),
        "RF(100)": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "GBR(100)": GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
    }


# ---------------------------------------------------------------------------
# Evaluation routines
# ---------------------------------------------------------------------------


def run_reg_fixed_split(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    outcome: str,
    feature_label: str,
    split_label: str,
) -> list[dict]:
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
            "  %-22s %-10s %-16s best=%-14s R²=%7.4f  r=%7.4f  (n=%d)",
            outcome, feature_label, split_label, best_name, best_r2, best_r, len(y_test),
        )

    return results


def run_reg_loto_cv(
    all_sessions: list[SessionTextData],
    X_all: np.ndarray,
    outcome: str,
    feature_label: str,
) -> list[dict]:
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
        tr_idx = [i for i, s in enumerate(all_sessions) if s.therapist_id != held_out and valid[i]]
        te_idx = [i for i, s in enumerate(all_sessions) if s.therapist_id == held_out and valid[i]]

        if len(tr_idx) < 5 or len(te_idx) == 0:
            continue

        reg = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        reg.fit(X_all[tr_idx], y_all[tr_idx])
        pred = reg.predict(X_all[te_idx])

        m = compute_metrics(y_all[te_idx], pred)
        m.update(
            held_out_therapist=held_out,
            n_train=len(tr_idx),
            n_test=len(te_idx),
            outcome=outcome,
            features=feature_label,
        )
        fold_results.append(m)
        all_true.extend(y_all[te_idx].tolist())
        all_pred.extend(pred.tolist())

    results: list[dict] = list(fold_results)
    if len(all_true) > 2:
        agg = compute_metrics(np.array(all_true), np.array(all_pred))
        agg.update(
            type="loto_aggregate",
            outcome=outcome,
            features=feature_label,
            n_valid=n_valid,
            n_folds=len(fold_results),
        )
        results.append(agg)
        log.info(
            "  LOTO AGG  %-22s %-10s  R²=%7.4f  r=%7.4f  MAE=%7.4f  (n=%d, %d folds)",
            outcome, feature_label, agg["r2"], agg["pearson_r"], agg["mae"], agg["n"], len(fold_results),
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

    # 1) Load sessions
    log.info("Loading sessions ...")
    datasets: dict[str, list[SessionTextData]] = {}
    for split in ("train", "val", "test"):
        datasets[split] = load_sessions(
            data_model_path=args.data_model,
            config_path=args.config,
            au_descriptions_dir=args.au_descriptions_dir,
            split=split,
        )

    train_s = datasets["train"]
    val_s = datasets["val"]
    test_s = datasets["test"]
    all_s = train_s + val_s + test_s

    if not train_s:
        log.error("Train split empty — aborting.")
        sys.exit(1)

    n_with_text = sum(1 for s in all_s if any((t or "").strip() for t in s.speech_summaries))
    n_with_au_text = sum(1 for s in all_s if any((t or "").strip() for t in s.au_descriptions))
    log.info(
        "Total sessions: %d  (with speech summaries: %d, with AU descriptions: %d)",
        len(all_s), n_with_text, n_with_au_text,
    )

    # Label availability overview
    log.info("Label availability (per interview type):")
    for itype, prefix in INTERVIEW_PREFIX.items():
        n_type = sum(1 for s in all_s if s.interview_type == itype)
        log.info("  --- %s (n=%d) ---", itype, n_type)
        for suffix in OUTCOME_SUFFIXES:
            full_key = f"{prefix}_{suffix}"
            _, valid = extract_targets(all_s, full_key)
            log.info("    %-25s  %d / %d", full_key, int(valid.sum()), n_type)

    # 2) Embeddings for each modality
    log.info("=== EXTRACTING TEXT EMBEDDINGS (speech summaries) ===")
    X_text_all = embed_session_texts(
        all_s,
        field="speech",
        model_name=args.embed_model,
        batch_size=args.embed_batch_size,
    )

    log.info("=== EXTRACTING AU-TEXT EMBEDDINGS (AU descriptions) ===")
    X_au_text_all = embed_session_texts(
        all_s,
        field="au",
        model_name=args.embed_model,
        batch_size=args.embed_batch_size,
    )

    n_tr = len(train_s)
    n_va = len(val_s)

    X_text_train = X_text_all[:n_tr]
    X_text_val = X_text_all[n_tr:n_tr + n_va]
    X_text_test = X_text_all[n_tr + n_va:]

    X_au_train = X_au_text_all[:n_tr]
    X_au_val = X_au_text_all[n_tr:n_tr + n_va]
    X_au_test = X_au_text_all[n_tr + n_va:]

    # 3) Ablation feature sets
    feature_sets: dict[str, dict[str, np.ndarray]] = {
        "AU_only": {
            "train": X_au_train,
            "val": X_au_val,
            "test": X_au_test,
            "all": X_au_text_all,
        },
        "Text_only": {
            "train": X_text_train,
            "val": X_text_val,
            "test": X_text_test,
            "all": X_text_all,
        },
        "AU+Text": {
            "train": np.hstack([X_au_train, X_text_train]),
            "val": np.hstack([X_au_val, X_text_val]),
            "test": np.hstack([X_au_test, X_text_test]),
            "all": np.hstack([X_au_text_all, X_text_all]),
        },
    }

    # 4) Multi-target regression over same target space as baseline_au_conv.py
    all_results: list[dict] = []
    loto_summary: list[dict] = []

    all_full_keys: list[tuple[str, str, str]] = [
        (itype, suffix, f"{prefix}_{suffix}")
        for itype, prefix in INTERVIEW_PREFIX.items()
        for suffix in OUTCOME_SUFFIXES
    ]

    for itype, prefix in INTERVIEW_PREFIX.items():
        log.info("")
        log.info("#" * 88)
        log.info("INTERVIEW TYPE: %s  (prefix=%s)", itype, prefix)
        log.info("#" * 88)

        for suffix in OUTCOME_SUFFIXES:
            full_key = f"{prefix}_{suffix}"
            log.info("")
            log.info("=" * 88)
            log.info("OUTCOME: %s", full_key)
            log.info("=" * 88)

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
                # train -> val
                if v_train.sum() >= 5 and v_val.sum() >= 2:
                    res = run_reg_fixed_split(
                        feat["train"][v_train], y_train[v_train],
                        feat["val"][v_val], y_val[v_val],
                        outcome=full_key,
                        feature_label=feat_name,
                        split_label="train→val",
                    )
                    all_results.extend(res)

                # train -> test
                if v_train.sum() >= 5 and v_test.sum() >= 2:
                    res = run_reg_fixed_split(
                        feat["train"][v_train], y_train[v_train],
                        feat["test"][v_test], y_test[v_test],
                        outcome=full_key,
                        feature_label=feat_name,
                        split_label="train→test",
                    )
                    all_results.extend(res)

                # train+val -> test
                if v_train.sum() + v_val.sum() >= 5 and v_test.sum() >= 2:
                    X_trv = np.vstack([feat["train"], feat["val"]])
                    y_trv = np.concatenate([y_train, y_val])
                    v_trv = np.concatenate([v_train, v_val])
                    res = run_reg_fixed_split(
                        X_trv[v_trv], y_trv[v_trv],
                        feat["test"][v_test], y_test[v_test],
                        outcome=full_key,
                        feature_label=feat_name,
                        split_label="train+val→test",
                    )
                    all_results.extend(res)

            log.info("--- LOTO-CV for %s ---", full_key)
            for feat_name, feat in feature_sets.items():
                loto_res = run_reg_loto_cv(all_s, feat["all"], full_key, feat_name)
                all_results.extend(loto_res)
                for r in loto_res:
                    if r.get("type") == "loto_aggregate":
                        loto_summary.append(r)

    # 5) Save
    results_path = output_dir / "baseline_embedding_regression_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info("Results saved to %s (%d entries)", results_path, len(all_results))

    # 6) Summary tables
    log.info("")
    log.info("=" * 132)
    log.info("SUMMARY — LOTO-CV AGGREGATES (all outcomes × modalities, sorted by R²)")
    log.info("=" * 132)
    log.info("%-26s %-12s %8s %10s %10s %10s %6s", "Outcome", "Features", "R²", "Pearson_r", "Pearson_p", "MAE", "n")
    log.info("-" * 132)

    loto_sorted = sorted(loto_summary, key=lambda r: r.get("r2", -999), reverse=True)
    for r in loto_sorted:
        log.info(
            "%-26s %-12s %8.4f %10.4f %10.4f %10.4f %6d",
            r["outcome"], r["features"], r["r2"], r["pearson_r"], r.get("pearson_p", 0.0), r["mae"], r["n"],
        )

    log.info("=" * 132)
    log.info("")
    log.info("=" * 132)
    log.info("BEST MODALITY PER OUTCOME (by LOTO R²)")
    log.info("=" * 132)

    best_per_outcome: dict[str, dict] = {}
    for r in loto_summary:
        key = r["outcome"]
        if key not in best_per_outcome or r["r2"] > best_per_outcome[key]["r2"]:
            best_per_outcome[key] = r

    for _itype, _suffix, full_key in all_full_keys:
        if full_key in best_per_outcome:
            r = best_per_outcome[full_key]
            log.info(
                "  %-26s -> %-10s  R²=%7.4f  r=%7.4f  MAE=%7.4f  (n=%d)",
                full_key, r["features"], r["r2"], r["pearson_r"], r["mae"], r["n"],
            )
        else:
            log.info("  %-26s -> no results", full_key)

    log.info("=" * 132)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Embedding baseline (AU-as-text + speech) for multi-target regression "
            "over all T3/T5/T7 numeric outcomes"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--data_model", type=Path, required=True, help="Path to data_model.yaml")
    p.add_argument("--config", type=Path, required=True, help="Path to config.yaml")
    p.add_argument("--au_descriptions_dir", type=Path, required=True, help="Directory containing AU-description JSON files")
    p.add_argument("--output_dir", type=Path, default=Path("results/baseline_embedding_regression"), help="Directory to save outputs")

    p.add_argument("--embed_model", type=str, default="all-MiniLM-L6-v2", help="Sentence-transformer model")
    p.add_argument("--embed_batch_size", type=int, default=256, help="Batch size for embedding")

    p.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING"])

    return p.parse_args()


if __name__ == "__main__":
    main()
