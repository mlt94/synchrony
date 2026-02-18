from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


INTERVIEW_TYPES = ["personal", "bindung", "wunder"]
PREFIX_TO_SOURCE_TYPE = {
    "T3": "personal",
    "T5": "bindung",
    "T7": "wunder",
}


def required_types_for_target(target_key: str) -> list[str]:
    prefix = target_key.split("_", 1)[0]
    if prefix == "T3":
        return ["personal"]
    if prefix == "T5":
        return ["personal", "bindung"]
    if prefix == "T7":
        return ["personal", "bindung", "wunder"]
    if prefix in {"T0", "T1"}:
        return ["personal", "bindung", "wunder"]
    return ["personal", "bindung", "wunder"]


@dataclass
class SessionFeatureRow:
    split: str
    patient_id: str
    therapist_id: str
    interview_type: str
    labels: dict[str, float] = field(default_factory=dict)
    baseline_labels: dict[str, Any] = field(default_factory=dict)
    au_feat: np.ndarray | None = None
    text_feat: np.ndarray | None = None


@dataclass
class PatientRecord:
    split: str
    patient_id: str
    therapist_id: str
    baseline_labels: dict[str, Any] = field(default_factory=dict)
    labels_by_type: dict[str, dict[str, float]] = field(default_factory=dict)
    features_by_type: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)


@dataclass
class TargetSpec:
    name: str
    task: str  # "regression" | "classification"
    classes: list[str] = field(default_factory=list)


def _is_finite_number(x: Any) -> bool:
    try:
        v = float(x)
        return np.isfinite(v)
    except (TypeError, ValueError):
        return False


def build_patient_records(rows: list[SessionFeatureRow]) -> list[PatientRecord]:
    grouped: dict[tuple[str, str, str], PatientRecord] = {}

    for row in rows:
        key = (row.split, row.therapist_id, row.patient_id)
        if key not in grouped:
            grouped[key] = PatientRecord(
                split=row.split,
                patient_id=row.patient_id,
                therapist_id=row.therapist_id,
                baseline_labels=dict(row.baseline_labels),
                labels_by_type={},
                features_by_type={},
            )

        rec = grouped[key]

        if not rec.baseline_labels and row.baseline_labels:
            rec.baseline_labels = dict(row.baseline_labels)

        rec.labels_by_type.setdefault(row.interview_type, {})
        rec.labels_by_type[row.interview_type].update(row.labels)

        rec.features_by_type.setdefault(row.interview_type, {})
        if row.au_feat is not None:
            rec.features_by_type[row.interview_type]["AU_only"] = row.au_feat
        if row.text_feat is not None:
            rec.features_by_type[row.interview_type]["Text_only"] = row.text_feat

    return list(grouped.values())


def get_target_value(rec: PatientRecord, target_key: str) -> Any:
    prefix = target_key.split("_", 1)[0]

    if prefix in {"T0", "T1"}:
        return rec.baseline_labels.get(target_key)

    source_type = PREFIX_TO_SOURCE_TYPE.get(prefix)
    if source_type is not None:
        return rec.labels_by_type.get(source_type, {}).get(target_key)

    # Fallback: check all type labels then baseline
    for itype in INTERVIEW_TYPES:
        value = rec.labels_by_type.get(itype, {}).get(target_key)
        if value is not None:
            return value
    return rec.baseline_labels.get(target_key)


def discover_target_specs(records: list[PatientRecord]) -> list[TargetSpec]:
    # Collect candidate keys from baseline + type labels
    key_set: set[str] = set()
    for rec in records:
        key_set.update(rec.baseline_labels.keys())
        for d in rec.labels_by_type.values():
            key_set.update(d.keys())

    specs: list[TargetSpec] = []

    for key in sorted(key_set):
        values = [get_target_value(rec, key) for rec in records]
        values = [v for v in values if v is not None]
        if not values:
            continue

        numeric_mask = [_is_finite_number(v) for v in values]
        numeric_ratio = sum(numeric_mask) / len(values)

        if numeric_ratio >= 0.99:
            specs.append(TargetSpec(name=key, task="regression", classes=[]))
        else:
            classes = sorted({str(v).strip() for v in values if str(v).strip()})
            if len(classes) >= 2:
                specs.append(TargetSpec(name=key, task="classification", classes=classes))

    return specs


def _aggregate_feature_for_target(
    rec: PatientRecord,
    target_key: str,
    base_modality: str,
) -> np.ndarray | None:
    req_types = required_types_for_target(target_key)

    chunks: list[np.ndarray] = []
    for itype in req_types:
        modality_map = rec.features_by_type.get(itype, {})
        feat = modality_map.get(base_modality)
        if feat is not None:
            chunks.append(feat)

    if not chunks:
        return None

    return np.mean(np.stack(chunks, axis=0), axis=0)


def build_features_and_targets(
    records: list[PatientRecord],
    target_key: str,
    feature_mode: str,
    class_to_idx: dict[str, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    feature_mode: AU_only | Text_only | AU+Text
    Returns X, y, valid_mask over records order.
    """
    if feature_mode not in {"AU_only", "Text_only", "AU+Text"}:
        raise ValueError(f"Invalid feature_mode: {feature_mode}")

    feats: list[np.ndarray] = []
    y_vals: list[float | int | None] = []
    valid: list[bool] = []

    for rec in records:
        if feature_mode == "AU+Text":
            f_au = _aggregate_feature_for_target(rec, target_key, "AU_only")
            f_tx = _aggregate_feature_for_target(rec, target_key, "Text_only")
            f = np.concatenate([f_au, f_tx]) if (f_au is not None and f_tx is not None) else None
        elif feature_mode == "AU_only":
            f = _aggregate_feature_for_target(rec, target_key, "AU_only")
        else:
            f = _aggregate_feature_for_target(rec, target_key, "Text_only")

        target_value = get_target_value(rec, target_key)

        if f is None or target_value is None:
            valid.append(False)
            feats.append(np.zeros(1, dtype=np.float32))
            y_vals.append(None)
            continue

        if class_to_idx is None:
            if not _is_finite_number(target_value):
                valid.append(False)
                feats.append(np.zeros(1, dtype=np.float32))
                y_vals.append(None)
                continue
            y = float(target_value)
        else:
            label = str(target_value).strip()
            if label not in class_to_idx:
                valid.append(False)
                feats.append(np.zeros(1, dtype=np.float32))
                y_vals.append(None)
                continue
            y = int(class_to_idx[label])

        valid.append(True)
        feats.append(np.asarray(f, dtype=np.float32))
        y_vals.append(y)

    valid_mask = np.array(valid, dtype=bool)
    if valid_mask.any():
        X = np.stack([feats[i] for i in range(len(feats)) if valid_mask[i]], axis=0)
        y = np.array([y_vals[i] for i in range(len(y_vals)) if valid_mask[i]])
    else:
        X = np.zeros((0, 1), dtype=np.float32)
        y = np.zeros((0,), dtype=np.float32 if class_to_idx is None else np.int64)

    return X, y, valid_mask


def compute_reg_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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


def compute_clf_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_labels: list[str]) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    acc = float(accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_micro = float(f1_score(y_true, y_pred, average="micro", zero_division=0))

    labels = list(range(len(class_labels)))
    f1_per = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    out = {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "f1_micro": f1_micro,
        "confusion_matrix": cm.tolist(),
        "n": len(y_true),
    }
    for i, c in enumerate(class_labels):
        out[f"f1_{c}"] = float(f1_per[i])
    return out


def get_regressors() -> dict[str, Any]:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Lasso, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR

    return {
        "Ridge(a=1)": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Ridge(a=10)": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "Ridge(a=0.1)": make_pipeline(StandardScaler(), Ridge(alpha=0.1)),
        "Lasso(a=0.1)": make_pipeline(StandardScaler(), Lasso(alpha=0.1, max_iter=5000)),
        "SVR(rbf)": make_pipeline(StandardScaler(), SVR(kernel="rbf", C=1.0)),
        "RF(100)": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "GBR(100)": GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
    }


def get_classifiers() -> dict[str, Any]:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    return {
        "LogReg(C=1)": make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=4000, random_state=42)),
        "LogReg(C=10)": make_pipeline(StandardScaler(), LogisticRegression(C=10.0, max_iter=4000, random_state=42)),
        "LogReg(C=0.1)": make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=4000, random_state=42)),
        "SVC(rbf)": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, random_state=42)),
        "SVC(rbf,C=10)": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10.0, random_state=42)),
        "RF(200)": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "GBC(100)": GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42),
    }
