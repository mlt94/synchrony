"""
Unified baseline for mixed-outcome prediction with explicit ablations across:
- Text input type: summarised vs unsummarised speech snippets
- AU input type: AU-text descriptions vs AU numeric OpenFace features

Ablations executed:
1) Summarised x AU-text
2) Summarised x AU-numbers
3) Unsummarised x AU-text
4) Unsummarised x AU-numbers

Evaluation protocol remains therapist-level LOTO.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
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
log = logging.getLogger("baseline")


AU_R_COLS: list[str] = [
    "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r",
    "AU09_r", "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r",
    "AU20_r", "AU23_r", "AU25_r", "AU26_r", "AU45_r",
]
N_AUS: int = len(AU_R_COLS)
MIN_FRAMES_FOR_CONV = 3

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
    speech_summaries: list[str] = field(default_factory=list)
    speech_unsummaries: list[str] = field(default_factory=list)
    au_descriptions: list[str] = field(default_factory=list)
    labels: dict[str, float] = field(default_factory=dict)
    baseline_labels: dict[str, Any] = field(default_factory=dict)
    patient_openface_path: Path | None = None
    therapist_openface_path: Path | None = None


@dataclass
class UnsTurn:
    start_s: float
    end_s: float
    text: str


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
    for t in list(tokens):
        expanded.add(_strip_known_prefixes(t))
        if t.startswith("results_"):
            expanded.add(t[len("results_"):])
        if t.startswith("translate_"):
            expanded.add(t[len("translate_"):])
    return {x for x in expanded if x}


def _parse_unsummarised_records(records: Any) -> list[UnsTurn]:
    if not isinstance(records, list):
        return []

    out: list[UnsTurn] = []
    for row in records:
        if not isinstance(row, dict):
            continue

        txt = ""
        for k in ("text", "snippet", "utterance", "content", "transcript"):
            v = row.get(k)
            if isinstance(v, str) and v.strip():
                txt = v.strip()
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
    files: list[Path] = []
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted(path.rglob("*.json"))
    else:
        raise FileNotFoundError(f"Unsummarised path does not exist: {path}")

    index: dict[str, list[UnsTurn]] = {}
    loaded = 0

    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        turns = _parse_unsummarised_records(data)
        if not turns:
            continue

        loaded += 1
        candidates = _session_id_candidates(fp.stem)
        candidates.update(_session_id_candidates(fp.parent.name))

        for key in candidates:
            prev = index.get(key)
            if prev is None or len(turns) > len(prev):
                index[key] = turns

    log.info("Unsummarised index: loaded %d files, %d session keys", loaded, len(index))
    return index


def _resolve_unsummarised_turns(
    transcript_path: Path,
    uns_index: dict[str, list[UnsTurn]],
) -> list[UnsTurn]:
    candidates = _session_id_candidates(transcript_path.stem)
    candidates.update(_session_id_candidates(transcript_path.parent.name))

    for c in candidates:
        if c in uns_index:
            return uns_index[c]

    return []


def _compose_unsummarised_texts(summary_turns: list[TurnInfo], uns_turns: list[UnsTurn]) -> list[str]:
    if not summary_turns:
        return []
    if not uns_turns:
        return [""] * len(summary_turns)

    out: list[str] = []
    for s in summary_turns:
        parts: list[str] = []
        for u in uns_turns:
            if u.end_s > s.start_s and u.start_s < s.end_s:
                parts.append(u.text)
        out.append(" ".join(parts).strip())

    return out


def load_openface_au(path: Path, confidence_threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    import pandas as pd

    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    mask = (df["success"] == 1) & (df["confidence"] >= confidence_threshold)
    aus = df.loc[mask, AU_R_COLS].values.astype(np.float32)
    timestamps = df.loc[mask, "timestamp"].values.astype(np.float64)
    return aus, timestamps


def load_sessions(
    data_model_path: Path,
    config_path: Path,
    au_descriptions_dir: Path,
    unsummarised_text_path: Path,
    split: str,
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

        baseline_raw = interview.get("baseline", {}) or {}
        baseline_labels: dict[str, Any] = {}
        for k, v in baseline_raw.items():
            if v is None:
                continue
            baseline_labels[k] = v

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
            turns: list[TurnInfo] = []
            speech_summaries: list[str] = []
            au_text: list[str] = []

            for turn in turns_data:
                tidx = turn.get("turn_index")
                try:
                    tidx_i = int(tidx)
                except (TypeError, ValueError):
                    continue

                start_s = float(turn.get("start_ms", 0)) / 1000.0
                end_s = float(turn.get("end_ms", 0)) / 1000.0
                summary = (turn.get("summary", "") or "").strip()

                turns.append(TurnInfo(turn_index=tidx_i, start_s=start_s, end_s=end_s, summary=summary))
                speech_summaries.append(summary)

                patient_au = (au_index.get((str(patient_id), itype, tidx_i), "") or "").strip()
                therapist_au = (au_index.get((str(therapist_id), itype, tidx_i), "") or "").strip()
                if patient_au and therapist_au:
                    au_text.append(f"Patient AU: {patient_au}\nTherapist AU: {therapist_au}")
                else:
                    au_text.append(patient_au or therapist_au)

            if not turns:
                continue

            uns_turns = _resolve_unsummarised_turns(transcript_path, uns_index)
            speech_unsummaries = _compose_unsummarised_texts(turns, uns_turns)

            of_patient_raw = idata.get("patient_openface", "")
            of_therapist_raw = idata.get("therapist_openface", "")
            of_patient_path = _resolve_path(of_patient_raw) if of_patient_raw else None
            of_therapist_path = _resolve_path(of_therapist_raw) if of_therapist_raw else None

            sessions.append(
                SessionData(
                    split=split,
                    patient_id=patient_id,
                    therapist_id=therapist_id,
                    interview_type=itype,
                    turns=turns,
                    speech_summaries=speech_summaries,
                    speech_unsummaries=speech_unsummaries,
                    au_descriptions=au_text,
                    labels=labels,
                    baseline_labels=baseline_labels,
                    patient_openface_path=of_patient_path,
                    therapist_openface_path=of_therapist_path,
                )
            )

    log.info("[%s] Loaded %d sessions (skipped %d transcript issues)", split.upper(), len(sessions), skipped_no_transcript)
    return sessions


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


def extract_au_numeric_features(
    sessions: list[SessionData],
    au_extractor: DepthwiseConv1DExtractor,
    confidence_threshold: float,
) -> np.ndarray:
    conv_dim = au_extractor.output_dim
    stats_dim = N_AUS * 6
    single_dim = conv_dim + stats_dim
    out_dim = single_dim * 2

    X = np.zeros((len(sessions), out_dim), dtype=np.float32)
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for i, s in enumerate(sessions):
        session_feat_parts: list[np.ndarray] = []
        for path in (s.patient_openface_path, s.therapist_openface_path):
            if path is None:
                session_feat_parts.append(np.zeros(single_dim, dtype=np.float32))
                continue

            key = str(path)
            if key not in cache:
                try:
                    cache[key] = load_openface_au(path, confidence_threshold=confidence_threshold)
                except Exception as e:
                    log.warning("Failed OpenFace load (%s): %s", path, e)
                    session_feat_parts.append(np.zeros(single_dim, dtype=np.float32))
                    continue

            aus, timestamps = cache[key]
            segments = segment_au_by_turns(aus, timestamps, s.turns)

            turn_vecs: list[np.ndarray] = []
            for seg in segments:
                if seg.shape[0] >= MIN_FRAMES_FOR_CONV:
                    x = torch.tensor(seg.T, dtype=torch.float32).unsqueeze(0)
                    feat = au_extractor(x).squeeze(0).cpu().numpy().astype(np.float32)
                else:
                    feat = np.zeros(conv_dim, dtype=np.float32)
                turn_vecs.append(np.concatenate([feat, _au_stats(seg)]))

            if turn_vecs:
                session_feat_parts.append(np.mean(turn_vecs, axis=0))
            else:
                session_feat_parts.append(np.zeros(single_dim, dtype=np.float32))

        X[i] = np.concatenate(session_feat_parts, axis=0)

    return X


def embed_session_texts(
    sessions: list[SessionData],
    field: str,
    model_name: str,
    batch_size: int,
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    if field not in {"summarised", "unsummarised", "au_text"}:
        raise ValueError("field must be one of {'summarised','unsummarised','au_text'}")

    st_model = SentenceTransformer(model_name)
    dim = st_model.get_sentence_embedding_dimension()

    all_texts: list[str] = []
    boundaries: list[tuple[int, int]] = []
    offset = 0

    for s in sessions:
        if field == "summarised":
            texts = s.speech_summaries
        elif field == "unsummarised":
            texts = s.speech_unsummaries
        else:
            texts = s.au_descriptions

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


def evaluate_records(
    records: list,
    output_path: Path,
    ablation_name: str,
    text_input: str,
    au_input: str,
) -> list[dict]:
    all_results: list[dict] = []
    specs = discover_target_specs(records)

    log.info("[%s] Discovered %d targets (%d regression, %d classification)",
             ablation_name,
             len(specs),
             sum(1 for s in specs if s.task == "regression"),
             sum(1 for s in specs if s.task == "classification"))

    feature_mode = "AU+Text"

    for spec in specs:
        class_to_idx = {c: i for i, c in enumerate(spec.classes)} if spec.task == "classification" else None

        if spec.task == "regression":
            from sklearn.linear_model import Ridge
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler

            all_true: list[float] = []
            all_pred: list[float] = []
            therapist_ids = sorted({r.therapist_id for r in records})

            X_all, y_all, _ = build_features_and_targets(records, spec.name, feature_mode, None)
            if len(y_all) >= 10:
                rec_all_valid = [r for r in records if build_features_and_targets([r], spec.name, feature_mode, None)[1].shape[0] == 1]
                for held in therapist_ids:
                    tr_idx = [i for i, r in enumerate(rec_all_valid) if r.therapist_id != held]
                    te_idx = [i for i, r in enumerate(rec_all_valid) if r.therapist_id == held]
                    if len(tr_idx) < 5 or len(te_idx) == 0:
                        continue
                    reg = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
                    reg.fit(X_all[tr_idx], y_all[tr_idx])
                    pred = reg.predict(X_all[te_idx])
                    m = compute_reg_metrics(y_all[te_idx], pred)
                    m.update(
                        task="regression",
                        target=spec.name,
                        features=feature_mode,
                        ablation=ablation_name,
                        text_input=text_input,
                        au_input=au_input,
                        held_out_therapist=held,
                        n_train=len(tr_idx),
                        n_test=len(te_idx),
                    )
                    all_results.append(m)
                    all_true.extend(y_all[te_idx].tolist())
                    all_pred.extend(pred.tolist())

                if len(all_true) > 2:
                    agg = compute_reg_metrics(np.array(all_true), np.array(all_pred))
                    agg.update(
                        task="regression",
                        target=spec.name,
                        features=feature_mode,
                        ablation=ablation_name,
                        text_input=text_input,
                        au_input=au_input,
                        type="loto_aggregate",
                    )
                    all_results.append(agg)

        else:
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler

            all_true: list[int] = []
            all_pred: list[int] = []
            therapist_ids = sorted({r.therapist_id for r in records})

            X_all, y_all, _ = build_features_and_targets(records, spec.name, feature_mode, class_to_idx)
            if len(y_all) >= 10:
                rec_all_valid = [r for r in records if build_features_and_targets([r], spec.name, feature_mode, class_to_idx)[1].shape[0] == 1]
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
                    m.update(
                        task="classification",
                        target=spec.name,
                        features=feature_mode,
                        ablation=ablation_name,
                        text_input=text_input,
                        au_input=au_input,
                        held_out_therapist=held,
                        n_train=len(tr_idx),
                        n_test=len(te_idx),
                    )
                    all_results.append(m)
                    all_true.extend(y_all[te_idx].tolist())
                    all_pred.extend(pred.tolist())

                if len(all_true) > 2:
                    agg = compute_clf_metrics(np.array(all_true), np.array(all_pred), spec.classes)
                    agg.update(
                        task="classification",
                        target=spec.name,
                        features=feature_mode,
                        ablation=ablation_name,
                        text_input=text_input,
                        au_input=au_input,
                        type="loto_aggregate",
                    )
                    all_results.append(agg)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)

    return all_results


def main() -> None:
    args = parse_args()
    log.setLevel(getattr(logging, args.log_level.upper()))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets: dict[str, list[SessionData]] = {}
    for split in ("train", "val", "test"):
        datasets[split] = load_sessions(
            args.data_model,
            args.config,
            args.au_descriptions_dir,
            args.unsummarised_text_path,
            split,
        )

    all_s = datasets["train"] + datasets["val"] + datasets["test"]
    if not datasets["train"]:
        log.error("Train split empty.")
        sys.exit(1)

    log.info("Extracting AU numeric features...")
    au_extractor = DepthwiseConv1DExtractor(
        n_channels=N_AUS,
        n_kernels_per_channel=args.n_kernels,
        kernel_sizes=[int(x.strip()) for x in args.kernel_sizes.split(",") if x.strip()],
        seed=args.seed,
    )
    X_au_numeric = extract_au_numeric_features(all_s, au_extractor, confidence_threshold=args.openface_confidence_threshold)

    log.info("Extracting summarised speech embeddings...")
    X_text_summarised = embed_session_texts(all_s, field="summarised", model_name=args.embed_model, batch_size=args.embed_batch_size)

    log.info("Extracting unsummarised speech embeddings...")
    X_text_unsummarised = embed_session_texts(all_s, field="unsummarised", model_name=args.embed_model, batch_size=args.embed_batch_size)

    log.info("Extracting AU-text embeddings...")
    X_au_text = embed_session_texts(all_s, field="au_text", model_name=args.embed_model, batch_size=args.embed_batch_size)

    ablations = [
        ("Summarised x AU-text", "summarised", "au_text", X_text_summarised, X_au_text),
        ("Summarised x AU-numbers", "summarised", "au_numbers", X_text_summarised, X_au_numeric),
        ("Unsummarised x AU-text", "unsummarised", "au_text", X_text_unsummarised, X_au_text),
        ("Unsummarised x AU-numbers", "unsummarised", "au_numbers", X_text_unsummarised, X_au_numeric),
    ]

    merged_results: list[dict] = []

    for ablation_name, text_name, au_name, X_text, X_au in ablations:
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
                    au_feat=X_au[i],
                    text_feat=X_text[i],
                )
            )

        patient_records = build_patient_records(rows)
        log.info("[%s] Built %d patient records", ablation_name, len(patient_records))

        per_ablation_path = out_dir / (re.sub(r"[^a-zA-Z0-9]+", "_", ablation_name.lower()).strip("_") + "_results.json")
        results = evaluate_records(
            patient_records,
            per_ablation_path,
            ablation_name=ablation_name,
            text_input=text_name,
            au_input=au_name,
        )
        log.info("[%s] Saved %d rows -> %s", ablation_name, len(results), per_ablation_path)
        merged_results.extend(results)

    merged_path = out_dir / "baseline_mixed_outcomes_results.json"
    with open(merged_path, "w", encoding="utf-8") as f:
        json.dump(merged_results, f, indent=2, default=str)

    log.info("Saved merged %d rows -> %s", len(merged_results), merged_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unified baseline with ablations: summarised/unsummarised x AU-text/AU-numbers",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--data_model", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--au_descriptions_dir", type=Path, required=True)
    p.add_argument("--unsummarised_text_path", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, default=Path("results/baseline_mixed"))

    p.add_argument("--openface_confidence_threshold", type=float, default=0.5)
    p.add_argument("--n_kernels", type=int, default=4)
    p.add_argument("--kernel_sizes", type=str, default="7,15,31")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--embed_model", type=str, default="all-MiniLM-L6-v2")
    p.add_argument("--embed_batch_size", type=int, default=256)

    p.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING"])

    return p.parse_args()


if __name__ == "__main__":
    main()
