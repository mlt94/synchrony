"""
Dataset and data-loading utilities for the Hierarchical Multimodal Attention
Network.

Each **sample** is one therapy *session* (= one ``(patient, interview_type)``
pair).  A session contains *N* speech turns; for each turn we carry two text
fields:

* **speech summary** – from the transcript JSON referenced in
  ``data_model.yaml``.
* **AU description** – from the generated AU-description JSON files whose
  directory is passed at runtime.

Targets are the two session-level BLRI scores (``BLRI_ges_Pr``,
``BLRI_ges_In``).

Padding strategy
~~~~~~~~~~~~~~~~
A :class:`BucketBatchSampler` sorts sessions by their number of turns and
groups similarly-sized sessions into each mini-batch.  Because the user has
accepted very small batch sizes (default 2), within-batch padding is minimal.
"""

from __future__ import annotations

import json
import math
import os
import platform
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset, DataLoader, Sampler

# ---------------------------------------------------------------------------
# BLRI label key mapping
# ---------------------------------------------------------------------------
# The label prefix differs for each interview type.
BLRI_LABEL_MAP: dict[str, tuple[str, str]] = {
    "bindung":  ("T5_BLRI_ges_Pr", "T5_BLRI_ges_In"),
    "personal": ("T3_BLRI_ges_Pr", "T3_BLRI_ges_In"),
    "wunder":   ("T7_BLRI_ges_Pr", "T7_BLRI_ges_In"),
}


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _wsl_to_windows(p: str) -> str:
    """Convert ``/mnt/c/…`` to ``C:\\…``."""
    if p.startswith("/mnt/") and len(p) > 6 and p[5].isalpha() and p[6] == "/":
        drive = p[5].upper()
        rest = p[7:].replace("/", "\\")
        return f"{drive}:\\{rest}"
    return p


def _windows_to_wsl(p: str) -> str:
    """Convert ``C:\\…`` to ``/mnt/c/…``."""
    if len(p) >= 3 and p[1] == ":" and p[2] in ("\\/",):
        drive = p[0].lower()
        rest = p[3:].replace("\\", "/")
        return f"/mnt/{drive}/{rest}"
    return p


def _resolve_path(raw: str) -> Path | None:
    """Try the raw path, then the converted form (WSL ↔ Windows)."""
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


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _is_nan(x: Any) -> bool:
    if x is None:
        return True
    try:
        return not np.isfinite(float(x))
    except (TypeError, ValueError):
        return True


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Session dataclass
# ---------------------------------------------------------------------------

@dataclass
class SessionSample:
    """One therapy session ready for model consumption."""

    patient_id: str
    therapist_id: str
    interview_type: str
    speech_summaries: list[str]     # one per turn
    au_descriptions: list[str]      # one per turn (matched by turn_index)
    blri_pr: float | None = None    # patient perspective
    blri_in: float | None = None    # interviewer perspective
    num_turns: int = 0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.num_turns = len(self.speech_summaries)


# ---------------------------------------------------------------------------
# AU-description index
# ---------------------------------------------------------------------------

def load_au_descriptions(
    au_dir: str | Path,
) -> dict[tuple[str, str, int], str]:
    """Load all AU-description JSONs from *au_dir* into a lookup table.

    Returns a dict keyed by ``(subject_id, interview_type, turn_index)``
    mapping to the description string. ``subject_id`` may be the patient id,
    therapist id, or another id field present in the AU-description entries.
    Supports both the raw
    ``generated_descriptions`` key and the combined
    ``original_timeseries_description`` key.
    """
    au_dir = Path(au_dir)
    index: dict[tuple[str, str, int], str] = {}

    for json_file in au_dir.glob("*.json"):
        try:
            data = _load_json(json_file)
        except Exception:
            continue

        if not isinstance(data, list):
            continue

        for entry in data:
            subject_id = (
                entry.get("patient_id")
                or entry.get("therapist_id")
                or entry.get("subject_id")
                or entry.get("speaker_id")
            )
            itype = entry.get("interview_type")
            tidx = entry.get("turn_index")
            if subject_id is None or itype is None or tidx is None:
                continue
            # Prefer raw description, fall back to combined key
            desc = (
                entry.get("generated_descriptions")
                or entry.get("generated_rationale")
                or entry.get("original_timeseries_description")
                or ""
            )
            if isinstance(desc, str) and desc.strip():
                index[(str(subject_id), str(itype), int(tidx))] = desc.strip()

    return index


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PsychotherapyDataset(Dataset):
    """PyTorch dataset yielding :class:`SessionSample` instances.

    Parameters
    ----------
    data_model_path : str | Path
        Path to ``data_model.yaml``.
    au_descriptions_dir : str | Path
        Directory containing AU-description JSON files.
    split : str
        One of ``"train"``, ``"val"``, ``"test"``.
    config_path : str | Path
        Path to ``config.yaml`` (contains therapist-based splits).
    """

    def __init__(
        self,
        data_model_path: str | Path,
        au_descriptions_dir: str | Path,
        split: str,
        config_path: str | Path,
    ):
        super().__init__()
        self.split = split

        # Load config splits
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        split_key = {"train": "train", "val": "val", "test": "test"}[split]
        allowed_therapists: set[str] = set(config["psychotherapy_splits"][split_key])

        # Load data model
        with open(data_model_path, "r", encoding="utf-8") as f:
            data_model = yaml.safe_load(f)

        # Build AU-description index
        au_index = load_au_descriptions(au_descriptions_dir)

        # Build sessions -------------------------------------------------
        self.sessions: list[SessionSample] = []
        skipped_incomplete_modalities = 0

        for interview in data_model["interviews"]:
            therapist_id = interview["therapist"]["therapist_id"]
            if therapist_id not in allowed_therapists:
                continue

            patient_id = interview["patient"]["patient_id"]

            for itype, idata in interview.get("types", {}).items():
                # --- BLRI labels ---
                labels = idata.get("labels", {})
                pr_key, in_key = BLRI_LABEL_MAP.get(itype, (None, None))
                blri_pr = labels.get(pr_key) if pr_key else None
                blri_in = labels.get(in_key) if in_key else None
                if _is_nan(blri_pr):
                    blri_pr = None
                if _is_nan(blri_in):
                    blri_in = None

                # Skip sessions where both BLRI targets are missing
                if blri_pr is None and blri_in is None:
                    continue

                # --- Transcript ---
                transcript_raw = idata.get("transcript", "")
                transcript_path = _resolve_path(transcript_raw) if transcript_raw else None
                if transcript_path is None or not transcript_path.exists():
                    continue

                try:
                    turns_data = _load_json(transcript_path)
                except Exception:
                    continue

                if not isinstance(turns_data, list) or len(turns_data) == 0:
                    continue

                # Sort by turn_index
                turns_data = sorted(turns_data, key=lambda t: t.get("turn_index", 0))

                speech_summaries: list[str] = []
                au_descriptions: list[str] = []
                session_incomplete = False

                for turn in turns_data:
                    tidx = turn.get("turn_index")
                    summary = turn.get("summary", "").strip()
                    if not summary:
                        session_incomplete = True
                        break

                    try:
                        tidx_int = int(tidx)
                    except (TypeError, ValueError):
                        session_incomplete = True
                        break

                    au_desc = au_index.get((patient_id, itype, tidx_int), "").strip()
                    if not au_desc:
                        session_incomplete = True
                        break

                    speech_summaries.append(summary)
                    au_descriptions.append(au_desc)

                if session_incomplete:
                    skipped_incomplete_modalities += 1
                    continue

                if len(speech_summaries) == 0:
                    continue

                self.sessions.append(
                    SessionSample(
                        patient_id=patient_id,
                        therapist_id=therapist_id,
                        interview_type=itype,
                        speech_summaries=speech_summaries,
                        au_descriptions=au_descriptions,
                        blri_pr=float(blri_pr) if blri_pr is not None else None,
                        blri_in=float(blri_in) if blri_in is not None else None,
                        metadata={
                            "therapist_id": therapist_id,
                            "patient_id": patient_id,
                            "interview_type": itype,
                        },
                    )
                )

        print(
            f"[{split.upper()}] Loaded {len(self.sessions)} sessions  "
            f"(turns range: "
            f"{min((s.num_turns for s in self.sessions), default=0)}–"
            f"{max((s.num_turns for s in self.sessions), default=0)})"
        )
        if skipped_incomplete_modalities:
            print(
                f"[{split.upper()}] Skipped {skipped_incomplete_modalities} sessions "
                f"with missing speech summary or AU description."
            )

    # -- Dataset interface ---------------------------------------------------

    def __len__(self) -> int:
        return len(self.sessions)

    def __getitem__(self, idx: int) -> SessionSample:
        return self.sessions[idx]

    # -- helpers for sampler -------------------------------------------------

    def get_turn_counts(self) -> list[int]:
        """Return list of turn counts, one per session (same order)."""
        return [s.num_turns for s in self.sessions]


# ---------------------------------------------------------------------------
# Bucket-based batch sampler (minimises padding)
# ---------------------------------------------------------------------------

class BucketBatchSampler(Sampler[list[int]]):
    """Batch sampler that groups sessions with similar turn counts.

    1. Sort dataset indices by number of turns.
    2. Chunk sorted indices into batches of *batch_size*.
    3. Optionally shuffle the order of batches each epoch.

    This ensures that within each mini-batch the difference between the
    longest and shortest session is small → minimal padding.

    Parameters
    ----------
    turn_counts : list[int]
        Number of turns per dataset sample (same order as dataset).
    batch_size : int
        Maximum samples per batch.
    shuffle : bool
        Shuffle **batch order** (not intra-batch order) each epoch.
    drop_last : bool
        Drop the last incomplete batch.
    """

    def __init__(
        self,
        turn_counts: list[int],
        batch_size: int = 2,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        sorted_indices = sorted(range(len(turn_counts)), key=lambda i: turn_counts[i])
        self.batches: list[list[int]] = [
            sorted_indices[i : i + batch_size]
            for i in range(0, len(sorted_indices), batch_size)
        ]
        if self.drop_last and len(self.batches) and len(self.batches[-1]) < batch_size:
            self.batches = self.batches[:-1]

    def __iter__(self):
        batches = list(self.batches)
        if self.shuffle:
            random.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        return len(self.batches)


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def create_collate_fn(tokenizer, max_token_length: int = 128):
    """Return a collate function closed over *tokenizer*.

    The collate function:
    1. Flattens all speech summaries and AU descriptions across sessions in
       the batch.
    2. Tokenises them together (so BERT-level padding is minimal).
    3. Builds ``turn_counts`` and ``turn_mask`` for the session dimension.
    4. Stacks BLRI targets with ``NaN`` for missing values (handled by the
       loss function).
    """

    def collate_fn(batch: list[SessionSample]) -> dict[str, Any]:
        all_speech: list[str] = []
        all_au: list[str] = []
        turn_counts: list[int] = []
        targets: list[list[float]] = []
        metadata: list[dict] = []

        for session in batch:
            all_speech.extend(session.speech_summaries)
            # Use a placeholder for empty AU descriptions so BERT gets valid input
            all_au.extend(
                d if d else "[UNK]" for d in session.au_descriptions
            )
            turn_counts.append(session.num_turns)
            targets.append([
                session.blri_pr if session.blri_pr is not None else float("nan"),
                session.blri_in if session.blri_in is not None else float("nan"),
            ])
            metadata.append(session.metadata)

        # Tokenise
        speech_enc = tokenizer(
            all_speech,
            padding=True,
            truncation=True,
            max_length=max_token_length,
            return_tensors="pt",
        )
        au_enc = tokenizer(
            all_au,
            padding=True,
            truncation=True,
            max_length=max_token_length,
            return_tensors="pt",
        )

        turn_counts_t = torch.tensor(turn_counts, dtype=torch.long)
        max_turns = int(turn_counts_t.max().item())
        turn_mask = torch.arange(max_turns).unsqueeze(0) < turn_counts_t.unsqueeze(1)
        targets_t = torch.tensor(targets, dtype=torch.float32)

        return {
            "speech_input_ids": speech_enc["input_ids"],
            "speech_attention_mask": speech_enc["attention_mask"],
            "au_input_ids": au_enc["input_ids"],
            "au_attention_mask": au_enc["attention_mask"],
            "turn_counts": turn_counts_t,
            "turn_mask": turn_mask,
            "targets": targets_t,
            "metadata": metadata,
        }

    return collate_fn


# ---------------------------------------------------------------------------
# Convenience: create all three dataloaders
# ---------------------------------------------------------------------------

def create_dataloaders(
    data_model_path: str | Path,
    au_descriptions_dir: str | Path,
    config_path: str | Path,
    tokenizer,
    batch_size: int = 2,
    max_token_length: int = 128,
    num_workers: int = 0,
) -> dict[str, DataLoader]:
    """Instantiate train / val / test dataloaders.

    Returns
    -------
    dict with keys ``"train"``, ``"val"``, ``"test"`` mapping to
    :class:`~torch.utils.data.DataLoader` instances.
    """
    collate = create_collate_fn(tokenizer, max_token_length)

    loaders: dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        ds = PsychotherapyDataset(
            data_model_path, au_descriptions_dir, split, config_path,
        )
        sampler = BucketBatchSampler(
            ds.get_turn_counts(),
            batch_size=batch_size,
            shuffle=(split == "train"),
        )
        loaders[split] = DataLoader(
            ds,
            batch_sampler=sampler,
            collate_fn=collate,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    return loaders
