"""
Utilities for loading OpenFace Action Unit CSVs as PyTorch-ready sequences.

We only care about the following parsed metadata:
	- identifier (e.g., identifier for the interviewee)
	- type       (e.g., Wunder, Personal, Bindung)
	- person     (e.g., interviewer or interviewee)

This module provides:
	- parse_file_metadata: extract (identifier, type, person) from filename
	- AUSequenceDataset:   a torch.utils.data.Dataset yielding AU time-series per file
	- pad_collate:         collate function that pads variable-length sequences
	- make_dataloaders_by: helper to construct DataLoaders split by a given field
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


_IDENT_RE = re.compile(r"^[A-Za-z0-9]+$")


@dataclass(frozen=True)
class FileMeta:
	path: Path
	identifier: str
	type: str
	person: str  # "In" or "Pr"
	therapist: Optional[str] = None  # from labels CSV (e.g., ID_Interviewerin)


def parse_file_metadata(path: Path) -> Optional[FileMeta]:
	"""
	Parse identifier, type, and person from a filename.

	Accepts variations like:
	  - XXXX_YEAR_DATE_TYPE_PERSON.csv

	Strategy:
	  - Split on underscores
	  - First token  -> identifier
	  - Last token   -> person token
	  - Penultimate  -> type
	  - Ignore the date/year middle tokens
	"""
	name = path.name
	if name.lower().endswith(".csv"):
		name = name[:-4]
	tokens = name.split("_")
	if len(tokens) < 3:
		return None

	identifier = tokens[0]
	if not _IDENT_RE.match(identifier):
		return None

	# type is typically the penultimate token
	type_token = tokens[-2] if len(tokens) >= 2 else None
	person_token = tokens[-1]

	# Person can include a trailing cut number, e.g., Pr1, In2
	m = re.match(r"^(Pr|In)(\d*)$", person_token, flags=re.IGNORECASE)
	if m:
		person = m.group(1).title()  # normalize to 'Pr' or 'In'
	else:
		# Fall back: if last token is exactly Pr/In, else try to derive
		if person_token.lower() in {"pr", "in"}:
			person = person_token.title()
		else:
			# Could be cases like *_Pr.csv already handled; otherwise unknown
			return None

	file_type = type_token if type_token is not None else "unknown"
	return FileMeta(path=path, identifier=identifier, type=file_type, person=person)


def discover_csv_files(root: Path | str) -> List[Path]:
	root = Path(root)
	return sorted(p for p in root.glob("*.csv") if p.is_file())


def _load_labels_map(labels_csv: Path | str, id_col: str = "ID_Proband", th_col: str = "ID_Interviewerin") -> Dict[str, str]:
    """Load a mapping from identifier -> therapist id from a labels CSV."""
    df = pd.read_csv(labels_csv, usecols=[id_col, th_col]).dropna()
    return dict(zip(df[id_col].str.strip(), df[th_col].str.strip()))


def _select_au_columns(df: pd.DataFrame, prefer: str = "r") -> List[str]:
    """Select AU feature columns from a DataFrame based on preference ('r' or 'c')."""
    # Normalize column names to handle leading/trailing spaces
    df.columns = df.columns.str.strip()
    if prefer == "r":
        return [col for col in df.columns if col.endswith("_r")]
    return [col for col in df.columns if col.endswith("_c")]


class AUSequenceDataset(Dataset):
	"""Dataset producing full-sequence AU tensors per CSV file.

	Each item is a dictionary comprised of:
	  - x:        FloatTensor [T, D] (T frames, D AU features)
	  - length:   int (T)
	  - meta:     dict with keys {identifier, type, person, path}

	"""

	def __init__(
		self,
		root: str | Path,
		include_identifiers: Optional[Iterable[str]] = None,
		include_types: Optional[Iterable[str]] = None,
		include_persons: Optional[Iterable[str]] = None,
		exclude_identifiers: Optional[Iterable[str]] = None,
		au_prefer: str = "r",
		drop_na: bool = True,
		labels_csv: Optional[str | Path] = None,
		labels_identifier_column: str = "ID_Proband",
		labels_therapist_column: str = "ID_Interviewerin",
	) -> None:
		super().__init__()
		self.root = Path(root)
		self.au_prefer = au_prefer
		self.drop_na = drop_na
		self._labels_map: Optional[Dict[str, str]] = None
		if labels_csv is not None:
			self._labels_map = _load_labels_map(labels_csv, labels_identifier_column, labels_therapist_column)
        
		files = discover_csv_files(self.root)
        
		metas: List[FileMeta] = []
		for p in files:
			meta = parse_file_metadata(p)
			if meta is None:
				continue
			metas.append(meta)

		def _in(opt: Optional[Iterable[str]], val: str) -> bool:
			return True if opt is None else (val in set(opt))

		def _not_in(opt: Optional[Iterable[str]], val: str) -> bool:
			return True if opt is None else (val not in set(opt))

		filtered: List[FileMeta] = []
		for m in metas:
			if not _in(include_identifiers, m.identifier):
				continue
			if not _in(include_types, m.type):
				continue
			if not _in(include_persons, m.person):
				continue
			if not _not_in(exclude_identifiers, m.identifier):
				continue
			filtered.append(m)

		# Attach therapist id from labels map if provided
		if self._labels_map is not None:
			attached: List[FileMeta] = []
			for m in filtered:
				ther = self._labels_map.get(m.identifier)
				attached.append(FileMeta(path=m.path, identifier=m.identifier, type=m.type, person=m.person, therapist=ther))
			self._metas = attached
		else:
			self._metas = filtered

		# Inspect one file to determine AU columns
		self._au_columns: Optional[List[str]] = None
		first_columns_sample: List[str] | None = None
		for m in self._metas:
			try:
				head = pd.read_csv(m.path, nrows=0)
				if first_columns_sample is None:
					first_columns_sample = [str(c).strip() for c in head.columns]
				au_cols = _select_au_columns(head, prefer=self.au_prefer)
				if au_cols:
					# Preserve exact casing as in DataFrame for consistent indexing later
					self._au_columns = au_cols
					break
			except Exception:
				continue

		if self._au_columns is None:
			details = f"Found {len(self._metas)} candidate CSVs under {self.root}. "
			if first_columns_sample is not None:
				details += f"Example header columns: {first_columns_sample[:10]}"
			raise RuntimeError(
				"Could not determine AU columns from any CSV. Ensure files contain AU*_r or AU*_c columns. "
				+ details
			)

	@property
	def au_columns(self) -> List[str]:
		assert self._au_columns is not None
		return self._au_columns

	@property
	def metas(self) -> List[FileMeta]:
		return self._metas

	def __len__(self) -> int:
		return len(self._metas)

	def __getitem__(self, idx: int) -> Dict:
		meta = self._metas[idx]
		df = pd.read_csv(meta.path)
		# Normalize header names to match detection time (strip BOM and whitespace)
		df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
		cols = self.au_columns
		x = df[cols]
		if self.drop_na:
			x = x.dropna(axis=0, how="any")
		x_tensor = torch.as_tensor(x.values, dtype=torch.float32)
		return {
			"x": x_tensor,                # [T, D]
			"length": x_tensor.shape[0],  # T
			"meta": {
				"identifier": meta.identifier,
				"type": meta.type,
				"person": meta.person,
				"therapist": meta.therapist,
			},
		}


def pad_collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
	"""Pad variable-length sequences in a batch.

	Returns dict with:
	  - x:       FloatTensor [B, T_max, D]
	  - lengths: LongTensor  [B]
	  - meta:    list of per-item metadata dicts
	"""
	xs = [b["x"] for b in batch]
	lengths = torch.tensor([b["length"] for b in batch], dtype=torch.long)
	D = xs[0].shape[1] if xs else 0
	T_max = max((t.shape[0] for t in xs), default=0)
	B = len(xs)

	x_pad = xs[0].new_zeros((B, T_max, D)) if B > 0 else torch.zeros((0, 0, 0))
	for i, t in enumerate(xs):
		x_pad[i, : t.shape[0]] = t

	meta = [b["meta"] for b in batch]
	return {"x": x_pad, "lengths": lengths, "meta": meta}


def no_pad_collate(batch: List[Dict]) -> Dict[str, object]:
	"""Collate without padding: return lists of variable-length tensors.

	Returns dict with:
	  - x:       list[Tensor[T_i, D]]
	  - lengths: LongTensor [B]
	  - meta:    list of per-item metadata dicts
	"""
	xs = [b["x"] for b in batch]
	lengths = torch.tensor([b["length"] for b in batch], dtype=torch.long)
	meta = [b["meta"] for b in batch]
	return {"x": xs, "lengths": lengths, "meta": meta}


def _subset_indices_by_field(dataset: AUSequenceDataset, field: str, allowed_values: Iterable[str]) -> List[int]:
	allowed = set(allowed_values)
	field = field.lower()
	idxs: List[int] = []
	for i, m in enumerate(dataset.metas):
		val = getattr(m, field)
		if val is not None and val in allowed:
			idxs.append(i)
	return idxs


def make_dataloaders_by(
	root: str | Path,
	field: str,
	train_values: Sequence[str],
	val_values: Optional[Sequence[str]] = None,
	test_values: Optional[Sequence[str]] = None,
	*,
	include_types: Optional[Iterable[str]] = None,
	include_persons: Optional[Iterable[str]] = None,
	include_identifiers: Optional[Iterable[str]] = None,
	exclude_identifiers: Optional[Iterable[str]] = None,
	au_prefer: str = "r",
	batch_size: int = 4,
	num_workers: int = 0,
	shuffle_train: bool = True,
	drop_na: bool = True,
	pad: bool = True,
	labels_csv: Optional[str | Path] = None,
	labels_identifier_column: str = "ID_Proband",
	labels_therapist_column: str = "ID_Interviewerin",
) -> Dict[str, DataLoader]:
	"""
	Create DataLoaders splitting on a specific metadata field.

	field: one of {"identifier", "type", "person"}
	train/val/test_values: which values of that field go to each split.
	Other include_* filters are applied before splitting.
	"""
	if field not in {"identifier", "type", "person", "therapist"}:
		raise ValueError("field must be one of {'identifier','type','person','therapist'}")

	base_ds = AUSequenceDataset(
		root,
		include_identifiers=include_identifiers,
		include_types=include_types,
		include_persons=include_persons,
		exclude_identifiers=exclude_identifiers,
		au_prefer=au_prefer,
		drop_na=drop_na,
		labels_csv=labels_csv,
		labels_identifier_column=labels_identifier_column,
		labels_therapist_column=labels_therapist_column,
	)

	loaders: Dict[str, DataLoader] = {}

	collate = pad_collate if pad else no_pad_collate

	def _make_loader(indices: List[int], shuffle: bool) -> DataLoader:
		subset = torch.utils.data.Subset(base_ds, indices)
		return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate)

	train_idx = _subset_indices_by_field(base_ds, field, train_values)
	loaders["train"] = _make_loader(train_idx, shuffle_train)

	if val_values is not None:
		val_idx = _subset_indices_by_field(base_ds, field, val_values)
		loaders["val"] = _make_loader(val_idx, shuffle=False)

	if test_values is not None:
		test_idx = _subset_indices_by_field(base_ds, field, test_values)
		loaders["test"] = _make_loader(test_idx, shuffle=False)

	return loaders


# Convenience: quick scan of available values
def summarize_dataset(root: str | Path) -> Dict[str, List[str]]:
	"""Return sorted unique values for identifier/type/person in a folder of CSVs."""
	metas = [m for p in discover_csv_files(root) for m in ([parse_file_metadata(p)] if parse_file_metadata(p) else [])]
	ids = sorted({m.identifier for m in metas})
	types = sorted({m.type for m in metas})
	persons = sorted({m.person for m in metas})
	return {"identifier": ids, "type": types, "person": persons}

