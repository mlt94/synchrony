"""Populate data_model.yaml from labels, OpenFace outputs, and transcripts.

Rules provided by user:
- Therapist ID is labels.iloc[0, 1]
- Patient ID is labels.iloc[0, 0]
- Use that identifier (e.g., A0EA) to find OpenFace CSVs in OpenFace_Output_MSB where trailing token indicates role: Pr (patient) or In (therapist)
- Find transcripts in output_files/ using the patient identifier and choose per interview type with this priority:
  1) Prefer filenames containing "geschnitten"
  2) Else prefer names containing the type words: wunder, personal, bindung
  3) Else prefer names containing the abbreviations: WF (Wunder), STIP (Personal), BRFI (Bindung)
- Three interview types: Bindung (BRFI), Personal (STIP), Wunder (WF)

The resulting YAML structure (now across ALL rows in labels):

interviews:
	- therapist:
			therapist_id: <id>
		patient:
			patient_id: <id>
		types:
			bindung:
				therapist_openface: <path or null>
				patient_openface: <path or null>
				transcript: <path or null>
			personal:
				therapist_openface: <path or null>
				patient_openface: <path or null>
				transcript: <path or null>
			wunder:
				therapist_openface: <path or null>
				patient_openface: <path or null>
				transcript: <path or null>
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from typing import Iterable, Optional

import pandas as pd
import yaml


TYPE_TOKEN_MAP = {
	"bindung": {
		"primary": ["bindung"],
		"aliases": ["brfi"],
	},
	"personal": {
		"primary": ["personal"],
		"aliases": ["stip"],
	},
	"wunder": {
		"primary": ["wunder", "wunderfrage"],
		"aliases": ["wf"],
	},
}

# Omit these identifiers entirely when building the model
OMIT_IDS = {"C3IJ", "C4OF", "S5EA", "K8OM"}

# Columns to drop from labels before inserting into data model
LABELS_COLUMNS_TO_DROP = [
	'T0_Pr_measured', 'T1_Pr_measured', 'T1_In_measured',
	'T2_MEA_measured', 'T2_OpenFace_measured',
	'T3_Pr_measured', 'T3_In_measured',
	'T4_MEA_measured', 'T4_OpenFace_measured', 'T4_STiP_measured',
	'T5_Pr_measured', 'T5_In_measured',
	'T6_MEA_measured', 'T6_OpenFace_measured', 'T6_LSM_measured',
	'T7_Pr_measured', 'T7_In_measured',
	'T0_STARTED_Pr', 'T1_STARTED_Pr', 'T1_STARTED_In',
	'T3_STARTED_Pr', 'T3_STARTED_In', 'T4_STARTED_In',
	'T5_STARTED_Pr', 'T5_STARTED_In', 'T7_STARTED_Pr', 'T7_STARTED_In',
]


def _lower(s: str) -> str:
	return s.lower()


def _contains_all(text: str, tokens: Iterable[str]) -> bool:
	t = text.lower()
	return all(tok in t for tok in tokens)


def read_labels_all_rows(labels_csv: Path) -> tuple[list[tuple[str, str]], pd.DataFrame]:
	"""Read labels and return (id_pairs, labels_df_cleaned).
	
	Returns:
		id_pairs: list of (therapist_id, patient_id) tuples
		labels_df: cleaned DataFrame with dropped columns, indexed by row number
	"""
	df = pd.read_csv(labels_csv)
	if df.shape[1] < 2 or df.shape[0] < 1:
		raise ValueError("labels CSV must have at least 1 row and 2 columns")
	
	# Drop unwanted columns
	cols_to_drop = [c for c in LABELS_COLUMNS_TO_DROP if c in df.columns]
	df_clean = df.drop(columns=cols_to_drop)
	
	pairs: list[tuple[str, str]] = []
	for i in range(df.shape[0]):
		c0 = df.iloc[i, 0]
		c1 = df.iloc[i, 1]
		# skip empty/NaN rows
		if pd.isna(c0) or pd.isna(c1):
			continue
		patient_id = str(c0).strip()
		therapist_id = str(c1).strip()
		if not patient_id or not therapist_id:
			continue
		# Skip rows where either identifier is omitted
		if patient_id in OMIT_IDS or therapist_id in OMIT_IDS:
			continue
		pairs.append((therapist_id, patient_id))
	return pairs, df_clean


def pick_openface_for_type(base_id: str, of_dir: Path, type_name: str, *, of_files: Optional[list[Path]] = None) -> tuple[Optional[Path], Optional[Path]]:
	"""Return (therapist_csv, patient_csv) for a given type.

	We search recursively under of_dir for CSVs with:
	  - filename containing the base_id
	  - filename contains any primary/alias tokens for the interview type
	  - ends with _In.csv (therapist) and _Pr.csv (patient)
	If multiple candidates exist, we pick the one whose filename has the most token hits;
	ties broken by preferring longer filenames and then lexicographically.
	"""
	type_info = TYPE_TOKEN_MAP[type_name]
	tokens_all = [*_lower_tokens(type_info["primary"]), *_lower_tokens(type_info["aliases"])]

	therapist_cands: list[Path] = []
	patient_cands: list[Path] = []
	search_space = of_files if of_files is not None else list(of_dir.rglob("*.csv"))
	for p in search_space:
		name = p.name.lower()
		if base_id.lower() not in name:
			continue
		# require at least one of the type tokens to reduce cross-type mismatches
		if not any(tok in name for tok in tokens_all):
			continue
		if name.endswith("_in.csv"):
			therapist_cands.append(p)
		elif name.endswith("_pr.csv"):
			patient_cands.append(p)

	def _score(path: Path) -> tuple[int, int, str]:
		n = path.name.lower()
		hits = sum(tok in n for tok in tokens_all)
		return (hits, len(n), n)

	therapist = sorted(therapist_cands, key=_score, reverse=True)[0] if therapist_cands else None
	patient = sorted(patient_cands, key=_score, reverse=True)[0] if patient_cands else None
	return therapist, patient


def _lower_tokens(tokens: Iterable[str]) -> list[str]:
	return [t.lower() for t in tokens]


def pick_transcript_for_type(base_id: str, transcripts_dir: Path, type_name: str, *, transcript_files: Optional[list[Path]] = None) -> Optional[Path]:
	"""Pick the best transcript JSON for a type, following the user's priority rules.

	Priority tiers within files that already contain the base_id:
	  1) contains "geschnitten" and contains any of type primary/alias tokens
	  2) contains any type primary tokens (wunder/personal/bindung)
	  3) contains any type alias tokens (WF/STIP/BRFI)
	As a final fallback, if nothing is found for the type, pick the best "geschnitten" match for the ID (any type).
	"""
	type_info = TYPE_TOKEN_MAP[type_name]
	prim = _lower_tokens(type_info["primary"])
	alias = _lower_tokens(type_info["aliases"])

	search_space = transcript_files if transcript_files is not None else list(transcripts_dir.rglob("results_*.json"))
	cands = [p for p in search_space if base_id.lower() in p.name.lower()]
	if not cands:
		return None

	def score(p: Path) -> tuple[int, int, int, str]:
		n = p.name.lower()
		has_id = base_id.lower() in n
		has_ges = int("geschnitten" in n)
		prim_hits = sum(t in n for t in prim)
		alias_hits = sum(t in n for t in alias)
		# primary tiering: prefer geschnitten + prim, then prim, then alias
		# We encode as a tuple so Python's sort can use lexicographic ordering.
		return (
			has_ges and prim_hits > 0,
			prim_hits,
			alias_hits,
			n,
		)

	# Tiered filtering
	tier1 = [p for p in cands if ("geschnitten" in p.name.lower()) and any(t in p.name.lower() for t in prim + alias)]
	if tier1:
		return sorted(tier1, key=score, reverse=True)[0]
	tier2 = [p for p in cands if any(t in p.name.lower() for t in prim)]
	if tier2:
		return sorted(tier2, key=score, reverse=True)[0]
	tier3 = [p for p in cands if any(t in p.name.lower() for t in alias)]
	if tier3:
		return sorted(tier3, key=score, reverse=True)[0]
	# Final fallback: any geschnitten for ID
	tier4 = [p for p in cands if "geschnitten" in p.name.lower()]
	if tier4:
		return sorted(tier4, key=lambda p: p.name.lower(), reverse=True)[0]
	return None


def extract_labels_for_row(df: pd.DataFrame, row_idx: int) -> dict:
	"""Extract label columns for a given row, organized by timepoint.
	
	Returns a dict with keys: 'baseline', 'after_personal', 'after_bindung', 'after_wunder'.
	Each contains a dict of {column_name: value} for that timepoint.
	
	Rules:
	- T0, T1 columns → baseline
	- T3 columns → after_personal
	- T5 columns → after_bindung
	- T7 columns → after_wunder
	- T2, T4, T6 columns are excluded (already dropped)
	- First two columns (patient_id, therapist_id) are excluded
	"""
	if row_idx >= len(df):
		return {"baseline": {}, "after_personal": {}, "after_bindung": {}, "after_wunder": {}}
	
	row = df.iloc[row_idx]
	cols = df.columns[2:]  # Skip first two ID columns
	
	baseline = {}
	after_personal = {}
	after_bindung = {}
	after_wunder = {}
	
	for col in cols:
		val = row[col]
		# Skip NaN values
		if pd.isna(val):
			continue
		
		col_upper = col.upper()
		# Convert numpy types to native Python types for YAML compatibility
		if hasattr(val, 'item'):
			val = val.item()
		
		if col_upper.startswith('T0') or col_upper.startswith('T1'):
			baseline[col] = val
		elif col_upper.startswith('T3'):
			after_personal[col] = val
		elif col_upper.startswith('T5'):
			after_bindung[col] = val
		elif col_upper.startswith('T7'):
			after_wunder[col] = val
	
	return {
		"baseline": baseline,
		"after_personal": after_personal,
		"after_bindung": after_bindung,
		"after_wunder": after_wunder,
	}


def build_entry(therapist_id: str, patient_id: str, of_dir: Path, transcripts_dir: Path, labels_dict: dict, *, of_files: Optional[list[Path]] = None, transcript_files: Optional[list[Path]] = None) -> dict:
	# The base identifier commonly shared across both roles; prefer patient_id for transcripts per user instruction
	base_id = patient_id if patient_id else therapist_id

	entry: dict = {
		"therapist": {
			"therapist_id": therapist_id,
		},
		"patient": {
			"patient_id": patient_id,
		},
		"baseline": labels_dict.get("baseline", {}),
		"types": {
			"bindung": {},
			"personal": {},
			"wunder": {},
		},
	}

	for tname in ("bindung", "personal", "wunder"):
		t_of_in, t_of_pr = pick_openface_for_type(base_id, of_dir, tname, of_files=of_files)
		t_json = pick_transcript_for_type(patient_id, transcripts_dir, tname, transcript_files=transcript_files)
		entry["types"][tname] = {
			"therapist_openface": str(t_of_in) if t_of_in else None,
			"patient_openface": str(t_of_pr) if t_of_pr else None,
			"transcript": str(t_json) if t_json else None,
		}
	
	# Insert timepoint-specific labels after each interview type
	entry["types"]["personal"]["labels"] = labels_dict.get("after_personal", {})
	entry["types"]["bindung"]["labels"] = labels_dict.get("after_bindung", {})
	entry["types"]["wunder"]["labels"] = labels_dict.get("after_wunder", {})

	return entry


def main(argv: list[str] | None = None) -> int:
	parser = argparse.ArgumentParser(description="Populate data_model.yaml from inputs")
	parser.add_argument(
		"--labels_csv",
		type=Path,
		default=Path("C:/Users/User/Desktop/martins/source_files/meta/labels_cleaned.csv"),
		help="Path to labels_cleaned.csv (first row: [patient, therapist])",
	)
	parser.add_argument(
		"--openface_dirs",
		type=Path,
		nargs="*",
		default=None,
		help="One or more directories containing OpenFace CSV outputs (searched recursively)",
	)
	parser.add_argument(
		"--transcripts_dir",
		type=Path,
		default=Path("C:/Users/User/Desktop/martins/output_files"),
		help="Directory containing results_*.json transcripts (searched recursively)",
	)
	parser.add_argument(
		"--output_yaml",
		type=Path,
		default=Path(__file__).resolve().parents[2] / "data_model.yaml",
		help="Destination YAML file to write",
	)
	parser.add_argument(
		"--force",
		action="store_true",
		help="Overwrite output YAML if it already exists (by default, skip when present).",
	)
	args = parser.parse_args(argv)

	try:
		pairs, labels_df = read_labels_all_rows(args.labels_csv)
	except Exception as e:
		print(f"[populate] Failed to read labels: {e}")
		return 1

	# Resolve OpenFace directories list
	of_dirs: list[Path] = []
	if args.openface_dirs:
		of_dirs.extend([d.resolve() for d in args.openface_dirs if d])
	else:
		# Sensible defaults: original raw location and the unpacked destination
		of_dirs = [
			Path("C:/Users/User/Desktop/martins/output_files/Openface_Output_MSB").resolve(),
		]

	# Pre-index files once for efficiency (merge from all dirs, de-duplicate)
	of_files_set = set()
	for d in of_dirs:
		if d.exists():
			for p in d.rglob("*.csv"):
				of_files_set.add(p.resolve())
	of_files = sorted(of_files_set)
	transcript_files = list(args.transcripts_dir.rglob("results_*.json")) if args.transcripts_dir.exists() else []

	interviews: list[dict] = []
	for idx, (therapist_id, patient_id) in enumerate(pairs):
		# Extract labels for this row
		labels_dict = extract_labels_for_row(labels_df, idx)
		
		entry = build_entry(
			therapist_id,
			patient_id,
			args.openface_dirs,
			args.transcripts_dir,
			labels_dict,
			of_files=of_files,
			transcript_files=transcript_files,
		)
		interviews.append(entry)

	model = {"interviews": interviews}

	# Ensure parent dir exists
	args.output_yaml.parent.mkdir(parents=True, exist_ok=True)

	# Create only if missing unless --force specified
	if args.output_yaml.exists() and not args.force:
		print(f"[populate] {args.output_yaml} already exists; skipping write (use --force to overwrite).")
		return 0

	with args.output_yaml.open("w", encoding="utf-8") as f:
		yaml.safe_dump(model, f, allow_unicode=True, sort_keys=False)
	action = "Overwrote" if args.output_yaml.exists() else "Created"
	print(f"[populate] {action} {args.output_yaml} with {len(interviews)} interview entries")
	return 0


if __name__ == "__main__":
	sys.exit(main())
