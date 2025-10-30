"""Unpack and clean OpenFace outputs into a normalized structure.

Input: a source directory with OpenFace CSV outputs (possibly nested).
Output: a destination directory with per-identifier subfolders and cleaned filenames.

Rules (from header):

- Drop all of S5EA, D3NM (diarization indistinguishable voices).
- For N1EL: remove two recordings that contain "Personal1".
- For C8LA: remove any recordings for the first Personal variant (e.g., "Personal1", "Personal_ln1", "Personal_pr1", "Personal_In1").
- For O0EA: filenames use hyphens instead of underscores; convert hyphens to underscores in destination.
- For B4CA and F1EY: Trailing .csv is cleaned --> B4CA_2024-07-08_Personal_In.csv.csv
- A3EW has a trailing 2 in its bindung condition; this is removed

This gives 76 dyads with 3 interviews each!

Safety & UX:
- Supports --dry-run (no changes), --copy (default) or --move, and --overwrite to replace existing files.
- Creates per-ID folders under the output directory to avoid collisions.
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable


EXCLUDE_IDS = {"S5EA", "D3NM"}
SPECIAL_DROP = {
	"N1EL": [re.compile(r"personal1", re.IGNORECASE)],
	# C8LA: drop any first Personal variant: Personal1, Personal_ln1, Personal_pr1, Personal_in1
	"C8LA": [re.compile(r"personal(?:[_-]?(?:ln|pr|in))?1", re.IGNORECASE)],
}
NEEDS_NAME_NORMALIZATION = {"B4CA", "F1EY", "C3IJ", "I9LB"}
HYPHEN_TO_UNDERSCORE_IDS = {"O0EA"}


def identifier_from_name(name: str) -> str | None:
	base = Path(name).stem
	# Split on underscores or hyphens, first token is assumed identifier
	parts = re.split(r"[-_]", base)
	if parts and parts[0]:
		return parts[0].upper()
	return None


def normalize_interview_tokens(fname: str, *, force_for_ids: Iterable[str] | None = None, id_hint: str | None = None) -> str:
	"""Normalize interview type tokens in filename. Only applied when id_hint is in force_for_ids, otherwise conservative.

	- Wunderfrage/WF -> Wunder
	- STIP -> Personal
	- BRFI -> Bindung
	- For specific IDs, also replace hyphens with underscores when requested.
	"""
	new = fname
	low = fname.lower()
	# Basic token mapping
	mapping = {
		r"wunderfrage": "Wunder",
		r"\bwf\b": "Wunder",
		r"\bstip\b": "Personal",
		r"\bbrfi\b": "Bindung",
	}
	for pat, repl in mapping.items():
		new = re.sub(pat, repl, new, flags=re.IGNORECASE)

	# Hyphen to underscore for specific IDs (O0EA)
	if id_hint and id_hint.upper() in HYPHEN_TO_UNDERSCORE_IDS:
		new = new.replace("-", "_")

	# For B4CA, F1EY, apply stronger normalization (ensure proper casing and separators)
	if id_hint and id_hint.upper() in (force_for_ids or NEEDS_NAME_NORMALIZATION):
		# Collapse multiple underscores
		new = re.sub(r"_+", "_", new)
		# Title-case the tokens we care about
		new = re.sub(r"bindung", "Bindung", new, flags=re.IGNORECASE)
		new = re.sub(r"personal", "Personal", new, flags=re.IGNORECASE)
		new = re.sub(r"wunder", "Wunder", new, flags=re.IGNORECASE)
		new = re.sub(r"_in(\.csv)$", r"_In\1", new, flags=re.IGNORECASE)
		new = re.sub(r"_pr(\.csv)$", r"_Pr\1", new, flags=re.IGNORECASE)

	# A3EW: remove trailing '2' after the Bindung token (e.g., 'Bindung2' -> 'Bindung')
	if id_hint and id_hint.upper() == "A3EW":
		# Replace 'Bindung2', 'Bindung_2', 'Bindung-2' when the 2 directly follows the Bindung token
		new = re.sub(r"(?i)(bindung)[ _-]?2(?=[_.-]|$)", r"\1", new)
		# Also handle role suffixes like '_In2' or '_Pr2' -> '_In' / '_Pr'
		new = re.sub(r"(?i)_(in|pr)2(\.csv)$", r"_\1\2", new)
		new = re.sub(r"(?i)_(in|pr)2(?=[_.-])", r"_\1", new)

	# C8LA: for remaining Personal files, remove trailing '2' in role suffix (_In2/_Pr2)
	if id_hint and id_hint.upper() == "C8LA":
		new = re.sub(r"(?i)_(in|pr)2(\.csv)$", r"_\1\2", new)
		new = re.sub(r"(?i)_(in|pr)2(?=[_.-]|$)", r"_\1", new)
		# Normalize role casing to _In/_Pr
		new = re.sub(r"(?i)_in(\.csv)$", r"_In\1", new)
		new = re.sub(r"(?i)_pr(\.csv)$", r"_Pr\1", new)
	# Collapse duplicate .csv extensions (e.g., ..._In.csv.csv -> ..._In.csv)
	new = re.sub(r"(\.csv)+$", ".csv", new, flags=re.IGNORECASE)
	return new


@dataclass
class PlanItem:
	src: Path
	dst: Path
	reason: str


def should_drop_file(file: Path, id_code: str) -> tuple[bool, str | None]:
	if id_code in EXCLUDE_IDS:
		return True, f"exclude-id:{id_code}"
	rules = SPECIAL_DROP.get(id_code)
	if rules:
		low = file.name.lower()
		for pat in rules:
			if pat.search(low):
				return True, f"special-drop:{id_code}:{pat.pattern}"
	return False, None


def build_plan(input_dir: Path, output_dir: Path, *, move: bool, overwrite: bool) -> list[PlanItem]:
	plan: list[PlanItem] = []
	for f in input_dir.rglob("*.csv"):
		if not f.is_file():
			continue
		id_code = identifier_from_name(f.name)
		if not id_code:
			continue
		drop, why = should_drop_file(f, id_code)
		if drop:
			# skipped
			continue

		# Destination directory per ID
		dest_dir = output_dir / id_code
		dest_dir.mkdir(parents=True, exist_ok=True)

		dest_name = f.name
		dest_name = normalize_interview_tokens(dest_name, id_hint=id_code)
		# Ensure .csv extension
		if not dest_name.lower().endswith(".csv"):
			dest_name = dest_name + ".csv"

		dst = dest_dir / dest_name
		reason = "move" if move else "copy"

		if dst.exists() and not overwrite:
			# Skip if already present and no overwrite requested
			continue

		plan.append(PlanItem(src=f, dst=dst, reason=reason))
	return plan


def execute_plan(plan: list[PlanItem], *, dry_run: bool) -> None:
	for item in plan:
		print(f"[unpack] {item.reason.upper()} {item.src} -> {item.dst}")
		if dry_run:
			continue
		item.dst.parent.mkdir(parents=True, exist_ok=True)
		if item.reason == "move":
			shutil.move(str(item.src), str(item.dst))
		else:
			shutil.copy2(str(item.src), str(item.dst))


def main() -> int:
	parser = argparse.ArgumentParser(description="Unpack and clean OpenFace outputs")
	parser.add_argument("--input_dir", type=Path, required=True, help="Source directory with OpenFace outputs")
	parser.add_argument("--output_dir", type=Path, required=True, help="Destination directory for cleaned outputs (not created)")
	parser.add_argument("--move", action="store_true", help="Move files instead of copying (default is copy)")
	parser.add_argument("--overwrite", action="store_true", help="Overwrite existing destination files if present")
	parser.add_argument("--dry_run", action="store_true", help="Print plan without making changes")
	args = parser.parse_args()

	input_dir = args.input_dir.resolve()
	output_dir = args.output_dir.resolve()
	output_dir.mkdir(parents=True, exist_ok=True)

	plan = build_plan(input_dir, output_dir, move=args.move, overwrite=args.overwrite)
	if not plan:
		print("[unpack] Nothing to do (no files or all skipped)")
		return 0

	print(f"[unpack] Planned {len(plan)} operations (move={args.move}, dry_run={args.dry_run}, overwrite={args.overwrite})")
	execute_plan(plan, dry_run=args.dry_run)
	print("[unpack] Done")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())