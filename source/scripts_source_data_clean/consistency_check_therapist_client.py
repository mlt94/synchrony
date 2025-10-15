"""Utilities to inspect and fix therapist/client label assignments.

The language pipeline currently assigns speaker roles by assuming the most
verbose speaker is the client. This script helps to review that output and to
swap labels for sessions where the assumption breaks.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


@dataclass
class TranscriptRecord:
	"""Container for a transcript JSON file."""

	path: Path
	entries: List[dict]


def load_transcripts(transcripts_root: Path) -> Dict[str, TranscriptRecord]:
	"""Load all diarization/ASR transcripts produced by the language pipeline."""

	transcripts: Dict[str, TranscriptRecord] = {}
	for session_dir in sorted(transcripts_root.iterdir()):
		if not session_dir.is_dir():
			continue

		json_path = session_dir / f"results_{session_dir.name}.json"
		if not json_path.exists():
			# Fall back to any results_*.json within the session directory.
			matches = list(session_dir.glob("results_*.json"))
			if not matches:
				continue
			json_path = matches[0]

		try:
			with json_path.open("r", encoding="utf-8") as handle:
				entries = json.load(handle)
		except (json.JSONDecodeError, OSError) as exc:
			print(f"[consistency_check] Failed to read {json_path}: {exc}")
			continue

		stem = json_path.stem.replace("results_", "", 1)
		transcripts[stem] = TranscriptRecord(path=json_path, entries=entries)

	return transcripts

def print_first_k_transcripts(transcripts: Dict[str, TranscriptRecord], k: int) -> None:
	"""Print transcripts that open with a client label, showing only the first *k* turns."""

	if k < 0:
		print("[consistency_check] --print_k cannot be negative; showing full transcripts.")
		k = 0

	filtered = [
		(session_id, record)
		for session_id, record in sorted(transcripts.items())
		if record.entries
		and record.entries[0].get("speaker_id") == "client"
	]

	if not filtered:
		print("[consistency_check] No transcripts with a client-labelled opening speaker found.")
		return

	for session_id, record in filtered:
		print(f"\nSession: {session_id}")
		print(f"File: {record.path}")

		entries_to_show = record.entries if k == 0 else record.entries[:k]
		for turn_idx, entry in enumerate(entries_to_show, start=1):
			speaker = entry.get("speaker_id", "?")
			text = str(entry.get("text", "")).strip()
			print(f"  Turn {turn_idx}: [{speaker}] {text}")

		remaining = len(record.entries) - len(entries_to_show)
		if remaining > 0:
			print(f"  ... ({remaining} additional turns omitted; increase --print_k to view more)")


def _normalise_session_name(name: str) -> str:
	"""Strip known prefixes/suffixes so both stem and filename work."""

	stem = Path(name).stem
	if stem.startswith("results_"):
		stem = stem[len("results_") :]
	return stem


def swap_speaker_labels(
	transcripts: Dict[str, TranscriptRecord],
	targets: Iterable[str],
	dry_run: bool = False,
) -> None:
	"""Swap client/therapist labels for selected transcripts."""

	desired = {_normalise_session_name(name) for name in targets}
	if not desired:
		print("[consistency_check] No transcripts requested for swapping.")
		return

	for session_id in desired:
		record = transcripts.get(session_id)
		if record is None:
			print(f"[consistency_check] Transcript '{session_id}' not found; skipping.")
			continue

		modified = False
		for entry in record.entries:
			speaker = entry.get("speaker_id")
			if speaker == "client":
				entry["speaker_id"] = "therapist"
				modified = True
			elif speaker == "therapist":
				entry["speaker_id"] = "client"
				modified = True

		if not modified:
			print(f"[consistency_check] Transcript '{session_id}' had no swappable labels.")
			continue

		if dry_run:
			print(f"[consistency_check] Dry run: labels for '{session_id}' would be swapped.")
			continue

		try:
			with record.path.open("w", encoding="utf-8") as handle:
				json.dump(record.entries, handle, indent=4, ensure_ascii=False)
			print(f"[consistency_check] Swapped labels written to {record.path}.")
		except OSError as exc:
			print(f"[consistency_check] Failed to write {record.path}: {exc}")


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Inspect and fix speaker labels.")
	parser.add_argument(
		"--transcripts_dir",
		type=Path,
		required=True,
		help="Directory containing session subdirectories with transcript JSON files.",
	)
	parser.add_argument(
		"--print_k",
		type=int,
		default=5,
		help="Number of turns to display per transcript (0 shows all turns).",
	)
	parser.add_argument(
		"--swap",
		nargs="*",
		default=[],
		help="Session identifiers or transcript filenames whose labels should be swapped.",
	)
	parser.add_argument(
		"--dry_run",
		action="store_true",
		help="Preview swaps without modifying any files.",
	)
	return parser.parse_args()


def main() -> None:
	args = _parse_args()

	transcripts_dir = args.transcripts_dir.expanduser().resolve()
	if not transcripts_dir.exists():
		raise SystemExit(f"Transcript directory not found: {transcripts_dir}")

	transcripts = load_transcripts(transcripts_dir)
	if not transcripts:
		print("[consistency_check] No transcripts discovered. Ensure the directory contains session subdirectories with results_*.json files.")
		return
	print_first_k_transcripts(transcripts, args.print_k)

	if args.swap:
		swap_speaker_labels(transcripts, args.swap, dry_run=args.dry_run)


if __name__ == "__main__":
	main()