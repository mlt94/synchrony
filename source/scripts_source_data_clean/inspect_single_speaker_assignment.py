"""Inspect results_*.json files to identify recordings with speaker imbalance.

This script recursively scans a directory for results_*.json files (ASR output)
and reports those where the "therapist" and "client" speaker counts are imbalanced
(i.e., one speaker has less than 10% of total speech turns).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


def count_speaker_turns(json_path: Path) -> Optional[dict[str, int]]:
	"""Parse a results_*.json file and count speaker speech segments.
	
	Each entry in the JSON is counted as one speech segment for that speaker.
	
	Args:
		json_path: Path to the results_*.json file
		
	Returns:
		Dictionary with speaker segment counts {'therapist': int, 'client': int}, or None if error
	"""
	try:
		with open(json_path, "r", encoding="utf-8") as f:
			records = json.load(f)
		
		if not isinstance(records, list) or len(records) == 0:
			return None
		
		speaker_counts = {"therapist": 0, "client": 0}
		
		for record in records:
			if not isinstance(record, dict):
				continue
			
			speaker_id = record.get("speaker_id", "").lower().strip()
			
			# Only count known speakers
			if speaker_id not in speaker_counts:
				continue
			
			# Count each speech segment
			speaker_counts[speaker_id] += 1
		
		return speaker_counts
	except Exception as e:
		print(f"[warning] Error reading {json_path}: {e}", file=sys.stderr)
		return None


def has_imbalance(counts: dict[str, int], threshold: float = 0.1) -> bool:
	"""Check if any speaker has less than threshold (10%) of total segments.
	
	Args:
		counts: Dictionary with speaker counts
		threshold: Minimum fraction (default: 0.1 = 10%)
		
	Returns:
		True if ANY speaker has < threshold, False otherwise
	"""
	total = sum(counts.values())
	if total == 0:
		return False
	
	for count in counts.values():
		fraction = count / total
		if fraction < threshold:
			return True
	
	return False


def scan_for_imbalanced_results(root_dir: Path, threshold: float = 0.1, verbose: bool = False, debug: bool = False) -> list[tuple[Path, dict[str, int], dict[str, float]]]:
	"""Recursively scan root_dir for results_*.json files with speaker imbalance.
	
	Args:
		root_dir: Root directory to scan recursively
		threshold: Minimum fraction for balance (default: 0.1 = 10%)
		verbose: If True, print progress info
		debug: If True, print all files with their speaker counts
		
	Returns:
		List of (json_path, counts, percentages) tuples for imbalanced files
	"""
	if not root_dir.exists():
		print(f"[error] Directory not found: {root_dir}", file=sys.stderr)
		return []
	
	# Find all files matching pattern: results_*.json (files that START with "results_")
	json_files = sorted(root_dir.glob("**/results_*.json"))
	
	# Double-check: filter to ensure filename starts with "results_"
	json_files = [f for f in json_files if f.name.startswith("results_") and f.name.endswith(".json")]
	
	if not json_files:
		print(f"[info] No results_*.json files found in {root_dir}")
		return []
	
	if verbose or debug:
		print(f"[info] Found {len(json_files)} results_*.json files")
		if debug:
			print(f"[debug] Files to process:")
			for jf in json_files[:10]:  # Show first 10 as sample
				print(f"  - {jf.name}")
			if len(json_files) > 10:
				print(f"  ... and {len(json_files) - 10} more")
			print()
	
	imbalanced_files = []
	
	for json_path in json_files:
		counts = count_speaker_turns(json_path)
		
		if counts is None:
			if debug:
				print(f"  {json_path.name}: [parse error or empty]")
			continue
		
		total = sum(counts.values())
		if total == 0:
			if debug:
				print(f"  {json_path.name}: 0 turns")
			continue
		
		# Calculate percentages
		percentages = {s: (count / total * 100) for s, count in counts.items()}
		
		# Debug: show all files
		if debug:
			pct_str = ", ".join(f"{s}: {count} ({pct:.1f}%)" for s, (count, pct) in 
							   [(s, (counts[s], percentages[s])) for s in sorted(counts.keys())])
			print(f"  {json_path.name}: {total} segments -> {pct_str}")
		
		# Check for imbalance (ANY speaker < threshold)
		if has_imbalance(counts, threshold):
			imbalanced_files.append((json_path, counts, percentages))
			if verbose and not debug:
				pct_str = ", ".join(f"{s}: {count} ({pct:.1f}%)" for s, (count, pct) in 
								   [(s, (counts[s], percentages[s])) for s in sorted(counts.keys())])
				print(f"[imbalance] {json_path.name} -> {pct_str}")
	
	return imbalanced_files


def main():
	parser = argparse.ArgumentParser(
		description="Identify results_*.json files with speaker imbalance where one speaker has <10% of speech segments"
	)
	parser.add_argument(
		"--input_dir",
		type=Path,
		default=Path.cwd(),
		help="Root directory to scan recursively for results_*.json files (default: current directory)",
	)
	parser.add_argument(
		"--threshold",
		type=float,
		default=0.1,
		help="Minimum fraction for a speaker to be considered balanced (default: 0.1 = 10%%)",
	)
	parser.add_argument(
		"--verbose",
		action="store_true",
		help="Print progress and verbose output",
	)
	parser.add_argument(
		"--debug",
		action="store_true",
		help="Print debug info for ALL results_*.json files (not just imbalanced ones)",
	)
	args = parser.parse_args()
	
	input_dir = args.input_dir.resolve()
	
	imbalanced_files = scan_for_imbalanced_results(
		input_dir, 
		threshold=args.threshold,
		verbose=args.verbose, 
		debug=args.debug
	)
	
	total_json_files = len(list(input_dir.glob("**/results_*.json")))
	
	# Report imbalanced files
	if not args.debug:  # Don't repeat if we already printed in debug mode
		if imbalanced_files:
			print(f"\n=== Files with speaker imbalance (<{args.threshold*100:.0f}% threshold) ===")
			for json_path, counts, percentages in imbalanced_files:
				total = sum(counts.values())
				pct_str = ", ".join(f"{s}: {counts[s]} ({percentages[s]:.1f}%)" for s in sorted(counts.keys()))
				print(f"{json_path.name} -> {pct_str}")
		else:
			print(f"[info] No files with speaker imbalance (<{args.threshold*100:.0f}%) found")
	
	# Summary
	print(f"\n[summary]")
	print(f"  Total results_*.json files: {total_json_files}")
	print(f"  Imbalanced (<{args.threshold*100:.0f}%): {len(imbalanced_files)}")
	
	return 0


if __name__ == "__main__":
	sys.exit(main())
