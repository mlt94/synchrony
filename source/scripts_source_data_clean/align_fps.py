"""
Align OpenFace CSV time series to 60 FPS using linear interpolation based only on the 'timestamp' column.

Inputs:
  - Directory of CSV files named like: IDENTIFIER_YYYY-MM-DD_Interviewtype_In.csv
  - Excel file frame_info.xlsx with columns:
		Pseudonym, FPS BRFI, FPS STiP, FPS WF

Behavior:
  - For each CSV, parse identifier and interview type from the filename
  - Look up the source FPS in the Excel by (identifier, interview type)
	- Resample to 60 FPS with linear interpolation for numeric columns using 'timestamp' only
	- Preserve 'timestamp' (recomputed at 60 FPS) and 'frame' if present (frame is re-generated sequentially)
  - Non-numeric columns are carried over from the nearest original frame
  - Save to a new output directory with the same filename

Usage:
  python align_fps.py --in /path/to/csvs --frame-info /path/to/frame_info.xlsx \
					  [--out /path/to/output] [--target-fps 60]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


# ------------------------------
# Filename parsing
# ------------------------------

@dataclass(frozen=True)
class ParsedName:
	identifier: str
	interview_type: str  # e.g., Bindung | Personal | Wunder | etc.


def parse_identifier_and_type(path: Path) -> Optional[ParsedName]:
	"""Parse identifier and interview type from a filename.

	Expected pattern: IDENTIFIER_YYYY-MM-DD_Interviewtype_In.csv
	"""
	name = path.name
	if name.lower().endswith(".csv"):
		name = name[:-4]
	tokens = name.split("_")
	if len(tokens) < 3:
		return None
	identifier = tokens[0]
	interview_type = tokens[-2]  # penultimate token
	return ParsedName(identifier=identifier.strip(), interview_type=interview_type.strip())


# ------------------------------
# Frame info (Excel) helpers
# ------------------------------

TYPE_TO_COLUMN = {
	"bindung": "FPS BRFI",
	"personal": "FPS STiP",
	"wunder": "FPS WF",
}


def normalize_type_for_column(t: str) -> Optional[str]:
	"""Map various interview type strings to the frame_info.xlsx column name.

	Returns the column name to use in the Excel ('FPS BRFI'/'FPS STiP'/'FPS WF'), or None if unknown.
	"""
	tl = t.strip().lower()
	if "bind" in tl:
		return TYPE_TO_COLUMN["bindung"]
	if tl.startswith("pers") or tl.startswith("st") or "personal" in tl:
		return TYPE_TO_COLUMN["personal"]
	if tl.startswith("wun") or "wf" in tl:
		return TYPE_TO_COLUMN["wunder"]
	return None


def load_frame_info(xlsx_path: Path) -> pd.DataFrame:
	df = pd.read_excel(xlsx_path)
	# Standardize column names whitespace
	df.columns = [str(c).strip() for c in df.columns]
	# Normalize Pseudonym as string key
	if "Pseudonym" not in df.columns:
		raise ValueError("frame_info.xlsx must contain a 'Pseudonym' column")
	df["Pseudonym"] = df["Pseudonym"].astype(str).str.strip()
	return df


def get_source_fps(frame_info: pd.DataFrame, identifier: str, interview_type: str) -> Optional[float]:
	col = normalize_type_for_column(interview_type)
	if col is None:
		return None
	if col not in frame_info.columns:
		raise ValueError(f"Expected column '{col}' not found in frame_info.xlsx")
	row = frame_info.loc[frame_info["Pseudonym"].astype(str).str.strip() == str(identifier).strip()]
	if row.empty:
		return None
	val = row.iloc[0][col]
	try:
		fps = float(val)
	except Exception:
		return None
	return fps if fps > 0 else None


# ------------------------------
# Resampling logic
# ------------------------------

def _compute_time_axis_from_timestamp(df: pd.DataFrame) -> Tuple[np.ndarray, str, Optional[str]]:
	"""Return times (seconds), the exact timestamp column name used, and (if present) the frame column name.

	Requires a 'timestamp' column (case-insensitive). If missing, raises ValueError.
	"""
	# Normalize header names to strip BOM/whitespace for detection (do not overwrite original df.columns order)
	norm_cols = [str(c).replace("\ufeff", "").strip() for c in df.columns]
	# Map lower -> original name
	col_map = {c.lower(): orig for c, orig in zip(norm_cols, df.columns)}
	ts_key = next((k for k in col_map if k == "timestamp"), None)
	if ts_key is None:
		raise ValueError("CSV must contain a 'timestamp' column (case-insensitive)")

	ts_col = col_map[ts_key]
	# Frame column is optional; we only use timestamp for interpolation
	fr_col = None
	if "frame" in col_map:
		fr_col = col_map["frame"]

	times = pd.to_numeric(df[ts_col], errors="coerce").to_numpy(dtype=float)
	# If there are NaNs in timestamp, interpolate timestamps over index
	if np.isnan(times).any():
		idx = np.arange(len(times), dtype=float)
		mask = ~np.isnan(times)
		times = np.interp(idx, idx[mask], times[mask])

	# Ensure strictly increasing by sorting; keep stable order mapping for downstream
	order = np.argsort(times)
	times = times[order]
	# Monotonic correction for any duplicate timestamps
	diffs = np.diff(times)
	if np.any(diffs <= 0):
		eps = 1e-9
		for i in range(1, len(times)):
			if times[i] <= times[i - 1]:
				times[i] = times[i - 1] + eps

	return times, ts_col, fr_col


def resample_to_target(df: pd.DataFrame, src_fps: float, target_fps: float = 60.0) -> pd.DataFrame:
	"""Resample dataframe to target fps using linear interpolation for numeric columns, using only 'timestamp'.

	- Recompute 'timestamp' and 'frame' (if present originally, frame is regenerated)
	- Preserve column order from the original
	- Non-numeric columns are aligned by nearest original sample
	"""
	if src_fps <= 0:
		raise ValueError("src_fps must be > 0")

	# Clean header BOM/whitespace to reliably detect 'timestamp' and 'frame'
	df = df.copy()
	df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]

	times_old, ts_col, fr_col = _compute_time_axis_from_timestamp(df)
	t0 = times_old[0] if len(times_old) > 0 else 0.0
	times_rel = times_old - t0
	if len(times_rel) == 0:
		return df.copy()

	# Sort the dataframe by timestamp to align values with times_rel
	df_sorted = df.sort_values(by=ts_col).reset_index(drop=True)

	t_end = times_rel[-1]
	# Compute number of samples so last new time <= t_end
	n_new = int(np.floor(t_end * target_fps)) + 1
	new_rel = np.arange(n_new, dtype=float) / float(target_fps)
	new_abs = t0 + new_rel

	# Identify columns
	original_cols = list(df.columns)

	# Separate numeric and non-numeric (exclude frame/timestamp from numeric set)
	numeric_cols = df_sorted.select_dtypes(include=[np.number]).columns.tolist()
	numeric_cols = [c for c in numeric_cols if c not in {ts_col, fr_col}]
	non_numeric_cols = [c for c in original_cols if c not in numeric_cols]
	if ts_col in non_numeric_cols:
		non_numeric_cols.remove(ts_col)
	if fr_col and fr_col in non_numeric_cols:
		non_numeric_cols.remove(fr_col)

	# Interpolate numeric columns on the time grid
	interp_data: Dict[str, np.ndarray] = {}
	for col in numeric_cols:
		y = pd.to_numeric(df_sorted[col], errors="coerce").to_numpy(dtype=float)
		# Replace missing at ends by nearest valid to make np.interp happy
		mask = ~np.isnan(y)
		if not mask.any():
			interp_data[col] = np.full_like(new_rel, np.nan)
			continue
		# For NaNs at ends, clamp to first/last valid
		first_idx = np.argmax(mask)
		last_idx = len(y) - 1 - np.argmax(mask[::-1])
		if first_idx > 0:
			y[:first_idx] = y[first_idx]
		if last_idx < len(y) - 1:
			y[last_idx + 1 :] = y[last_idx]
		# Fill interior NaNs via linear interpolation along the original time order
		if np.isnan(y).any():
			# Interpolate y over indices of valid points (maintain order aligned with times_rel)
			idx = np.arange(len(y), dtype=float)
			y = np.interp(idx, idx[mask], y[mask])
		interp_data[col] = np.interp(new_rel, times_rel, y)

	# Align non-numeric columns by nearest original time
	nonnum_df = pd.DataFrame({})
	if non_numeric_cols:
		non_src = df_sorted[non_numeric_cols].copy()
		non_src["__time__"] = times_rel
		new_time_df = pd.DataFrame({"__time__": new_rel})
		# nearest original sample
		nonnum_df = pd.merge_asof(
			new_time_df.sort_values("__time__"),
			non_src.sort_values("__time__"),
			on="__time__",
			direction="nearest",
		).drop(columns=["__time__"])  # type: ignore[assignment]

	# Build output with original order
	out_cols: Dict[str, np.ndarray] = {}
	for col in original_cols:
		if col == ts_col:
			out_cols[col] = new_abs
		elif fr_col and col == fr_col:
			# Generate new sequential frame indices starting at 0
			out_cols[col] = np.arange(n_new, dtype=int)
		elif col in interp_data:
			out_cols[col] = interp_data[col]
		elif non_numeric_cols and col in nonnum_df.columns:
			out_cols[col] = nonnum_df[col].to_numpy()
		else:
			# Column type unknown; try to carry forward the first value
			arr = np.array([df_sorted[col].iloc[0]] * n_new)
			out_cols[col] = arr

	out_df = pd.DataFrame(out_cols)
	return out_df


# ------------------------------
# CLI glue
# ------------------------------

def process_directory(in_dir: Path, out_dir: Path, frame_info_xlsx: Path, target_fps: float = 60.0) -> None:
	out_dir.mkdir(parents=True, exist_ok=True)
	frame_info = load_frame_info(frame_info_xlsx)

	csv_paths = sorted(p for p in in_dir.glob("*.csv") if p.is_file())
	if not csv_paths:
		print(f"No CSV files found in {in_dir}")
		return

	for p in csv_paths:
		parsed = parse_identifier_and_type(p)
		if parsed is None:
			print(f"[skip] Cannot parse identifier/type from: {p.name}")
			continue

		src_fps = get_source_fps(frame_info, parsed.identifier, parsed.interview_type)
		if src_fps is None:
			print(
				f"[skip] Missing FPS for identifier='{parsed.identifier}', type='{parsed.interview_type}' in frame_info.xlsx"
			)
			continue

		try:
			df = pd.read_csv(p)
		except Exception as e:
			print(f"[skip] Failed to read {p.name}: {e}")
			continue

		if abs(src_fps - target_fps) < 1e-6:
			out_path = out_dir / p.name
			try:
				df.to_csv(out_path, index=False)
				print(f"[copy] {p.name} (already {target_fps} FPS)")
			except Exception as e:
				print(f"[error] Writing {out_path.name}: {e}")
			continue

		try:
			out_df = resample_to_target(df, src_fps=src_fps, target_fps=target_fps)
		except Exception as e:
			print(f"[error] Resampling {p.name}: {e}")
			continue

		out_path = out_dir / p.name
		try:
			out_df.to_csv(out_path, index=False)
			print(f"[ok] {p.name}: {src_fps} -> {target_fps} FPS (rows {len(df)} -> {len(out_df)})")
		except Exception as e:
			print(f"[error] Writing {out_path.name}: {e}")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
	ap = argparse.ArgumentParser(description="Resample OpenFace CSVs to 60 FPS using frame_info.xlsx metadata")
	ap.add_argument("--in", dest="in_dir", required=True, help="Input directory containing CSV files")
	ap.add_argument("--frame-info", dest="frame_info", required=True, help="Path to frame_info.xlsx")
	ap.add_argument("--out", dest="out_dir", default=None, help="Output directory for resampled CSVs (default: <in>/aligned_60fps)")
	ap.add_argument("--target-fps", dest="target_fps", type=float, default=60.0, help="Target FPS (default: 60)")
	return ap.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
	args = parse_args(argv)
	in_dir = Path(args.in_dir).expanduser().resolve()
	out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (in_dir / "aligned_60fps")
	frame_info = Path(args.frame_info).expanduser().resolve()

	if not in_dir.exists() or not in_dir.is_dir():
		print(f"Input directory does not exist or is not a directory: {in_dir}")
		return 2
	if not frame_info.exists():
		print(f"Frame info Excel not found: {frame_info}")
		return 2

	process_directory(in_dir, out_dir, frame_info, target_fps=float(args.target_fps))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

#python /home/mlut/synchrony/source/scripts_source_data_clean/align_fps.py --in /path/to/openface/csvs --frame-info /path/to/frame_info.xlsx
#python /home/mlut/synchrony/source/scripts_source_data_clean/align_fps.py --in /path/to/openface/csvs --frame-info /path/to/frame_info.xlsx --out /path/to/output --target-fps 60