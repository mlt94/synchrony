"""
Align OpenFace CSV time series down to a target FPS (default 24) using rolling-mean smoothing
and linear interpolation based on the 'timestamp' column.

Inputs:
  - Directory of CSV files named like: IDENTIFIER_YYYY-MM-DD_Interviewtype_In.csv
  - Excel file frame_info.xlsx with columns:
		Pseudonym, FPS BRFI, FPS STiP, FPS WF

Behavior:
  - For each CSV, parse identifier and interview type from the filename
  - Look up the source FPS in the Excel by (identifier, interview type)
	- Downsample to the target FPS using rolling-mean smoothing and interpolation on 'timestamp'
	- Preserve 'timestamp' (recomputed on the new grid) and 'frame' if present (frame is re-generated sequentially)
  - Non-numeric columns are carried over from the nearest original frame
  - Save to a new output directory with the same filename

The interpolation and downsampling is done according to:
- Rolling mean smoothing
Original timestamps: T₀, T₁, …, T_{N−1}
Original values for some AU: X₀ … X_{N−1}
Window size: w_raw = ceil(source_fps / target_fps) ->> I enforce odd so we can center, for instance 30/24 --> 3
This operatin outputs new AU values (Y) calculated as the mean of its neighbors; the values then need to be transformed onto the new timestep grid

-- Time resampling
t0 = T₀
Duration = T_{N−1} − T₀
Step = Δ = 1 / target_fps
Generate new relative times (new_timestep vector): τ_k = k · Δ, for k = 0, 1, …, K where K are the integers denoting the corresponding timestep
Z_k (new AU/feature vectors) = Y_i + (Y{i+1} − Y_i) * (T'k − T_i) / (T{i+1} − T_i)
Y_i + (Y{i+1} − Y_i) --> how much does the new AU values change between two known points?
(T'k − T_i) / (T{i+1} − T_i) --> how far are these two points separated in time?

We thus compute a weighted average of the two smoothed neighbor values, weighted according to how far away the target value was between its neighbors.

Usage:
  python align_fps.py --in /path/to/csvs --frame-info /path/to/frame_info.xlsx \
				  [--out /path/to/output] [--target-fps 24]
"""

from __future__ import annotations

import argparse
import shutil
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

	Expected pattern (flexible with '_' or '-' separators):
	  IDENTIFIER_YYYY-MM-DD_Interviewtype_In.csv
	  IDENTIFIER_YYYY-MM-DD-Interviewtype-In.csv
	If the second-to-last token is 'geschnitten', the interview type is the token before it.
	"""
	name = path.name
	if name.lower().endswith(".csv"):
		name = name[:-4]
	# Normalize separators so anomalous '-' delimiters become '_' tokens
	tokens = [tok for tok in name.replace("-", "_").split("_") if tok]
	if len(tokens) < 3:
		return None
	identifier = tokens[0]
	# Handle cases where the penultimate token is 'geschnitten'; use the preceding token
	penultimate = tokens[-2].lower()
	if penultimate == "geschnitten":
		if len(tokens) < 4:
			return None
		interview_type = tokens[-3]
	else:
		interview_type = tokens[-2]
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


def resample_to_target(df: pd.DataFrame, src_fps: float, target_fps: float = 24.0) -> pd.DataFrame:
	"""Resample dataframe to target fps using rolling-mean smoothing plus interpolation on 'timestamp'.

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

	# Determine smoothing window (odd, >=1) based on ratio between source and target FPS
	ratio = src_fps / float(target_fps)
	if ratio <= 1:
		window = 1
	else:
		window = int(np.ceil(ratio))
		if window % 2 == 0:
			window += 1

	# Interpolate numeric columns on the time grid
	interp_data: Dict[str, np.ndarray] = {}
	for col in numeric_cols:
		series = pd.to_numeric(df_sorted[col], errors="coerce")
		series = series.interpolate(method="linear", limit_direction="both")
		if window > 1:
			series = series.rolling(window=window, min_periods=1, center=True).mean()
		y = series.to_numpy(dtype=float)
		mask = ~np.isnan(y)
		if not mask.any():
			interp_data[col] = np.full_like(new_rel, np.nan)
			continue
		if np.isnan(y).any():
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

def process_directory(in_dir: Path, out_dir: Path, frame_info_xlsx: Path, target_fps: float = 24.0) -> None:
	out_dir.mkdir(parents=True, exist_ok=True)
	log_path = out_dir / "align_fps_log.txt"
	with log_path.open("w", encoding="utf-8") as log_file:
		def log(msg: str) -> None:
			print(msg)
			log_file.write(msg + "\n")
			log_file.flush()

		frame_info = load_frame_info(frame_info_xlsx)
		csv_paths = sorted(p for p in in_dir.glob("*.csv") if p.is_file())
		if not csv_paths:
			log(f"No CSV files found in {in_dir}")
			return

		log(f"Processing {len(csv_paths)} file(s) to {target_fps} FPS")

		for p in csv_paths:
			parsed = parse_identifier_and_type(p)
			if parsed is None:
				log(f"[skip] Cannot parse identifier/type from: {p.name}")
				continue

			src_fps = get_source_fps(frame_info, parsed.identifier, parsed.interview_type)
			if src_fps is None:
				log(
					f"[skip] Missing FPS for identifier='{parsed.identifier}', type='{parsed.interview_type}' in frame_info.xlsx"
				)
				continue

			try:
				df = pd.read_csv(p)
			except Exception as e:
				log(f"[skip] Failed to read {p.name}: {e}")
				continue

			if abs(src_fps - target_fps) < 1e-6:
				out_path = out_dir / p.name
				try:
					shutil.copy2(p, out_path)
					log(f"[copy] {p.name} (already {target_fps} FPS)")
				except Exception as e:
					log(f"[error] Copying {p.name} -> {out_path}: {e}")
				continue

			try:
				out_df = resample_to_target(df, src_fps=src_fps, target_fps=target_fps)
			except Exception as e:
				log(f"[error] Resampling {p.name}: {e}")
				continue

			out_path = out_dir / p.name
			try:
				out_df.to_csv(out_path, index=False)
				log(f"[ok] {p.name}: {src_fps} -> {target_fps} FPS (rows {len(df)} -> {len(out_df)})")
			except Exception as e:
				log(f"[error] Writing {out_path.name}: {e}")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
	ap = argparse.ArgumentParser(description="Downsample OpenFace CSVs to a target FPS using frame_info.xlsx metadata")
	ap.add_argument("--in", dest="in_dir", required=True, help="Input directory containing CSV files")
	ap.add_argument("--frame-info", dest="frame_info", required=True, help="Path to frame_info.xlsx")
	ap.add_argument("--out", dest="out_dir", default=None, help="Output directory for resampled CSVs (default: <in>/aligned_24fps)")
	ap.add_argument("--target-fps", dest="target_fps", type=float, default=24.0, help="Target FPS (default: 24)")
	return ap.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
	args = parse_args(argv)
	in_dir = Path(args.in_dir).expanduser().resolve()
	out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (in_dir / "aligned_24fps")
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
#python /home/mlut/synchrony/source/scripts_source_data_clean/align_fps.py --in /path/to/openface/csvs --frame-info /path/to/frame_info.xlsx --out /path/to/output --target-fps 24