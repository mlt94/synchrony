"""
Align OpenFace CSV time series down to a target FPS (default 24) using rolling-mean smoothing
and linear interpolation based on the 'timestamp' column.

Inputs:
  - Directory of CSV files named like: IDENTIFIER_YYYY-MM-DD_Interviewtype_targetperson.csv
  - Excel file frame_info.xlsx with columns:
		Pseudonym, FPS BRFI, FPS STiP, FPS WF
The interpolation and downsampling is done according to:
- Rolling mean smoothing (pandas.rolling, only where source_fps > target_fps)
Original timestamps: T₀, T₁, …, T_{N−1}
Original values for some AU: X₀ … X_{N−1}
Window size: w_raw = ceil(source_fps / target_fps) ->> I enforce odd so we can center, for instance 30/24 --> 3
This operatin outputs new AU values (Y) calculated as the mean of its neighbors; the values then need to be transformed onto the new timestep grid
This is done to smooth out jitters in the high frequency of the AU estimates

-- Time resampling (np.interp(new_timestep vector τ_k, old_timesteps T_i, values from rolling mean y))
Step = Δ = 1 / target_fps
Generate new relative times (new_timestep vector): τ_k = k · Δ, for k = 0, 1, …, K where K are the integers denoting the corresponding timestep
Z_k (new AU/feature vectors) = Y_i + (Y{i+1} − Y_i) * (T'k − T_i) / (T{i+1} − T_i)
Y_i + (Y{i+1} − Y_i) --> how much does the new AU values change between two known points?
(T'k − T_i) / (T{i+1} − T_i) --> how far are these two points separated in time?

We thus compute a weighted average of the two smoothed neighbor values, weighted according to how far away the target value was between its neighbors.

python source/scripts_source_data_clean/align_fps.py --in /home/data_shares/genface/data/MentalHealth/msb/OpenFace_Output_MSB/ --out /home/data_shares/
genface/data/MentalHealth/msb/ --target-fps 30 --frame-info /home/data_shares/genface/data
/MentalHealth/msb/frame_info.xlsx

the --in argument specifies the path where all the .csv files should be located, you know, the openface outputs
the --target-fps specifies the target fps rate
the --frame-info is the xlsx file that stores the fps rates from the .csv located (those we reference in --in)
the --out is the path where you want the new, aligned files saved (a directory will be created, but you need to specify the path)
In addition an "align_fps_log.txt" is created which stores one line pr file designating if it was copied, aligned or or if it failed. Super useful to know if something went wrong!

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

	  IDENTIFIER_YYYY-MM-DD_Interviewtype_In.csv
	  IDENTIFIER_YYYY-MM-DD-Interviewtype-In.csv
	If the second-to-last token is 'geschnitten', the interview type is the token before it.
	"""
	name = path.name
	if name.lower().endswith(".csv"):
		name = name[:-4]
	# Some of the file names contains '-' delimiters ; make all --> '_'
	tokens = [tok for tok in name.replace("-", "_").split("_") if tok]
	if len(tokens) < 3:
		return None
	identifier = tokens[0]
	# Handle cases where the second-to-last token is 'geschnitten'; use the preceding token
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

	Returns the column name to use in the Excel ('FPS BRFI'/'FPS STiP'/'FPS WF').
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
	df.columns = [str(c).strip() for c in df.columns]
	df["Pseudonym"] = df["Pseudonym"].astype(str).str.strip()
	return df


def get_source_fps(frame_info: pd.DataFrame, identifier: str, interview_type: str) -> Optional[float]:
	col = normalize_type_for_column(interview_type)
	row = frame_info.loc[frame_info["Pseudonym"].astype(str).str.strip() == str(identifier).strip()]
	if row.empty:
		return None
	val = row.iloc[0][col] #the corresponding fps rate for the identifier for the interview type
	fps = float(val)
	return fps


# ------------------------------
# Resampling logic
# ------------------------------

def _compute_time_axis_from_timestamp(df: pd.DataFrame) -> Tuple[np.ndarray, str, Optional[str]]:
	"""Return times (seconds), the exact timestamp column name used, and (if present) the frame column name.

	Requires a 'timestamp' column (case-insensitive). 
	"""
	# Normalize header names to strip BOM/whitespace for detection (do not overwrite original df.columns order)
	norm_cols = [str(c).replace("\ufeff", "").strip() for c in df.columns]
	# Map lower -> original name
	col_map = {c.lower(): orig for c, orig in zip(norm_cols, df.columns)}
	ts_key = next((k for k in col_map if k == "timestamp"), None)
	if ts_key is None:
		raise KeyError("Required 'timestamp' column not found (case-insensitive)")
	ts_col = col_map[ts_key]
	fr_col = col_map.get("frame")

	times = pd.to_numeric(df[ts_col], errors="raise").to_numpy(dtype=float)
	return times, ts_col, fr_col


def resample_to_target(df: pd.DataFrame, src_fps: float, target_fps: float = 24.0) -> pd.DataFrame:
	"""Resample dataframe to target fps using rolling-mean (pandas.rolling) smoothing plus interpolation on 'timestamp' (np.interp).

	- Recompute 'timestamp' and 'frame' (if present originally, frame is regenerated)
	- Preserve column order from the original
	- Non-numeric columns are aligned by nearest original sample
	"""
	# Clean headers for whitespace and similiar
	df = df.copy()
	df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]

	times_old, ts_col, fr_col = _compute_time_axis_from_timestamp(df)
	t0 = times_old[0] #to account for the few cases where first timesteps may not be 0
	times_rel = times_old - t0 #original timesteps

	# Sort the dataframe by timestamp to align values with times_rel
	df_sorted = df.sort_values(by=ts_col).reset_index(drop=True)

	t_end = times_rel[-1]
	# Compute number of new frames n_new and create the new timestep grid new_rel
	n_new = int(np.floor(t_end * target_fps)) + 1 
	new_rel = np.arange(n_new, dtype=float) / float(target_fps) #the new time steps
	new_rel = np.round(new_rel, 6) #many decimals when going from, for instance 60 fps to 24
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
		window = 1 #if src_fps is smaller than target fps, dont do rolling mean
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
		#new_rel --> vector of new timesteps, times_rel --> original timesteps, y--> values after rolling mean

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
			# Write clean, zero-based seconds for timestamp (no huge epoch-like numbers)
			out_cols[col] = new_rel
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

			# Even if FPS matches, rewrite file to normalize timestamp and ensure comma-separated output
			if abs(src_fps - target_fps) < 1e-6:
				try:
					out_df = resample_to_target(df, src_fps=src_fps, target_fps=target_fps)
				except Exception as e:
					log(f"[error] Normalizing (same-FPS) {p.name}: {e}")
					continue

			try:
				out_df = resample_to_target(df, src_fps=src_fps, target_fps=target_fps)
			except Exception as e:
				log(f"[error] Resampling {p.name}: {e}")
				continue

			out_path = out_dir / p.name
			try:
				# Enforce comma-separated CSV output
				out_df.to_csv(out_path, index=False, sep=";")
				log(f"[ok] {p.name}: {src_fps} -> {target_fps} FPS (rows {len(df)} -> {len(out_df)})")
			except Exception as e:
				log(f"[error] Writing {out_path.name}: {e}")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
	ap = argparse.ArgumentParser(description="Downsample OpenFace CSVs to a target FPS using frame_info.xlsx metadata")
	ap.add_argument("--in", dest="in_dir", required=True, help="Input directory containing CSV files")
	ap.add_argument("--frame-info", dest="frame_info", required=True, help="Path to frame_info.xlsx")
	ap.add_argument("--out", dest="out_dir", required=True, default=None, help="Output directory for resampled CSVs")
	ap.add_argument("--target-fps", dest="target_fps", type=float, default=24.0, help="Target FPS (default: 24)")
	return ap.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
	args = parse_args(argv)
	in_dir = Path(args.in_dir).expanduser().resolve()
	out_dir_name = f"aligned_{args.target_fps}fps"
	out_dir = Path(args.out_dir).expanduser().resolve() / out_dir_name
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