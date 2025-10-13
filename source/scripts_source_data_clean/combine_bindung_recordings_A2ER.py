from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.astype(str).str.strip()
    return df


def rebase_segment(
    df: pd.DataFrame,
    timestamp_col: str,
    frame_col: str | None,
    ts_offset: float,
    frame_offset: int | None,
) -> pd.DataFrame:
    df = df.copy()
    ts = pd.to_numeric(df[timestamp_col], errors="coerce")
    first_ts = ts.dropna().iloc[0]
    df[timestamp_col] = ts - first_ts + ts_offset

    if frame_col is not None and frame_offset is not None:
        frames = pd.to_numeric(df[frame_col], errors="coerce")
        first_frame = int(frames.dropna().iloc[0])
        df[frame_col] = (frames - first_frame + frame_offset).astype(int)

    return df


def combine(base_path: Path, seg_path: Path) -> None:
    base = load_csv(base_path)
    seg = load_csv(seg_path)

    timestamp_col = next((c for c in base.columns if c.lower() == "timestamp"), None)
    if timestamp_col is None:
        raise ValueError(f"'timestamp' column not found in {base_path.name}")

    frame_col = next((c for c in base.columns if c.lower() == "frame"), None)

    base.sort_values(timestamp_col, inplace=True, ignore_index=True)
    seg.sort_values(timestamp_col, inplace=True, ignore_index=True)

    ts_base = pd.to_numeric(base[timestamp_col], errors="coerce")
    step = ts_base.diff().dropna()
    step = float(step[step > 0].median()) if not step.empty else 0.0
    last_ts = ts_base.dropna().iloc[-1]

    frame_offset = None
    if frame_col is not None:
        frames = pd.to_numeric(base[frame_col], errors="coerce")
        if not frames.dropna().empty:
            frame_offset = int(frames.dropna().iloc[-1]) + 1

    seg_adjusted = rebase_segment(seg, timestamp_col, frame_col, last_ts + step, frame_offset)

    combined = pd.concat([base, seg_adjusted], ignore_index=True)
    combined.sort_values(timestamp_col, inplace=True, ignore_index=True)
    combined.to_csv(base_path, index=False)
    seg_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser("Combine A2ER Bindung segments (_2 appended to base files).")
    parser.add_argument("--root", required=True, type=Path, help="Directory holding the CSV files.")
    args = parser.parse_args()

    root = args.root.resolve()
    pairs = [
        ("A2ER_2024_02-07_Bindung_In.csv", "A2ER_2024_02-07_Bindung_In_2.csv"),
        ("A2ER_2024_02-07_Bindung_Pr.csv", "A2ER_2024_02-07_Bindung_Pr_2.csv"),
    ]

    for base_name, seg_name in pairs:
        base_path = root / base_name
        seg_path = root / seg_name
        combine(base_path, seg_path)
        print(f"[combine] {base_name} (merged {seg_name})")


if __name__ == "__main__":
    main()