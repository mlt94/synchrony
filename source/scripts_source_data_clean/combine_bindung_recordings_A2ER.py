from __future__ import annotations

import argparse
from pathlib import Path
import json
import re

import pandas as pd

##This script is needed as we have two source files for the same interview;
#This script fixes it; remember that it still outputs two .json files, you only need the first

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


def combine(base_path: Path, seg_path: Path) -> float:
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

    offset_sec = last_ts + step
    seg_adjusted = rebase_segment(seg, timestamp_col, frame_col, offset_sec, frame_offset)

    combined = pd.concat([base, seg_adjusted], ignore_index=True)
    combined.sort_values(timestamp_col, inplace=True, ignore_index=True)
    combined.to_csv(base_path, index=False)
    seg_path.unlink()
    return float(offset_sec)


def _normalise_stem(name: str) -> str:
    """Normalise a stem for matching.

    - Replace runs of hyphens or whitespace with underscores
    - Leave other characters intact
    """
    # collapse hyphens and spaces to single underscore
    out = re.sub(r"[-\s]+", "_", str(name))
    return out


def _find_json_for_stem(transcripts_dir: Path, stem: str) -> Path | None:
    """Find a results_*.json that corresponds to a file stem.

    Tries exact 'results_{stem}.json', then any 'results_{stem}*.json' recursively.
    Also tries hyphen/underscore normalisation.
    """
    s = _normalise_stem(stem)
    exact = transcripts_dir / f"results_{s}.json"
    if exact.exists():
        return exact
    # recursive search for close matches
    candidates = list(transcripts_dir.rglob(f"results_{s}*.json"))
    if candidates:
        return sorted(candidates)[0]
    # try the original (non-normalised) stem
    exact2 = transcripts_dir / f"results_{stem}.json"
    if exact2.exists():
        return exact2
    candidates2 = list(transcripts_dir.rglob(f"results_{stem}*.json"))
    if candidates2:
        return sorted(candidates2)[0]
    return None


def _fuzzy_find_results(transcripts_dir: Path, include_tokens: list[str]) -> Path | None:
    """Fuzzy search for a results_*.json whose path contains all include_tokens (case-insensitive).

    Scans recursively for results_*.json and scores by token coverage; returns best match.
    """
    toks = [t.lower() for t in include_tokens if t]
    all_results = list(transcripts_dir.rglob("results_*.json"))
    best: tuple[int, Path] | None = None
    for p in all_results:
        s = str(p).lower()
        score = sum(1 for t in toks if t in s)
        if score and (best is None or score > best[0]):
            best = (score, p)
    # require at least 2 tokens to match to avoid overly broad picks
    if best and best[0] >= min(2, len(toks)):
        return best[1]
    return None


def _load_transcript(json_path: Path) -> list[dict]:
    with json_path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Transcript is not a list: {json_path}")
    return data


def _offset_transcript(entries: list[dict], offset_ms: float) -> list[dict]:
    out: list[dict] = []
    for e in entries:
        ne = dict(e)
        for key in ("start", "end"):
            if key in ne:
                try:
                    ne[key] = float(ne[key]) + float(offset_ms)
                except Exception:
                    # leave as-is if not numeric
                    pass
        out.append(ne)
    return out


def _write_transcript(json_path: Path, entries: list[dict]) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open('w', encoding='utf-8') as f:
        json.dump(entries, f, indent=4, ensure_ascii=False)


def _map_bindung_stem_to_geschnitten(stem: str, which: int) -> str:
    """Given a CSV stem like '..._Bindung_In' or '...-Bindung-Pr', return
    a candidate transcript stem using '..._Bindung_geschnitten{which}'.

    Hyphens/underscores are normalised to underscores in the returned stem.
    which must be 1 (base) or 2 (segment).
    """
    s = _normalise_stem(stem)
    which = 1 if which == 1 else 2
    # Replace the last token (In/Pr/_2 etc.) when preceded by 'Bindung'
    # Common CSV stems seen: ..._Bindung_In, ..._Bindung_Pr, and their _2 variants.
    if '_Bindung_In_2' in s or '_Bindung_Pr_2' in s:
        s = s.rsplit('_', 1)[0]  # drop trailing _2
    s = (s
         .replace('_Bindung_In', f'_Bindung_geschnitten{which}')
         .replace('_Bindung_Pr', f'_Bindung_geschnitten{which}'))
    return s


def _find_json_for_bindung(transcripts_dir: Path, base_stem: str, which: int) -> Path | None:
    """Fuzzy finder tailored for A2ER Bindung sessions.

    Looks for JSON like: results_<ID>*Bindung*geschnitten{which}*.json anywhere under transcripts_dir.
    <ID> is parsed as the first token of the stem (e.g., 'A2ER').
    """
    s = _normalise_stem(base_stem)
    # subject/session prefix: first token up to underscore
    prefix = s.split("_", 1)[0] if "_" in s else s
    tokens = [prefix.lower(), "bindung", f"geschnitten{1 if which == 1 else 2}"]
    return _fuzzy_find_results(transcripts_dir, tokens)


def main() -> None:
    parser = argparse.ArgumentParser("Combine A2ER Bindung segments (_2 appended to base files).")
    parser.add_argument("--root", required=True, type=Path, help="Directory holding the CSV files.")
    parser.add_argument(
        "--transcripts-dir",
        type=Path,
        default=None,
        help="Optional: directory holding results_*.json transcripts to be merged alongside CSVs (searched recursively).",
    )
    parser.add_argument(
        "--delete-seg-json",
        action="store_true",
        help="If set, delete the segment JSON after merging into the base JSON.",
    )
    args = parser.parse_args()

    root = args.root.resolve()

    # Auto-discover pairs: any file ending with "_2.csv" paired with its base (without suffix),
    # matching stems after normalising hyphens/underscores.
    csv_files = [p for p in root.glob('*.csv') if p.is_file()]
    stem_to_path_norm: dict[str, Path] = { _normalise_stem(p.stem): p for p in csv_files }
    pairs: list[tuple[Path, Path]] = []
    for seg in csv_files:
        stem = seg.stem
        if not stem.endswith('_2'):
            continue
        base_stem = stem[:-2]  # drop trailing "_2"
        base_norm = _normalise_stem(base_stem)
        base = stem_to_path_norm.get(base_norm)
        if base is None:
            print(f"[discover] Skipping segment without base match: {seg.name}")
            continue
        pairs.append((base, seg))

    if not pairs:
        print(f"[discover] No '*_2.csv' segment pairs found in {root}")
        return

    print(f"[discover] Found {len(pairs)} pair(s):")
    for b, s in pairs:
        print(f"  - {b.name}  +  {s.name}")

    for base_path, seg_path in pairs:
        offset_sec = combine(base_path, seg_path)
        print(f"[combine] {base_path.name} (merged {seg_path.name}); offset_sec={offset_sec:.6f}")

        # Merge transcripts if directory provided
        if args.transcripts_dir:
            tdir = args.transcripts_dir.resolve()
            base_stem = base_path.stem
            seg_stem = seg_path.stem
            # Prefer the user-specified Bindung mapping to '..._geschnitten1/2'
            mapped_base = _map_bindung_stem_to_geschnitten(base_stem, which=1)
            mapped_seg = _map_bindung_stem_to_geschnitten(seg_stem, which=2)
            base_json = (
                _find_json_for_stem(tdir, mapped_base)
                or _find_json_for_stem(tdir, base_stem)
                or _find_json_for_bindung(tdir, base_stem, which=1)
            )
            seg_json = (
                _find_json_for_stem(tdir, mapped_seg)
                or _find_json_for_stem(tdir, seg_stem)
                or _find_json_for_bindung(tdir, seg_stem, which=2)
            )

            if base_json is None:
                print(f"[combine-json] Base transcript not found for stem '{base_stem}' in {tdir}")
                continue
            if seg_json is None:
                print(f"[combine-json] Segment transcript not found for stem '{seg_stem}' in {tdir}")
                continue

            # Safety: if base and segment resolve to the same file, skip to avoid corrupting/deleting needed file
            try:
                if base_json.resolve() == seg_json.resolve():
                    print(
                        f"[combine-json][WARN] Base and segment transcripts point to the SAME file: {base_json}. "
                        "Skipping transcript merge and deletion for this pair."
                    )
                    continue
            except Exception:
                pass

            try:
                base_entries = _load_transcript(base_json)
                seg_entries = _load_transcript(seg_json)
                # JSON timestamps are in milliseconds; CSV offset is seconds
                offset_ms = float(offset_sec) * 1000.0
                seg_adj = _offset_transcript(seg_entries, offset_ms)

                # Deduplicate: avoid adding items already present in base (by start,end,text rounded)
                def _key(e: dict) -> tuple:
                    start = float(e.get('start', 0.0))
                    end = float(e.get('end', 0.0))
                    txt = str(e.get('text', '')).strip()
                    return (round(start, 3), round(end, 3), txt)

                base_keys = { _key(e) for e in base_entries }
                seg_adj_dedup = [e for e in seg_adj if _key(e) not in base_keys]

                merged = sorted(base_entries + seg_adj_dedup, key=lambda x: (float(x.get('start', 0)), float(x.get('end', 0))))
                _write_transcript(base_json, merged)
                print(f"[combine-json] Wrote merged transcript: {base_json} (from {seg_json}); added={len(seg_adj_dedup)} skipped_dupes={len(seg_adj) - len(seg_adj_dedup)}")
                if args.delete_seg_json:
                    try:
                        # Prevent deleting if somehow same as base
                        if base_json.resolve() == seg_json.resolve():
                            print(f"[combine-json][WARN] Prevented deletion: segment equals base: {seg_json}")
                        else:
                            seg_json.unlink()
                            print(f"[combine-json] Deleted segment transcript: {seg_json}")
                    except Exception as de:
                        print(f"[combine-json] Failed to delete {seg_json}: {de}")
            except Exception as je:
                print(f"[combine-json] Error merging transcripts for '{base_stem}': {je}")


if __name__ == "__main__":
    main()