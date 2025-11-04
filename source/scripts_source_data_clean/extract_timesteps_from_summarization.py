"""Extract time-step metadata from summarisation outputs.

This script scans a root `input_dir` for JSON files matching a pattern (default
`summaries_speaker_turns_*.json`), removes the "summary" field from each
entry in the JSON array, and writes the cleaned JSON to a matching path under
`output_dir` while preserving the relative directory structure.

Example:
  python extract_timesteps_from_summarization.py --input_dir ./output_files/transcripts --output_dir ./output_files/transcripts_cleaned

The script writes files atomically (write to .tmp then os.replace).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Iterable


def process_file(in_path: Path, out_path: Path, *, force: bool = False) -> bool:
    """Read JSON array from in_path, remove 'summary' key from each object,
    write cleaned JSON array to out_path. Returns True if written, False if skipped.
    """
    if not in_path.exists():
        print(f"[skip] Input file not found: {in_path}")
        return False

    if out_path.exists() and not force:
        print(f"[skip] Output exists: {out_path}")
        return False

    try:
        with open(in_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[error] Failed to read {in_path}: {e}")
        return False

    if not isinstance(data, list):
        print(f"[warn] Expected JSON array in {in_path}; skipping")
        return False

    # Remove 'summary' from each object if present
    cleaned = []
    for obj in data:
        if isinstance(obj, dict):
            if "summary" in obj:
                obj = {k: v for k, v in obj.items() if k != "summary"}
        cleaned.append(obj)

    # Ensure parent exists
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Write atomically
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cleaned, f, indent=4, ensure_ascii=False)
        os.replace(str(tmp), str(out_path))
        print(f"[ok] Wrote {out_path} (entries: {len(cleaned)})")
        return True
    except Exception as e:
        print(f"[error] Failed to write {out_path}: {e}")
        # attempt cleanup
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        return False


def discover_files(root: Path, pattern: str) -> Iterable[Path]:
    """Yield files under `root` matching glob `pattern` (recursive)."""
    if not root.exists():
        return []
    return root.rglob(pattern)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Extract timesteps from summarisation JSONs (remove summaries)")
    p.add_argument("--input_dir", type=Path, required=True, help="Root folder with summarisation JSON files")
    p.add_argument("--output_dir", type=Path, required=True, help="Root folder to write cleaned JSON files (preserves relative paths)")
    p.add_argument("--pattern", type=str, default="summaries_speaker_turns_*.json", help="Glob pattern to find summary JSONs (default: %(default)s)")
    p.add_argument("--force", action="store_true", help="Overwrite existing files in output_dir")
    args = p.parse_args(argv)

    input_root: Path = args.input_dir.resolve()
    output_root: Path = args.output_dir.resolve()

    files = list(discover_files(input_root, args.pattern))
    if not files:
        print(f"[info] No files found matching {args.pattern} under {input_root}")
        return 0

    for fpath in files:
        # compute relative path from input_root and apply to output_root
        try:
            rel = fpath.relative_to(input_root)
        except Exception:
            # fallback: use name only
            rel = Path(fpath.name)

        out_path = output_root.joinpath(rel)
        # ensure extension is .json
        out_path = out_path.with_suffix('.json')

        process_file(fpath, out_path, force=args.force)

    print(f"[done] Processed {len(files)} files. Output root: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
