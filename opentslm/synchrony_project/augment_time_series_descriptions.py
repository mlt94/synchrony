"""
Data augmentation strategy for AU descriptions: substitute AU numbers to expand coverage.

THE PROBLEM:
- Training data only mentions AU04, AU06, AU12, AU15, because the VLM cannot attend to more features in a plot/heatmap
- Model has 17 AUs as input but learns to only generate text about 4 AUs

THE SOLUTION:
- Create augmented versions of training samples
- Systematically replace mentioned AUs with other AUs from the 17 available
- Keep the same temporal patterns/descriptions but for different AUs
- Model learns the STRUCTURE of AU description, applicable to any AU

EXAMPLE:
Original: "AU12 peaks toward end, AU06 shows elevation"
Augmented: "AU17 peaks toward end, AU09 shows elevation"
           "AU02 peaks toward end, AU25 shows elevation"
           etc.

This way:
1. Model sees descriptions for ALL 17 AUs during training
2. Model learns to generate AU## where ## comes from attention to time-series
3. No train-test mismatch
"""

import json
import re
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict


# All 17 AUs available in OpenFace output (from your data)
ALL_AUS = [
    'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 
    'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r',
    'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r'
]

# AUs that appear in current training data
ORIGINAL_AUS = ['AU04_r', 'AU06_r', 'AU12_r', 'AU15_r']


def extract_au_numbers(text: str) -> List[str]:
    """Extract all AU## numbers mentioned in text."""
    return re.findall(r'\bAU\d+\b', text)


def _get_non_overwriting_path(path: Path) -> Path:
    """Return a Path that does not overwrite an existing file.

    If `path` does not exist, return it. Otherwise append `_augmented`,
    then `_augmented_1`, `_augmented_2`, ... until a free filename is found.
    """
    path = Path(path)
    if not path.exists():
        return path

    parent = path.parent
    stem = path.stem
    suffix = path.suffix

    # First try stem_augmented + suffix
    candidate = parent / f"{stem}_augmented{suffix}"
    if not candidate.exists():
        return candidate

    # Otherwise try numbered suffixes
    i = 1
    while True:
        candidate = parent / f"{stem}_augmented_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def create_au_substitution_mapping(
    original_aus: List[str],
    available_aus: List[str],
    seed: int = None
) -> Dict[str, str]:
    """
    Create a mapping to substitute original AUs with different ones.
    
    Args:
        original_aus: AUs that appear in the text (e.g., ['AU12', 'AU06'])
        available_aus: All available AUs to choose from
        seed: Random seed for reproducibility
    
    Returns:
        Mapping like {'AU12': 'AU17', 'AU06': 'AU09'}
    """
    if seed is not None:
        random.seed(seed)
    
    # AUs we can substitute TO (exclude the originals to create variation)
    available_for_substitution = [au for au in available_aus if au not in original_aus]
    
    # Shuffle and take as many as we need
    random.shuffle(available_for_substitution)
    
    mapping = {}
    for i, orig_au in enumerate(original_aus):
        if i < len(available_for_substitution):
            # Extract just the number part for mapping
            orig_num = re.search(r'\d+', orig_au).group()
            new_num = re.search(r'\d+', available_for_substitution[i]).group()
            mapping[f'AU{orig_num}'] = f'AU{new_num}'
    
    return mapping


def substitute_aus_in_text(text: str, mapping: Dict[str, str]) -> str:
    """
    Replace AU numbers in text according to mapping.
    
    Args:
        text: Text containing AU references
        mapping: Dict like {'AU12': 'AU17', 'AU06': 'AU09'}
    
    Returns:
        Text with substituted AU numbers
    """
    result = text
    
    # Sort by length (descending) to avoid partial replacements
    # e.g., replace "AU12" before "AU1"
    for orig_au in sorted(mapping.keys(), key=len, reverse=True):
        new_au = mapping[orig_au]
        # Use word boundaries to avoid replacing parts of numbers
        result = re.sub(rf'\b{orig_au}\b', new_au, result)
    
    return result


def augment_training_sample(
    sample: Dict,
    augmentation_factor: int = 2,
    all_aus: List[str] = ALL_AUS
) -> List[Dict]:
    """
    Create multiple augmented versions of a training sample.
    
    Args:
        sample: Original training sample dict
        augmentation_factor: How many augmented versions to create
        all_aus: List of all available AUs
    
    Returns:
        List containing [augmented1, augmented2, ...] (original excluded)
    """
    augmented_samples = []  # Only augmented versions, no original
    
    # Extract AUs mentioned in the description
    combined_desc = sample.get('combined_description', '')
    original_summary = sample.get('original_summary', '')
    timeseries_desc = sample.get('original_timeseries_description', '')
    
    mentioned_aus = list(set(extract_au_numbers(combined_desc)))
    
    if not mentioned_aus:
        # No AUs to substitute, skip this sample (don't include original)
        return augmented_samples
    
    # Create multiple augmented versions with different AU substitutions
    for i in range(augmentation_factor):
        mapping = create_au_substitution_mapping(
            mentioned_aus, 
            all_aus,
            seed=hash(f"{sample.get('patient_id')}_{sample.get('turn_index')}_{i}")
        )
        
        # Create augmented sample
        aug_sample = sample.copy()
        
        # Substitute AUs in all text fields
        if combined_desc:
            aug_sample['combined_description'] = substitute_aus_in_text(combined_desc, mapping)
        if timeseries_desc:
            aug_sample['original_timeseries_description'] = substitute_aus_in_text(timeseries_desc, mapping)
        
        # Mark as augmented
        aug_sample['augmented'] = True
        aug_sample['augmentation_mapping'] = mapping
        aug_sample['augmentation_index'] = i
        
        augmented_samples.append(aug_sample)
    
    return augmented_samples


def process_file_with_augmentation(
    input_path: Path,
    output_path: Path,
    augmentation_factor: int = 2,
    dry_run: bool = False
) -> Dict:
    """Process a single combined description file with AU substitution augmentation."""
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    stats = {
        "file": input_path.name,
        "original_samples": len(data),
        "augmented_samples": 0,
        "total_samples": 0,
        "example_original": None,
        "example_augmented": None
    }
    
    augmented_data = []
    
    for sample in data:
        augmented_versions = augment_training_sample(
            sample, 
            augmentation_factor=augmentation_factor,
            all_aus=[au.replace('_r', '') for au in ALL_AUS]
        )
        
        # Store example for display
        if stats["example_original"] is None and len(augmented_versions) > 1:
            stats["example_original"] = augmented_versions[0].get('combined_description', '')
            stats["example_augmented"] = augmented_versions[1].get('combined_description', '')
            stats["example_mapping"] = augmented_versions[1].get('augmentation_mapping', {})
        
        augmented_data.extend(augmented_versions)
        stats["augmented_samples"] += len(augmented_versions) - 1
    
    stats["total_samples"] = len(augmented_data)
    
    # Write augmented data to a NEW file (do not overwrite existing files)
    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # If output_path already exists, generate a non-conflicting filename
        final_output = _get_non_overwriting_path(output_path)

        with open(final_output, 'w', encoding='utf-8') as f:
            json.dump(augmented_data, f, indent=2, ensure_ascii=False)
        # Return the actual path used for writing
        stats['output_path'] = str(final_output)
    else:
        # In dry-run, show what filename WOULD be used (without writing)
        suggested = _get_non_overwriting_path(output_path)
        stats['suggested_output_path'] = str(suggested)
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Augment training data by substituting AU numbers to expand coverage"
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path(r"C:\Users\User\Desktop\martins\results\combined_descriptions"),
        help="Directory containing combined description JSON files"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(r"C:\Users\User\Desktop\martins\results\combined_descriptions_augmented"),
        help="Output directory for augmented files"
    )
    parser.add_argument(
        "--augmentation_factor",
        type=int,
        default=2,
        help="Number of augmented versions per sample (default: 2). Only augmented samples will be in output."
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what would be created without writing files"
    )
    parser.add_argument(
        "--pattern",
        default="*_combined.json",
        help="File pattern to match (default: *_combined.json)"
    )
    
    args = parser.parse_args()
    
    print("🔄 AU Substitution Data Augmentation")
    print("=" * 80)
    print(f"Strategy: Substitute AU04/06/12/15 -> other AUs from 17 available")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Augmentation factor: {args.augmentation_factor} versions per sample")
    print(f"Output contains: ONLY augmented samples (originals excluded)")
    print(f"Dry run: {args.dry_run}")
    print()
    
    # Find all matching files
    files = list(args.input_dir.glob(args.pattern))
    
    if not files:
        print(f"❌ No files found matching pattern '{args.pattern}' in {args.input_dir}")
        return
    
    print(f"Found {len(files)} files to process\n")
    
    all_stats = []
    
    for input_path in files:
        # Keep original filename but write into the output directory
        output_path = args.output_dir / input_path.name
        
        print(f"Processing: {input_path.name}")
        stats = process_file_with_augmentation(
            input_path, output_path, args.augmentation_factor, args.dry_run
        )
        all_stats.append(stats)
        
        print(f"  Original samples: {stats['original_samples']}")
        print(f"  Augmented samples: {stats['augmented_samples']}")
        print(f"  Total samples: {stats['total_samples']} ({stats['total_samples']/stats['original_samples']:.1f}x)")
        
        if stats.get("example_original"):
            print(f"\n  📝 Example AU substitution:")
            print(f"  Mapping: {stats.get('example_mapping', {})}")
            print(f"\n  ORIGINAL: {stats['example_original'][:250]}...")
            print(f"\n  AUGMENTED: {stats['example_augmented'][:250]}...")
            print()
        # Show suggested/actual output path
        if 'suggested_output_path' in stats:
            print(f"  Suggested output path (dry-run): {stats['suggested_output_path']}")
        if 'output_path' in stats:
            print(f"  Written output path: {stats['output_path']}")
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 Summary")
    print("=" * 80)
    total_original = sum(s['original_samples'] for s in all_stats)
    total_augmented = sum(s['augmented_samples'] for s in all_stats)
    total_final = sum(s['total_samples'] for s in all_stats)
    
    print(f"Files processed: {len(all_stats)}")
    print(f"Original samples: {total_original}")
    print(f"Augmented samples: {total_augmented}")
    print(f"Total output samples: {total_final} (only augmented, originals excluded)")
    
    print(f"\n🎯 Coverage Analysis:")
    print(f"  Before: Model only sees AU04, AU06, AU12, AU15 (4 AUs)")
    print(f"  After: Model sees all 17 AUs with {args.augmentation_factor} variations each")
    print(f"  Result: Model learns to describe ANY AU, not just the original 4")
    
    if args.dry_run:
        print("\n⚠️  DRY RUN - No files were actually modified")
        print("Run without --dry_run to apply changes")
    else:
        print(f"\n✅ Augmented files saved to: {args.output_dir}")
        print("\n💡 Next steps:")
        print("1. Combine original files + augmented files (or use augmented alone)")
        print("2. Update data_model.yaml to point to combined/augmented JSON files")
        print("3. Train with expanded data (covering all 17 AUs)")
        print("4. Model will learn to generate AU## for ANY AU in the time-series input")


if __name__ == "__main__":
    main()
