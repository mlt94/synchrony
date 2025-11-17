"""
Generate comprehensive AU heatmaps showing ALL 17 action units from OpenFace.
Based on generate_time_series_descriptions.py but focused on visualization only.

This script:
1. Reads data_model.yaml to get interview information
2. Extracts ALL AU time series from OpenFace CSVs for specific speech turn windows
3. Generates side-by-side heatmaps (therapist left, client right) with all 17 AUs as rows
4. Saves each turn as a separate high-resolution image

ALL 17 OpenFace Action Units:
AU01_r (Inner Brow Raiser), AU02_r (Outer Brow Raiser), AU04_r (Brow Lowerer),
AU05_r (Upper Lid Raiser), AU06_r (Cheek Raiser), AU07_r (Lid Tightener),
AU09_r (Nose Wrinkler), AU10_r (Upper Lip Raiser), AU12_r (Lip Corner Puller),
AU14_r (Dimpler), AU15_r (Lip Corner Depressor), AU17_r (Chin Raiser),
AU20_r (Lip Stretcher), AU23_r (Lip Tightener), AU25_r (Lips Part),
AU26_r (Jaw Drop), AU45_r (Blink)
"""

import sys
import os
import numpy as np
import pandas as pd
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import matplotlib.pyplot as plt
import yaml

# Add OpenTSLM src to path
opentslm_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "opentslm"))
src_dir = os.path.join(opentslm_dir, "src")
sys.path.insert(0, opentslm_dir)
sys.path.insert(0, src_dir)


def load_data_model(yaml_path: Path) -> Dict:
    """Load the data_model.yaml file."""
    print(f"Loading data model from {yaml_path}...")
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    print(f"✅ Loaded {len(data['interviews'])} interviews")
    return data


def load_speech_turns(json_path: Path) -> List[Dict]:
    """Load speech turns from transcript JSON."""
    with open(json_path, 'r') as f:
        turns = json.load(f)
    return turns


def extract_au_window(csv_path: Path, start_ms: float, end_ms: float, au_columns: List[str]) -> pd.DataFrame:
    """Extract AU data from OpenFace CSV for a specific time window.
    
    Args:
        csv_path: Path to OpenFace CSV
        start_ms: Start time in milliseconds
        end_ms: End time in milliseconds
        au_columns: List of AU column names to extract
    
    Returns:
        DataFrame with timestamp and AU columns for the specified window
    """
    # Read CSV with whitespace handling
    df = pd.read_csv(csv_path, skipinitialspace=True)
    
    # Convert timestamp from seconds to milliseconds
    df['timestamp_ms'] = df['timestamp'] * 1000
    
    # Filter to time window
    mask = (df['timestamp_ms'] >= start_ms) & (df['timestamp_ms'] <= end_ms)
    window_df = df.loc[mask, ['timestamp_ms'] + au_columns].copy()
    
    return window_df


def bin_time_series(data: pd.DataFrame, au_name: str, num_bins: int = 8) -> np.ndarray:
    """Bin time series data into equal temporal bins and compute mean activation per bin.
    
    Args:
        data: DataFrame with timestamp_ms and AU columns
        au_name: Name of the AU column to bin
        num_bins: Number of temporal bins
    
    Returns:
        Array of mean activations per bin (length = num_bins)
    """
    if len(data) == 0:
        return np.zeros(num_bins)
    
    # Create bin indices for each row
    bin_indices = np.linspace(0, len(data), num_bins + 1, dtype=int)
    
    # Compute mean activation for each bin
    binned_values = []
    for i in range(num_bins):
        start_idx = bin_indices[i]
        end_idx = bin_indices[i + 1]
        if end_idx > start_idx:
            bin_mean = data[au_name].iloc[start_idx:end_idx].mean()
            binned_values.append(bin_mean)
        else:
            binned_values.append(0.0)
    
    return np.array(binned_values)


def generate_comprehensive_heatmap(
    therapist_csv: Path,
    patient_csv: Path,
    turn: Dict,
    au_names: List[str],
    output_path: Path,
    num_bins: int = 8
) -> bool:
    """Generate comprehensive heatmap showing ALL action units for therapist and client.
    
    Args:
        therapist_csv: Path to therapist OpenFace CSV
        patient_csv: Path to patient OpenFace CSV
        turn: Speech turn dict with start_ms, end_ms, speaker_id
        au_names: List of ALL AU names to plot (17 AUs)
        output_path: Where to save the plot
        num_bins: Number of temporal bins (default 8)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        start_ms = turn['start_ms']
        end_ms = turn['end_ms']
        speaker_id = turn['speaker_id']
        turn_index = turn['turn_index']
        
        # Extract AU data for both therapist and patient
        therapist_data = extract_au_window(therapist_csv, start_ms, end_ms, au_names)
        patient_data = extract_au_window(patient_csv, start_ms, end_ms, au_names)
        
        # Check if we have data
        if therapist_data.empty or patient_data.empty:
            print(f"⚠️ No data found for turn {turn_index} ({start_ms}-{end_ms}ms)")
            return False
        
        # Create binned heatmaps for each AU
        therapist_heatmap = np.zeros((len(au_names), num_bins))
        client_heatmap = np.zeros((len(au_names), num_bins))
        
        for i, au_name in enumerate(au_names):
            therapist_heatmap[i, :] = bin_time_series(therapist_data, au_name, num_bins)
            client_heatmap[i, :] = bin_time_series(patient_data, au_name, num_bins)
        
        # Create figure with two side-by-side heatmaps (taller for 17 AUs)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 12))
        
        # Find global min/max for consistent color scaling
        vmin = min(therapist_heatmap.min(), client_heatmap.min())
        vmax = max(therapist_heatmap.max(), client_heatmap.max())
        
        # Therapist heatmap (left)
        im1 = ax1.imshow(therapist_heatmap, aspect='auto', cmap='Blues', 
                         interpolation='nearest', vmin=vmin, vmax=vmax)
        ax1.set_title('THERAPIST AU Activation (All 17 AUs)', fontsize=16, fontweight='bold', pad=20)
        ax1.set_ylabel('Action Unit', fontsize=13, fontweight='bold')
        ax1.set_xlabel('Time Progression (Start → End)', fontsize=13, fontweight='bold')
        ax1.set_yticks(range(len(au_names)))
        ax1.set_yticklabels(au_names, fontsize=10)
        ax1.set_xticks(range(num_bins))
        ax1.set_xticklabels(['Start', 'Early', 'Early-Mid', 'Mid', 'Mid-Late', 'Late', 'Very Late', 'End'][:num_bins], 
                            fontsize=10, rotation=45, ha='right')
        
        # Add colorbar for therapist
        cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.set_label('Activation Level', fontsize=12, fontweight='bold')
        
        # Add value annotations on therapist heatmap (smaller font for 17 AUs)
        for i in range(len(au_names)):
            for j in range(num_bins):
                text = ax1.text(j, i, f'{therapist_heatmap[i, j]:.2f}',
                               ha="center", va="center", color="black", fontsize=7)
        
        # Client heatmap (right)
        im2 = ax2.imshow(client_heatmap, aspect='auto', cmap='Oranges', 
                         interpolation='nearest', vmin=vmin, vmax=vmax)
        ax2.set_title('CLIENT AU Activation (All 17 AUs)', fontsize=16, fontweight='bold', pad=20)
        ax2.set_ylabel('Action Unit', fontsize=13, fontweight='bold')
        ax2.set_xlabel('Time Progression (Start → End)', fontsize=13, fontweight='bold')
        ax2.set_yticks(range(len(au_names)))
        ax2.set_yticklabels(au_names, fontsize=10)
        ax2.set_xticks(range(num_bins))
        ax2.set_xticklabels(['Start', 'Early', 'Early-Mid', 'Mid', 'Mid-Late', 'Late', 'Very Late', 'End'][:num_bins], 
                            fontsize=10, rotation=45, ha='right')
        
        # Add colorbar for client
        cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        cbar2.set_label('Activation Level', fontsize=12, fontweight='bold')
        
        # Add value annotations on client heatmap (smaller font for 17 AUs)
        for i in range(len(au_names)):
            for j in range(num_bins):
                text = ax2.text(j, i, f'{client_heatmap[i, j]:.2f}',
                               ha="center", va="center", color="black", fontsize=7)
        
        # Overall title
        plt.suptitle(f"Turn {turn_index}: {speaker_id.capitalize()} speaking ({start_ms:.0f}-{end_ms:.0f}ms)\n" + 
                     f"Comprehensive AU Heatmap: Mean activation across {num_bins} temporal phases", 
                     fontsize=17, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Error generating heatmap for turn {turn.get('turn_index', '?')}: {e}")
        import traceback
        traceback.print_exc()
        return False


def process_interview(
    interview: Dict,
    interview_type: str,
    output_dir: Path,
    au_names: List[str],
    max_turns: int = None
) -> int:
    """Process a single interview type for heatmap generation.
    
    Args:
        interview: Interview dict from data_model.yaml
        interview_type: One of 'bindung', 'personal', 'wunder'
        output_dir: Directory to save heatmaps
        au_names: List of ALL AU names to analyze (17 AUs)
        max_turns: Maximum number of turns to process (None = all)
    
    Returns:
        Number of heatmaps generated
    """
    if interview_type not in interview['types']:
        print(f"⚠️ Interview type '{interview_type}' not found, skipping")
        return 0
    
    type_data = interview['types'][interview_type]
    therapist_csv = Path(type_data['therapist_openface'])
    patient_csv = Path(type_data['patient_openface'])
    transcript_json = Path(type_data['transcript'])
    
    # Validate paths
    if not therapist_csv.exists():
        print(f"❌ Therapist CSV not found: {therapist_csv}")
        return 0
    if not patient_csv.exists():
        print(f"❌ Patient CSV not found: {patient_csv}")
        return 0
    if not transcript_json.exists():
        print(f"❌ Transcript JSON not found: {transcript_json}")
        return 0
    
    # Load speech turns
    turns = load_speech_turns(transcript_json)
    
    # Limit turns if requested
    if max_turns:
        turns = turns[:max_turns]
    
    therapist_id = interview['therapist']['therapist_id']
    patient_id = interview['patient']['patient_id']
    
    # Create output directory for this interview
    interview_output_dir = output_dir / f"{patient_id}_{interview_type}"
    interview_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nProcessing {patient_id}/{therapist_id} - {interview_type}: {len(turns)} turns")
    print(f"Output directory: {interview_output_dir}")
    
    generated_count = 0
    
    for turn in tqdm(turns, desc=f"{patient_id} {interview_type}"):
        turn_index = turn['turn_index']
        speaker = turn['speaker_id']
        
        # Generate heatmap
        output_path = interview_output_dir / f"turn_{turn_index:03d}_{speaker}.png"
        success = generate_comprehensive_heatmap(
            therapist_csv, patient_csv, turn, au_names, output_path
        )
        
        if success:
            generated_count += 1
    
    print(f"✅ Generated {generated_count}/{len(turns)} heatmaps for {patient_id} {interview_type}")
    return generated_count


def main():
    parser = argparse.ArgumentParser(
        description="Generate comprehensive AU heatmaps showing ALL 17 action units for each speech turn"
    )
    parser.add_argument(
        "--data_model",
        type=Path,
        required=True,
        help="Path to data_model.yaml"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output directory (will create 'all_heatmaps' subdirectory)"
    )
    parser.add_argument(
        "--interview_types",
        nargs="+",
        default=["personal", "bindung", "wunder"],
        help="Interview types to process"
    )
    parser.add_argument(
        "--max_interviews",
        type=int,
        default=None,
        help="Maximum number of interviews to process (None = all)"
    )
    parser.add_argument(
        "--max_turns_per_interview",
        type=int,
        default=None,
        help="Maximum turns per interview (None = all)"
    )
    
    args = parser.parse_args()
    
    # Define ALL 17 OpenFace Action Units
    ALL_AU_COLUMNS = [
        'AU01_r',  # Inner Brow Raiser
        'AU02_r',  # Outer Brow Raiser
        'AU04_r',  # Brow Lowerer
        'AU05_r',  # Upper Lid Raiser
        'AU06_r',  # Cheek Raiser
        'AU07_r',  # Lid Tightener
        'AU09_r',  # Nose Wrinkler
        'AU10_r',  # Upper Lip Raiser
        'AU12_r',  # Lip Corner Puller (Smile)
        'AU14_r',  # Dimpler
        'AU15_r',  # Lip Corner Depressor
        'AU17_r',  # Chin Raiser
        'AU20_r',  # Lip Stretcher
        'AU23_r',  # Lip Tightener
        'AU25_r',  # Lips Part
        'AU26_r',  # Jaw Drop
        'AU45_r',  # Blink
    ]
    
    print("🎨 Starting comprehensive AU heatmap generation (ALL 17 AUs)")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Data model: {args.data_model}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Interview types: {args.interview_types}")
    print(f"  AU columns: {len(ALL_AU_COLUMNS)} AUs (all available)")
    print()
    
    # Create output directory structure
    all_heatmaps_dir = args.output_dir / "all_heatmaps"
    all_heatmaps_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory created: {all_heatmaps_dir}")
    
    # Load data model
    data_model = load_data_model(args.data_model)
    interviews = data_model['interviews']
    
    if args.max_interviews:
        interviews = interviews[:args.max_interviews]
    
    # Process all interviews
    total_heatmaps = 0
    
    for interview_idx, interview in enumerate(interviews):
        patient_id = interview['patient']['patient_id']
        therapist_id = interview['therapist']['therapist_id']
        
        print(f"\n{'='*80}")
        print(f"Interview {interview_idx + 1}/{len(interviews)}: {patient_id}/{therapist_id}")
        print(f"{'='*80}")
        
        for interview_type in args.interview_types:
            count = process_interview(
                interview,
                interview_type,
                all_heatmaps_dir,
                ALL_AU_COLUMNS,
                max_turns=args.max_turns_per_interview
            )
            total_heatmaps += count
    
    print(f"\n{'='*80}")
    print(f"✅ Complete! Generated {total_heatmaps} comprehensive AU heatmaps")
    print(f"   Output location: {all_heatmaps_dir}")
    print(f"   Each heatmap shows ALL 17 action units across 8 temporal bins")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
