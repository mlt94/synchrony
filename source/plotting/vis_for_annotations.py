"""
Generate PDF annotation files for human evaluation of model predictions.

This script creates PDF files that combine:
1. Comprehensive AU heatmap (all 17 AUs for therapist and client)
2. Ground truth description (what was actually said in the transcript)
3. Model-generated prediction (from test_predictions.jsonl)

The PDFs are designed for human annotators to label the accuracy of model predictions.

Usage:
    python vis_for_annotations.py \
        --data_model data_model.yaml \
        --predictions_file results/gemma_3_270m/.../test_predictions.jsonl \
        --output_dir results/annotations \
        --dyad_interviews A2ER_wunder B7UH_personal A0EA_bindung
        
Note: Only processes test set data. Will error if requested dyad/interview not in test set.
"""

import sys
import os
import numpy as np
import pandas as pd
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import yaml
import textwrap
import io
from PIL import Image

# Increase PIL's decompression bomb limit for large heatmaps
Image.MAX_IMAGE_PIXELS = None  # Disable the limit entirely (safe for trusted local generation)

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


def load_predictions(jsonl_path: Path) -> Dict[Tuple[str, str, str, int], Dict]:
    """Load test predictions from JSONL file.
    
    Returns:
        Dict mapping (patient_id, therapist_id, interview_type, turn_index) to prediction dict
    """
    print(f"\nLoading predictions from {jsonl_path}...")
    predictions = {}
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            pred = json.loads(line.strip())
            key = (
                pred['patient_id'],
                pred['therapist_id'],
                pred['interview_type'],
                pred['turn_index']
            )
            predictions[key] = pred
    
    print(f"✅ Loaded {len(predictions)} test predictions")
    return predictions


def load_speech_turns(json_path: Path) -> List[Dict]:
    """Load speech turns from transcript JSON."""
    with open(json_path, 'r') as f:
        turns = json.load(f)
    return turns


def parse_dyad_interview_list(dyad_interviews: List[str]) -> List[Tuple[str, str]]:
    """Parse list of dyad_interview strings into (dyad_id, interview_type) tuples.
    
    Args:
        dyad_interviews: List like ["A2ER_wunder", "B7UH_personal", "A0EA_bindung"]
    
    Returns:
        List of (dyad_id, interview_type) tuples
    """
    parsed = []
    for item in dyad_interviews:
        parts = item.split('_')
        if len(parts) != 2:
            raise ValueError(f"Invalid format: '{item}'. Expected format: 'DYAD_TYPE' (e.g., 'A2ER_wunder')")
        
        dyad_id = parts[0].upper()
        interview_type = parts[1].lower()
        
        if interview_type not in ['personal', 'bindung', 'wunder']:
            raise ValueError(f"Invalid interview type: '{interview_type}'. Must be 'personal', 'bindung', or 'wunder'")
        
        parsed.append((dyad_id, interview_type))
    
    return parsed


def extract_au_window(csv_path: Path, start_ms: float, end_ms: float, au_columns: List[str]) -> pd.DataFrame:
    """Extract AU data from OpenFace CSV for a specific time window."""
    df = pd.read_csv(csv_path, skipinitialspace=True)
    df['timestamp_ms'] = df['timestamp'] * 1000
    mask = (df['timestamp_ms'] >= start_ms) & (df['timestamp_ms'] <= end_ms)
    window_df = df.loc[mask, ['timestamp_ms'] + au_columns].copy()
    return window_df


def bin_time_series(data: pd.DataFrame, au_name: str, num_bins: int = 8) -> np.ndarray:
    """Bin time series data into equal temporal bins and compute mean activation per bin."""
    if len(data) == 0:
        return np.zeros(num_bins)
    
    bin_indices = np.linspace(0, len(data), num_bins + 1, dtype=int)
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


def generate_heatmap_figure(
    therapist_csv: Path,
    patient_csv: Path,
    turn: Dict,
    au_names: List[str],
    num_bins: int = 8,
    fontsize = 40
) -> plt.Figure:
    """Generate comprehensive heatmap figure showing ALL action units.
    
    Returns:
        matplotlib Figure object (caller must close it)
    """
    start_ms = turn['start_ms']
    end_ms = turn['end_ms']
    speaker_id = turn['speaker_id']
    turn_index = turn['turn_index']
    
    # Extract AU data
    therapist_data = extract_au_window(therapist_csv, start_ms, end_ms, au_names)
    patient_data = extract_au_window(patient_csv, start_ms, end_ms, au_names)
    
    if therapist_data.empty or patient_data.empty:
        raise ValueError(f"No data found for turn {turn_index} ({start_ms}-{end_ms}ms)")
    
    # Create binned heatmaps
    therapist_heatmap = np.zeros((len(au_names), num_bins))
    client_heatmap = np.zeros((len(au_names), num_bins))
    
    for i, au_name in enumerate(au_names):
        therapist_heatmap[i, :] = bin_time_series(therapist_data, au_name, num_bins)
        client_heatmap[i, :] = bin_time_series(patient_data, au_name, num_bins)
    

    height = max(55.2, len(au_names) * 10)  # Scale height with number of AUs
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(48, height))  # Stacked vertically, wider
    
    vmin = min(therapist_heatmap.min(), client_heatmap.min())
    vmax = max(therapist_heatmap.max(), client_heatmap.max())
    
    # Calculate time bin labels (in seconds, absolute time from recording start)
    start_s = start_ms / 1000.0
    end_s = end_ms / 1000.0
    bin_edges_abs_s = np.linspace(start_s, end_s, num_bins + 1)
    bin_labels = [f"{bin_edges_abs_s[i]:.0f}-{bin_edges_abs_s[i+1]:.0f}s" for i in range(num_bins)]
    
    # Therapist heatmap (TOP)
    im1 = ax1.imshow(therapist_heatmap, aspect='auto', cmap='Blues', 
                     interpolation='nearest', vmin=vmin, vmax=vmax)
    ax1.set_title('THERAPIST AU Activation', fontsize=fontsize, fontweight='bold', pad=15)
    ax1.set_ylabel('Action Unit', fontsize=fontsize, fontweight='bold')
    ax1.set_xlabel('Time Progression', fontsize=fontsize, fontweight='bold')
    ax1.set_yticks(range(len(au_names)))
    ax1.set_yticklabels(au_names, fontsize=fontsize)
    ax1.set_xticks(range(num_bins))
    ax1.set_xticklabels(bin_labels, fontsize=fontsize, rotation=45, ha='right')
    
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.03, pad=0.04)
    cbar1.ax.tick_params(labelsize=fontsize) 
    cbar1.set_label('Activation', fontsize=fontsize, fontweight='bold')
    
    # Add values
    for i in range(len(au_names)):
        for j in range(num_bins):
            ax1.text(j, i, f'{therapist_heatmap[i, j]:.2f}',
                    ha="center", va="center", color="black", fontsize=fontsize)
    
    # Client heatmap (BOTTOM)
    im2 = ax2.imshow(client_heatmap, aspect='auto', cmap='Oranges', 
                     interpolation='nearest', vmin=vmin, vmax=vmax)
    ax2.set_title('CLIENT AU Activation', fontsize=fontsize, fontweight='bold', pad=15)
    ax2.set_ylabel('Action Unit', fontsize=fontsize, fontweight='bold')
    ax2.set_xlabel('Time Progression', fontsize=fontsize, fontweight='bold')
    ax2.set_yticks(range(len(au_names)))
    ax2.set_yticklabels(au_names, fontsize=fontsize)
    ax2.set_xticks(range(num_bins))
    ax2.set_xticklabels(bin_labels, fontsize=fontsize, rotation=45, ha='right')
    
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.03, pad=0.04)
    cbar2.ax.tick_params(labelsize=fontsize) 
    cbar2.set_label('Activation', fontsize=fontsize, fontweight='bold')
    
    # Add values
    for i in range(len(au_names)):
        for j in range(num_bins):
            ax2.text(j, i, f'{client_heatmap[i, j]:.2f}',
                    ha="center", va="center", color="black", fontsize=fontsize)
    
    # Title
    plt.suptitle(f"Turn {turn_index}: {speaker_id.capitalize()} speaking ({start_ms:.0f}-{end_ms:.0f}ms)", 
                 fontsize=fontsize, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    return fig


def wrap_text(text: str, width: int = 80) -> str:
    """Wrap text to specified width."""
    return '\n'.join(textwrap.wrap(text, width=width))


def create_annotation_pdf(
    patient_id: str,
    therapist_id: str,
    interview_type: str,
    turn: Dict,
    prediction: Dict,
    therapist_csv: Path,
    patient_csv: Path,
    au_names: List[str],
    pdf: PdfPages
):
    """Create a single annotation page in the PDF.
    
    Args:
        patient_id: Patient ID
        therapist_id: Therapist ID
        interview_type: Interview type
        turn: Turn dict from transcript (includes 'summary' field)
        prediction: Prediction dict with 'generated' field
        therapist_csv: Path to therapist OpenFace CSV
        patient_csv: Path to patient OpenFace CSV
        au_names: List of AU names
        pdf: PdfPages object to add page to
    """
    turn_index = turn['turn_index']
    
    try:
        # Create a figure in LANDSCAPE A4 orientation (11.69 x 8.27 inches)
        fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape
        
        # Use 2 columns: left for heatmap, right for text
        gs = fig.add_gridspec(2, 2, height_ratios=[0.3, 1], width_ratios=[1.2, 1], 
                              hspace=0.2, wspace=0.3)
        # Use 2 columns: left for heatmap, right for text
        gs = fig.add_gridspec(2, 2, height_ratios=[0.3, 1], width_ratios=[1.2, 1], 
                              hspace=0.2, wspace=0.3)
        
        # Metadata section - spans both columns
        ax_meta = fig.add_subplot(gs[0, :])
        ax_meta.axis('off')
        metadata_text = (
            f"ANNOTATION SHEET\n"
            f"Dyad: {patient_id} / {therapist_id} | Interview: {interview_type.capitalize()} | "
            f"Turn: {turn_index} | Speaker: {turn['speaker_id'].capitalize()} | "
            f"Duration: {turn['start_ms'] / 1000:.2f}-{turn['end_ms'] / 1000:.2f}s ({(turn['end_ms'] - turn['start_ms']) / 1000:.2f}s)"
        )
        ax_meta.text(0.5, 0.5, metadata_text, ha='center', va='center', 
                     fontsize=12, fontweight='bold', family='monospace',
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        # Heatmap section - LEFT column
        ax_heatmap = fig.add_subplot(gs[1, 0])
        ax_heatmap.axis('off')
        
        # Generate heatmap as separate figure
        heatmap_fig = generate_heatmap_figure(therapist_csv, patient_csv, turn, au_names, fontsize=65)
        
        # Convert heatmap figure to image and display in main figure
        import io
        from PIL import Image
        
        buf = io.BytesIO()
        heatmap_fig.savefig(buf, format='png', bbox_inches='tight', dpi=300, facecolor='white')
        buf.seek(0)
        img = Image.open(buf)
        ax_heatmap.imshow(img, interpolation='none')  # No interpolation for crisp rendering
        buf.close()
        plt.close(heatmap_fig)  # Close the temporary figure
        
        # Text section - RIGHT column: Ground truth and prediction
        ax_text = fig.add_subplot(gs[1, 1])
        ax_text.axis('off')
        
        # Use the summary from the transcript turn as ground truth
        ground_truth = turn.get('summary', 'N/A')
        model_output = prediction.get('generated', 'N/A')
        
        # Wrap text for readability (shorter width for side column)
        gt_wrapped = wrap_text(ground_truth, width=60)
        pred_wrapped = wrap_text(model_output, width=60)
        
        text_content = (
            f"GROUND TRUTH:\n"
            f"{gt_wrapped}\n\n"
            f"{'='*60}\n\n"
            f"MODEL PREDICTION:\n"
            f"{pred_wrapped}\n\n"
            f"{'='*60}\n\n"
            f"ANNOTATION:\n"
            f"Rate accuracy, completeness,\n"
            f"and relevance of prediction."
        )
        
        ax_text.text(0.05, 0.95, text_content, ha='left', va='top', 
                     fontsize=8, family='monospace', wrap=True,
                     bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.2))
        
        # Save this page to PDF with high quality
        pdf.savefig(fig, bbox_inches='tight', dpi=300)
        plt.close(fig)
        
    except Exception as e:
        print(f"❌ Error creating annotation for turn {turn_index}: {e}")
        import traceback
        traceback.print_exc()


def process_dyad_interview(
    patient_id: str,
    therapist_id: str,
    interview_type: str,
    interview: Dict,
    predictions: Dict,
    output_pdf: Path,
    au_names: List[str]
) -> int:
    """Process a single dyad/interview and create annotation PDF.
    
    Returns:
        Number of annotations created
    """
    print(f"\nProcessing {patient_id}/{therapist_id} - {interview_type}...")
    
    # Get interview data
    if interview_type not in interview['types']:
        raise ValueError(f"Interview type '{interview_type}' not found for {patient_id}/{therapist_id}")
    
    type_data = interview['types'][interview_type]
    therapist_csv = Path(type_data['therapist_openface'])
    patient_csv = Path(type_data['patient_openface'])
    transcript_json = Path(type_data['transcript'])
    
    # Validate paths
    if not therapist_csv.exists():
        raise FileNotFoundError(f"Therapist CSV not found: {therapist_csv}")
    if not patient_csv.exists():
        raise FileNotFoundError(f"Patient CSV not found: {patient_csv}")
    if not transcript_json.exists():
        raise FileNotFoundError(f"Transcript JSON not found: {transcript_json}")
    
    # Load speech turns
    turns = load_speech_turns(transcript_json)
    
    # Filter to only turns that exist in predictions (test set)
    test_turns = []
    for turn in turns:
        key = (patient_id, therapist_id, interview_type, turn['turn_index'])
        if key in predictions:
            test_turns.append(turn)
    
    if not test_turns:
        raise ValueError(
            f"No test set predictions found for {patient_id}/{therapist_id} {interview_type}!\n"
            f"This dyad/interview is not in the test set. Please only request test set cases."
        )
    
    print(f"  Found {len(test_turns)} turns in test set (out of {len(turns)} total)")
    
    # Create PDF
    annotation_count = 0
    
    with PdfPages(output_pdf) as pdf:
        for turn in tqdm(test_turns, desc=f"Creating PDFs"):
            key = (patient_id, therapist_id, interview_type, turn['turn_index'])
            prediction = predictions[key]
            
            create_annotation_pdf(
                patient_id, therapist_id, interview_type,
                turn, prediction,
                therapist_csv, patient_csv,
                au_names, pdf
            )
            annotation_count += 1
    
    print(f"✅ Created {annotation_count} annotation sheets in {output_pdf}")
    return annotation_count


def main():
    parser = argparse.ArgumentParser(
        description="Generate PDF annotation files for human evaluation of model predictions"
    )
    parser.add_argument(
        "--data_model",
        type=Path,
        required=True,
        help="Path to data_model.yaml"
    )
    parser.add_argument(
        "--predictions_file",
        type=Path,
        required=True,
        help="Path to test_predictions.jsonl file"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output directory (will create 'annotations' subdirectory)"
    )
    parser.add_argument(
        "--dyad_interviews",
        nargs="+",
        required=True,
        help="List of dyad_interview pairs (e.g., A2ER_wunder B7UH_personal A0EA_bindung)"
    )
    parser.add_argument(
        "--au_columns",
        nargs="+",
        default=None,
        help="Specific AU columns to display (e.g., AU12_r AU06_r AU04_r AU15_r). If not provided, uses all 17 AUs"
    )
    
    args = parser.parse_args()
    
    # Define ALL 17 OpenFace Action Units
    ALL_AU_COLUMNS = [
        'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r',
        'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r',
        'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r'
    ]
    
    # Use specified AUs or default to all
    au_columns_to_use = args.au_columns if args.au_columns else ALL_AU_COLUMNS
    
    # Validate AU columns
    invalid_aus = [au for au in au_columns_to_use if au not in ALL_AU_COLUMNS]
    if invalid_aus:
        raise ValueError(f"Invalid AU columns: {invalid_aus}. Valid options: {ALL_AU_COLUMNS}")
    
    print("📋 Starting annotation PDF generation")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Data model: {args.data_model}")
    print(f"  Predictions: {args.predictions_file}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Dyad/interviews: {args.dyad_interviews}")
    print(f"  AU columns: {len(au_columns_to_use)} AUs - {au_columns_to_use}")
    print()
    
    # Validate inputs
    if not args.data_model.exists():
        raise FileNotFoundError(f"Data model not found: {args.data_model}")
    if not args.predictions_file.exists():
        raise FileNotFoundError(f"Predictions file not found: {args.predictions_file}")
    
    # Create output directory
    annotations_dir = args.output_dir / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {annotations_dir}")
    
    # Load data
    data_model = load_data_model(args.data_model)
    predictions = load_predictions(args.predictions_file)
    
    # Parse dyad/interview list
    dyad_interview_pairs = parse_dyad_interview_list(args.dyad_interviews)
    print(f"\n📝 Processing {len(dyad_interview_pairs)} dyad/interview pairs:")
    for dyad_id, interview_type in dyad_interview_pairs:
        print(f"  - {dyad_id}_{interview_type}")
    
    # Find matching interviews in data model
    interviews_dict = {
        (interview['patient']['patient_id'], interview['therapist']['therapist_id']): interview
        for interview in data_model['interviews']
    }
    
    # Process each requested dyad/interview
    total_annotations = 0
    
    for dyad_id, interview_type in dyad_interview_pairs:
        # Find the interview
        matching_interview = None
        patient_id = None
        therapist_id = None
        
        for (pid, tid), interview in interviews_dict.items():
            if pid == dyad_id or tid == dyad_id:
                matching_interview = interview
                patient_id = pid
                therapist_id = tid
                break
        
        if not matching_interview:
            raise ValueError(f"Dyad '{dyad_id}' not found in data model!")
        
        # Create output PDF
        output_pdf = annotations_dir / f"{patient_id}_{interview_type}_annotations.pdf"
        
        try:
            count = process_dyad_interview(
                patient_id, therapist_id, interview_type,
                matching_interview, predictions,
                output_pdf, au_columns_to_use
            )
            total_annotations += count
            
        except ValueError as e:
            print(f"\n❌ ERROR: {e}")
            print(f"   Skipping {patient_id}/{therapist_id} {interview_type}")
            continue
    
    print(f"\n{'='*80}")
    print(f"✅ Complete! Generated {total_annotations} annotation sheets across {len(dyad_interview_pairs)} PDFs")
    print(f"   Output location: {annotations_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
