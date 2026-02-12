"""
Generate time-series descriptions using the Qwen2.5-VL multimodal model.
This is the first out of two-steps to create comparable "rationales" as the OpenTSLM paper have.
In this step, we give a Qwen2.5-VL model a heatmap of action unit time series, and simply asks it to describe it.
Its akin to what the opentslm authors did with gpt-4o.

The next step is to combine these descriptions with the transcripts to form coherent rationales or descriptions,
as I like to call them (see combine_transcripts_with_time_series_descriptions.py for this)

This script:
1. Reads data_model.yaml to get interview information
2. Extracts AU time series from OpenFace CSVs for specific speech turn windows
3. Generates 4 subplots (2x2) with therapist and client AUs overlaid
4. Feeds plots into Qwen2.5-VL-72B-Instruct-AWQ for description generation
"""

import sys
import os
import torch
import numpy as np
import pandas as pd
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import AutoModelForVision2Seq, AutoProcessor
try:
    from transformers import AwqConfig
except Exception:
    AwqConfig = None
import yaml
from PIL import Image

# Add OpenTSLM src to path
opentslm_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_dir = os.path.join(opentslm_dir, "src")
sys.path.insert(0, opentslm_dir)
sys.path.insert(0, src_dir)


def setup_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    return device


def _build_max_memory(max_memory_gb: int) -> Dict[str, str]:
    if not torch.cuda.is_available():
        return {"cpu": "64GiB"}
    num_gpus = torch.cuda.device_count()
    max_memory = {i: f"{max_memory_gb}GiB" for i in range(num_gpus)}
    max_memory["cpu"] = "64GiB"
    return max_memory


def load_qwen_vl_model(model_id: str, max_memory_gb: int, attn_implementation: str):
    print(f"\nLoading Qwen2.5-VL model: {model_id}")

    quantization_config = None
    if AwqConfig is not None:
        quantization_config = AwqConfig(bits=4, group_size=128, zero_point=True, version="GEMM")

    max_memory = _build_max_memory(max_memory_gb)

    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        max_memory=max_memory,
        quantization_config=quantization_config,
        attn_implementation=attn_implementation,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    print("Model loaded")
    return model, processor


def load_data_model(yaml_path: Path) -> Dict:
    """Load the data_model.yaml file."""
    print(f"Loading data model from {yaml_path}...")
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    print(f"Loaded {len(data['interviews'])} interviews")
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


def generate_plot_for_turn(
    therapist_csv: Path,
    patient_csv: Path,
    turn: Dict,
    au_names: List[str],
    output_path: Path,
    num_bins: int = 8
) -> bool:
    """Generate heatmap visualization with binned AU activations for therapist and client.
    
    Args:
        therapist_csv: Path to therapist OpenFace CSV
        patient_csv: Path to patient OpenFace CSV
        turn: Speech turn dict with start_ms, end_ms, speaker_id
        au_names: List of AU names to plot (can be 4 or 17)
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
            print(f"Warning: No data found for turn {turn_index} ({start_ms}-{end_ms}ms)")
            return False
        
        # Create binned heatmaps for each AU
        therapist_heatmap = np.zeros((len(au_names), num_bins))
        client_heatmap = np.zeros((len(au_names), num_bins))
        
        for i, au_name in enumerate(au_names):
            therapist_heatmap[i, :] = bin_time_series(therapist_data, au_name, num_bins)
            client_heatmap[i, :] = bin_time_series(patient_data, au_name, num_bins)
        
        # Create figure with two side-by-side heatmaps
        # Dynamically adjust figure height based on number of AUs
        fig_height = max(6, len(au_names) * 0.5)  # At least 6 inches, scale with AU count
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, fig_height))
        
        # Find global min/max for consistent color scaling
        vmin = min(therapist_heatmap.min(), client_heatmap.min())
        vmax = max(therapist_heatmap.max(), client_heatmap.max())
        
        # Therapist heatmap (left)
        im1 = ax1.imshow(therapist_heatmap, aspect='auto', cmap='Blues', 
                         interpolation='nearest', vmin=vmin, vmax=vmax)
        ax1.set_title('THERAPIST AU Activation', fontsize=14, fontweight='bold', pad=15)
        ax1.set_ylabel('Action Unit', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Time Progression (Start → End)', fontsize=12, fontweight='bold')
        ax1.set_yticks(range(len(au_names)))
        ax1.set_yticklabels(au_names, fontsize=11)
        ax1.set_xticks(range(num_bins))
        ax1.set_xticklabels(['Start', 'Early', 'Early-Mid', 'Mid', 'Late-Mid', 'Late', 'Very Late', 'End'][:num_bins], 
                            fontsize=9, rotation=45, ha='right')
        
        # Add colorbar for therapist
        cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.set_label('Activation Level', fontsize=11, fontweight='bold')
        
        # Add value annotations on therapist heatmap (smaller font for many AUs)
        font_size = 8 if len(au_names) <= 4 else 6
        for i in range(len(au_names)):
            for j in range(num_bins):
                text = ax1.text(j, i, f'{therapist_heatmap[i, j]:.2f}',
                               ha="center", va="center", color="black", fontsize=font_size)
        
        # Client heatmap (right)
        im2 = ax2.imshow(client_heatmap, aspect='auto', cmap='Oranges', 
                         interpolation='nearest', vmin=vmin, vmax=vmax)
        ax2.set_title('CLIENT AU Activation', fontsize=14, fontweight='bold', pad=15)
        ax2.set_ylabel('Action Unit', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Time Progression (Start → End)', fontsize=12, fontweight='bold')
        ax2.set_yticks(range(len(au_names)))
        ax2.set_yticklabels(au_names, fontsize=11)
        ax2.set_xticks(range(num_bins))
        ax2.set_xticklabels(['Start', 'Early', 'Early-Mid', 'Mid', 'Mid-Late', 'Late', 'Very Late', 'End'][:num_bins], 
                            fontsize=9, rotation=45, ha='right')
        
        # Add colorbar for client
        cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        cbar2.set_label('Activation Level', fontsize=11, fontweight='bold')
        
        # Add value annotations on client heatmap (smaller font for many AUs)
        for i in range(len(au_names)):
            for j in range(num_bins):
                text = ax2.text(j, i, f'{client_heatmap[i, j]:.2f}',
                               ha="center", va="center", color="black", fontsize=font_size)
        
        # Overall title
        plt.suptitle(f"Turn {turn_index}: {speaker_id.capitalize()} speaking ({start_ms:.0f}-{end_ms:.0f}ms)\n" + 
                     f"Heatmap shows mean AU activation across {num_bins} temporal phases", 
                     fontsize=15, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return True
        
    except Exception as e:
        print(f"Error generating plot for turn {turn.get('turn_index', '?')}: {e}")
        return False


def generate_description_with_model(model, processor, image_path: Path, turn: Dict, au_names: List[str]) -> str:
    """Generate time-series description using Qwen2.5-VL."""
    
    pre_prompt = """Describe these AU heatmaps (left=therapist blue, right=client orange). 
Each row is one AU across 8 time bins. Write one compact sentence per AU comparing patterns.
ONLY consider the AUs which shows either 1) high variability within patient or client or 2) strong difference between therapist and client
Consider a maximum of 4 action units. 
Format: "AU##: therapist [pattern], client [pattern], [key difference]."
No markdown, bullets, or headers. ONLY output your description."""
    
    au_list = ", ".join(au_names)
    context = f"""Turn {turn['turn_index']} ({turn['speaker_id']}), {turn['start_ms']:.0f}-{turn['end_ms']:.0f}ms
AUs: {au_list}
Description:"""
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"{pre_prompt}\n\n{context}"},
            ],
        }
    ]
    
    try:
        image = Image.open(image_path).convert("RGB")
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[prompt],
            images=[image],
            return_tensors="pt",
        )

        input_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(input_device)

        # Fixed token limit: model instructed to focus only on most volatile AUs
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=125,
            do_sample=False,
        )
        generated_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()

        if prompt in generated_text:
            description = generated_text.replace(prompt, "", 1).strip()
        else:
            description = generated_text
        
        # Optimized: Minimal post-processing - remove only essential unwanted elements
        # Remove common prefixes
        if description.lower().startswith(("here's", "here is", "the patterns")):
            if ':' in description[:50]:
                description = description.split(':', 1)[1].lstrip()
        
        # Remove markdown and normalize whitespace
        import re
        description = description.replace('**', '').replace('*', '')
        description = description.replace('\n', ' ')
        description = re.sub(r'\s+', ' ', description).strip()
        
        return description
    except Exception as e:
        print(f"Error generating description: {e}")
        return f"Error: {str(e)}"


def process_interview(
    interview: Dict,
    interview_type: str,
    model,
    processor,
    output_dir: Path,
    au_names: List[str],
    max_turns: int = None
) -> List[Dict[str, Any]]:
    """Process a single interview type for description generation.
    
    Args:
        interview: Interview dict from data_model.yaml
        interview_type: One of 'bindung', 'personal', 'wunder'
        model: Qwen2.5-VL model
        processor: Qwen2.5-VL processor
        output_dir: Directory to save plots and results
        au_names: List of AU names to analyze
        max_turns: Maximum number of turns to process (None = all)
    
    Returns:
        List of results dicts
    """
    if interview_type not in interview['types']:
        print(f"Warning: Interview type '{interview_type}' not found, skipping")
        return []
    
    type_data = interview['types'][interview_type]
    therapist_csv = Path(type_data['therapist_openface'])
    patient_csv = Path(type_data['patient_openface'])
    transcript_json = Path(type_data['transcript'])
    
    # Validate paths
    if not therapist_csv.exists():
        print(f"Error: Therapist CSV not found: {therapist_csv}")
        return []
    if not patient_csv.exists():
        print(f"Error: Patient CSV not found: {patient_csv}")
        return []
    if not transcript_json.exists():
        print(f"Error: Transcript JSON not found: {transcript_json}")
        return []
    
    # Load speech turns
    turns = load_speech_turns(transcript_json)
    
    # Limit turns if requested
    if max_turns:
        turns = turns[:max_turns]
    
    results = []
    therapist_id = interview['therapist']['therapist_id']
    patient_id = interview['patient']['patient_id']
    
    # Create .temp directory for plots
    temp_dir = output_dir / ".temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nProcessing {patient_id}/{therapist_id} - {interview_type}: {len(turns)} turns")
    
    for turn in tqdm(turns, desc=f"{patient_id} {interview_type}"):
        turn_index = turn['turn_index']
        
        # Generate plot in .temp directory
        plot_path = temp_dir / f"{patient_id}_{interview_type}_turn{turn_index:03d}.jpg"
        success = generate_plot_for_turn(therapist_csv, patient_csv, turn, au_names, plot_path)
        
        if not success:
            continue
        
        # Generate description
        description = generate_description_with_model(model, processor, plot_path, turn, au_names)
        
        # Collect result
        result = {
            "patient_id": patient_id,
            "therapist_id": therapist_id,
            "interview_type": interview_type,
            "turn_index": turn_index,
            "speaker_id": turn['speaker_id'],
            "start_ms": turn['start_ms'],
            "end_ms": turn['end_ms'],
            "duration_ms": turn['end_ms'] - turn['start_ms'],
            "generated_descriptions": description,  
            "plot_path": str(plot_path)
        }
        results.append(result)
    
    return results


def save_results(results: List[Dict[str, Any]], output_path: Path):
    """Save the generated time-series descriptions to JSON."""
    print(f"\nSaving {len(results)} results to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print("Results saved")
    
    if results:
        avg_duration = np.mean([r['duration_ms'] for r in results])
        avg_description_len = np.mean([len(r['generated_descriptions']) for r in results])
        print("\nSummary:")
        print(f"  Total turns processed: {len(results)}")
        print(f"  Average turn duration: {avg_duration:.0f}ms")
        print(f"  Average description length: {avg_description_len:.0f} characters")


def main():
    parser = argparse.ArgumentParser(
        description="Generate AU time-series descriptions from data_model.yaml using Qwen2.5-VL"
    )
    parser.add_argument(
        "--data_model",
        type=Path,
        default=Path("/home/mlut/synchrony/data_model.yaml"),
        help="Path to data_model.yaml"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/home/data_shares/genface/data/MentalHealth/msb/results/"),
        help="Output directory for plots and results"
    )
    parser.add_argument(
        "--interview_types",
        nargs="+",
        default=["wunder", "personal", "bindung"],
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
    parser.add_argument(
        "--au_columns",
        nargs="+",
        default=['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 
                 'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 
                 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r'],
        help="AU columns to analyze (default: all 17 AUs)"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="Qwen/Qwen2.5-VL-72B-Instruct-AWQ",
        help="Qwen2.5-VL model ID (default: Qwen/Qwen2.5-VL-72B-Instruct-AWQ)"
    )
    parser.add_argument(
        "--max_memory_gb",
        type=int,
        default=75,
        help="Per-GPU max memory for device_map auto (GiB, default: 75)"
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        help="Attention implementation (default: flash_attention_2)"
    )
    
    args = parser.parse_args()
    
    print("Starting AU time-series description generation with Qwen2.5-VL")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Data model: {args.data_model}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Interview types: {args.interview_types}")
    print(f"  AU columns: {args.au_columns}")
    print(f"  Model ID: {args.model_id}")
    print(f"  Max memory per GPU (GiB): {args.max_memory_gb}")
    print(f"  Attention impl: {args.attn_implementation}")
    print()
    
    # Setup
    device = setup_device()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data model
    data_model = load_data_model(args.data_model)
    interviews = data_model['interviews']
    
    if args.max_interviews:
        interviews = interviews[:args.max_interviews]
    
    # Initialize Qwen2.5-VL model
    model, processor = load_qwen_vl_model(
        model_id=args.model_id,
        max_memory_gb=args.max_memory_gb,
        attn_implementation=args.attn_implementation,
    )
    
    # Process all interviews
    all_results = []
    saved_files = []  # Track saved JSON files
    
    for interview_idx, interview in enumerate(interviews):
        patient_id = interview['patient']['patient_id']
        therapist_id = interview['therapist']['therapist_id']
        
        print(f"\n{'='*80}")
        print(f"Interview {interview_idx + 1}/{len(interviews)}: {patient_id}/{therapist_id}")
        print(f"{'='*80}")
        
        for interview_type in args.interview_types:
            results = process_interview(
                interview,
                interview_type,
                model,
                processor,
                args.output_dir,
                args.au_columns,
                max_turns=args.max_turns_per_interview
            )
            all_results.extend(results)
            
            # Save immediately after each interview session (3 per dyad)
            if results:
                output_json = args.output_dir / f"{patient_id}_{interview_type}_descriptions.json"
                save_results(results, output_json)
                saved_files.append(str(output_json))
    
    print(f"\nComplete! Generated time-series descriptions for {len(all_results)} speech turns across {len(saved_files)} interview sessions")


if __name__ == "__main__":
    main()
