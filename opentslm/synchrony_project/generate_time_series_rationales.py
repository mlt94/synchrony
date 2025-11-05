"""
Refactored script to generate time-series rationales using the LLaVA multimodal pipeline.
This script:
1. Reads data_model.yaml to get interview information
2. Extracts AU time series from OpenFace CSVs for specific speech turn windows
3. Generates 6 subplots with therapist and client AUs overlaid
4. Feeds plots into LLaVA 7B for rationale generation (optimized for RTX 4070)
"""

import sys
import os
import torch
import numpy as np
import pandas as pd
import random
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image
import yaml

# Add OpenTSLM src to path
opentslm_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_dir = os.path.join(opentslm_dir, "src")
sys.path.insert(0, opentslm_dir)
sys.path.insert(0, src_dir)


def setup_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    return device


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


def generate_plot_for_turn(
    therapist_csv: Path,
    patient_csv: Path,
    turn: Dict,
    au_names: List[str],
    output_path: Path
) -> bool:
    """Generate a 2x3 subplot with therapist and patient AUs overlaid.
    
    Args:
        therapist_csv: Path to therapist OpenFace CSV
        patient_csv: Path to patient OpenFace CSV
        turn: Speech turn dict with start_ms, end_ms, speaker_id
        au_names: List of 6 AU names to plot
        output_path: Where to save the plot
    
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
        
        # Create 2x3 subplot
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        axes = axes.flatten()
        
        colors_therapist = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#BC4B51']
        colors_patient = ['#5BB9DB', '#C26BA2', '#FFA531', '#E76D4D', '#9AC97E', '#DC7B81']
        
        for i, au_name in enumerate(au_names):
            ax = axes[i]
            
            # Plot therapist AU - solid line with thicker linewidth
            ax.plot(therapist_data['timestamp_ms'], therapist_data[au_name], 
                   linewidth=3, color=colors_therapist[i], label='Therapist', alpha=0.9)
            
            # Plot patient AU - distinct marker style with thinner line
            ax.plot(patient_data['timestamp_ms'], patient_data[au_name], 
                   linewidth=2, color=colors_patient[i], label='Client', 
                   alpha=0.7, linestyle=':', marker='o', markersize=3, markevery=5)
            
            ax.set_title(f"{au_name} - Speaker: {speaker_id.capitalize()}", 
                        fontsize=11, pad=10, fontweight='bold')
            ax.set_xlabel("Time (ms)", fontsize=9)
            ax.set_ylabel("Activation", fontsize=9)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(loc='upper right', fontsize=8)
            ax.tick_params(labelsize=8)
        
        plt.suptitle(f"Turn {turn_index}: {speaker_id.capitalize()} speaking ({start_ms:.0f}-{end_ms:.0f}ms)", 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Error generating plot for turn {turn.get('turn_index', '?')}: {e}")
        return False


def generate_rationale_with_llava(processor, model, image_path: Path, turn: Dict, au_names: List[str]) -> str:
    """Generate rationale using the LLaVA model."""
    
    pre_prompt = """You are shown facial Action Unit (AU) activation plots for both therapist and client during a psychotherapy speech turn.

Describe ONLY the activation patterns for each individual AU shown in the plots. For EACH AU, state very briefly:
- The activation level (low, moderate, high)
- How the activation pattern differs between therapist and client

Go through each AU systematically. DO NOT include introductory phrases like "Here's a description" or concluding statements about overall synchrony. ONLY describe what you observe in each AU plot."""
    
    au_list = ", ".join(au_names)
    context = f"""Speech Turn {turn['turn_index']}: {turn['speaker_id'].capitalize()} speaking
Time window: {turn['start_ms']:.0f}-{turn['end_ms']:.0f}ms
Action Units shown: {au_list}

Description: """
    
    prompt = f"[INST] <image>\n{pre_prompt}\n\n{context} [/INST]"
    
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        
        # Correct processor call: text and images as keyword arguments
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
        
        # Generate with optimized settings for RTX 4070
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                temperature=None,  # Greedy decoding when do_sample=False
                pad_token_id=processor.tokenizer.pad_token_id
            )
        
        # Decode and extract response
        generated_text = processor.decode(output[0], skip_special_tokens=True)
        
        # Extract only the assistant's response (after [/INST])
        if "[/INST]" in generated_text:
            rationale = generated_text.split("[/INST]")[-1].strip()
        else:
            rationale = generated_text.strip()
        
        # Remove common unwanted prefixes
        unwanted_prefixes = [
            "Here's a concise description of the patterns in the provided plots",
            "Here's a description of the patterns",
            "Here's a concise description",
            "Here is a description",
            "Here is a concise description",
            "Here's what I observe",
        ]
        for prefix in unwanted_prefixes:
            if rationale.lower().startswith(prefix.lower()):
                # Remove prefix and any following punctuation/whitespace
                rationale = rationale[len(prefix):].lstrip(':.,; ')
                break
        
        # Remove common unwanted conclusions (last sentence patterns)
        unwanted_conclusions = [
            "Overall, the patterns diverge",
            "Overall, the patterns converge",
            "Overall,",
            "In summary,",
            "This suggests",
            "This indicates",
        ]
        sentences = rationale.split('.')
        if len(sentences) > 1:
            last_sentence = sentences[-1].strip()
            for conclusion in unwanted_conclusions:
                if last_sentence.lower().startswith(conclusion.lower()):
                    rationale = '.'.join(sentences[:-1]) + '.'
                    break
        
        return rationale.strip()
    except Exception as e:
        print(f"❌ Error generating rationale: {e}")
        return f"Error: {str(e)}"


def process_interview(
    interview: Dict,
    interview_type: str,
    processor,
    model,
    output_dir: Path,
    au_names: List[str],
    max_turns: int = None
) -> List[Dict[str, Any]]:
    """Process a single interview type for rationale generation.
    
    Args:
        interview: Interview dict from data_model.yaml
        interview_type: One of 'bindung', 'personal', 'wunder'
        processor: LLaVA processor
        model: LLaVA model
        output_dir: Directory to save plots and results
        au_names: List of AU names to analyze
        max_turns: Maximum number of turns to process (None = all)
    
    Returns:
        List of results dicts
    """
    if interview_type not in interview['types']:
        print(f"⚠️ Interview type '{interview_type}' not found, skipping")
        return []
    
    type_data = interview['types'][interview_type]
    therapist_csv = Path(type_data['therapist_openface'])
    patient_csv = Path(type_data['patient_openface'])
    transcript_json = Path(type_data['transcript'])
    
    # Validate paths
    if not therapist_csv.exists():
        print(f"❌ Therapist CSV not found: {therapist_csv}")
        return []
    if not patient_csv.exists():
        print(f"❌ Patient CSV not found: {patient_csv}")
        return []
    if not transcript_json.exists():
        print(f"❌ Transcript JSON not found: {transcript_json}")
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
        
        # Generate rationale
        rationale = generate_rationale_with_llava(processor, model, plot_path, turn, au_names)
        
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
            "generated_rationale": rationale,
            "plot_path": str(plot_path)
        }
        results.append(result)
    
    return results


def save_results(results: List[Dict[str, Any]], output_path: Path):
    """Save the generated rationales to JSON."""
    print(f"\n💾 Saving {len(results)} results to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ Results saved")
    
    if results:
        avg_duration = np.mean([r['duration_ms'] for r in results])
        avg_rationale_len = np.mean([len(r['generated_rationale']) for r in results])
        print(f"\n📊 Summary:")
        print(f"  Total turns processed: {len(results)}")
        print(f"  Average turn duration: {avg_duration:.0f}ms")
        print(f"  Average rationale length: {avg_rationale_len:.0f} characters")


def main():
    parser = argparse.ArgumentParser(
        description="Generate AU rationales from data_model.yaml using LLaVA-NeXT 7B (optimized for RTX 4070)"
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
        default=['AU04_r', 'AU15_r', 'AU06_r', 'AU12_r', 'AU01_r', 'AU07_r'],
        help="AU columns to analyze"
    )
    
    args = parser.parse_args()
    
    print("🚀 Starting AU rationale generation with LLaVA-NeXT 7B (RTX 4070 optimized)")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Data model: {args.data_model}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Interview types: {args.interview_types}")
    print(f"  AU columns: {args.au_columns}")
    print()
    
    # Setup
    device = setup_device()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data model
    data_model = load_data_model(args.data_model)
    interviews = data_model['interviews']
    
    if args.max_interviews:
        interviews = interviews[:args.max_interviews]
    
    # Initialize LLaVA model (optimized for RTX 4070)
    print(f"\n🔧 Loading LLaVA-NeXT 7B model...")
    model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
    
    processor = LlavaNextProcessor.from_pretrained(model_id)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True
    )
    
    if device == "cpu":
        model = model.to(device)
    
    print(f"✅ Model loaded on {device}")
    if device == "cuda":
        print(f"   GPU Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    # Process all interviews
    all_results = []
    results_by_dyad_type = {}  # Store results grouped by (patient_id, interview_type)
    
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
                processor,
                model,
                args.output_dir,
                args.au_columns,
                max_turns=args.max_turns_per_interview
            )
            all_results.extend(results)
            
            # Group by (patient_id, interview_type)
            key = (patient_id, interview_type)
            if key not in results_by_dyad_type:
                results_by_dyad_type[key] = []
            results_by_dyad_type[key].extend(results)
    
    # Save results per dyad and interview type
    for (dyad_id, interview_type), dyad_results in results_by_dyad_type.items():
        output_json = args.output_dir / f"{dyad_id}_{interview_type}_rationales.json"
        save_results(dyad_results, output_json)
    
    print(f"\n✅ Complete! Generated rationales for {len(all_results)} speech turns across {len(results_by_dyad_type)} dyad-interview combinations")


if __name__ == "__main__":
    main()
