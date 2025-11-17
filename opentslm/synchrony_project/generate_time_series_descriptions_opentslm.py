"""
Generate time-series descriptions using trained OpenTSLM model.
This is the first step to create "rationales" for psychotherapy AU time series.

This script:
1. Reads data_model.yaml to get interview information
2. Extracts AU time series from OpenFace CSVs for specific speech turn windows
3. Formats the data in the same way as psychotherapy_loader (17 AUs × 2 people)
4. Feeds into trained OpenTSLM Gemma model for description generation
5. Outputs JSON files compatible with combine_transcripts_with_time_series_descriptions.py

NO PLOTS/HEATMAPS: Model processes raw time series directly like during training.
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
import yaml

# Add OpenTSLM src to path
opentslm_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_dir = os.path.join(opentslm_dir, "src")
sys.path.insert(0, opentslm_dir)
sys.path.insert(0, src_dir)

from model.llm.OpenTSLMFlamingo import OpenTSLMFlamingo
from prompt.text_prompt import TextPrompt
from prompt.text_time_series_prompt import TextTimeSeriesPrompt
from prompt.full_prompt import FullPrompt


# All 17 AUs available in OpenFace output
ALL_AU_COLUMNS = [
    'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r',
    'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r',
    'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r'
]

# TEST: Use only 1 AU for maximum simplicity
TEST_AU_COLUMNS = ['AU12_r']  # Lip corner puller (smile)


def setup_device():
    """Setup compute device."""
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")
    return device


def load_data_model(yaml_path: Path) -> Dict:
    """Load the data_model.yaml file."""
    print(f"Loading data model from {yaml_path}...")
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    print(f"✅ Loaded {len(data['interviews'])} interviews")
    return data


def load_speech_turns(json_path: Path) -> List[Dict]:
    """Load speech turns from transcript JSON."""
    with open(json_path, 'r', encoding='utf-8') as f:
        turns = json.load(f)
    return turns


def extract_au_window(csv_path: Path, start_ms: float, end_ms: float, au_columns: List[str]) -> Dict[str, np.ndarray]:
    """Extract AU data from OpenFace CSV for a specific time window.
    
    Args:
        csv_path: Path to OpenFace CSV
        start_ms: Start time in milliseconds
        end_ms: End time in milliseconds
        au_columns: List of AU column names to extract
    
    Returns:
        Dict mapping AU column name to numpy array of values, or None if insufficient data
    """
    try:
        # Read CSV with whitespace handling
        df = pd.read_csv(csv_path, skipinitialspace=True)
        
        # Normalize column names
        df.columns = df.columns.str.strip()
        
        # Find timestamp column
        timestamp_col = None
        for col in df.columns:
            if col.lower() == 'timestamp':
                timestamp_col = col
                break
        
        if timestamp_col is None:
            print(f"⚠️ No timestamp column found in {csv_path}")
            return None
        
        # Convert timestamp from seconds to milliseconds
        df['timestamp_ms'] = pd.to_numeric(df[timestamp_col], errors='coerce') * 1000
        df = df.dropna(subset=['timestamp_ms'])
        
        if df.empty:
            return None
        
        # Filter to time window
        mask = (df['timestamp_ms'] >= start_ms) & (df['timestamp_ms'] <= end_ms)
        window_df = df.loc[mask]
        
        # Need at least 10 frames
        if len(window_df) < 10:
            return None
        
        # Extract AU vectors
        au_vectors = {}
        for au_col in au_columns:
            if au_col in window_df.columns:
                au_signal = pd.to_numeric(window_df[au_col], errors='coerce').to_numpy()
                # Remove NaN values
                au_signal = au_signal[~np.isnan(au_signal)]
                if len(au_signal) > 0:
                    au_vectors[au_col] = au_signal.astype(np.float32)
        
        return au_vectors if au_vectors else None
        
    except Exception as e:
        print(f"❌ Error extracting AU window from {csv_path}: {e}")
        return None


def normalize_time_series(series: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Normalize time series to zero mean and unit std.
    
    Returns:
        (normalized_series, mean, std)
    """
    mean = float(series.mean())
    std = float(series.std())
    if std > 1e-8:  # Use small epsilon to avoid division by near-zero
        normalized = (series - mean) / std
    else:
        normalized = series - mean
    return normalized, mean, std


def truncate_time_series(series: np.ndarray, max_length: int = 10400) -> np.ndarray:
    """Truncate time series to maximum supported length."""
    if len(series) > max_length:
        return series[:max_length]
    return series


def generate_description_with_opentslm(
    model: OpenTSLMFlamingo,
    therapist_au_vectors: Dict[str, np.ndarray],
    patient_au_vectors: Dict[str, np.ndarray],
    au_columns: List[str],
    turn: Dict
) -> str:
    """Generate time-series description using OpenTSLM model.
    
    Args:
        model: Trained OpenTSLM model
        therapist_au_vectors: Dict of AU name -> time series for therapist
        patient_au_vectors: Dict of AU name -> time series for patient
        au_columns: List of AU column names (in order)
        turn: Turn metadata dict
    
    Returns:
        Generated description text
    """
    try:
        # Build the prompt matching the LLaVA format for consistency
        pre_prompt_text = """You are an expert in analyzing facial Action Units (AUs) from psychotherapy sessions. 
You will receive time series data for AU12 (smile/lip corner puller) from both therapist and patient during a speech turn.

Describe the AU12 activation patterns. Write one compact sentence commenting on therapist and patient patterns, including notable differences.
Format: "AU12: therapist [pattern], patient [pattern], [key difference]."
ONLY output your description. Make sure to comment on BOTH therapist and patient."""
        
        au_list = ", ".join([au.replace('_r', '') for au in au_columns])
        
        post_prompt_text = f"""Turn {turn['turn_index']} ({turn['speaker_id']}), {turn['start_ms']:.0f}-{turn['end_ms']:.0f}ms
AUs: {au_list}
Description:"""
        
        pre_prompt = TextPrompt(pre_prompt_text)
        post_prompt = TextPrompt(post_prompt_text)
        
        # Build time series prompts for each AU (therapist first, then patient)
        ts_prompts = []
        
        for au_col in au_columns:
            # Therapist AU
            if au_col in therapist_au_vectors:
                series = therapist_au_vectors[au_col]
                series = truncate_time_series(series)
                normalized, mean, std = normalize_time_series(series)
                
                ts_text = f"Therapist {au_col.replace('_r', '')} (mean={mean:.3f}, std={std:.3f})"
                ts_prompts.append(TextTimeSeriesPrompt(ts_text, normalized.tolist()))
            else:
                print(f"⚠️ Missing therapist data for {au_col} in turn {turn['turn_index']}")
            
            # Patient AU
            if au_col in patient_au_vectors:
                series = patient_au_vectors[au_col]
                series = truncate_time_series(series)
                normalized, mean, std = normalize_time_series(series)
                
                ts_text = f"Patient {au_col.replace('_r', '')} (mean={mean:.3f}, std={std:.3f})"
                ts_prompts.append(TextTimeSeriesPrompt(ts_text, normalized.tolist()))
            else:
                print(f"⚠️ Missing patient data for {au_col} in turn {turn['turn_index']}")
        
        # Build full prompt
        prompt = FullPrompt(pre_prompt, ts_prompts, post_prompt)
        
        # Debug: Print prompt structure for first turn
        if turn['turn_index'] == 0:
            print(f"\n[DEBUG] Prompt structure for turn 0:")
            print(f"  Pre-prompt length: {len(pre_prompt_text)} chars")
            print(f"  Number of time series: {len(ts_prompts)} (expected: {len(au_columns) * 2})")
            print(f"  Post-prompt: {post_prompt_text[:100]}...")
            for i, ts_prompt in enumerate(ts_prompts):
                print(f"  TS {i}: {ts_prompt.text[:80]}...")
        
        # Generate description with explicit max_new_tokens
        # Using 500 tokens which should be ~350-400 words
        description = model.eval_prompt(prompt, max_new_tokens=500)
        
        # Clean up the output
        description = description.strip()
        
        # Debug: print first generation to check output
        if turn['turn_index'] == 0:
            print(f"\n[DEBUG] Sample generation for turn 0:")
            print(f"  Generated text length: {len(description)} chars")
            print(f"  Full output: '{description}'")
            print(f"  First 500 chars: {description[:500]}")
        
        return description
        
    except Exception as e:
        print(f"❌ Error generating description: {e}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}"


def process_interview(
    interview: Dict,
    interview_type: str,
    model: OpenTSLMFlamingo,
    output_dir: Path,
    au_columns: List[str],
    max_turns: int = None
) -> List[Dict[str, Any]]:
    """Process a single interview type for description generation.
    
    Args:
        interview: Interview dict from data_model.yaml
        interview_type: One of 'bindung', 'personal', 'wunder'
        model: Trained OpenTSLM model
        output_dir: Directory to save results
        au_columns: List of AU column names to analyze (all 17)
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
    
    print(f"\nProcessing {patient_id}/{therapist_id} - {interview_type}: {len(turns)} turns")
    
    for turn in tqdm(turns, desc=f"{patient_id} {interview_type}"):
        turn_index = turn['turn_index']
        start_ms = turn['start_ms']
        end_ms = turn['end_ms']
        
        # Extract AU data for both therapist and patient
        therapist_au_vectors = extract_au_window(therapist_csv, start_ms, end_ms, au_columns)
        patient_au_vectors = extract_au_window(patient_csv, start_ms, end_ms, au_columns)
        
        if not therapist_au_vectors or not patient_au_vectors:
            print(f"⚠️ Insufficient data for turn {turn_index} ({start_ms:.0f}-{end_ms:.0f}ms)")
            continue
        
        # Generate description
        description = generate_description_with_opentslm(
            model,
            therapist_au_vectors,
            patient_au_vectors,
            au_columns,
            turn
        )
        
        # Collect result
        result = {
            "patient_id": patient_id,
            "therapist_id": therapist_id,
            "interview_type": interview_type,
            "turn_index": turn_index,
            "speaker_id": turn['speaker_id'],
            "start_ms": start_ms,
            "end_ms": end_ms,
            "duration_ms": end_ms - start_ms,
            "generated_descriptions": description
        }
        results.append(result)
    
    return results


def save_results(results: List[Dict[str, Any]], output_path: Path):
    """Save the generated time-series descriptions to JSON.
    
    Args:
        results: List of result dicts to save
        output_path: Path to output JSON file
    """
    print(f"\n💾 Saving {len(results)} results to {output_path}...")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Results saved (total: {len(results)} turns)")
    
    if results:
        avg_duration = np.mean([r['duration_ms'] for r in results])
        avg_description_len = np.mean([len(r['generated_descriptions']) for r in results])
        print(f"📊 Summary:")
        print(f"  Total turns: {len(results)}")
        print(f"  Avg duration: {avg_duration:.0f}ms")
        print(f"  Avg description length: {avg_description_len:.0f} chars")


def main():
    parser = argparse.ArgumentParser(
        description="Generate AU time-series descriptions using trained OpenTSLM model"
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
        help="Output directory for results"
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        default=Path("results/gemma_3_270m/OpenTSLMFlamingo/stage2_captioning/checkpoints/best_model.pt"),
        help="Path to trained OpenTSLM model checkpoint"
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
        "--use_all_aus",
        action="store_true",
        help="Use all 17 AUs (default: use only 4 AUs for testing)"
    )
    
    args = parser.parse_args()
    
    # Decide which AUs to use
    au_columns = ALL_AU_COLUMNS if args.use_all_aus else TEST_AU_COLUMNS
    
    print("🚀 Starting AU time-series description generation with OpenTSLM")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Data model: {args.data_model}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Model path: {args.model_path}")
    print(f"  Interview types: {args.interview_types}")
    print(f"  AU columns: {len(au_columns)} AUs ({au_columns})")
    print(f"  Total time series per turn: {len(au_columns) * 2} (AUs × 2 people)")
    print()
    
    # Setup
    device = setup_device()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data model
    data_model = load_data_model(args.data_model)
    interviews = data_model['interviews']
    
    if args.max_interviews:
        interviews = interviews[:args.max_interviews]
    
    print(f"\n🔧 Loading trained OpenTSLM model with Llama-3.2-1B backbone...")
    model = OpenTSLMFlamingo(
        device=device,
        llm_id="meta-llama/Llama-3.2-1B"  # Must match the backbone used during training
    )
    
    if not args.model_path.exists():
        print(f"❌ Model checkpoint not found: {args.model_path}")
        print(f"   Please train the model first or provide correct path")
        return
    
    model.load_from_file(str(args.model_path))
    print("✅ Model loaded successfully")
    print(f"   Using {len(au_columns)} AUs × 2 people = {len(au_columns) * 2} time series per turn")
    
    # Process all interviews
    all_results = []
    
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
                args.output_dir,
                au_columns,  # Use the selected AU columns
                max_turns=args.max_turns_per_interview
            )
            
            # Save immediately after processing this interview type
            if results:
                output_json = args.output_dir / f"{patient_id}_{interview_type}_descriptions.json"
                save_results(results, output_json)
                all_results.extend(results)
            else:
                print(f"⚠️ No results generated for {patient_id} {interview_type}")
    
    print(f"\n{'='*80}")
    print(f"✅ Complete! Generated time-series descriptions for {len(all_results)} speech turns")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
