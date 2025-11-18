"""
Test simple generation with OpenTSLMFlamingo on a single AU time series.

This script loads a trained OpenTSLM model and tests it on:
- Dyad: A3EY
- Interview: Bindung
- Turn: 0
- AU: AU12_r (therapist only)

The prompt follows the exact OpenTSLM training format with mean and std normalization.
"""

import sys
import os
import torch
import numpy as np
import pandas as pd
import json
import argparse
from pathlib import Path
import yaml

# Add OpenTSLM src to path - must add src_dir FIRST
opentslm_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_dir = os.path.join(opentslm_dir, "src")
sys.path.insert(0, src_dir)
sys.path.insert(0, opentslm_dir)

# Import without src. prefix since src_dir is in sys.path
from model.llm.OpenTSLMFlamingo import OpenTSLMFlamingo
from prompt.text_prompt import TextPrompt
from prompt.text_time_series_prompt import TextTimeSeriesPrompt
from prompt.full_prompt import FullPrompt


def setup_device():
    """Setup compute device."""
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")
    return device


def load_data_model(yaml_path: Path) -> dict:
    """Load the data_model.yaml file."""
    print(f"Loading data model from {yaml_path}...")
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data


def load_speech_turns(json_path: Path) -> list:
    """Load speech turns from transcript JSON."""
    with open(json_path, 'r', encoding='utf-8') as f:
        turns = json.load(f)
    return turns


def extract_au_window(csv_path: Path, start_ms: float, end_ms: float, au_column: str) -> np.ndarray:
    """Extract AU data from OpenFace CSV for a specific time window.
    
    Args:
        csv_path: Path to OpenFace CSV
        start_ms: Start time in milliseconds
        end_ms: End time in milliseconds
        au_column: AU column name to extract
    
    Returns:
        Numpy array of AU values, or None if insufficient data
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
            print(f"⚠️ Only {len(window_df)} frames in window (need ≥10)")
            return None
        
        # Extract AU vector
        if au_column in window_df.columns:
            au_signal = pd.to_numeric(window_df[au_column], errors='coerce').to_numpy()
            # Remove NaN values
            au_signal = au_signal[~np.isnan(au_signal)]
            if len(au_signal) > 0:
                return au_signal.astype(np.float32)
        
        return None
        
    except Exception as e:
        print(f"❌ Error extracting AU window from {csv_path}: {e}")
        return None


def normalize_time_series(series: np.ndarray) -> tuple:
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


def main():
    parser = argparse.ArgumentParser(
        description="Test OpenTSLM model with single AU time series"
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        required=True,
        help="Path to trained model checkpoint (best_model.pt)"
    )
    parser.add_argument(
        "--llm_id",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="Base LLM ID (e.g., 'meta-llama/Llama-3.2-1B' or 'google/gemma-2-2b')"
    )
    parser.add_argument(
        "--data_model",
        type=Path,
        default=Path("/home/mlut/synchrony/data_model.yaml"),
        help="Path to data_model.yaml"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🧪 Testing OpenTSLM Model Generation")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Model path: {args.model_path}")
    print(f"  LLM backbone: {args.llm_id}")
    print(f"  Data model: {args.data_model}")
    print(f"  Test case: A3EY, Bindung, Turn 0, AU12_r (therapist)")
    print()
    
    # Setup device
    device = setup_device()
    
    # Load model
    print(f"\n🔧 Loading OpenTSLM model...")
    model = OpenTSLMFlamingo(device=device, llm_id=args.llm_id)
    
    if not args.model_path.exists():
        print(f"❌ Model checkpoint not found: {args.model_path}")
        return 1
    
    model.load_from_file(str(args.model_path))
    print("✅ Model loaded successfully")
    
    # Load data model
    data_model = load_data_model(args.data_model)
    
    # Find A3EY dyad
    a3ey_interview = None
    for interview in data_model['interviews']:
        if interview['patient']['patient_id'] == 'A3EY':
            a3ey_interview = interview
            break
    
    if not a3ey_interview:
        print("❌ Could not find A3EY in data_model.yaml")
        return 1
    
    print(f"\n📋 Found dyad:")
    print(f"  Patient: {a3ey_interview['patient']['patient_id']}")
    print(f"  Therapist: {a3ey_interview['therapist']['therapist_id']}")
    
    # Get Bindung interview data
    if 'bindung' not in a3ey_interview['types']:
        print("❌ Bindung interview not found for A3EY")
        return 1
    
    bindung_data = a3ey_interview['types']['bindung']
    therapist_csv = Path(bindung_data['therapist_openface'])
    transcript_json = Path(bindung_data['transcript'])
    
    print(f"\n📂 Data paths:")
    print(f"  Therapist OpenFace: {therapist_csv}")
    print(f"  Transcript: {transcript_json}")
    
    # Load transcript and get turn 0
    turns = load_speech_turns(transcript_json)
    if len(turns) == 0:
        print("❌ No turns found in transcript")
        return 1
    
    turn_0 = turns[0]
    print(f"\n📝 Turn 0 details:")
    print(f"  Speaker: {turn_0['speaker_id']}")
    print(f"  Start: {turn_0['start_ms']:.0f}ms")
    print(f"  End: {turn_0['end_ms']:.0f}ms")
    print(f"  Duration: {turn_0['end_ms'] - turn_0['start_ms']:.0f}ms")
    
    # Extract AU12_r for therapist
    print(f"\n⏳ Extracting AU12_r from therapist...")
    au_series = extract_au_window(
        therapist_csv,
        turn_0['start_ms'],
        turn_0['end_ms'],
        'AU12_r'
    )
    
    if au_series is None:
        print("❌ Failed to extract AU12_r data")
        return 1
    
    print(f"✅ Extracted {len(au_series)} frames")
    print(f"  Raw data range: [{au_series.min():.3f}, {au_series.max():.3f}]")
    
    # Truncate if needed
    au_series = truncate_time_series(au_series)
    
    # Normalize (following OpenTSLM training format)
    normalized, mean, std = normalize_time_series(au_series)
    print(f"  Mean: {mean:.4f}, Std: {std:.4f}")
    print(f"  Normalized range: [{normalized.min():.3f}, {normalized.max():.3f}]")
    
    # Build prompt following EXACT OpenTSLM training format
    print(f"\n🎯 Building prompt...")
    
    pre_prompt_text = """You are an expert in analyzing facial Action Units (AUs) from psychotherapy sessions.
You will receive time series data for AU12 (smile/lip corner puller) from the therapist.

Task: Describe the AU12 activation pattern briefly."""
    
    post_prompt_text = f"""
Speech Turn {turn_0['turn_index']}: {turn_0['speaker_id']} speaking
Time: {turn_0['start_ms']/1000:.1f}s - {turn_0['end_ms']/1000:.1f}s

Now describe the AU12 activation pattern:"""
    
    pre_prompt = TextPrompt(pre_prompt_text)
    post_prompt = TextPrompt(post_prompt_text)
    
    # Create time series prompt with EXACT format from training
    # Format: "Facial AU activation for {au_name} (therapist), it has mean {mean:.4f} and std {std:.4f}:"
    ts_text = f"Facial AU activation for AU12_r (therapist), it has mean {mean:.4f} and std {std:.4f}:"
    ts_prompt = TextTimeSeriesPrompt(ts_text, normalized.tolist())
    
    # Build full prompt
    prompt = FullPrompt(pre_prompt, [ts_prompt], post_prompt)
    
    print(f"  Pre-prompt length: {len(pre_prompt_text)} chars")
    print(f"  Time series length: {len(normalized)} points")
    print(f"  Post-prompt length: {len(post_prompt_text)} chars")
    
    # Generate description
    print(f"\n🚀 Generating description...")
    description = model.eval_prompt(prompt, max_new_tokens=100)
    
    print(f"\n{'=' * 80}")
    print(f"📄 GENERATED DESCRIPTION:")
    print(f"{'=' * 80}")
    print(description.strip())
    print(f"{'=' * 80}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())




