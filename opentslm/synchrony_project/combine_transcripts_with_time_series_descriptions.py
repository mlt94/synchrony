"""
Combine time-series descriptions (AU patterns) with transcript summaries (speech content)
using Gemma to describe associations.

This script:
1. Loads data_model.yaml containing interview metadata and transcript paths
2. Loads time-series descriptions from ituhpc_timeseries_rationales/*.json (contains "generated_rationale" key - legacy naming)
3. Matches entries by patient_id, interview_type, turn_index
4. Uses Gemma to describe associations between AU patterns and speech content
5. Saves combined results to output JSON files
"""

import sys
import os
import torch
import json
import argparse
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
from transformers import pipeline
from collections import defaultdict


def setup_device():
    """Determine best available device."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    return device


def load_data_model(yaml_path: Path) -> Dict:
    """Load the data_model.yaml file."""
    print(f"\n📂 Loading data model from {yaml_path}...")
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    print(f"✅ Loaded {len(data['interviews'])} interviews")
    return data


def load_timeseries_descriptions(descriptions_dir: Path) -> Dict[str, List[Dict]]:
    """Load all time-series description JSON files from directory.
    
    Returns:
        Dict mapping (patient_id, interview_type) -> list of description entries
    """
    descriptions_by_key = defaultdict(list)
    
    print(f"\n📂 Loading time-series descriptions from {descriptions_dir}...")
    json_files = list(descriptions_dir.glob("*.json"))
    
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for entry in data:
            patient_id = entry['patient_id']
            interview_type = entry['interview_type']
            key = (patient_id, interview_type)
            descriptions_by_key[key].append(entry)
    
    print(f"✅ Loaded {len(json_files)} description files covering {len(descriptions_by_key)} patient-interview combinations")
    return descriptions_by_key


def match_entries(
    descriptions: List[Dict],
    summaries: List[Dict],
    tolerance_ms: int = 100
) -> List[Tuple[Dict, Dict]]:
    """Match time-series description and summary entries by turn_index and time window.
    
    Args:
        descriptions: List of time-series description entries
        summaries: List of summary entries (already extracted from wrapper)
        tolerance_ms: Time window tolerance in milliseconds for matching
    
    Returns:
        List of (description, summary) tuples for matched entries
    """
    matches = []
    
    # Build lookup by turn_index for summaries
    summaries_by_turn = {s['turn_index']: s for s in summaries}
    
    for desc in descriptions:
        turn_idx = desc['turn_index']
        start_ms = desc['start_ms']
        end_ms = desc['end_ms']
        
        # Try exact turn index match first
        if turn_idx in summaries_by_turn:
            summ = summaries_by_turn[turn_idx]
            # Verify time windows are close
            if (abs(summ['start_ms'] - start_ms) <= tolerance_ms and
                abs(summ['end_ms'] - end_ms) <= tolerance_ms):
                matches.append((desc, summ))
                continue
        
        # Fallback: search by time window
        for summ in summaries:
            if (abs(summ['start_ms'] - start_ms) <= tolerance_ms and
                abs(summ['end_ms'] - end_ms) <= tolerance_ms):
                matches.append((desc, summ))
                break
    
    return matches


def create_combination_prompt(
    timeseries_description: str, 
    summary: str, 
    speaker_id: str
) -> str:
    """Create prompt for Gemma to combine time-series description and summary."""
    
    prompt = f"""
    You are describing the communication in a speech turn from a psychotherapy session. 
    Your task is to describe the associations between what was said and the facial expressions.

Data for this turn:

Speech content summary: {summary} (spoken by {speaker_id})

Facial Action Unit (AU) patterns: {timeseries_description}

Instructions:
- Begin by describing the speech content very briefly
- Then briefly note any salient facial Action Units (AUs) that stand out — do not over-analyze every AU, only mention the most relevant ones, and dont write what facial movement the AU references.
- Do **not** over-analyze or speculate; be very true to what is actually present in the data available. 
- Do not reflect on the emotional bond, synchrony or similar aspects of the interaction.
- Write your description as a single, natural paragraph — do not use bullet points, numbered steps, new lines or section headings.

Description:"""
    
    return prompt


def combine_with_gemma(
    text_pipe,
    timeseries_description: str,
    summary: str,
    speaker_id: str
) -> str:
    """Use Gemma pipeline to combine time-series description and summary into coherent text."""
    
    prompt = create_combination_prompt(
        timeseries_description, summary, speaker_id
    )
    
    outputs = text_pipe(
        prompt,
        max_new_tokens=200,
        return_full_text=False,
        do_sample=False
    )
    
    # Extract generated text
    if isinstance(outputs, list) and outputs:
        combined = str(outputs[0].get("generated_text", "")).strip()
    elif isinstance(outputs, dict):
        combined = str(outputs.get("generated_text", "")).strip()
    else:
        combined = str(outputs).strip()
      
    return combined


def process_patient_interview(
    interview_data: Dict,
    interview_type: str,
    timeseries_descriptions: List[Dict],
    text_pipe,
    max_turns: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Process all turns for a single patient-interview combination.
    
    Args:
        interview_data: Interview data from data_model.yaml
        interview_type: Type of interview (bindung, personal, wunder)
        timeseries_descriptions: List of time-series description entries
        text_pipe: Gemma pipeline (optimized for speed)
        max_turns: Maximum number of turns to process (None = all)
    """
    
    results = []
    
    patient_id = interview_data['patient']['patient_id']
    therapist_id = interview_data['therapist']['therapist_id']
    
    # Load summaries from transcript path in data model
    if interview_type not in interview_data.get('types', {}):
        print(f"⚠️ Interview type '{interview_type}' not found for {patient_id}")
        return results
    
    transcript_path = Path(interview_data['types'][interview_type]['transcript'])
    if not transcript_path.exists():
        print(f"⚠️ Transcript not found: {transcript_path}")
        return results
    
    with open(transcript_path, 'r', encoding='utf-8') as f:
        summaries = json.load(f)
    
    # Match time-series descriptions with summaries
    matches = match_entries(timeseries_descriptions, summaries)
    
    if not matches:
        print(f"⚠️ No matches found for {patient_id} {interview_type}")
        return results
    
    # Limit turns if requested (for debugging)
    if max_turns is not None and len(matches) > max_turns:
        matches = matches[:max_turns]
        print(f"ℹ️  Limited to first {max_turns} turns for debugging")
    
    print(f"\nProcessing {patient_id} - {interview_type}: {len(matches)} matched turns")
    
    skipped_empty = 0
    
    for desc, summ in tqdm(matches, desc=f"{patient_id} {interview_type}"):
        try:
            # Skip if time-series description is empty or missing
            description_text = desc.get('generated_rationale', '').strip()  # Note: JSON key is legacy 'generated_rationale'
            if not description_text:
                skipped_empty += 1
                continue
            
            # Skip if description is an error message
            if description_text.lower().startswith('error:'):
                skipped_empty += 1
                continue
            
            # Skip if summary is empty
            summary_text = summ.get('summary', '').strip()
            if not summary_text:
                skipped_empty += 1
                continue
            
            # Use pipeline API to generate combined description
            combined = combine_with_gemma(
                text_pipe,
                description_text,
                summary_text,
                desc['speaker_id']
            )
            
            result = {
                "patient_id": patient_id,
                "therapist_id": therapist_id,
                "interview_type": interview_type,
                "turn_index": desc['turn_index'],
                "speaker_id": desc['speaker_id'],
                "start_ms": desc['start_ms'],
                "end_ms": desc['end_ms'],
                "duration_ms": desc['duration_ms'],
                "original_timeseries_description": desc['generated_rationale'],  # Note: JSON key is legacy 'generated_rationale'
                "original_summary": summ['summary'],
                "combined_description": combined
            }
            results.append(result)
            
        except Exception as e:
            print(f"❌ Error processing turn {desc['turn_index']}: {e}")
            continue
    
    if skipped_empty > 0:
        print(f"⚠️  Skipped {skipped_empty} turn(s) with empty description or summary")
    
    return results


def save_results(results: List[Dict[str, Any]], output_path: Path):
    """Save combined results to JSON."""
    print(f"\n💾 Saving {len(results)} results to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✅ Results saved")


def main():
    parser = argparse.ArgumentParser(
        description="Combine time-series descriptions with transcript summaries using Gemma"
    )
    parser.add_argument(
        "--data_model",
        type=Path,
        default=Path("C:/Users/User/Desktop/martins/synchrony/data_model.yaml"),
        help="Path to data_model.yaml"
    )
    parser.add_argument(
        "--descriptions_dir",
        type=Path,
        default=Path("C:/Users/User/Desktop/martins/data_transfers/ituhpc_timeseries_rationales"),
        help="Directory containing time-series description JSON files"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("C:/Users/User/Desktop/martins/results/combined_descriptions"),
        help="Output directory for combined results"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/gemma-2-9b-it",
        help="Gemma model to use (default: gemma-2-9b-it - good balance of speed and quality)"
    )
    parser.add_argument(
        "--max_interviews",
        type=int,
        default=None,
        help="Maximum number of interviews to process (None = all)"
    )
    parser.add_argument(
        "--max_turns",
        type=int,
        default=None,
        help="Maximum number of speech turns to process per interview (None = all, useful for debugging)"
    )
    parser.add_argument(
        "--interview_types",
        nargs="+",
        default=["bindung", "personal", "wunder"],
        help="Interview types to process"
    )
    
    args = parser.parse_args()
    
    print("🚀 Starting time-series description + summary combination with Gemma")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Data model: {args.data_model}")
    print(f"  Time-series descriptions dir: {args.descriptions_dir}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Model: {args.model_name}")
    print(f"  Interview types: {args.interview_types}")
    if args.max_turns:
        print(f"  Max turns per interview: {args.max_turns} (debugging mode)")
    print()
    
    # Setup
    device = setup_device()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    data_model = load_data_model(args.data_model)
    descriptions_by_key = load_timeseries_descriptions(args.descriptions_dir)
    
    interviews = data_model['interviews']
    if args.max_interviews:
        interviews = interviews[:args.max_interviews]
    
    print(f"\n📊 Processing {len(interviews)} interviews")
    
    # OPTIMIZED: Use pipeline API with 4-bit quantization for 12GB VRAM GPUs
    # 4-bit quantization reduces VRAM from ~18GB to ~5-6GB with minimal quality loss
    print(f"\n🔧 Loading {args.model_name} with pipeline API and 4-bit quantization...")
    
    # Convert device to pipeline format
    if device == "cuda":
        device_arg = 0  # First GPU
        
        # 4-bit quantization config for efficient VRAM usage
        from transformers import BitsAndBytesConfig
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,  # Nested quantization for even better compression
            bnb_4bit_quant_type="nf4"  # Normal Float 4-bit (optimal for LLMs)
        )
        
        print(f"   Using 4-bit quantization (NF4) to fit in 12GB VRAM")
    else:
        device_arg = -1  # CPU
        quantization_config = None
    
    text_pipe = pipeline(
        "text-generation",
        model=args.model_name,
        #device=device_arg,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        model_kwargs={"quantization_config": quantization_config} if quantization_config else {}
    )
    
    print(f"✅ Model loaded on {device}")
    if device == "cuda":
        print(f"   GPU Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    # Process all interviews
    all_results = []
    processed_count = 0
    
    for interview_idx, interview in enumerate(interviews):
        patient_id = interview['patient']['patient_id']
        therapist_id = interview['therapist']['therapist_id']
        
        print(f"\n{'='*80}")
        print(f"Interview {interview_idx + 1}/{len(interviews)}: {patient_id}/{therapist_id}")
        print(f"{'='*80}")
        
        for interview_type in args.interview_types:
            # Check if output file already exists (skip if already processed)
            output_file = args.output_dir / f"{patient_id}_{interview_type}_combined.json"
            if output_file.exists():
                print(f"⏭️  Skipping {patient_id} {interview_type} - output file already exists")
                continue
            
            # Check if we have time-series descriptions for this combination
            key = (patient_id, interview_type)
            if key not in descriptions_by_key:
                print(f"⚠️ No time-series descriptions found for {patient_id} {interview_type}, skipping")
                continue
            
            # Check if interview type exists in data model
            if interview_type not in interview.get('types', {}):
                print(f"⚠️ Interview type '{interview_type}' not in data model for {patient_id}, skipping")
                continue
            
            results = process_patient_interview(
                interview,
                interview_type,
                descriptions_by_key[key],
                text_pipe,
                max_turns=args.max_turns
            )
            
            if results:
                all_results.extend(results)
                processed_count += 1
                
                # Save per patient-interview
                save_results(results, output_file)
    
    print(f"\n✅ Complete! Combined {len(all_results)} turns across {processed_count} patient-interview combinations")
    print(f"📁 Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
