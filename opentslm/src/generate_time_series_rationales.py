"""
Script to generate rationales for psychotherapy dataset using a trained OpenTSLMFlamingo model.
This creates the time-series rationales that we - in another script - will augment with the transcripts

The script:
1. Loads the psychotherapy dataset using PsychotherapyCoTQADataset (which handles prompt construction)
2. Loads the trained OpenTSLMFlamingo model from checkpoint
3. Runs inference to generate rationales based on time-series data
4. Saves rationales to CSV/JSON for incorporation into the training dataset

Note: This script uses the existing dataset class to avoid duplicating prompt construction logic.

Usage:
    python generate_rationales.py --num_samples 100 --split train

Requirements:
    - Trained OpenTSLMFlamingo model checkpoint
    - The psychotherapy data loader and dataset
    - Required dependencies: torch, pandas, numpy
"""

import sys
import os
import torch
import pandas as pd
import numpy as np
import random
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

# Add OpenTSLM src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "opentslm", "src")))

from model.llm.OpenTSLMFlamingo import OpenTSLMFlamingo
from time_series_datasets.psychotherapy.psychotherapyCoTQADataset import PsychotherapyCoTQADataset
from prompt.full_prompt import FullPrompt
from prompt.text_prompt import TextPrompt
from prompt.text_time_series_prompt import TextTimeSeriesPrompt
from time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate

def setup_device():
    """Setup the device for model inference."""
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    return device


def load_trained_model(checkpoint_path: str, device: str, llm_id: str = "google/gemma-3-270m"):
    """Load the trained OpenTSLMFlamingo model from checkpoint."""
    print(f"Loading trained OpenTSLMFlamingo model from {checkpoint_path}...")
    
    model = OpenTSLMFlamingo(
        device=device,
        llm_id=llm_id,
        cross_attn_every_n_layers=1,
    )
    
    model.load_from_file(checkpoint_path)
    model.eval()
    print(f"✅ Model loaded successfully")
    return model


def load_dataset(split: str = "train", max_samples: int = None):
    """Load psychotherapy dataset using the proper QADataset class."""
    print(f"Loading psychotherapy dataset ({split} split)...")
    
    # Use the dataset class - it already handles prompt construction
    # We use format_sample_str=False to get structured FullPrompt objects
    dataset = PsychotherapyCoTQADataset(
        split=split,
        EOS_TOKEN="",  # We don't need EOS for generation
        format_sample_str=False,
        max_samples=max_samples
    )
    
    print(f"✅ Loaded {len(dataset)} samples")
    return dataset


def generate_rationale_with_model(
    model: OpenTSLMFlamingo,
    prompt: FullPrompt,
    max_new_tokens: int = 30000
) -> str:
    """Generate a rationale using the trained OpenTSLMFlamingo model."""
    with torch.no_grad():
        generated_text = model.eval_prompt(prompt, max_new_tokens=max_new_tokens)
    return generated_text.strip()


def run_rationale_generation(
    model: OpenTSLMFlamingo,
    dataset: PsychotherapyCoTQADataset,
    num_samples: int = None,
    max_new_tokens: int = 30000,
    random_seed: int = 42,
) -> List[Dict[str, Any]]:
    """Generate rationales for psychotherapy samples using the trained model."""
    print(f"Generating rationales for {num_samples or 'all'} samples...")
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    # Select samples
    dataset_size = len(dataset)
    if num_samples is None or num_samples >= dataset_size:
        selected_indices = list(range(dataset_size))
    else:
        selected_indices = random.sample(range(dataset_size), num_samples)
    
    results = []
    
    with torch.no_grad():
        for i, idx in enumerate(tqdm(selected_indices, desc="Generating rationales")):
            try:
                # Get the sample dict from the dataset
                sample = dataset[idx]
                # Reconstruct FullPrompt from the dict (same as HARCoT dataset format)
                # The dataset returns a dict with keys: pre_prompt, time_series, time_series_text, post_prompt
                prompt = FullPrompt(
                    TextPrompt(sample["pre_prompt"]),
                    [TextTimeSeriesPrompt(text, ts) for text, ts in zip(sample["time_series_text"], sample["time_series"])],
                    TextPrompt(sample["post_prompt"])
                )

                # Generate rationale using the trained model
                generated_rationale = generate_rationale_with_model(
                    model, prompt, max_new_tokens
                )
                
                # Collect result with all metadata
                result = {
                    "sample_index": idx,
                    "therapist_id": sample.get("therapist_id"),
                    "patient_id": sample.get("patient_id"),
                    "interview_type": sample.get("interview_type"),
                    "window_start": sample.get("window_start"),
                    "window_end": sample.get("window_end"),
                    "generated_rationale": generated_rationale,
                    "window_duration": sample.get("window_end", 0) - sample.get("window_start", 0),
                }
                
                results.append(result)
                
            except Exception as e:
                print(f"\n❌ Error processing sample {idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print(f"✅ Successfully generated rationales for {len(results)} samples")
    return results


def save_results(results: List[Dict[str, Any]], output_path: str):
    """Save the generated rationales JSON."""
    print(f"Saving results to {output_path}...")
    
    json_path = output_path + ".json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ JSON saved to {json_path}")
    
    # Print summary
    print(f"\n📊 Summary:")
    print(f"Total samples processed: {len(results)}")
    if results:
        avg_rationale_length = np.mean([len(r["generated_rationale"]) for r in results])
        print(f"Average rationale length: {avg_rationale_length:.1f} characters")
        
        # Show first rationale as example
        print(f"\n📝 Example rationale (sample {results[0]['sample_index']}):")
        print(f"Therapist: {results[0]['therapist_id']}, Patient: {results[0]['patient_id']}")
        print(f"Window: {results[0]['window_start']:.2f}s - {results[0]['window_end']:.2f}s")
        print(f"Rationale: {results[0]['generated_rationale'][:300]}...")


def main():
    """Main function to run rationale generation."""
    parser = argparse.ArgumentParser(
        description="Generate rationales for psychotherapy dataset using trained OpenTSLMFlamingo model"
    )
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default="/home/mlut/synchrony/opentslm/results/gemma_3_270m/OpenTSLMFlamingo/stage2_captioning/checkpoints/best_model.pt",
        help="Path to trained OpenTSLMFlamingo checkpoint"
    )
    parser.add_argument(
        "--llm_id", 
        type=str, 
        default="google/gemma-3-270m",
        help="Base LLM ID used for training"
    )
    parser.add_argument(
        "--split", 
        type=str, 
        default="train", 
        choices=["train", "validation", "test"],
        help="Dataset split to process"
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=None,
        help="Number of samples to process (None = all)"
    )
    parser.add_argument(
        "--max_new_tokens", 
        type=int, 
        default=300,
        help="Maximum tokens to generate per rationale"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="psychotherapy_rationales",
        help="Output file prefix (without extension)"
    )
    parser.add_argument(
        "--random_seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    print("🚀 Starting psychotherapy rationale generation with OpenTSLMFlamingo...")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  LLM ID: {args.llm_id}")
    print(f"  Split: {args.split}")
    print(f"  Num samples: {args.num_samples or 'all'}")
    print(f"  Max new tokens: {args.max_new_tokens}")
    print(f"  Output: {args.output}")
    print()
    
    # Setup
    device = setup_device()
    
    # Load trained model
    model = load_trained_model(args.checkpoint, device, args.llm_id)
    
    # Load dataset (uses the QADataset class with proper prompt construction)
    dataset = load_dataset(split=args.split, max_samples=args.num_samples)
    
    # Generate rationales
    results = run_rationale_generation(
        model,
        dataset,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        random_seed=args.random_seed,
    )
    
    # Save results
    output_path = f"/home/mlut/synchrony/.garbage/{args.output}_{args.split}"
    save_results(results, output_path)

if __name__ == "__main__":
    main()
