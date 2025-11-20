"""
Baseline for Stage6 Synchrony Experiments using Gemma 7B.

This baseline:
1. Loads combined descriptions (transcript summary + AU descriptions) WITHOUT ground truth labels
2. Uses Gemma 7B to predict empathy category (equally empathic vs discrepancy)
3. Evaluates predictions against ground truth BLRI-derived labels
4. Supports train/val/test splits matching the stage6 dataset

The key difference from stage6 training:
- Stage6: OpenTSLMFlamingo learns from raw AU time series + transcript
- Baseline: Gemma 7B predicts from text descriptions only (no raw time series)
"""

import sys
import os
import torch
import json
import argparse
import yaml
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import re


def setup_device():
    """Determine best available device."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    return device


def load_model(model_name: str, device: str):
    """Load Gemma model and tokenizer."""
    print(f"\n📦 Loading model: {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    # Load model with appropriate precision
    if device == "cuda":
        # Use bfloat16 for better performance on modern GPUs
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        model.to(device)
    
    model.eval()
    print(f"✅ Model loaded successfully")
    
    return tokenizer, model


def load_data_model(yaml_path: Path) -> Dict:
    """Load the data_model.yaml file."""
    print(f"\n📂 Loading data model from {yaml_path}...")
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    print(f"✅ Loaded {len(data['interviews'])} interviews")
    return data


def load_combined_descriptions(answer_dir: Path, data_model: Dict) -> List[Dict]:
    """Load combined descriptions from answer JSON files.
    
    These files contain:
    - combined_description: The generated text (transcript + AU description)
    - empathy_category: Ground truth label derived from BLRI
    - blri_difference: Numeric BLRI score difference
    """
    all_entries = []
    
    print(f"\n📂 Loading combined descriptions from {answer_dir}...")
    
    for interview in data_model['interviews']:
        for interview_type, type_data in interview.get('types', {}).items():
            answer_path = type_data.get('answer')
            if not answer_path:
                continue
            
            answer_path = Path(answer_path)
            if not answer_path.exists():
                print(f"⚠️  Answer file not found: {answer_path}")
                continue
            
            with open(answer_path, 'r', encoding='utf-8') as f:
                entries = json.load(f)
            
            for entry in entries:
                # Skip entries without combined_description or empathy_category
                if not entry.get('combined_description') or not entry.get('empathy_category'):
                    continue
                
                all_entries.append(entry)
    
    print(f"✅ Loaded {len(all_entries)} entries with combined descriptions")
    return all_entries


def load_config_splits(config_path: Path) -> Dict[str, List[str]]:
    """Load therapist splits from config_opentslm.yaml.
    
    Returns:
        Dict with keys 'train', 'val', 'test' containing therapist IDs
    """
    print(f"\n📂 Loading splits from {config_path}...")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    splits = config['psychotherapy_splits']
    print(f"✅ Loaded splits: {len(splits['train'])} train, {len(splits['val'])} val, {len(splits['test'])} test therapists")
    
    return splits


def create_therapist_splits(entries: List[Dict], config_path: Path) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split data by therapist using predefined splits from config_opentslm.yaml.
    
    Args:
        entries: List of all entries
        config_path: Path to config_opentslm.yaml
        
    Returns:
        Tuple of (train_list, val_list, test_list)
    """
    # Load predefined splits
    splits_config = load_config_splits(config_path)
    
    train_therapists = splits_config['train']
    val_therapists = splits_config['val']
    test_therapists = splits_config['test']
    
    # Group entries by therapist
    therapist_entries = defaultdict(list)
    for entry in entries:
        therapist_id = entry['therapist_id']
        therapist_entries[therapist_id].append(entry)
    
    # Assign entries to splits based on predefined therapist IDs
    train_list = []
    val_list = []
    test_list = []
    
    for tid in train_therapists:
        if tid in therapist_entries:
            train_list.extend(therapist_entries[tid])
    
    for tid in val_therapists:
        if tid in therapist_entries:
            val_list.extend(therapist_entries[tid])
    
    for tid in test_therapists:
        if tid in therapist_entries:
            test_list.extend(therapist_entries[tid])
    
    # Check for unassigned therapists
    all_split_therapists = set(train_therapists + val_therapists + test_therapists)
    all_data_therapists = set(therapist_entries.keys())
    unassigned = all_data_therapists - all_split_therapists
    
    if unassigned:
        print(f"⚠️  Warning: {len(unassigned)} therapists not in config splits: {unassigned}")
    
    print(f"\n📊 Split statistics:")
    print(f"   Train: {len(train_list)} samples from {len([t for t in train_therapists if t in therapist_entries])} therapists")
    print(f"   Val: {len(val_list)} samples from {len([t for t in val_therapists if t in therapist_entries])} therapists")
    print(f"   Test: {len(test_list)} samples from {len([t for t in test_therapists if t in therapist_entries])} therapists")
    
    return train_list, val_list, test_list


def create_baseline_prompt(
    combined_description: str,
    speaker_id: str,
    original_summary: str
) -> str:
    """Create prompt for Gemma to predict empathy category.
    
    Key difference from training: We REMOVE the ground truth answer and ask Gemma to predict.
    """
    
    prompt = f"""You are analyzing a speech turn from a psychotherapy session.
Your task is to predict the relational empathy dynamic based on the speech content and facial expressions.

There are two possible categories:
1. "equally empathic" - client and therapist feel equally empathic to one another
2. "discrepancy" - there is a discrepancy in empathy levels

Data for this turn:

Speech content summary: {original_summary} (spoken by {speaker_id})

Combined description: {combined_description}

Based on the above information, predict the empathy category.

You MUST end your response with exactly one of these two phrases:
"Answer: equally empathic" OR "Answer: discrepancy"

Your prediction:"""
    
    return prompt


def predict_with_gemma(
    tokenizer,
    model,
    device: str,
    combined_description: str,
    speaker_id: str,
    original_summary: str
) -> str:
    """Use Gemma to predict empathy category from combined description.
    
    Returns:
        Predicted category: "equally empathic" or "discrepancy"
    """
    
    prompt = create_baseline_prompt(combined_description, speaker_id, original_summary)
    
    # Format as chat if available
    messages = [{"role": "user", "content": prompt}]
    
    if hasattr(tokenizer, "apply_chat_template"):
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        formatted_prompt = prompt
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,  # Greedy decoding for consistency
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode only the generated tokens
    generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    # Extract prediction from response
    prediction = extract_prediction(response)
    
    return prediction, response


def extract_prediction(response: str) -> str:
    """Extract empathy category from model response.
    
    Args:
        response: Model's generated text
        
    Returns:
        "equally empathic" or "discrepancy" or "unknown"
    """
    response_lower = response.lower()
    
    # Look for explicit "Answer: X" pattern first
    answer_match = re.search(r'answer:\s*(equally empathic|discrepancy)', response_lower)
    if answer_match:
        return answer_match.group(1)
    
    # Look for the phrases anywhere in the response (last occurrence wins)
    if "discrepancy" in response_lower:
        return "discrepancy"
    if "equally empathic" in response_lower or "equal empathy" in response_lower:
        return "equally empathic"
    
    # Default to unknown if neither found
    return "unknown"


def evaluate_split(
    entries: List[Dict],
    tokenizer,
    model,
    device: str,
    split_name: str,
    max_samples: Optional[int] = None,
    save_predictions: bool = False,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """Evaluate model on a data split.
    
    Args:
        entries: List of data entries
        tokenizer: Model tokenizer
        model: Model
        device: Device to run on
        split_name: Name of split (train/val/test)
        max_samples: Maximum samples to evaluate (None = all)
        save_predictions: Whether to save predictions to file
        output_dir: Directory to save predictions
        
    Returns:
        Dictionary of evaluation metrics
    """
    
    if max_samples is not None and len(entries) > max_samples:
        entries = entries[:max_samples]
        print(f"ℹ️  Limited to {max_samples} samples for {split_name}")
    
    print(f"\n🔬 Evaluating {split_name} set ({len(entries)} samples)...")
    
    predictions = []
    ground_truths = []
    full_responses = []
    
    for entry in tqdm(entries, desc=f"Predicting {split_name}"):
        try:
            # Get combined description WITHOUT ground truth label
            combined_desc = entry['combined_description']
            
            # Remove any "Answer: X" from the combined description if present
            # (to ensure we're not giving the model the answer)
            combined_desc_clean = re.sub(r'Answer:\s*(equally empathic|discrepancy)', '', combined_desc, flags=re.IGNORECASE).strip()
            
            speaker_id = entry['speaker_id']
            original_summary = entry.get('original_summary', '')
            
            # Predict
            prediction, response = predict_with_gemma(
                tokenizer,
                model,
                device,
                combined_desc_clean,
                speaker_id,
                original_summary
            )
            
            # Get ground truth
            ground_truth = entry['empathy_category']
            
            predictions.append(prediction)
            ground_truths.append(ground_truth)
            full_responses.append({
                "patient_id": entry['patient_id'],
                "therapist_id": entry['therapist_id'],
                "interview_type": entry['interview_type'],
                "turn_index": entry['turn_index'],
                "prediction": prediction,
                "ground_truth": ground_truth,
                "full_response": response,
                "blri_difference": entry.get('blri_difference'),
                "combined_description": entry['combined_description']  # Include original with label
            })
            
        except Exception as e:
            print(f"❌ Error processing entry: {e}")
            predictions.append("unknown")
            ground_truths.append(entry.get('empathy_category', 'unknown'))
            continue
    
    # Calculate turn-level metrics
    turn_metrics = calculate_metrics(predictions, ground_truths, split_name, level="turn")
    
    # Calculate interview-level metrics (majority vote aggregation)
    interview_metrics = calculate_interview_level_metrics(full_responses, split_name)
    
    # Combine metrics
    metrics = {
        "turn_level": turn_metrics,
        "interview_level": interview_metrics
    }
    
    # Save predictions if requested
    if save_predictions and output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        predictions_file = output_dir / f"{split_name}_predictions.json"
        with open(predictions_file, 'w', encoding='utf-8') as f:
            json.dump(full_responses, f, indent=2, ensure_ascii=False)
        print(f"💾 Predictions saved to {predictions_file}")
    
    return metrics


def calculate_interview_level_metrics(responses: List[Dict], split_name: str) -> Dict[str, Any]:
    """Calculate metrics at interview level using majority voting.
    
    Since BLRI scores are per interview, we aggregate turn-level predictions
    using majority voting and compare against interview-level ground truth.
    
    Args:
        responses: List of response dictionaries with predictions and metadata
        split_name: Name of split (for display)
        
    Returns:
        Dictionary of interview-level metrics
    """
    # Group responses by interview (patient_id, therapist_id, interview_type)
    interview_groups = defaultdict(list)
    for resp in responses:
        if resp['prediction'] != 'unknown':  # Skip unknown predictions
            key = (resp['patient_id'], resp['therapist_id'], resp['interview_type'])
            interview_groups[key].append(resp)
    
    # Aggregate predictions per interview using majority vote
    interview_predictions = []
    interview_ground_truths = []
    interview_details = []
    
    for interview_key, turns in interview_groups.items():
        # Count predictions
        pred_counts = defaultdict(int)
        for turn in turns:
            pred_counts[turn['prediction']] += 1
        
        # Majority vote (tie goes to first alphabetically)
        majority_pred = max(pred_counts.items(), key=lambda x: (x[1], x[0]))[0]
        
        # Ground truth should be the same for all turns in the interview
        ground_truth = turns[0]['ground_truth']
        blri_diff = turns[0]['blri_difference']
        
        interview_predictions.append(majority_pred)
        interview_ground_truths.append(ground_truth)
        interview_details.append({
            'patient_id': interview_key[0],
            'therapist_id': interview_key[1],
            'interview_type': interview_key[2],
            'n_turns': len(turns),
            'prediction_counts': dict(pred_counts),
            'majority_prediction': majority_pred,
            'ground_truth': ground_truth,
            'blri_difference': blri_diff
        })
    
    # Calculate metrics
    if not interview_predictions:
        print(f"❌ No valid interview predictions for {split_name}!")
        return {"error": "No valid predictions"}
    
    accuracy = accuracy_score(interview_ground_truths, interview_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        interview_ground_truths,
        interview_predictions,
        average='weighted',
        zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(interview_ground_truths, interview_predictions, 
                          labels=["equally empathic", "discrepancy"])
    
    metrics = {
        "n_interviews": len(interview_predictions),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm.tolist(),
        "interview_details": interview_details
    }
    
    # Print results
    print(f"\n📊 {split_name.upper()} Interview-Level Results (Majority Vote):")
    print(f"   Total interviews: {len(interview_predictions)}")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1 Score: {f1:.4f}")
    print(f"\n   Confusion Matrix:")
    print(f"   {'':20s} | Pred: Empathic | Pred: Discrepancy")
    print(f"   {'-'*60}")
    print(f"   {'True: Empathic':20s} | {cm[0][0]:14d} | {cm[0][1]:17d}")
    print(f"   {'True: Discrepancy':20s} | {cm[1][0]:14d} | {cm[1][1]:17d}")
    
    return metrics


def calculate_metrics(predictions: List[str], ground_truths: List[str], split_name: str, level: str = "turn") -> Dict[str, Any]:
    """Calculate evaluation metrics at turn level.
    
    Args:
        predictions: List of predicted categories
        ground_truths: List of ground truth categories
        split_name: Name of split (for display)
        level: Level of evaluation ("turn" or "interview")
        
    Returns:
        Dictionary of metrics
    """
    
    # Filter out "unknown" predictions for fair evaluation
    valid_indices = [i for i, p in enumerate(predictions) if p != "unknown"]
    
    if not valid_indices:
        print(f"❌ No valid predictions for {split_name}!")
        return {"error": "No valid predictions"}
    
    filtered_preds = [predictions[i] for i in valid_indices]
    filtered_truths = [ground_truths[i] for i in valid_indices]
    
    # Calculate metrics
    accuracy = accuracy_score(filtered_truths, filtered_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        filtered_truths, 
        filtered_preds, 
        average='weighted',
        zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(filtered_truths, filtered_preds, labels=["equally empathic", "discrepancy"])
    
    # Count unknowns
    n_unknown = len(predictions) - len(valid_indices)
    
    metrics = {
        "split": split_name,
        "n_total": len(predictions),
        "n_valid": len(valid_indices),
        "n_unknown": n_unknown,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm.tolist()
    }
    
    # Print results
    print(f"\n📊 {split_name.upper()} Turn-Level Results:")
    print(f"   Total turns: {len(predictions)}")
    print(f"   Valid predictions: {len(valid_indices)} ({100*len(valid_indices)/len(predictions):.1f}%)")
    print(f"   Unknown predictions: {n_unknown}")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1 Score: {f1:.4f}")
    print(f"\n   Confusion Matrix:")
    print(f"   {'':20s} | Pred: Empathic | Pred: Discrepancy")
    print(f"   {'-'*60}")
    print(f"   {'True: Empathic':20s} | {cm[0][0]:14d} | {cm[0][1]:17d}")
    print(f"   {'True: Discrepancy':20s} | {cm[1][0]:14d} | {cm[1][1]:17d}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Baseline evaluation for Stage6 using Gemma 7B"
    )
    parser.add_argument(
        "--data_model",
        type=Path,
        required=True,
        help="Path to data_model.yaml"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("../config_opentslm.yaml"),
        help="Path to config_opentslm.yaml with therapist splits (default: ../config_opentslm.yaml)"
    )
    parser.add_argument(
        "--answer_dir",
        type=Path,
        required=True,
        help="Directory containing answer JSON files (with combined descriptions)"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./baseline_results"),
        help="Output directory for results and predictions"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/gemma-7b-it",
        help="Model to use (default: google/gemma-7b-it)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples per split (None = all, useful for debugging)"
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        choices=["train", "val", "test", "all"],
        default="all",
        help="Which split to evaluate (default: all)"
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Save individual predictions to JSON files"
    )
    
    args = parser.parse_args()
    
    # Setup
    device = setup_device()
    tokenizer, model = load_model(args.model_name, device)
    
    # Load data
    data_model = load_data_model(args.data_model)
    entries = load_combined_descriptions(args.answer_dir, data_model)
    
    if not entries:
        print("❌ No entries loaded. Check your data paths.")
        return
    
    # Create splits using config file
    train_list, val_list, test_list = create_therapist_splits(entries, args.config)
    
    # Evaluate requested splits
    all_metrics = {}
    
    if args.eval_split in ["train", "all"]:
        train_metrics = evaluate_split(
            train_list, tokenizer, model, device, "train",
            max_samples=args.max_samples,
            save_predictions=args.save_predictions,
            output_dir=args.output_dir
        )
        all_metrics["train"] = train_metrics
    
    if args.eval_split in ["val", "all"]:
        val_metrics = evaluate_split(
            val_list, tokenizer, model, device, "val",
            max_samples=args.max_samples,
            save_predictions=args.save_predictions,
            output_dir=args.output_dir
        )
        all_metrics["val"] = val_metrics
    
    if args.eval_split in ["test", "all"]:
        test_metrics = evaluate_split(
            test_list, tokenizer, model, device, "test",
            max_samples=args.max_samples,
            save_predictions=args.save_predictions,
            output_dir=args.output_dir
        )
        all_metrics["test"] = test_metrics
    
    # Save overall metrics
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = args.output_dir / "baseline_metrics.json"
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n💾 Overall metrics saved to {metrics_file}")
    
    print("\n✅ Baseline evaluation complete!")


if __name__ == "__main__":
    main()
