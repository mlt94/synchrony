"""
Evaluate Stage6 Synchrony predictions using binary classification metrics.

This script:
1. Reads test_predictions.jsonl from Stage6 output
2. Extracts predictions from "generated" field (last "Answer: [category]")
3. Extracts ground truth from "gold" field (last "Answer: [category]")
4. Computes turn-level and interview-level metrics
5. Saves results to specified output directory
"""

import json
import argparse
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


def extract_answer_category(text: str) -> str:
    """Extract the last 'Answer: [category]' from text.
    
    Args:
        text: String potentially containing multiple "Answer: ..." patterns
        
    Returns:
        Either "equally empathic", "discrepancy", or "unknown"
    """
    if not text or not isinstance(text, str):
        return "unknown"
    
    # Find all occurrences of "Answer: X" pattern
    # Match "equally empathic" or "discrepancy" (case insensitive)
    pattern = r'Answer:\s*(equally empathic|discrepancy)'
    matches = re.findall(pattern, text, re.IGNORECASE)
    
    if matches:
        # Return the LAST match (most recent answer)
        return matches[-1].lower()
    
    # Fallback: look for the words directly
    text_lower = text.lower()
    if "discrepancy" in text_lower:
        return "discrepancy"
    if "equally empathic" in text_lower or "equal empathy" in text_lower:
        return "equally empathic"
    
    return "unknown"


def load_predictions(jsonl_path: Path) -> List[Dict[str, Any]]:
    """Load predictions from JSONL file."""
    predictions = []
    print(f"\n📂 Loading predictions from {jsonl_path}...")
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                obj = json.loads(line.strip())
                predictions.append(obj)
            except json.JSONDecodeError as e:
                print(f"⚠️  Warning: Failed to parse line {line_num}: {e}")
                continue
    
    print(f"✅ Loaded {len(predictions)} predictions")
    return predictions


def extract_predictions_and_ground_truth(predictions: List[Dict]) -> Tuple[List[str], List[str], List[Dict]]:
    """Extract predicted and ground truth categories from predictions.
    
    Returns:
        Tuple of (predictions, ground_truths, processed_entries)
    """
    pred_list = []
    gt_list = []
    processed = []
    
    for entry in predictions:
        generated = entry.get("generated", "")
        gold = entry.get("gold", "")
        
        # Extract categories
        pred_category = extract_answer_category(generated)
        gt_category = extract_answer_category(gold)
        
        pred_list.append(pred_category)
        gt_list.append(gt_category)
        
        processed.append({
            "patient_id": entry.get("patient_id"),
            "therapist_id": entry.get("therapist_id"),
            "interview_type": entry.get("interview_type"),
            "turn_index": entry.get("turn_index"),
            "prediction": pred_category,
            "ground_truth": gt_category,
            "generated_text": generated,
            "gold_text": gold
        })
    
    return pred_list, gt_list, processed


def calculate_turn_level_metrics(predictions: List[str], ground_truths: List[str]) -> Dict[str, Any]:
    """Calculate turn-level metrics."""
    # Filter out unknown predictions/labels
    valid_indices = [
        i for i in range(len(predictions)) 
        if predictions[i] != "unknown" and ground_truths[i] != "unknown"
    ]
    
    if not valid_indices:
        return {
            "error": "No valid predictions",
            "n_total": len(predictions),
            "n_valid": 0,
            "n_unknown_pred": sum(1 for p in predictions if p == "unknown"),
            "n_unknown_gt": sum(1 for g in ground_truths if g == "unknown")
        }
    
    filtered_preds = [predictions[i] for i in valid_indices]
    filtered_gts = [ground_truths[i] for i in valid_indices]
    
    # Calculate metrics
    accuracy = accuracy_score(filtered_gts, filtered_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        filtered_gts, 
        filtered_preds, 
        average='weighted',
        zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(
        filtered_gts, 
        filtered_preds, 
        labels=["equally empathic", "discrepancy"]
    )
    
    n_unknown_pred = len(predictions) - len(valid_indices)
    
    metrics = {
        "n_total": len(predictions),
        "n_valid": len(valid_indices),
        "n_unknown_pred": sum(1 for p in predictions if p == "unknown"),
        "n_unknown_gt": sum(1 for g in ground_truths if g == "unknown"),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": ["equally empathic", "discrepancy"]
    }
    
    return metrics


def calculate_interview_level_metrics(processed_entries: List[Dict]) -> Dict[str, Any]:
    """Calculate interview-level metrics using majority voting."""
    # Group by interview
    interview_groups = defaultdict(list)
    
    for entry in processed_entries:
        if entry['prediction'] != 'unknown' and entry['ground_truth'] != 'unknown':
            key = (entry['patient_id'], entry['therapist_id'], entry['interview_type'])
            interview_groups[key].append(entry)
    
    if not interview_groups:
        return {"error": "No valid interview predictions"}
    
    interview_predictions = []
    interview_ground_truths = []
    interview_details = []
    
    for interview_key, turns in interview_groups.items():
        # Count predictions
        pred_counts = defaultdict(int)
        for turn in turns:
            pred_counts[turn['prediction']] += 1
        
        # Majority vote (break ties by lexicographic order)
        majority_pred = max(pred_counts.items(), key=lambda x: (x[1], x[0]))[0]
        
        # Ground truth should be consistent per interview, take first
        ground_truth = turns[0]['ground_truth']
        
        interview_predictions.append(majority_pred)
        interview_ground_truths.append(ground_truth)
        interview_details.append({
            'patient_id': interview_key[0],
            'therapist_id': interview_key[1],
            'interview_type': interview_key[2],
            'n_turns': len(turns),
            'prediction_counts': dict(pred_counts),
            'majority_prediction': majority_pred,
            'ground_truth': ground_truth
        })
    
    # Calculate metrics
    accuracy = accuracy_score(interview_ground_truths, interview_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        interview_ground_truths,
        interview_predictions,
        average='weighted',
        zero_division=0
    )
    
    cm = confusion_matrix(
        interview_ground_truths, 
        interview_predictions, 
        labels=["equally empathic", "discrepancy"]
    )
    
    metrics = {
        "n_interviews": len(interview_predictions),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": ["equally empathic", "discrepancy"],
        "interview_details": interview_details
    }
    
    return metrics


def print_metrics(turn_metrics: Dict, interview_metrics: Dict):
    """Print metrics in a readable format."""
    print("\n" + "="*60)
    print("TURN-LEVEL METRICS")
    print("="*60)
    
    if "error" in turn_metrics:
        print(f"❌ {turn_metrics['error']}")
        print(f"   Total turns: {turn_metrics['n_total']}")
        print(f"   Unknown predictions: {turn_metrics.get('n_unknown_pred', 0)}")
        print(f"   Unknown ground truth: {turn_metrics.get('n_unknown_gt', 0)}")
    else:
        print(f"Total turns: {turn_metrics['n_total']}")
        print(f"Valid predictions: {turn_metrics['n_valid']} ({100*turn_metrics['n_valid']/turn_metrics['n_total']:.1f}%)")
        print(f"Unknown predictions: {turn_metrics['n_unknown_pred']}")
        print(f"Unknown ground truth: {turn_metrics['n_unknown_gt']}")
        print(f"\nAccuracy:  {turn_metrics['accuracy']:.4f}")
        print(f"Precision: {turn_metrics['precision']:.4f}")
        print(f"Recall:    {turn_metrics['recall']:.4f}")
        print(f"F1 Score:  {turn_metrics['f1']:.4f}")
        
        cm = turn_metrics['confusion_matrix']
        print(f"\nConfusion Matrix:")
        print(f"{'':20s} | Pred: Empathic | Pred: Discrepancy")
        print(f"{'-'*60}")
        print(f"{'True: Empathic':20s} | {cm[0][0]:14d} | {cm[0][1]:17d}")
        print(f"{'True: Discrepancy':20s} | {cm[1][0]:14d} | {cm[1][1]:17d}")
    
    print("\n" + "="*60)
    print("INTERVIEW-LEVEL METRICS (Majority Voting)")
    print("="*60)
    
    if "error" in interview_metrics:
        print(f"❌ {interview_metrics['error']}")
    else:
        print(f"Total interviews: {interview_metrics['n_interviews']}")
        print(f"\nAccuracy:  {interview_metrics['accuracy']:.4f}")
        print(f"Precision: {interview_metrics['precision']:.4f}")
        print(f"Recall:    {interview_metrics['recall']:.4f}")
        print(f"F1 Score:  {interview_metrics['f1']:.4f}")
        
        cm = interview_metrics['confusion_matrix']
        print(f"\nConfusion Matrix:")
        print(f"{'':20s} | Pred: Empathic | Pred: Discrepancy")
        print(f"{'-'*60}")
        print(f"{'True: Empathic':20s} | {cm[0][0]:14d} | {cm[0][1]:17d}")
        print(f"{'True: Discrepancy':20s} | {cm[1][0]:14d} | {cm[1][1]:17d}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Stage6 Synchrony predictions with binary classification metrics"
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Path to test_predictions.jsonl from Stage6 output"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output directory for evaluation results"
    )
    parser.add_argument(
        "--save_processed",
        action="store_true",
        help="Save processed predictions with extracted categories"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {args.output_dir}")
    
    # Load predictions
    predictions = load_predictions(args.predictions)
    
    if not predictions:
        print("❌ No predictions loaded. Exiting.")
        return
    
    # Extract predictions and ground truth
    pred_list, gt_list, processed = extract_predictions_and_ground_truth(predictions)
    
    # Calculate turn-level metrics
    turn_metrics = calculate_turn_level_metrics(pred_list, gt_list)
    
    # Calculate interview-level metrics
    interview_metrics = calculate_interview_level_metrics(processed)
    
    # Combine metrics
    all_metrics = {
        "turn_level": turn_metrics,
        "interview_level": interview_metrics,
        "metadata": {
            "predictions_file": str(args.predictions),
            "total_entries": len(predictions)
        }
    }
    
    # Print metrics
    print_metrics(turn_metrics, interview_metrics)
    
    # Save metrics
    metrics_file = args.output_dir / "evaluation_metrics.json"
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Metrics saved to: {metrics_file}")
    
    # Save processed predictions if requested
    if args.save_processed:
        processed_file = args.output_dir / "processed_predictions.json"
        with open(processed_file, 'w', encoding='utf-8') as f:
            json.dump(processed, f, indent=2, ensure_ascii=False)
        print(f"💾 Processed predictions saved to: {processed_file}")
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
