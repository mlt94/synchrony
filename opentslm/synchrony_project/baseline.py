"""
NO-LEAKAGE Baseline for Stage6 Synchrony Experiments.

This baseline ensures NO label leakage by:
1. Stage 1: Generate heatmap plots from raw OpenFace AU data → Gemma 3 27B VLM describes them
   (exactly like generate_time_series_descriptions.py)
2. Stage 2: Gemma 7B predicts empathy label from AU description + transcript summary
   (exactly like combine_transcripts_with_time_series_descriptions.py BUT without including the answer label)

Key differences from training data generation:
- generate_time_series_descriptions.py: Creates AU descriptions (Stage 1) ✓ REPLICATED
- combine_transcripts_with_time_series_descriptions.py: Combines AU + transcript BUT includes "Answer: {label}" ✗ EXCLUDED
- This baseline: Does Stage 1 + Stage 2 but NEVER shows the ground truth label to the model

This ensures fair evaluation: the model predicts without seeing answers.
"""

import sys
import os
import torch
import json
import argparse
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
from transformers import pipeline
import re


def setup_device():
    """Determine best available device."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    return device


def load_text_model(model_name: str, device: str):
    """Load Gemma text-only model and tokenizer (for Stage 2)."""
    print(f"\n📦 Loading text model: {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    # Load model with appropriate precision
    if device == "cuda":
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


def load_vlm_pipeline(model_name: str, device: str):
    """Load vision-language model pipeline (for Stage 1)."""
    print(f"\n📦 Loading VLM pipeline: {model_name}...")
    
    pipe = pipeline(
        "image-text-to-text",
        model=model_name,
        device=device,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
    )
    
    print(f"✅ VLM pipeline loaded successfully")
    return pipe


def load_data_model(yaml_path: Path) -> Dict:
    """Load the data_model.yaml file."""
    print(f"\n📂 Loading data model from {yaml_path}...")
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    print(f"✅ Loaded {len(data['interviews'])} interviews")
    return data


def load_config_splits(config_path: Path) -> Dict[str, List[str]]:
    """Load therapist splits from config_opentslm.yaml."""
    print(f"\n📂 Loading splits from {config_path}...")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    splits = config['psychotherapy_splits']
    print(f"✅ Loaded splits: {len(splits['train'])} train, {len(splits['val'])} val, {len(splits['test'])} test therapists")
    
    return splits


def extract_au_window(csv_path: Path, start_ms: float, end_ms: float, au_columns: List[str]) -> pd.DataFrame:
    """Extract AU data from OpenFace CSV for a specific time window."""
    df = pd.read_csv(csv_path, skipinitialspace=True)
    df['timestamp_ms'] = df['timestamp'] * 1000
    mask = (df['timestamp_ms'] >= start_ms) & (df['timestamp_ms'] <= end_ms)
    window_df = df.loc[mask, ['timestamp_ms'] + au_columns].copy()
    return window_df


def bin_time_series(data: pd.DataFrame, au_name: str, num_bins: int = 8) -> np.ndarray:
    """Bin time series data into equal temporal bins."""
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


def generate_heatmap_plot(
    therapist_csv: Path,
    patient_csv: Path,
    turn: Dict,
    au_names: List[str],
    output_path: Path,
    num_bins: int = 8
) -> bool:
    """Generate heatmap visualization (exactly like generate_time_series_descriptions.py).
    
    Returns:
        True if successful, False otherwise
    """
    try:
        start_ms = turn['start_ms']
        end_ms = turn['end_ms']
        speaker_id = turn['speaker_id']
        turn_index = turn['turn_index']
        
        # Extract AU data
        therapist_data = extract_au_window(therapist_csv, start_ms, end_ms, au_names)
        patient_data = extract_au_window(patient_csv, start_ms, end_ms, au_names)
        
        if therapist_data.empty or patient_data.empty:
            return False
        
        # Create binned heatmaps
        therapist_heatmap = np.zeros((len(au_names), num_bins))
        client_heatmap = np.zeros((len(au_names), num_bins))
        
        for i, au_name in enumerate(au_names):
            therapist_heatmap[i, :] = bin_time_series(therapist_data, au_name, num_bins)
            client_heatmap[i, :] = bin_time_series(patient_data, au_name, num_bins)
        
        # Create figure with two side-by-side heatmaps
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
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
        
        # Add value annotations on therapist heatmap
        for i in range(len(au_names)):
            for j in range(num_bins):
                ax1.text(j, i, f'{therapist_heatmap[i, j]:.2f}',
                        ha="center", va="center", color="black", fontsize=8)
        
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
        
        # Add value annotations on client heatmap
        for i in range(len(au_names)):
            for j in range(num_bins):
                ax2.text(j, i, f'{client_heatmap[i, j]:.2f}',
                        ha="center", va="center", color="black", fontsize=8)
        
        # Overall title
        plt.suptitle(f"Turn {turn_index}: {speaker_id.capitalize()} speaking ({start_ms:.0f}-{end_ms:.0f}ms)\n" + 
                     f"Heatmap shows mean AU activation across {num_bins} temporal phases", 
                     fontsize=15, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Error generating plot: {e}")
        return False


def generate_heatmap_description(
    vlm_pipe,
    plot_path: Path,
    turn: Dict,
    au_names: List[str]
) -> str:
    """Generate AU pattern description from heatmap image using Gemma 3 27B VLM (Stage 1).
    
    This exactly replicates generate_time_series_descriptions.py approach.
    """
    try:
        pre_prompt = """Describe these AU heatmaps (left=therapist blue, right=client orange). 
Each row is one AU across 8 time bins. Write one compact sentence per AU comparing patterns.
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
                    {"type": "image", "url": str(plot_path)},
                    {"type": "text", "text": f"{pre_prompt}\n\n{context}"}
                ]
            }
        ]
        
        output = vlm_pipe(
            text=messages,
            max_new_tokens=80,
            do_sample=False
        )
        description = output[0]["generated_text"][-1]["content"].strip()
        
        # Clean formatting (minimal post-processing)
        if description.lower().startswith(("here's", "here is", "the patterns")):
            if ':' in description[:50]:
                description = description.split(':', 1)[1].lstrip()
        
        description = description.replace('**', '').replace('*', '')
        description = description.replace('\n', ' ')
        description = re.sub(r'\s+', ' ', description).strip()
        
        return description
        
    except Exception as e:
        print(f"❌ Error generating heatmap description: {e}")
        return f"Error: {str(e)}"


def predict_empathy_category(
    au_description: str,
    transcript_summary: str,
    speaker_id: str,
    tokenizer_7b,
    model_7b,
    device: str
) -> Tuple[str, str]:
    """Predict empathy category using Gemma 7B from AU description + transcript (Stage 2).
    
    NO LABEL LEAKAGE - The model only sees:
    - AU pattern description (from Stage 1)
    - Transcript summary
    
    It does NOT see the ground truth label.
    """
    
    prompt = f"""You are analyzing a speech turn from a psychotherapy session.
Your task is to predict the relational empathy dynamic based on the speech content and facial expressions.

There are two possible categories:
1. "equally empathic" - client and therapist feel equally empathic to one another
2. "discrepancy" - there is a discrepancy in empathy levels

Data for this turn:

Speech content summary: {transcript_summary} (spoken by {speaker_id})

Facial Action Unit (AU) patterns: {au_description}

Based on the above information, predict the empathy category.

You MUST end your response with exactly one of these two phrases:
"Answer: equally empathic" OR "Answer: discrepancy"

Your prediction:"""
    
    messages = [{"role": "user", "content": prompt}]
    
    if hasattr(tokenizer_7b, "apply_chat_template"):
        formatted_prompt = tokenizer_7b.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        formatted_prompt = prompt
    
    inputs = tokenizer_7b(formatted_prompt, return_tensors="pt").to(device)
    
    with torch.inference_mode():
        outputs = model_7b.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            pad_token_id=tokenizer_7b.pad_token_id,
            eos_token_id=tokenizer_7b.eos_token_id
        )
    
    generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer_7b.decode(generated_tokens, skip_special_tokens=True).strip()
    
    # Extract prediction
    prediction = extract_prediction(response)
    
    return prediction, response


def extract_prediction(response: str) -> str:
    """Extract empathy category from model response."""
    response_lower = response.lower()
    
    # Look for explicit "Answer: X" pattern first
    answer_match = re.search(r'answer:\s*(equally empathic|discrepancy)', response_lower)
    if answer_match:
        return answer_match.group(1)
    
    # Look for the phrases anywhere in the response
    if "discrepancy" in response_lower:
        return "discrepancy"
    if "equally empathic" in response_lower or "equal empathy" in response_lower:
        return "equally empathic"
    
    return "unknown"


def _is_nan_like(x) -> bool:
    """Return True if x is None or cannot be interpreted as a finite number."""
    if x is None:
        return True
    if isinstance(x, str):
        try:
            xv = float(x)
        except Exception:
            return True
        return not np.isfinite(xv)
    try:
        xv = float(x)
    except Exception:
        return True
    return not np.isfinite(xv)


def extract_blri_scores(interview_data: Dict, interview_type: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Extract BLRI scores for therapist and client from interview data.
    
    This matches combine_transcripts_with_time_series_descriptions.py exactly.
    
    Returns:
        Tuple of (therapist_blri, client_blri, blri_difference)
        blri_difference = therapist - client (positive = therapist finds client more empathic)
        Returns (None, None, None) if either score is missing or NaN-like
    """
    if interview_type not in interview_data.get('types', {}):
        return None, None, None
    
    type_data = interview_data['types'][interview_type]
    labels = type_data.get('labels', {})
    
    # Find BLRI keys (they vary by interview type: T3, T5, T7)
    therapist_blri = None
    client_blri = None
    
    for key, value in labels.items():
        if 'BLRI_ges_In' in key:  # Interviewer/Therapist
            therapist_blri = value
        elif 'BLRI_ges_Pr' in key:  # Patient/Client
            client_blri = value
    
    # Skip if either score is missing or NaN-like
    if therapist_blri is None or client_blri is None:
        return None, None, None
    if _is_nan_like(therapist_blri) or _is_nan_like(client_blri):
        return None, None, None
    
    # Calculate difference using numeric values
    therapist_val = float(therapist_blri)
    client_val = float(client_blri)
    blri_diff = therapist_val - client_val
    return therapist_val, client_val, blri_diff


def discretize_blri_difference(blri_diff: Optional[float]) -> str:
    """Convert BLRI difference to binary empathy category."""
    if blri_diff is None or _is_nan_like(blri_diff):
        return "unknown"
    if -6 <= blri_diff <= 6:
        return "equally empathic"
    else:
        return "discrepancy"


def load_interview_data_with_splits(
    data_model: Dict,
    config_path: Path,
    au_names: List[str]
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Load and split interview data by therapist.
    
    Returns:
        Tuple of (train_list, val_list, test_list) where each entry contains
        interview metadata and turn information.
    """
    splits_config = load_config_splits(config_path)
    
    train_therapists = splits_config['train']
    val_therapists = splits_config['val']
    test_therapists = splits_config['test']
    
    train_list = []
    val_list = []
    test_list = []
    
    print(f"\n📊 Loading interview data...")
    
    for interview in data_model['interviews']:
        therapist_id = interview['therapist']['therapist_id']
        patient_id = interview['patient']['patient_id']
        
        for interview_type, type_data in interview.get('types', {}).items():
            # Extract BLRI scores using the same method as combine_transcripts
            therapist_blri, client_blri, blri_diff = extract_blri_scores(interview, interview_type)
            
            # Skip if BLRI scores are missing or NaN
            if blri_diff is None:
                continue
            
            empathy_category = discretize_blri_difference(blri_diff)
            
            if empathy_category == "unknown":
                continue
            
            # Load transcript
            transcript_path = Path(type_data.get('transcript', ''))
            if not transcript_path.exists():
                continue
            
            with open(transcript_path, 'r', encoding='utf-8') as f:
                turns = json.load(f)
            
            # Get OpenFace CSVs
            therapist_csv = Path(type_data.get('therapist_openface', ''))
            patient_csv = Path(type_data.get('patient_openface', ''))
            
            if not therapist_csv.exists() or not patient_csv.exists():
                continue
            
            # Add each turn
            for turn in turns:
                entry = {
                    'patient_id': patient_id,
                    'therapist_id': therapist_id,
                    'interview_type': interview_type,
                    'turn_index': turn['turn_index'],
                    'speaker_id': turn['speaker_id'],
                    'start_ms': turn['start_ms'],
                    'end_ms': turn['end_ms'],
                    'summary': turn.get('summary', ''),
                    'therapist_csv': therapist_csv,
                    'patient_csv': patient_csv,
                    'blri_difference': blri_diff,
                    'empathy_category': empathy_category,
                    'au_names': au_names
                }
                
                # Assign to split based on therapist
                if therapist_id in train_therapists:
                    train_list.append(entry)
                elif therapist_id in val_therapists:
                    val_list.append(entry)
                elif therapist_id in test_therapists:
                    test_list.append(entry)
    
    print(f"   Train: {len(train_list)} turns")
    print(f"   Val: {len(val_list)} turns")
    print(f"   Test: {len(test_list)} turns")
    
    return train_list, val_list, test_list


def evaluate_split(
    entries: List[Dict],
    vlm_pipe,
    tokenizer_7b,
    model_7b,
    device: str,
    split_name: str,
    temp_dir: Path,
    max_samples: Optional[int] = None,
    save_predictions: bool = False,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """Evaluate model on a data split using the two-stage pipeline."""
    
    if max_samples is not None and len(entries) > max_samples:
        entries = entries[:max_samples]
        print(f"ℹ️  Limited to {max_samples} samples for {split_name}")
    
    print(f"\n🔬 Evaluating {split_name} set ({len(entries)} samples)...")
    print(f"   Stage 1: Generate heatmap plots → Gemma 3 27B VLM describes them")
    print(f"   Stage 2: Gemma 7B predicts empathy from AU description + transcript")
    
    predictions = []
    ground_truths = []
    full_responses = []
    
    # Create temp directory for plots
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    for entry in tqdm(entries, desc=f"Predicting {split_name}"):
        try:
            # Stage 1a: Generate heatmap plot
            plot_path = temp_dir / f"{entry['patient_id']}_{entry['interview_type']}_turn{entry['turn_index']:03d}.jpg"
            
            plot_success = generate_heatmap_plot(
                entry['therapist_csv'],
                entry['patient_csv'],
                entry,
                entry['au_names'],
                plot_path
            )
            
            if not plot_success:
                predictions.append("unknown")
                ground_truths.append(entry['empathy_category'])
                continue
            
            # Stage 1b: VLM describes the heatmap
            au_description = generate_heatmap_description(
                vlm_pipe,
                plot_path,
                entry,
                entry['au_names']
            )
            
            # Stage 2: Predict empathy category (Gemma 7B)
            prediction, response = predict_empathy_category(
                au_description,
                entry['summary'],
                entry['speaker_id'],
                tokenizer_7b,
                model_7b,
                device
            )
            
            ground_truth = entry['empathy_category']
            
            predictions.append(prediction)
            ground_truths.append(ground_truth)
            full_responses.append({
                "patient_id": entry['patient_id'],
                "therapist_id": entry['therapist_id'],
                "interview_type": entry['interview_type'],
                "turn_index": entry['turn_index'],
                "speaker_id": entry['speaker_id'],
                "prediction": prediction,
                "ground_truth": ground_truth,
                "au_description": au_description,  # Stage 1 output
                "full_response": response,  # Stage 2 output
                "blri_difference": entry['blri_difference'],
                "transcript_summary": entry['summary'],
                "plot_path": str(plot_path)
            })
            
        except Exception as e:
            print(f"❌ Error processing entry: {e}")
            predictions.append("unknown")
            ground_truths.append(entry.get('empathy_category', 'unknown'))
            continue
    
    # Calculate turn-level metrics
    turn_metrics = calculate_metrics(predictions, ground_truths, split_name, level="turn")
    
    # Calculate interview-level metrics
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
    """Calculate metrics at interview level using majority voting."""
    # Group responses by interview
    interview_groups = defaultdict(list)
    for resp in responses:
        if resp['prediction'] != 'unknown':
            key = (resp['patient_id'], resp['therapist_id'], resp['interview_type'])
            interview_groups[key].append(resp)
    
    interview_predictions = []
    interview_ground_truths = []
    interview_details = []
    
    for interview_key, turns in interview_groups.items():
        # Count predictions
        pred_counts = defaultdict(int)
        for turn in turns:
            pred_counts[turn['prediction']] += 1
        
        # Majority vote
        majority_pred = max(pred_counts.items(), key=lambda x: (x[1], x[0]))[0]
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
    """Calculate evaluation metrics at turn level."""
    # Filter out "unknown" predictions
    valid_indices = [i for i, p in enumerate(predictions) if p != "unknown"]
    
    if not valid_indices:
        print(f"❌ No valid predictions for {split_name}!")
        return {"error": "No valid predictions"}
    
    filtered_preds = [predictions[i] for i in valid_indices]
    filtered_truths = [ground_truths[i] for i in valid_indices]
    
    accuracy = accuracy_score(filtered_truths, filtered_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        filtered_truths, 
        filtered_preds, 
        average='weighted',
        zero_division=0
    )
    
    cm = confusion_matrix(filtered_truths, filtered_preds, labels=["equally empathic", "discrepancy"])
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
        description="NO-LEAKAGE Baseline: Two-stage Gemma pipeline (27B + 7B)"
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
        help="Path to config_opentslm.yaml with therapist splits"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./baseline_results"),
        help="Output directory for results"
    )
    parser.add_argument(
        "--model_vlm",
        type=str,
        default="google/gemma-3-27b-it",
        help="Gemma 3 27B vision-language model for AU description generation"
    )
    parser.add_argument(
        "--model_7b",
        type=str,
        default="google/gemma-7b-it",
        help="Gemma 7B model for empathy prediction"
    )
    parser.add_argument(
        "--au_names",
        type=str,
        nargs='+',
        default=["AU12_r", "AU06_r", "AU04_r", "AU15_r"],
        help="AU column names to analyze"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples per split (debug)"
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        choices=["train", "val", "test", "all"],
        default="test",
        help="Which split to evaluate"
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Save individual predictions"
    )
    
    args = parser.parse_args()
    
    # Setup
    device = setup_device()
    
    # Load models
    print("\n" + "="*60)
    print("STAGE 1: Loading Gemma 3 27B VLM for AU description generation")
    print("="*60)
    vlm_pipe = load_vlm_pipeline(args.model_vlm, device)
    
    print("\n" + "="*60)
    print("STAGE 2: Loading Gemma 7B for empathy prediction")
    print("="*60)
    tokenizer_7b, model_7b = load_text_model(args.model_7b, device)
    
    # Load data
    data_model = load_data_model(args.data_model)
    train_list, val_list, test_list = load_interview_data_with_splits(
        data_model, args.config, args.au_names
    )
    
    if not (train_list or val_list or test_list):
        print("❌ No data loaded. Check your paths.")
        return
    
    # Create temp directory for plots
    temp_dir = args.output_dir / ".temp"
    
    # Evaluate requested splits
    all_metrics = {}
    
    if args.eval_split in ["train", "all"]:
        train_metrics = evaluate_split(
            train_list, vlm_pipe, tokenizer_7b, model_7b, device, "train", temp_dir,
            max_samples=args.max_samples,
            save_predictions=args.save_predictions,
            output_dir=args.output_dir
        )
        all_metrics["train"] = train_metrics
    
    if args.eval_split in ["val", "all"]:
        val_metrics = evaluate_split(
            val_list, vlm_pipe, tokenizer_7b, model_7b, device, "val", temp_dir,
            max_samples=args.max_samples,
            save_predictions=args.save_predictions,
            output_dir=args.output_dir
        )
        all_metrics["val"] = val_metrics
    
    if args.eval_split in ["test", "all"]:
        test_metrics = evaluate_split(
            test_list, vlm_pipe, tokenizer_7b, model_7b, device, "test", temp_dir,
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
    
    print("\n✅ NO-LEAKAGE Baseline evaluation complete!")
    print("="*60)
    print("This baseline ensures ZERO label leakage:")
    print("  Stage 1: Generate heatmap plots → Gemma 3 27B VLM describes them")
    print("  Stage 2: Gemma 7B predicts from description + transcript only")
    print(f"\nPlots saved in: {temp_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
