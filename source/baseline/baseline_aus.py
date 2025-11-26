"""
Random Forest baseline for BLRI binary classification using AU features.

This script:
1. Loads interview data with train/val/test splits from config_opentslm.yaml
2. Extracts ALL AU features (both therapist and client) from OpenFace CSVs
3. Trains a Random Forest classifier to predict binary BLRI categories:
   - "equally empathic": BLRI difference in [-6, 6]
   - "discrepancy": BLRI difference outside [-6, 6]
4. Labels are per-interview, but classification is per-frame
5. Reports metrics at both frame-level and interview-level (via majority vote)
"""

import sys
import os
import numpy as np
import pandas as pd
import json
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, confusion_matrix
from tqdm import tqdm

# Add OpenTSLM paths
opentslm_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "opentslm"))
sys.path.insert(0, opentslm_dir)

# All 17 AU columns from OpenFace
ALL_AU_COLUMNS = [
    'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r',
    'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r',
    'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r'
]


def load_config_splits(config_path: Path) -> Dict:
    """Load train/val/test splits from config_opentslm.yaml."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['psychotherapy_splits']


def load_data_model(yaml_path: Path) -> Dict:
    """Load the data_model.yaml file."""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data


def _is_nan_like(value) -> bool:
    """Check if value is NaN, None, or string 'nan'."""
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    if isinstance(value, str) and value.lower() in ['nan', 'none', '']:
        return True
    return False


def extract_blri_scores(interview: Dict, interview_type: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Extract BLRI scores for therapist and client from interview data.
    
    Returns:
        Tuple of (therapist_blri, client_blri, blri_difference)
        blri_difference = therapist - client (positive = therapist finds client more empathic)
    """
    if interview_type not in interview.get('types', {}):
        return None, None, None
    
    type_data = interview['types'][interview_type]
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
    
    # Calculate difference
    therapist_val = float(therapist_blri)
    client_val = float(client_blri)
    blri_diff = therapist_val - client_val
    
    return therapist_val, client_val, blri_diff


def discretize_blri_difference(blri_diff: Optional[float]) -> Optional[str]:
    """Convert BLRI difference to binary empathy category.
    
    Returns:
        'equally empathic' if -6 <= blri_diff <= 6
        'discrepancy' otherwise
        None if blri_diff is None or NaN
    """
    if blri_diff is None or _is_nan_like(blri_diff):
        return None
    if -6 <= blri_diff <= 6:
        return "equally empathic"
    else:
        return "discrepancy"


def extract_au_features(
    therapist_csv: Path,
    patient_csv: Path,
    au_columns: List[str]
) -> pd.DataFrame:
    """Extract AU features from OpenFace CSVs for both therapist and client.
    
    Returns:
        DataFrame with columns: [therapist_AU01_r, ..., therapist_AU45_r, 
                                  client_AU01_r, ..., client_AU45_r]
        Each row is one frame (aligned by timestamp).
    """
    # Read OpenFace CSVs
    therapist_df = pd.read_csv(therapist_csv, skipinitialspace=True)
    patient_df = pd.read_csv(patient_csv, skipinitialspace=True)
    
    # Convert timestamp to milliseconds
    therapist_df['timestamp_ms'] = therapist_df['timestamp'] * 1000
    patient_df['timestamp_ms'] = patient_df['timestamp'] * 1000
    
    # Extract AU columns and rename with prefix
    therapist_aus = therapist_df[['timestamp_ms'] + au_columns].copy()
    patient_aus = patient_df[['timestamp_ms'] + au_columns].copy()
    
    # Rename columns to indicate speaker
    therapist_aus.columns = ['timestamp_ms'] + [f'therapist_{col}' for col in au_columns]
    patient_aus.columns = ['timestamp_ms'] + [f'client_{col}' for col in au_columns]
    
    # Merge on timestamp (inner join to ensure alignment)
    merged = pd.merge(therapist_aus, patient_aus, on='timestamp_ms', how='inner')
    
    return merged


def load_interview_data_with_splits(
    data_model: Dict,
    config_path: Path,
    au_columns: List[str]
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Load and split interview data by therapist.
    
    Returns:
        Tuple of (train_list, val_list, test_list) where each entry contains:
        - X: AU feature matrix (n_frames × n_features)
        - y: binary label array (n_frames,) - same label repeated for all frames
        - interview_id: unique identifier for grouping frames into interviews
    """
    splits_config = load_config_splits(config_path)
    
    train_therapists = splits_config['train']
    val_therapists = splits_config['val']
    test_therapists = splits_config['test']
    
    train_list = []
    val_list = []
    test_list = []
    
    print(f"\n📊 Loading interview data with AU features...")
    
    for interview in tqdm(data_model['interviews'], desc="Processing interviews"):
        therapist_id = interview['therapist']['therapist_id']
        patient_id = interview['patient']['patient_id']
        
        for interview_type, type_data in interview.get('types', {}).items():
            # Extract BLRI scores
            therapist_blri, client_blri, blri_diff = extract_blri_scores(interview, interview_type)
            
            # Skip if BLRI scores are missing
            if blri_diff is None:
                continue
            
            empathy_category = discretize_blri_difference(blri_diff)
            if empathy_category is None:
                continue
            
            # Get OpenFace CSVs
            therapist_csv = Path(type_data.get('therapist_openface', ''))
            patient_csv = Path(type_data.get('patient_openface', ''))
            
            if not therapist_csv.exists() or not patient_csv.exists():
                continue
            
            # Extract AU features for entire interview
            try:
                au_features = extract_au_features(therapist_csv, patient_csv, au_columns)
            except Exception as e:
                print(f"⚠️ Error extracting AUs for {patient_id}_{interview_type}: {e}")
                continue
            
            if len(au_features) == 0:
                continue
            
            # Prepare feature matrix (drop timestamp)
            X = au_features.drop(columns=['timestamp_ms']).values
            
            # Label is the same for all frames in this interview
            y = np.array([empathy_category] * len(X))
            
            # Unique interview identifier
            interview_id = f"{patient_id}_{interview_type}"
            
            entry = {
                'X': X,
                'y': y,
                'interview_id': interview_id,
                'patient_id': patient_id,
                'therapist_id': therapist_id,
                'interview_type': interview_type,
                'empathy_category': empathy_category,
                'blri_difference': blri_diff,
                'n_frames': len(X)
            }
            
            # Assign to split based on therapist
            if therapist_id in train_therapists:
                train_list.append(entry)
            elif therapist_id in val_therapists:
                val_list.append(entry)
            elif therapist_id in test_therapists:
                test_list.append(entry)
    
    # Report statistics
    print(f"\n✅ Data loading complete:")
    print(f"   Train: {len(train_list)} interviews, {sum(e['n_frames'] for e in train_list):,} frames")
    print(f"   Val: {len(val_list)} interviews, {sum(e['n_frames'] for e in val_list):,} frames")
    print(f"   Test: {len(test_list)} interviews, {sum(e['n_frames'] for e in test_list):,} frames")
    
    return train_list, val_list, test_list


def aggregate_to_arrays(entries: List[Dict]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Concatenate all entries into single X, y arrays with interview IDs.
    
    Returns:
        X: (total_frames, n_features)
        y: (total_frames,)
        interview_ids: List of interview IDs (one per frame)
    """
    X_list = []
    y_list = []
    interview_ids = []
    
    for entry in entries:
        X_list.append(entry['X'])
        y_list.append(entry['y'])
        interview_ids.extend([entry['interview_id']] * entry['n_frames'])
    
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    
    return X, y, interview_ids


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 100,
    max_depth: Optional[int] = 10,
    min_samples_split: int = 20,
    min_samples_leaf: int = 10,
    max_features: str = 'sqrt',
    random_state: int = 42
) -> RandomForestClassifier:
    """Train a Random Forest classifier with regularization to prevent overfitting.
    
    Key parameters to prevent overfitting:
    - max_depth: Limits tree depth (default: 10)
    - min_samples_split: Minimum samples required to split a node (default: 20)
    - min_samples_leaf: Minimum samples required at leaf node (default: 10)
    - max_features: Features to consider for best split (default: 'sqrt')
    """
    print(f"\n🌲 Training Random Forest with regularization...")
    print(f"   n_estimators={n_estimators}")
    print(f"   max_depth={max_depth}")
    print(f"   min_samples_split={min_samples_split}")
    print(f"   min_samples_leaf={min_samples_leaf}")
    print(f"   max_features={max_features}")
    
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state,
        n_jobs=-1,
        class_weight='balanced'  # Handle class imbalance
    )
    
    clf.fit(X_train, y_train)
    
    print(f"✅ Training complete")
    return clf


def evaluate_frame_level(
    clf: RandomForestClassifier,
    X: np.ndarray,
    y_true: np.ndarray,
    split_name: str
) -> Dict:
    """Evaluate classifier at frame level."""
    y_pred = clf.predict(X)
    
    # Calculate metrics
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    
    print(f"\n📊 {split_name} - Frame-Level Metrics:")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1-Score: {f1:.4f}")
    print(f"\n   Classification Report:")
    print(classification_report(y_true, y_pred, zero_division=0, digits=4))
    print(f"\n   Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'y_true': y_true,
        'y_pred': y_pred
    }


def evaluate_interview_level(
    clf: RandomForestClassifier,
    entries: List[Dict],
    split_name: str
) -> Dict:
    """Evaluate classifier at interview level using majority vote."""
    y_true_interview = []
    y_pred_interview = []
    interview_ids = []
    
    print(f"\n🗳️  {split_name} - Interview-Level Evaluation (Majority Vote):")
    
    for entry in entries:
        # Predict on all frames
        y_pred_frames = clf.predict(entry['X'])
        
        # Majority vote
        vote_counts = Counter(y_pred_frames)
        majority_pred = vote_counts.most_common(1)[0][0]
        
        # Ground truth (all frames have same label)
        ground_truth = entry['y'][0]
        
        y_true_interview.append(ground_truth)
        y_pred_interview.append(majority_pred)
        interview_ids.append(entry['interview_id'])
    
    y_true_interview = np.array(y_true_interview)
    y_pred_interview = np.array(y_pred_interview)
    
    # Calculate metrics
    f1 = f1_score(y_true_interview, y_pred_interview, average='weighted', zero_division=0)
    precision = precision_score(y_true_interview, y_pred_interview, average='weighted', zero_division=0)
    recall = recall_score(y_true_interview, y_pred_interview, average='weighted', zero_division=0)
    
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1-Score: {f1:.4f}")
    print(f"\n   Classification Report:")
    print(classification_report(y_true_interview, y_pred_interview, zero_division=0, digits=4))
    print(f"\n   Confusion Matrix:")
    print(confusion_matrix(y_true_interview, y_pred_interview))
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'y_true': y_true_interview,
        'y_pred': y_pred_interview,
        'interview_ids': interview_ids
    }


def main():
    parser = argparse.ArgumentParser(
        description="Random Forest baseline for BLRI binary classification using AU features"
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
        default=Path(__file__).parent.parent.parent / "opentslm" / "config_opentslm.yaml",
        help="Path to config_opentslm.yaml (default: opentslm/config_opentslm.yaml)"
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=100,
        help="Number of trees in Random Forest (default: 100)"
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=10,
        help="Maximum depth of trees to prevent overfitting (default: 10, try 5-15)"
    )
    parser.add_argument(
        "--min_samples_split",
        type=int,
        default=20,
        help="Minimum samples required to split a node (default: 20, higher = more regularization)"
    )
    parser.add_argument(
        "--min_samples_leaf",
        type=int,
        default=10,
        help="Minimum samples required at leaf node (default: 10, higher = more regularization)"
    )
    parser.add_argument(
        "--max_features",
        type=str,
        default='sqrt',
        help="Features to consider for best split: 'sqrt', 'log2', or float (default: 'sqrt')"
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Random Forest Baseline: AU-based BLRI Binary Classification")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Data model: {args.data_model}")
    print(f"  Config: {args.config}")
    print(f"  AU columns: ALL 17 AUs (both therapist and client) = 34 features")
    print(f"\nRandom Forest Hyperparameters:")
    print(f"  n_estimators: {args.n_estimators}")
    print(f"  max_depth: {args.max_depth}")
    print(f"  min_samples_split: {args.min_samples_split}")
    print(f"  min_samples_leaf: {args.min_samples_leaf}")
    print(f"  max_features: {args.max_features}")
    print(f"  random_state: {args.random_state}")
    
    # Load data model
    data_model = load_data_model(args.data_model)
    
    # Load data with train/val/test splits
    train_entries, val_entries, test_entries = load_interview_data_with_splits(
        data_model,
        args.config,
        ALL_AU_COLUMNS
    )
    
    # Convert to arrays
    X_train, y_train, train_interview_ids = aggregate_to_arrays(train_entries)
    X_val, y_val, val_interview_ids = aggregate_to_arrays(val_entries)
    X_test, y_test, test_interview_ids = aggregate_to_arrays(test_entries)
    
    print(f"\n📐 Feature dimensions:")
    print(f"   X_train: {X_train.shape}")
    print(f"   X_val: {X_val.shape}")
    print(f"   X_test: {X_test.shape}")
    
    # Train Random Forest
    clf = train_random_forest(
        X_train,
        y_train,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        random_state=args.random_state
    )
    
    # Evaluate on all splits
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    
    # Frame-level evaluation
    train_frame_metrics = evaluate_frame_level(clf, X_train, y_train, "Train")
    val_frame_metrics = evaluate_frame_level(clf, X_val, y_val, "Validation")
    test_frame_metrics = evaluate_frame_level(clf, X_test, y_test, "Test")
    
    # Interview-level evaluation (majority vote)
    train_interview_metrics = evaluate_interview_level(clf, train_entries, "Train")
    val_interview_metrics = evaluate_interview_level(clf, val_entries, "Validation")
    test_interview_metrics = evaluate_interview_level(clf, test_entries, "Test")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\nFrame-Level F1 Scores:")
    print(f"  Train: {train_frame_metrics['f1']:.4f}")
    print(f"  Val:   {val_frame_metrics['f1']:.4f}")
    print(f"  Test:  {test_frame_metrics['f1']:.4f}")
    
    print("\nInterview-Level F1 Scores (Majority Vote):")
    print(f"  Train: {train_interview_metrics['f1']:.4f}")
    print(f"  Val:   {val_interview_metrics['f1']:.4f}")
    print(f"  Test:  {test_interview_metrics['f1']:.4f}")
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
