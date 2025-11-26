"""
MLP baseline for BLRI regression using AU features.

This script:
1. Loads interview data with train/val/test splits from config_opentslm.yaml
2. Extracts ALL AU features (both therapist and client) from OpenFace CSVs
3. Trains an MLP regressor to predict client BLRI scores (T3/T5/T7_BLRI_ges_Pr)
4. Labels are per-interview, but prediction is per-frame
5. Reports metrics at both frame-level and interview-level (via mean aggregation)
"""

import sys
import os
import numpy as np
import pandas as pd
import json
import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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


def extract_client_blri(interview: Dict, interview_type: str) -> Optional[float]:
    """Extract client BLRI score from interview data.
    
    Interview types are: "personal" (T3), "bindung" (T5), "wunder" (T7)
    BLRI keys are: T3_BLRI_ges_Pr, T5_BLRI_ges_Pr, T7_BLRI_ges_Pr (Patient/Client)
    
    Returns:
        Client BLRI score (float) or None if missing/invalid
    """
    if interview_type not in interview.get('types', {}):
        return None
    
    type_data = interview['types'][interview_type]
    labels = type_data.get('labels', {})
    
    # Search for any key containing 'BLRI_ges_Pr' (Patient/Client BLRI)
    # This handles T3_BLRI_ges_Pr, T5_BLRI_ges_Pr, T7_BLRI_ges_Pr
    client_blri = None
    for key, value in labels.items():
        if 'BLRI_ges_Pr' in key:
            client_blri = value
            break
    
    # Skip if missing or NaN-like
    if client_blri is None or _is_nan_like(client_blri):
        return None
    
    return float(client_blri)


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
        - y: BLRI score array (n_frames,) - same score repeated for all frames
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
            # Extract client BLRI score
            client_blri = extract_client_blri(interview, interview_type)
            
            # Skip if BLRI score is missing
            if client_blri is None:
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
            y = np.array([client_blri] * len(X), dtype=np.float32)
            
            # Unique interview identifier
            interview_id = f"{patient_id}_{interview_type}"
            
            entry = {
                'X': X,
                'y': y,
                'interview_id': interview_id,
                'patient_id': patient_id,
                'therapist_id': therapist_id,
                'interview_type': interview_type,
                'client_blri': client_blri,
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


class MLPRegressor(nn.Module):
    """Multi-layer perceptron for BLRI regression."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64, 32], dropout: float = 0.3):
        super(MLPRegressor, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer (single value regression)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze(-1)


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hidden_dims: List[int] = [128, 64, 32],
    dropout: float = 0.3,
    learning_rate: float = 0.001,
    batch_size: int = 256,
    num_epochs: int = 15,
    device: str = 'cuda'
) -> MLPRegressor:
    """Train an MLP regressor for BLRI prediction.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data (for early stopping monitoring)
        hidden_dims: List of hidden layer dimensions
        dropout: Dropout rate for regularization
        learning_rate: Learning rate for Adam optimizer
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        device: 'cuda' or 'cpu'
    
    Returns:
        Trained MLP model
    """
    print(f"\n🧠 Training MLP Regressor...")
    print(f"   Architecture: {X_train.shape[1]} → {' → '.join(map(str, hidden_dims))} → 1")
    print(f"   Dropout: {dropout}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Batch size: {batch_size}")
    print(f"   Epochs: {num_epochs}")
    print(f"   Device: {device}")
    
    # Standardize features (important for neural networks)
    train_mean = X_train.mean(axis=0)
    train_std = X_train.std(axis=0) + 1e-8
    
    X_train_norm = (X_train - train_mean) / train_std
    X_val_norm = (X_val - train_mean) / train_std
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_norm).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val_norm).to(device)
    y_val_tensor = torch.FloatTensor(y_val).to(device)
    
    # Create model
    model = MLPRegressor(input_dim=X_train.shape[1], hidden_dims=hidden_dims, dropout=dropout).to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(num_epochs):
        model.train()
        
        # Shuffle training data
        perm = torch.randperm(len(X_train_tensor))
        X_train_shuffled = X_train_tensor[perm]
        y_train_shuffled = y_train_tensor[perm]
        
        # Mini-batch training
        train_loss = 0.0
        num_batches = 0
        
        for i in range(0, len(X_train_tensor), batch_size):
            batch_X = X_train_shuffled[i:i+batch_size]
            batch_y = y_train_shuffled[i:i+batch_size]
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = train_loss / num_batches
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()
        
        print(f"   Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
    
    # Load best model
    model.load_state_dict(best_model_state)
    print(f"\n✅ Training complete (Best Val Loss: {best_val_loss:.4f})")
    
    # Store normalization parameters in model
    model.train_mean = train_mean
    model.train_std = train_std
    
    return model


def evaluate_frame_level(
    model: MLPRegressor,
    X: np.ndarray,
    y_true: np.ndarray,
    split_name: str,
    device: str = 'cuda'
) -> Dict:
    """Evaluate regressor at frame level."""
    model.eval()
    
    # Normalize using training statistics
    X_norm = (X - model.train_mean) / model.train_std
    X_tensor = torch.FloatTensor(X_norm).to(device)
    
    with torch.no_grad():
        y_pred = model(X_tensor).cpu().numpy()
    
    # Calculate regression metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n📊 {split_name} - Frame-Level Metrics:")
    print(f"   MSE:  {mse:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   MAE:  {mae:.4f}")
    print(f"   R²:   {r2:.4f}")
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'y_true': y_true,
        'y_pred': y_pred
    }


def evaluate_interview_level(
    model: MLPRegressor,
    entries: List[Dict],
    split_name: str,
    device: str = 'cuda'
) -> Dict:
    """Evaluate regressor at interview level using mean aggregation."""
    y_true_interview = []
    y_pred_interview = []
    interview_ids = []
    
    print(f"\n📊 {split_name} - Interview-Level Evaluation (Mean Aggregation):")
    
    model.eval()
    
    for entry in entries:
        # Normalize and predict on all frames
        X_norm = (entry['X'] - model.train_mean) / model.train_std
        X_tensor = torch.FloatTensor(X_norm).to(device)
        
        with torch.no_grad():
            y_pred_frames = model(X_tensor).cpu().numpy()
        
        # Mean aggregation
        mean_pred = y_pred_frames.mean()
        
        # Ground truth (all frames have same label)
        ground_truth = entry['y'][0]
        
        y_true_interview.append(ground_truth)
        y_pred_interview.append(mean_pred)
        interview_ids.append(entry['interview_id'])
    
    y_true_interview = np.array(y_true_interview)
    y_pred_interview = np.array(y_pred_interview)
    
    # Calculate regression metrics
    mse = mean_squared_error(y_true_interview, y_pred_interview)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_interview, y_pred_interview)
    r2 = r2_score(y_true_interview, y_pred_interview)
    
    print(f"   MSE:  {mse:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   MAE:  {mae:.4f}")
    print(f"   R²:   {r2:.4f}")
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'y_true': y_true_interview,
        'y_pred': y_pred_interview,
        'interview_ids': interview_ids
    }


def main():
    parser = argparse.ArgumentParser(
        description="MLP baseline for BLRI regression using AU features"
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
        "--hidden_dims",
        type=int,
        nargs='+',
        default=[128, 64, 32],
        help="Hidden layer dimensions (default: 128 64 32)"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout rate for regularization (default: 0.3)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate for Adam optimizer (default: 0.001)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for training (default: 256)"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=15,
        help="Number of training epochs (default: 15)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help="Device: 'cuda' or 'cpu' (default: auto-detect)"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("MLP Baseline: AU-based BLRI Regression")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Data model: {args.data_model}")
    print(f"  Config: {args.config}")
    print(f"  AU columns: ALL 17 AUs (both therapist and client) = 34 features")
    print(f"  Target: Client BLRI scores (T3/T5/T7_BLRI_ges_Pr)")
    print(f"\nMLP Hyperparameters:")
    print(f"  Hidden dims: {args.hidden_dims}")
    print(f"  Dropout: {args.dropout}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Device: {args.device}")
    
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
    
    print(f"\n📊 Target statistics:")
    print(f"   Train BLRI - Mean: {y_train.mean():.2f}, Std: {y_train.std():.2f}, Range: [{y_train.min():.2f}, {y_train.max():.2f}]")
    print(f"   Val BLRI   - Mean: {y_val.mean():.2f}, Std: {y_val.std():.2f}, Range: [{y_val.min():.2f}, {y_val.max():.2f}]")
    print(f"   Test BLRI  - Mean: {y_test.mean():.2f}, Std: {y_test.std():.2f}, Range: [{y_test.min():.2f}, {y_test.max():.2f}]")
    
    # Train MLP
    model = train_mlp(
        X_train,
        y_train,
        X_val,
        y_val,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        device=args.device
    )
    
    # Evaluate on all splits
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    
    # Frame-level evaluation
    train_frame_metrics = evaluate_frame_level(model, X_train, y_train, "Train", args.device)
    val_frame_metrics = evaluate_frame_level(model, X_val, y_val, "Validation", args.device)
    test_frame_metrics = evaluate_frame_level(model, X_test, y_test, "Test", args.device)
    
    # Interview-level evaluation (mean aggregation)
    train_interview_metrics = evaluate_interview_level(model, train_entries, "Train", args.device)
    val_interview_metrics = evaluate_interview_level(model, val_entries, "Validation", args.device)
    test_interview_metrics = evaluate_interview_level(model, test_entries, "Test", args.device)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\nFrame-Level RMSE:")
    print(f"  Train: {train_frame_metrics['rmse']:.4f}")
    print(f"  Val:   {val_frame_metrics['rmse']:.4f}")
    print(f"  Test:  {test_frame_metrics['rmse']:.4f}")
    
    print("\nFrame-Level R²:")
    print(f"  Train: {train_frame_metrics['r2']:.4f}")
    print(f"  Val:   {val_frame_metrics['r2']:.4f}")
    print(f"  Test:  {test_frame_metrics['r2']:.4f}")
    
    print("\nInterview-Level RMSE (Mean Aggregation):")
    print(f"  Train: {train_interview_metrics['rmse']:.4f}")
    print(f"  Val:   {val_interview_metrics['rmse']:.4f}")
    print(f"  Test:  {test_interview_metrics['rmse']:.4f}")
    
    print("\nInterview-Level R²:")
    print(f"  Train: {train_interview_metrics['r2']:.4f}")
    print(f"  Val:   {val_interview_metrics['r2']:.4f}")
    print(f"  Test:  {test_interview_metrics['r2']:.4f}")
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
