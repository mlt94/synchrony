import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple


def load_psychotherapy_cot_splits(
    config_path: str = '/home/mlut/synchrony/opentslm/config_opentslm.yaml',
    data_model_path: str = '/home/mlut/synchrony/data_model.yaml',
    window_size_seconds: float = 30.0,
    step_size_seconds: float = 15.0,
    interview_types: List[str] = None,
    max_samples: int = None,  # Add max_samples parameter here
    feature_columns: List[str] = None  # Columns to extract from CSV files
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Load psychotherapy data from data_model.yaml and create train/val/test splits by therapist.
    
    Args:
        config_path: Path to config with therapist splits (train/val/test)
        data_model_path: Path to data_model.yaml
        window_size_seconds: Time window size
        step_size_seconds: Sliding window step
        interview_types: List of interview types to include (default: ['bindung', 'personal', 'wunder'])
        max_samples: Maximum samples per split (for debugging; None = no limit)
        feature_columns: List of column names to extract from CSV files (default: AU columns with '_r' suffix)
    
    Returns:
        Tuple of (train_samples, val_samples, test_samples) as lists of dicts
    """
    if interview_types is None:
        interview_types = ['bindung', 'personal', 'wunder']
    
    # Load configs
    with open(config_path) as f:
        config = yaml.safe_load(f)
    with open(data_model_path) as f:
        data_model = yaml.safe_load(f)
    
    train_therapists = set(config['psychotherapy_splits']['train'])
    val_therapists = set(config['psychotherapy_splits']['val'])
    test_therapists = set(config['psychotherapy_splits']['test'])
    
    # Process splits with early exit if max_samples reached
    train_samples = _process_split(
        data_model['interviews'], train_therapists, interview_types,
        window_size_seconds, step_size_seconds, max_samples, feature_columns
    )
    val_samples = _process_split(
        data_model['interviews'], val_therapists, interview_types,
        window_size_seconds, step_size_seconds, max_samples, feature_columns
    )
    test_samples = _process_split(
        data_model['interviews'], test_therapists, interview_types,
        window_size_seconds, step_size_seconds, max_samples, feature_columns
    )
    
    print(f"[psychotherapy_loader] Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")
    
    return train_samples, val_samples, test_samples


def _process_split(
    interviews: List[Dict],
    therapist_ids: set,
    interview_types: List[str],
    window_size_seconds: float,
    step_size_seconds: float,
    max_samples: int = None,  # Add parameter
    feature_columns: List[str] = None
) -> List[Dict]:
    """Process all interviews for therapists in the given split."""
    samples = []
    
    for entry in interviews:
        # Early exit if we have enough samples
        if max_samples is not None and len(samples) >= max_samples:
            print(f"[debug] Reached max_samples={max_samples}, stopping early")
            break
            
        therapist_id = entry['therapist']['therapist_id']
        if therapist_id not in therapist_ids:
            continue
        
        patient_id = entry['patient']['patient_id']
        baseline = entry.get('baseline', {})
        
        # Iterate through interview types (bindung, personal, wunder)
        for interview_type in interview_types:
            # Check again inside the inner loop
            if max_samples is not None and len(samples) >= max_samples:
                break
                
            if 'types' not in entry or interview_type not in entry['types']:
                continue
            
            type_data = entry['types'][interview_type]
            
            # Get CSV paths
            therapist_csv_path = type_data.get('therapist_openface')
            patient_csv_path = type_data.get('patient_openface')
            
            if not therapist_csv_path or not patient_csv_path:
                print(f"[warn] Missing CSV for {patient_id} ({interview_type})")
                continue
            
            therapist_csv = Path(therapist_csv_path)
            patient_csv = Path(patient_csv_path)
            
            if not therapist_csv.exists() or not patient_csv.exists():
                print(f"[warn] CSV not found: {patient_id} ({interview_type})")
                continue
            
            # Load data
            try:
                therapist_df = pd.read_csv(therapist_csv)
                patient_df = pd.read_csv(patient_csv)
            except Exception as e:
                print(f"[error] Loading CSVs for {patient_id} ({interview_type}): {e}")
                continue
            
            # Extract windows (pass max_samples limit)
            session_samples = _extract_windows(
                therapist_df, patient_df,
                patient_id, therapist_id, interview_type,
                window_size_seconds, step_size_seconds,
                type_data.get('labels', {}),
                baseline,
                max_samples - len(samples) if max_samples else None,  # Remaining budget
                feature_columns
            )
            samples.extend(session_samples)
    
    return samples


def _extract_windows(
    therapist_df: pd.DataFrame,
    patient_df: pd.DataFrame,
    patient_id: str,
    therapist_id: str,
    interview_type: str,
    window_size_seconds: float,
    step_size_seconds: float,
    labels: Dict,
    baseline: Dict,
    max_windows: int = None,
    feature_columns: List[str] = None
) -> List[Dict]:
    """Extract sliding windows from therapist and patient AU data."""
    samples = []
    
    # Normalize column names
    therapist_df.columns = therapist_df.columns.str.strip()
    patient_df.columns = patient_df.columns.str.strip()
    
    # Find timestamp column
    timestamp_col = None
    for col in therapist_df.columns:
        if col.lower() == 'timestamp':
            timestamp_col = col
            break
    
    if timestamp_col is None:
        print(f"[error] No 'timestamp' column found for {patient_id} ({interview_type})")
        return samples
    
    # Convert timestamps to numeric
    therapist_df[timestamp_col] = pd.to_numeric(therapist_df[timestamp_col], errors='coerce')
    patient_df[timestamp_col] = pd.to_numeric(patient_df[timestamp_col], errors='coerce')
    
    # Drop NaN timestamps
    therapist_df = therapist_df.dropna(subset=[timestamp_col])
    patient_df = patient_df.dropna(subset=[timestamp_col])
    
    if therapist_df.empty or patient_df.empty:
        return samples
    
    # Find common time range
    start_time = max(therapist_df[timestamp_col].min(), patient_df[timestamp_col].min())
    end_time = min(therapist_df[timestamp_col].max(), patient_df[timestamp_col].max())
    
    # Extract AU columns (only _r regression values by default, or use specified columns)
    if feature_columns is not None:
        # Use specified columns
        au_cols = [c for c in feature_columns if c in therapist_df.columns and c in patient_df.columns]
        if len(au_cols) < len(feature_columns):
            missing_cols = set(feature_columns) - set(au_cols)
            print(f"[warn] Missing columns for {patient_id} ({interview_type}): {missing_cols}")
    else:
        # Default: extract AU columns with _r suffix
        au_cols = [c for c in therapist_df.columns if 'AU' in c and '_r' in c]
    
    if not au_cols:
        print(f"[warn] No AU columns found for {patient_id} ({interview_type})")
        return samples
    
    # Sliding window
    current_time = start_time
    
    while current_time + window_size_seconds <= end_time:
        # Early exit if we have enough windows
        if max_windows is not None and len(samples) >= max_windows:
            break
            
        window_end = current_time + window_size_seconds
        
        # Extract windows
        therapist_window = therapist_df[
            (therapist_df[timestamp_col] >= current_time) &
            (therapist_df[timestamp_col] < window_end)
        ]
        patient_window = patient_df[
            (patient_df[timestamp_col] >= current_time) &
            (patient_df[timestamp_col] < window_end)
        ]
        
        if len(therapist_window) < 10 or len(patient_window) < 10:
            current_time += step_size_seconds
            continue
        
        # Extract each AU as a separate vector
        therapist_au_vectors = {}
        therapist_au_stats = {}
        
        for au_col in au_cols:
            au_signal = therapist_window[au_col].to_numpy()
            au_mean = float(au_signal.mean())
            au_std = float(au_signal.std())
            
            # # Normalize
            # if au_std > 1e-6:
            #     au_signal_norm = (au_signal - au_mean) / au_std
            # else:
            #     au_signal_norm = au_signal - au_mean
            
            therapist_au_vectors[au_col] = au_signal.tolist()
            therapist_au_stats[au_col] = {"mean": au_mean, "std": au_std}
        
        # Same for patient
        patient_au_vectors = {}
        patient_au_stats = {}
        
        for au_col in au_cols:
            au_signal = patient_window[au_col].to_numpy()
            # au_mean = float(au_signal.mean())
            # au_std = float(au_signal.std())
            
            # # Normalize
            # if au_std > 1e-6:
            #     au_signal_norm = (au_signal - au_mean) / au_std
            # else:
            #     au_signal_norm = au_signal - au_mean
            
            patient_au_vectors[au_col] = au_signal.tolist()
            patient_au_stats[au_col] = {"mean": au_mean, "std": au_std}
        
        # Create sample
        sample = {
            "patient_id": patient_id,
            "therapist_id": therapist_id,
            "interview_type": interview_type,
            "window_start": float(current_time),
            "window_end": float(window_end),
            "therapist_au_vectors": therapist_au_vectors,  # Dict: {AU_name: [normalized_values]}
            "therapist_au_stats": therapist_au_stats,      # Dict: {AU_name: {mean, std}}
            "patient_au_vectors": patient_au_vectors,
            "patient_au_stats": patient_au_stats,
            "au_columns": au_cols,
            "labels": labels,
            "baseline": baseline,
            "rationale": ""
        }
        samples.append(sample)
        
        current_time += step_size_seconds
    
    return samples
