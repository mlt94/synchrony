import yaml
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple


def load_psychotherapy_cot_splits(
    config_path: str = r"C:\Users\User\Desktop\martins\synchrony\opentslm\config_opentslm.yaml",
    data_model_path: str = r'C:\Users\User\Desktop\martins\synchrony\data_model.yaml',
    interview_types: List[str] = None,
    max_samples: int = None,
    feature_columns: List[str] = None
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Load psychotherapy data from data_model.yaml and create train/val/test splits by therapist.
    Uses exact time windows from combined description JSON files (start_ms to end_ms).
    The paths to combined description files are read from the 'transcript' key in data_model.yaml.
    
    Args:
        config_path: Path to config with therapist splits (train/val/test)
        data_model_path: Path to data_model.yaml (contains both OpenFace paths and transcript/combined description paths)
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
    
    # Load combined descriptions from data_model transcript paths
    print(f"[psychotherapy_loader] Loading combined descriptions from data_model.yaml transcript paths")
    combined_descriptions = _load_combined_descriptions_from_data_model(data_model, interview_types)
    print(f"[psychotherapy_loader] Loaded {len(combined_descriptions)} turn descriptions")
    
    # Load answers from data_model answer paths
    print(f"[psychotherapy_loader] Loading answers from data_model.yaml answer paths")
    answers = _load_answers_from_data_model(data_model, interview_types)
    print(f"[psychotherapy_loader] Loaded {len(answers)} answer descriptions")
    
    train_therapists = set(config['psychotherapy_splits']['train'])
    val_therapists = set(config['psychotherapy_splits']['val'])
    test_therapists = set(config['psychotherapy_splits']['test'])
    
    # Process splits with early exit if max_samples reached
    train_samples = _process_split(
        data_model['interviews'], train_therapists, interview_types,
        combined_descriptions, answers, max_samples, feature_columns
    )
    val_samples = _process_split(
        data_model['interviews'], val_therapists, interview_types,
        combined_descriptions, answers, max_samples, feature_columns
    )
    test_samples = _process_split(
        data_model['interviews'], test_therapists, interview_types,
        combined_descriptions, answers, max_samples, feature_columns
    )
    
    print(f"[psychotherapy_loader] Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")
    
    return train_samples, val_samples, test_samples


def _load_combined_descriptions_from_data_model(data_model: Dict, interview_types: List[str]) -> List[Dict]:
    """
    Load combined description JSON files using paths from data_model.yaml.
    
    Args:
        data_model: Loaded data_model.yaml dict
        interview_types: List of interview types to load
    
    Returns:
        List of all description entries (each with patient_id, therapist_id, interview_type, start_ms, end_ms, etc.)
    """
    all_descriptions = []
    
    for interview in data_model['interviews']:
        therapist_id = interview['therapist']['therapist_id']
        patient_id = interview['patient']['patient_id']
        
        for itype in interview_types:
            if itype not in interview.get('types', {}):
                continue
            
            transcript_path = interview['types'][itype].get('transcript')
            if not transcript_path:
                continue
            
            transcript_path = Path(transcript_path)
            
            # Try to find the file, handling potential encoding issues (e.g., ß -> 0)
            if not transcript_path.exists():
                # Try replacing ß with 0 (common encoding issue in German filenames)
                alt_path = Path(str(transcript_path).replace('ß', '0'))
                if alt_path.exists():
                    transcript_path = alt_path
                else:
                    print(f"[warn] Transcript not found: {transcript_path}")
                    continue
            
            # Load this combined description file
            try:
                with open(transcript_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Add therapist_id and patient_id to each entry if not present
                for entry in data:
                    if 'therapist_id' not in entry:
                        entry['therapist_id'] = therapist_id
                    if 'patient_id' not in entry:
                        entry['patient_id'] = patient_id
                    if 'interview_type' not in entry:
                        entry['interview_type'] = itype
                
                all_descriptions.extend(data)
            
            except Exception as e:
                print(f"[error] Loading {transcript_path.name}: {e}")
                continue
    
    print(f"[psychotherapy_loader] Loaded {len(all_descriptions)} total turn descriptions")
    return all_descriptions


def _load_answers_from_data_model(data_model: Dict, interview_types: List[str]) -> Dict[Tuple[str, str, str, int], str]:
    """
    Load answer JSON files using paths from data_model.yaml.
    
    Args:
        data_model: Loaded data_model.yaml dict
        interview_types: List of interview types to load
    
    Returns:
        Dict mapping (patient_id, therapist_id, interview_type, turn_index) -> combined_description
    """
    answers_dict = {}
    
    for interview in data_model['interviews']:
        therapist_id = interview['therapist']['therapist_id']
        patient_id = interview['patient']['patient_id']
        
        for itype in interview_types:
            if itype not in interview.get('types', {}):
                continue
            
            answer_path = interview['types'][itype].get('answer')
            if not answer_path:
                continue
            
            answer_path = Path(answer_path)
            
            # Try to find the file, handling potential encoding issues (e.g., ß -> 0)
            if not answer_path.exists():
                # Try replacing ß with 0 (common encoding issue in German filenames)
                alt_path = Path(str(answer_path).replace('ß', '0'))
                if alt_path.exists():
                    answer_path = alt_path
                else:
                    print(f"[warn] Answer file not found: {answer_path}")
                    continue
            
            # Load this answer file
            try:
                with open(answer_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Index answers by (patient_id, therapist_id, interview_type, turn_index)
                for entry in data:
                    turn_idx = entry.get('turn_index')
                    combined_desc = entry.get('combined_description', '')
                    
                    if turn_idx is not None and combined_desc:
                        key = (patient_id, therapist_id, itype, turn_idx)
                        answers_dict[key] = combined_desc
            
            except Exception as e:
                print(f"[error] Loading {answer_path.name}: {e}")
                continue
    
    print(f"[psychotherapy_loader] Indexed {len(answers_dict)} answers")
    return answers_dict


def _process_split(
    interviews: List[Dict],
    therapist_ids: set,
    interview_types: List[str],
    combined_descriptions: List[Dict],
    answers: Dict[Tuple[str, str, str, int], str],
    max_samples: int = None,
    feature_columns: List[str] = None
) -> List[Dict]:
    """Process all interviews for therapists in the given split using exact time windows from combined_descriptions."""
    samples = []
    
    for desc_entry in combined_descriptions:
        # Early exit if we have enough samples
        if max_samples is not None and len(samples) >= max_samples:
            print(f"[debug] Reached max_samples={max_samples}, stopping early")
            break
        
        patient_id = desc_entry['patient_id']
        therapist_id = desc_entry['therapist_id']
        interview_type = desc_entry['interview_type']
        
        # Check if this therapist is in the current split
        if therapist_id not in therapist_ids:
            continue
        
        # Check if this interview type should be included
        if interview_type not in interview_types:
            continue
        
        # Find the corresponding interview in data_model
        interview_data = None
        for entry in interviews:
            if (entry['patient']['patient_id'] == patient_id and
                entry['therapist']['therapist_id'] == therapist_id):
                interview_data = entry
                break
        
        if interview_data is None:
            print(f"[warn] Interview not found in data_model: {patient_id}/{therapist_id}")
            continue
        
        if interview_type not in interview_data.get('types', {}):
            continue
        
        type_data = interview_data['types'][interview_type]
        
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
        
        # Extract the specific window from this description entry
        sample = _extract_single_window(
            therapist_df, patient_df,
            desc_entry,
            type_data.get('labels', {}),
            interview_data.get('baseline', {}),
            feature_columns,
            answers
        )
        
        if sample is not None:
            samples.append(sample)
    
    return samples


def _extract_single_window(
    therapist_df: pd.DataFrame,
    patient_df: pd.DataFrame,
    desc_entry: Dict,
    labels: Dict,
    baseline: Dict,
    feature_columns: List[str] = None,
    answers: Dict[Tuple[str, str, str, int], str] = None
) -> Dict:
    """
    Extract a single time window from therapist and patient AU data based on the
    exact start_ms and end_ms from the combined description entry.
    
    Args:
        therapist_df: Therapist OpenFace CSV data
        patient_df: Patient OpenFace CSV data
        desc_entry: Description entry with patient_id, therapist_id, start_ms, end_ms, combined_description, etc.
        labels: Labels from data_model
        baseline: Baseline from data_model
        feature_columns: List of column names to extract (default: AU columns with '_r' suffix)
    
    Returns:
        Sample dict with AU vectors and rationale, or None if extraction fails
    """
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
        print(f"[error] No 'timestamp' column found for {desc_entry['patient_id']} ({desc_entry['interview_type']})")
        return None
    
    # Convert timestamps to numeric (in seconds)
    therapist_df[timestamp_col] = pd.to_numeric(therapist_df[timestamp_col], errors='coerce')
    patient_df[timestamp_col] = pd.to_numeric(patient_df[timestamp_col], errors='coerce')
    
    # Drop NaN timestamps
    therapist_df = therapist_df.dropna(subset=[timestamp_col])
    patient_df = patient_df.dropna(subset=[timestamp_col])
    
    if therapist_df.empty or patient_df.empty:
        return None
    
    # Convert milliseconds to seconds
    start_time = desc_entry['start_ms'] / 1000.0
    end_time = desc_entry['end_ms'] / 1000.0
    
    # Extract AU columns (only _r regression values by default, or use specified columns)
    if feature_columns is not None:
        # Use specified columns
        au_cols = [c for c in feature_columns if c in therapist_df.columns and c in patient_df.columns]
        if len(au_cols) < len(feature_columns):
            missing_cols = set(feature_columns) - set(au_cols)
            print(f"[warn] Missing columns for {desc_entry['patient_id']} ({desc_entry['interview_type']}): {missing_cols}")
    else:
        # Default: extract AU columns with _r suffix
        au_cols = [c for c in therapist_df.columns if 'AU' in c and '_r' in c]
    
    if not au_cols:
        print(f"[warn] No AU columns found for {desc_entry['patient_id']} ({desc_entry['interview_type']})")
        return None
    
    # Extract windows for this specific time range
    therapist_window = therapist_df[
        (therapist_df[timestamp_col] >= start_time) &
        (therapist_df[timestamp_col] < end_time)
    ]
    patient_window = patient_df[
        (patient_df[timestamp_col] >= start_time) &
        (patient_df[timestamp_col] < end_time)
    ]
    
    if len(therapist_window) < 10 or len(patient_window) < 10:
        print(f"[warn] Insufficient data points for {desc_entry['patient_id']} ({desc_entry['interview_type']}) turn {desc_entry['turn_index']}: therapist={len(therapist_window)}, patient={len(patient_window)}")
        return None
    
    # Extract each AU as a separate vector for therapist
    therapist_au_vectors = {}
    therapist_au_stats = {}
    
    for au_col in au_cols:
        au_signal = therapist_window[au_col].to_numpy()
        au_mean = float(au_signal.mean())
        au_std = float(au_signal.std())
        
        therapist_au_vectors[au_col] = au_signal.tolist()
        therapist_au_stats[au_col] = {"mean": au_mean, "std": au_std}
    
    # Same for patient
    patient_au_vectors = {}
    patient_au_stats = {}
    
    for au_col in au_cols:
        au_signal = patient_window[au_col].to_numpy()
        au_mean = float(au_signal.mean())
        au_std = float(au_signal.std())
        
        patient_au_vectors[au_col] = au_signal.tolist()
        patient_au_stats[au_col] = {"mean": au_mean, "std": au_std}
    
    # Get the transcript summary (original_summary from transcript JSON)
    original_summary = desc_entry.get('original_summary', desc_entry.get('summary', ''))
    
    # Get the answer (combined_description from answer JSON)
    answer_key = (desc_entry['patient_id'], desc_entry['therapist_id'], 
                  desc_entry['interview_type'], desc_entry['turn_index'])
    answer_text = answers.get(answer_key, '') if answers else ''
    
    # Create sample with data from transcript and answer files
    sample = {
        "patient_id": desc_entry['patient_id'],
        "therapist_id": desc_entry['therapist_id'],
        "interview_type": desc_entry['interview_type'],
        "turn_index": desc_entry['turn_index'],
        "speaker_id": desc_entry['speaker_id'],
        "window_start": start_time,
        "window_end": end_time,
        "start_ms": desc_entry['start_ms'],
        "end_ms": desc_entry['end_ms'],
        "therapist_au_vectors": therapist_au_vectors,
        "therapist_au_stats": therapist_au_stats,
        "patient_au_vectors": patient_au_vectors,
        "patient_au_stats": patient_au_stats,
        "au_columns": au_cols,
        "labels": labels,
        "baseline": baseline,
        "original_summary": original_summary,
        "answer": answer_text,
        "rationale": answer_text  # Use answer as rationale for backward compatibility
    }
    
    return sample
