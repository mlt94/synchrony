import yaml
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple


def load_psychotherapy_cot_splits(
    config_path: str = r"/home/mlut/PsyTSLM/opentslm/config_opentslm.yaml",
    data_model_path: str = r'/home/mlut/PsyTSLM/data_model.yaml',
    interview_types: List[str] = None,
    max_samples: int = None,
    feature_columns: List[str] = None,
    max_seq_length: int = 4096
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Load psychotherapy data from data_model.yaml and create train/val/test splits by therapist.
    
    This function orchestrates the complete data loading pipeline:
    1. Loads transcript JSONs (for original_summary context)
    2. Loads answer JSONs (for combined_description targets)
    3. Loads OpenFace CSVs (for AU time series)
    4. Extracts exact time windows and creates samples
    5. Implements two-level caching for fast subsequent loads
    
    Args:
        config_path: Path to config with therapist splits (train/val/test therapist IDs)
        data_model_path: Path to data_model.yaml (contains all file paths)
        interview_types: List of interview types to include (default: ['bindung', 'personal', 'wunder'])
        max_samples: Maximum samples per split (for debugging; None = no limit)
        feature_columns: List of AU column names to extract (default: all AU*_r columns)
                        Example: ['AU04_r'] for debug mode with single feature
        max_seq_length: Maximum sequence length after downsampling (default: 4096 to match model limit)
                       Sequences longer than this will be uniformly downsampled
    
    Returns:
        Tuple of (train_samples, val_samples, test_samples) as lists of dicts.
        Each sample contains: patient_id, therapist_id, interview_type, timing info,
        therapist/patient AU vectors and stats, original_summary, answer text, labels
    """
    import pickle
    import hashlib
    
    # Disk caching: Create cache file based on parameters
    # This avoids reloading and processing JSONs/CSVs on every run (saves minutes)
    cache_key = f"{config_path}_{data_model_path}_{interview_types}_{max_samples}_{feature_columns}_{max_seq_length}"
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
    cache_file = Path(f"psychotherapy_cache_{cache_hash}.pkl")
    
    # Try to load from disk cache first
    if cache_file.exists():
        print(f"[psychotherapy_loader] Loading from cache: {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    if interview_types is None:
        interview_types = ['bindung', 'personal', 'wunder']
    
    if feature_columns is not None:
        print(f"[psychotherapy_loader] Using specified features: {feature_columns}")
    
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
    
    # Check alignment between descriptions and answers
    _check_data_alignment(combined_descriptions, answers)
    
    train_therapists = set(config['psychotherapy_splits']['train'])
    val_therapists = set(config['psychotherapy_splits']['val'])
    test_therapists = set(config['psychotherapy_splits']['test'])
    
    # Process splits with early exit if max_samples reached
    train_samples = _process_split(
        data_model['interviews'], train_therapists, interview_types,
        combined_descriptions, answers, max_samples, feature_columns, max_seq_length
    )
    val_samples = _process_split(
        data_model['interviews'], val_therapists, interview_types,
        combined_descriptions, answers, max_samples, feature_columns, max_seq_length
    )
    test_samples = _process_split(
        data_model['interviews'], test_therapists, interview_types,
        combined_descriptions, answers, max_samples, feature_columns, max_seq_length
    )
    
    print(f"[psychotherapy_loader] Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")
    
    result = (train_samples, val_samples, test_samples)
    
    # Save to cache
    print(f"[psychotherapy_loader] Saving to cache: {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump(result, f)
    
    return result


def _load_combined_descriptions_from_data_model(data_model: Dict, interview_types: List[str]) -> List[Dict]:
    """
    Load transcript JSON files using paths from data_model.yaml 'transcript' key.
    
    These files contain the original_summary field which provides context about
    what was said during each speech turn. Each entry represents one turn with
    precise timing (start_ms, end_ms).
    
    Args:
        data_model: Loaded data_model.yaml dict
        interview_types: List of interview types to load ('bindung', 'personal', 'wunder')
    
    Returns:
        List of all turn entries, each containing: patient_id, therapist_id, 
        interview_type, turn_index, speaker_id, start_ms, end_ms, original_summary
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
    Load answer JSON files using paths from data_model.yaml 'answer' key.
    
    These files contain the combined_description field which is the target text
    that the model should generate - a description of facial expressions in context
    of what was said.
    
    Args:
        data_model: Loaded data_model.yaml dict
        interview_types: List of interview types to load ('bindung', 'personal', 'wunder')
    
    Returns:
        Dict mapping (patient_id, therapist_id, interview_type, turn_index) -> combined_description.
        This allows fast lookup of the answer for each turn.
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
                
                # Track skipped entries for diagnostics
                skipped_count = 0
                
                # Index answers by (patient_id, therapist_id, interview_type, turn_index)
                for entry in data:
                    turn_idx = entry.get('turn_index')
                    combined_desc = entry.get('combined_description', '')
                    
                    if turn_idx is not None and combined_desc:
                        key = (patient_id, therapist_id, itype, turn_idx)
                        answers_dict[key] = combined_desc
                    else:
                        skipped_count += 1
                
                if skipped_count > 0:
                    print(f"[info] Skipped {skipped_count} answer entries from {answer_path.name} (missing turn_index or empty combined_description)")
            
            except Exception as e:
                print(f"[error] Loading {answer_path.name}: {e}")
                continue
    
    print(f"[psychotherapy_loader] Indexed {len(answers_dict)} answers")
    return answers_dict


def _check_data_alignment(combined_descriptions: List[Dict], answers: Dict) -> None:
    """Check for mismatches between turn descriptions and answers."""
    # Build set of description keys
    desc_keys = set()
    duplicate_descs = []
    
    for desc in combined_descriptions:
        key = (desc['patient_id'], desc['therapist_id'], 
               desc['interview_type'], desc['turn_index'])
        if key in desc_keys:
            duplicate_descs.append(key)
        desc_keys.add(key)
    
    # Check for descriptions without answers
    missing_answers = []
    for desc in combined_descriptions:
        key = (desc['patient_id'], desc['therapist_id'], 
               desc['interview_type'], desc['turn_index'])
        if key not in answers:
            missing_answers.append(key)
    
    # Check for answers without descriptions
    answer_keys = set(answers.keys())
    orphaned_answers = answer_keys - desc_keys
    
    # Report findings
    print(f"\n{'='*60}")
    print(f"DATA ALIGNMENT CHECK:")
    print(f"{'='*60}")
    print(f"Total turn descriptions: {len(combined_descriptions)}")
    print(f"Unique turn descriptions: {len(desc_keys)}")
    print(f"Total answers: {len(answers)}")
    
    if duplicate_descs:
        print(f"\n[warn] {len(duplicate_descs)} DUPLICATE turn descriptions found!")
        print(f"[warn] First 5 duplicates: {duplicate_descs[:5]}")
    
    if missing_answers:
        print(f"\n[warn] {len(missing_answers)} turn descriptions have NO matching answers")
        print(f"[warn] First 5 missing: {missing_answers[:5]}")
    
    if orphaned_answers:
        print(f"\n[warn] {len(orphaned_answers)} answers have NO matching turn descriptions")
        print(f"[warn] First 5 orphaned: {list(orphaned_answers)[:5]}")
    
    matched = len(desc_keys & answer_keys)
    print(f"\n✓ Successfully matched pairs: {matched}")
    print(f"{'='*60}\n")


def _process_split(
    interviews: List[Dict],
    therapist_ids: set,
    interview_types: List[str],
    combined_descriptions: List[Dict],
    answers: Dict[Tuple[str, str, str, int], str],
    max_samples: int = None,
    feature_columns: List[str] = None,
    max_seq_length: int = 4096
) -> List[Dict]:
    """
    Process all interviews for therapists in the given split.
    
    For each turn description, this function:
    1. Finds the corresponding interview in data_model
    2. Loads OpenFace CSVs (with caching to avoid repeated I/O)
    3. Extracts the exact time window specified by start_ms/end_ms
    4. Computes AU statistics and creates a sample dict
    
    Args:
        interviews: List of interview entries from data_model.yaml
        therapist_ids: Set of therapist IDs in this split (train/val/test)
        interview_types: Types of interviews to include
        combined_descriptions: List of all turn descriptions (from transcript JSONs)
        answers: Dict mapping turn keys to answer text (from answer JSONs)
        max_samples: Optional limit for debugging
        feature_columns: Optional list of AU columns to extract
        max_seq_length: Maximum sequence length; longer sequences are downsampled (default: 4096)
    
    Returns:
        List of sample dicts ready for training
    """
    samples = []
    # CSV cache: Prevents loading same CSV 60+ times for multiple turns in same interview
    # Key: (therapist_csv_path, patient_csv_path), Value: (therapist_df, patient_df)
    csv_cache = {}
    
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
        
        # Check CSV cache first (avoids repeated loading)
        cache_key = (str(therapist_csv), str(patient_csv))
        if cache_key in csv_cache:
            therapist_df, patient_df = csv_cache[cache_key]
        else:
            # Load CSV files with robust Python engine for better error handling
            try:
                print(f"[debug] Loading CSVs for {patient_id} ({interview_type}): Therapist...")
                therapist_df = pd.read_csv(therapist_csv, engine='python')
                print(f"[debug] Loading CSVs for {patient_id} ({interview_type}): Patient...")
                patient_df = pd.read_csv(patient_csv, engine='python')
                print(f"[debug] Successfully loaded CSVs for {patient_id} ({interview_type})")
                # Cache the loaded DataFrames for subsequent turns in this interview
                csv_cache[cache_key] = (therapist_df, patient_df)
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
            answers,
            max_seq_length
        )
        
        if sample is not None:
            samples.append(sample)
    
    # Report processing results
    processed_count = len([d for d in combined_descriptions if d['therapist_id'] in therapist_ids])
    print(f"[info] Processed {processed_count} descriptions → Created {len(samples)} samples")
    
    return samples


def _extract_single_window(
    therapist_df: pd.DataFrame,
    patient_df: pd.DataFrame,
    desc_entry: Dict,
    labels: Dict,
    baseline: Dict,
    feature_columns: List[str] = None,
    answers: Dict[Tuple[str, str, str, int], str] = None,
    max_seq_length: int = 4096
) -> Dict:
    """
    Extract a single time window from therapist and patient OpenFace AU data.
    
    This function:
    1. Extracts the time window specified by start_ms/end_ms from the turn description
    2. Filters to AU columns (default: all AU*_r regression values)
    3. Extracts AU vectors and computes mean/std for each AU for both people
    4. Downsamples sequences longer than max_seq_length using uniform sampling
    5. Looks up the answer text from the answers dict
    6. Returns a complete sample dict ready for the dataset
    
    Args:
        therapist_df: Therapist OpenFace CSV DataFrame
        patient_df: Patient OpenFace CSV DataFrame
        desc_entry: Turn description with timing, speaker info, original_summary
        labels: Labels from data_model (session-level metadata)
        baseline: Baseline from data_model (session-level metadata)
        feature_columns: Optional list of AU column names (e.g., ['AU04_r'] for debug)
        answers: Dict to lookup answer text by (patient_id, therapist_id, type, turn_idx)
        max_seq_length: Maximum sequence length; longer sequences are downsampled (default: 4096)
    
    Returns:
        Sample dict with all necessary fields, or None if extraction fails
        (e.g., insufficient data points, missing columns, corrupted data)
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
    
    # Drop NaN timestamps - wrap in try/except to handle corrupted data
    try:
        therapist_df = therapist_df.dropna(subset=[timestamp_col])
        patient_df = patient_df.dropna(subset=[timestamp_col])
    except Exception as e:
        print(f"[error] Failed to process data for {desc_entry['patient_id']} ({desc_entry['interview_type']}): {e}")
        return None
    
    if therapist_df.empty or patient_df.empty:
        return None
    
    # Convert milliseconds to seconds for time window extraction
    start_time = desc_entry['start_ms'] / 1000.0
    end_time = desc_entry['end_ms'] / 1000.0
    
    # Determine which AU columns to extract
    if feature_columns is not None:
        # Use specified columns (e.g., for debug mode with single AU)
        au_cols = [c for c in feature_columns if c in therapist_df.columns and c in patient_df.columns]
        if len(au_cols) < len(feature_columns):
            missing_cols = set(feature_columns) - set(au_cols)
            print(f"[warn] Missing columns for {desc_entry['patient_id']} ({desc_entry['interview_type']}): {missing_cols}")
    else:
        # Default: extract all AU regression columns (AU*_r suffix)
        # These are continuous intensity values, more informative than binary presence (_c)
        au_cols = [c for c in therapist_df.columns if 'AU' in c and '_r' in c]
    
    if not au_cols:
        print(f"[warn] No AU columns found for {desc_entry['patient_id']} ({desc_entry['interview_type']})")
        return None
    
    # Extract time windows for this specific speech turn
    # Wrap in try/except to handle corrupted or malformed data gracefully
    try:
        therapist_window = therapist_df[
            (therapist_df[timestamp_col] >= start_time) &
            (therapist_df[timestamp_col] < end_time)
        ]
        patient_window = patient_df[
            (patient_df[timestamp_col] >= start_time) &
            (patient_df[timestamp_col] < end_time)
        ]
    except Exception as e:
        print(f"[error] Failed to extract window for {desc_entry['patient_id']} ({desc_entry['interview_type']}): {e}")
        return None
    
    # Validate we have sufficient data points (minimum 10 frames per person)
    if len(therapist_window) < 10 or len(patient_window) < 10:
        print(f"[warn] Insufficient data points for {desc_entry['patient_id']} ({desc_entry['interview_type']}) turn {desc_entry['turn_index']}: therapist={len(therapist_window)}, patient={len(patient_window)}")
        return None
    
    # Extract each AU as a separate vector with statistics for therapist
    therapist_au_vectors = {}
    therapist_au_stats = {}
    
    try:
        for au_col in au_cols:
            au_signal = therapist_window[au_col].to_numpy()
            au_mean = float(au_signal.mean())
            au_std = float(au_signal.std())
            
            therapist_au_vectors[au_col] = au_signal.tolist()
            therapist_au_stats[au_col] = {"mean": au_mean, "std": au_std}
    except Exception as e:
        print(f"[error] Failed to extract therapist AUs for {desc_entry['patient_id']} ({desc_entry['interview_type']}): {e}")
        return None
    
    # Extract each AU as a separate vector with statistics for patient
    patient_au_vectors = {}
    patient_au_stats = {}
    
    try:
        for au_col in au_cols:
            au_signal = patient_window[au_col].to_numpy()
            au_mean = float(au_signal.mean())
            au_std = float(au_signal.std())
            
            patient_au_vectors[au_col] = au_signal.tolist()
            patient_au_stats[au_col] = {"mean": au_mean, "std": au_std}
    except Exception as e:
        print(f"[error] Failed to extract patient AUs for {desc_entry['patient_id']} ({desc_entry['interview_type']}): {e}")
        return None
    
    # Check time series length and downsample if necessary to maintain model compatibility
    # Model has max_patches limit which translates to max sequence length
    # Use uniform downsampling to preserve temporal structure better than truncation
    
    # Get the length of the first AU (all AUs should have same length)
    sample_au = list(therapist_au_vectors.values())[0] if therapist_au_vectors else []
    actual_length = len(sample_au)
    
    if actual_length > max_seq_length:
        print(f"⚠️  WARNING: Time series length {actual_length} exceeds maximum {max_seq_length}")
        print(f"   Patient: {desc_entry['patient_id']}, Therapist: {desc_entry['therapist_id']}")
        print(f"   Interview: {desc_entry['interview_type']}, Turn: {desc_entry['turn_index']}")
        print(f"   Time window: {desc_entry['start_ms']}ms - {desc_entry['end_ms']}ms ({(desc_entry['end_ms'] - desc_entry['start_ms'])/1000:.1f}s)")
        print(f"   Downsampling all AU vectors from {actual_length} to {max_seq_length} frames using uniform sampling")
        
        # Compute uniform sampling indices to preserve temporal structure
        # This is better than truncation as it maintains representation across the full time window
        indices = np.linspace(0, actual_length - 1, max_seq_length, dtype=int)
        
        # Downsample all therapist AU vectors
        for au_col in therapist_au_vectors:
            original_signal = np.array(therapist_au_vectors[au_col])
            downsampled_signal = original_signal[indices]
            therapist_au_vectors[au_col] = downsampled_signal.tolist()
            # Recalculate stats on downsampled data
            therapist_au_stats[au_col]["mean"] = float(downsampled_signal.mean())
            therapist_au_stats[au_col]["std"] = float(downsampled_signal.std())
        
        # Downsample all patient AU vectors
        for au_col in patient_au_vectors:
            original_signal = np.array(patient_au_vectors[au_col])
            downsampled_signal = original_signal[indices]
            patient_au_vectors[au_col] = downsampled_signal.tolist()
            # Recalculate stats on downsampled data
            patient_au_stats[au_col]["mean"] = float(downsampled_signal.mean())
            patient_au_stats[au_col]["std"] = float(downsampled_signal.std())
    
    # Get the transcript summary from the turn description
    # This provides context about what was said during this speech turn
    original_summary = desc_entry.get('original_summary', desc_entry.get('summary', ''))
    
    # Look up the answer text (target output for model)
    # This is the combined_description from the answer JSON file
    answer_key = (desc_entry['patient_id'], desc_entry['therapist_id'], 
                  desc_entry['interview_type'], desc_entry['turn_index'])
    answer_text = answers.get(answer_key, '') if answers else ''
    
    # Skip samples without answers - these are incomplete data
    if not answer_text:
        # Uncomment for debugging: print(f"[debug] Skipping turn {desc_entry['turn_index']} - no answer")
        return None
    
    # Create complete sample dict with all necessary fields
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
        "answer": answer_text
    }
    
    return sample
