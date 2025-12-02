from typing import List, Tuple, Literal
import sys
import os
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from prompt.text_time_series_prompt import TextTimeSeriesPrompt

from time_series_datasets.QADataset import QADataset
from time_series_datasets.psychotherapy.psychotherapy_loader import load_psychotherapy_cot_splits

class PsychotherapyCoTQADataset(QADataset):
    """
    Psychotherapy CoT QA Dataset for analyzing therapist-patient facial AU time-series.
    
    This dataset processes synchronized facial action units (AUs) from therapist and patient
    during psychotherapy sessions. Each sample represents one speech turn with:
    - Original transcript summary as context
    - Synchronized AU time series for therapist and patient
    - Combined facial description as the target answer
    
    The dataset handles variable-length sequences within each turn by applying forward-fill
    padding to ensure all AU channels have identical length for tensor operations.
    
    Args:
        split: One of "train", "test", or "validation"
        EOS_TOKEN: End-of-sequence token for the language model
        format_sample_str: Whether to format samples as strings
        time_series_format_function: Optional function to format time series
        max_samples: Maximum number of samples to load (for debugging)
        feature_columns: List of AU column names to use (e.g., ['AU04_r'] for debugging)
        max_seq_length: Maximum sequence length; longer sequences are downsampled (default: 4096)
    """

    def __init__(self, 
                 split: Literal["train", "test", "validation"],
                 EOS_TOKEN: str,
                 format_sample_str: bool = False, 
                 time_series_format_function=None,
                 max_samples: int = None,
                 feature_columns: List[str] = None,
                 max_seq_length: int = 4096):
        self.max_samples = max_samples
        self.feature_columns = feature_columns
        self.max_seq_length = max_seq_length
        super().__init__(
            split=split,
            EOS_TOKEN=EOS_TOKEN, 
            format_sample_str=format_sample_str, 
            time_series_format_function=time_series_format_function
        )

    def _load_splits(self) -> Tuple[List, List, List]:
        """Load train/val/test splits as plain Python lists."""
        train_list, val_list, test_list = load_psychotherapy_cot_splits(
            max_samples=self.max_samples,
            feature_columns=self.feature_columns,
            max_seq_length=self.max_seq_length
        )
        
        return train_list, val_list, test_list

    def _get_answer(self, row) -> str:
        """Get the answer from the data_model yaml file (combined_description)."""
        return row.get("answer", "")

    def _get_pre_prompt(self, row) -> str:
        """Generate the pre-prompt instruction with transcript summary."""
        original_summary = row.get("original_summary", "")
        speaker_id = row.get("speaker_id", "unknown")
        
        prompt = f"""Describe the relational dynamic in this psychotherapy turn. Focus on the speech content and the 2-3 most salient facial Action Units (AUs). Be concise and avoid repetition.

Speech content: {original_summary} (spoken by {speaker_id})

Write a brief paragraph (3-5 sentences maximum):
1. Summarize the speech content in ONE sentence
2. Note only the MOST salient AU patterns that differ between client and therapist
3. End with: "Answer: [equally empathic OR discrepancy]"

Be concise. Do NOT describe every AU. Do NOT repeat phrases.

"""
        return prompt

    def _get_post_prompt(self, row) -> str:
        """Generate the post-prompt."""
        return "Description: "

    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        """
        Convert the time series data into a list of TextTimeSeriesPrompt objects.
        Creates one prompt per AU sequence for both therapist and patient, following HAR pattern.
        """
        # Get AU column names
        au_cols = row.get("au_columns", [])
        
        # Unpack data from dictionaries
        therapist_au_vectors = row["therapist_au_vectors"]
        therapist_au_stats = row["therapist_au_stats"]
        patient_au_vectors = row["patient_au_vectors"]
        patient_au_stats = row["patient_au_stats"]
        
        
        all_signals = []
        all_means = []
        all_stds = []
        all_labels = []
        
        # Add therapist AUs
        for au_name in au_cols:
            signal = therapist_au_vectors[au_name]
            stats = therapist_au_stats[au_name]
            
            all_signals.append(signal)
            all_means.append(stats["mean"])
            all_stds.append(stats["std"])
            all_labels.append(f"therapist for {au_name}")
        
        # Add patient AUs
        for au_name in au_cols:
            signal = patient_au_vectors[au_name]
            stats = patient_au_stats[au_name]
            
            all_signals.append(signal)
            all_means.append(stats["mean"])
            all_stds.append(stats["std"])
            all_labels.append(f"patient for {au_name}")

        # Pad sequences to the same length within each speech turn.
        # This is necessary because therapist and patient AU sequences may have slightly 
        # different lengths (e.g., 662 vs 659 frames) due to timing differences.
        # We use forward-fill (repeat last value) to ensure all signals have identical
        # length for tensor creation while preserving signal characteristics.
        # Note: Length check and truncation for MAX_SUPPORTED_LENGTH happens in the loader
        max_length = max(len(sig) for sig in all_signals)
        
        padded_signals = []
        for signal in all_signals:
            if len(signal) < max_length:
                # Pad with the last value (forward fill)
                padded = signal + [signal[-1]] * (max_length - len(signal))
            else:
                padded = signal
            padded_signals.append(padded)
        
        # Create tensor from padded signals
        series = torch.tensor(padded_signals, dtype=torch.float32)
        
        # Check for invalid data before normalization
        if torch.isnan(series).any() or torch.isinf(series).any():
            print(f"❌ Invalid data detected in Psychotherapy CoT sample")
            print(f"Row keys: {row.keys()}")
            print(f"Series shape: {series.shape}")
            print(f"NaN positions: {torch.isnan(series).nonzero()}")
            print(f"Inf positions: {torch.isinf(series).nonzero()}")
            raise ValueError("Invalid data detected")
        
        # Normalize the tensor using torch operations (matching HAR CoT approach)
        # Note: We use the pre-computed means and stds from the loader for consistency
        # with the original (unpadded) data statistics
        means_tensor = torch.tensor(all_means, dtype=torch.float32).unsqueeze(1)
        stds_tensor = torch.tensor(all_stds, dtype=torch.float32).unsqueeze(1)
        
        # Clamp stds to avoid division by zero (matching HAR CoT)
        min_std = 1e-8
        stds_tensor = torch.clamp(stds_tensor, min=min_std)
        
        # Normalize: (x - mean) / std
        series_norm = (series - means_tensor) / stds_tensor
        
        # Check for invalid data after normalization
        if torch.isnan(series_norm).any() or torch.isinf(series_norm).any():
            print(f"❌ NaN/Inf detected after normalization")
            print(f"Original series shape: {series.shape}")
            print(f"Means: {means_tensor.squeeze()}")
            print(f"Stds: {stds_tensor.squeeze()}")
            print(f"Normalized series: {series_norm}")
            print(f"NaN positions: {torch.isnan(series_norm).nonzero()}")
            print(f"Inf positions: {torch.isinf(series_norm).nonzero()}")
            raise ValueError("NaN/Inf detected after normalization")
        
        # Create prompts (one per AU, using normalized tensor data)
        # Following HAR CoT pattern: convert normalized tensor to list
        prompts = []
        au_idx = 0
        
        # Therapist AUs
        for au_name in au_cols:
            mean = all_means[au_idx]
            std = all_stds[au_idx]
            normalized_series = series_norm[au_idx].tolist()  # Convert tensor to list
            
            text_prompt = f"Facial AU activation for {au_name} (therapist), it has mean {mean:.4f} and std {std:.4f}:"
            prompts.append(TextTimeSeriesPrompt(text_prompt, normalized_series))
            au_idx += 1
        
        # Patient AUs
        for au_name in au_cols:
            mean = all_means[au_idx]
            std = all_stds[au_idx]
            normalized_series = series_norm[au_idx].tolist()  # Convert tensor to list
            
            text_prompt = f"Facial AU activation for {au_name} (patient), it has mean {mean:.4f} and std {std:.4f}:"
            prompts.append(TextTimeSeriesPrompt(text_prompt, normalized_series))
            au_idx += 1
        
        return prompts

    def _format_sample(self, row):
        """Format the sample with additional metadata."""
        sample = super()._format_sample(row)
        sample["patient_id"] = row["patient_id"]
        sample["therapist_id"] = row["therapist_id"]
        sample["interview_type"] = row["interview_type"]
        sample["turn_index"] = row.get("turn_index")
        sample["window_start"] = row["window_start"]
        sample["window_end"] = row["window_end"]
        sample["labels"] = row.get("labels", {})
        sample["baseline"] = row.get("baseline", {})
        sample["answer"] = row.get("answer", {})
        return sample


# Test the dataset
if __name__ == "__main__":
    dataset = PsychotherapyCoTQADataset(split="train", EOS_TOKEN=";", max_samples=5)    
    print(len(dataset))
    # Show sample data
    if len(dataset) > 0:
        print("\n" + "="*50 + "\n")
        print("Sample data from training set:")
        sample = dataset[0]
        print("Sample keys:", sample.keys())
        print("Patient ID:", sample.get("patient_id"))
        print("Therapist ID:", sample.get("therapist_id"))
        print("Interview type:", sample.get("interview_type"))
        print("Window:", f"{sample.get('window_start'):.2f}s - {sample.get('window_end'):.2f}s")
        print("Answer", sample.get("answer"))
        print(sample["time_series_text"][0])
        print(sample["time_series"][0])
        print(sample["time_series_text"][1])
        print(sample["time_series"][1])
  

