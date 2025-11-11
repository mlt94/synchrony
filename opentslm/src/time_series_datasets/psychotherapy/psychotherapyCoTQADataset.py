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
    Psychotherapy CoT QA Dataset for therapist-patient facial AU time-series.
    Follows the same pattern as HARCoTQADataset for multi-feature handling.
    """

    def __init__(self, 
                 split: Literal["train", "test", "validation"],
                 EOS_TOKEN: str,
                 format_sample_str: bool = False, 
                 time_series_format_function=None,
                 max_samples: int = None,
                 feature_columns: List[str] = None):
        self.max_samples = max_samples
        self.feature_columns = feature_columns
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
            feature_columns=self.feature_columns
        )
        
        return train_list, val_list, test_list

    def _get_answer(self, row) -> str:
        """Get the answer from the answer JSON file (combined_description)."""
        return row.get("answer", "")

    def _get_pre_prompt(self, row) -> str:
        """Generate the pre-prompt instruction with transcript summary."""
        speaker_id = row.get("speaker_id", "unknown")
        start_ms = row.get("start_ms", 0)
        end_ms = row.get("end_ms", 0)
        original_summary = row.get("original_summary", "")
        
        # Determine speaker role
        therapist_id = row.get("therapist_id", "")
        speaker_role = "therapist" if speaker_id == therapist_id else "patient"
        
        # Format time in seconds
        start_sec = start_ms / 1000.0
        end_sec = end_ms / 1000.0
        
        prompt = f"""You are given facial action unit data in 17 dimensions for both a therapist and patient during a psychotherapy session.

Context: During this speech turn (from {start_sec:.1f}s to {end_sec:.1f}s), the {speaker_role} said:
"{original_summary}"

Your task is to describe the associations between what was said and the facial expressions.
Instructions:
- Begin by describing the speech content very briefly
- Then briefly note any salient facial Action Units (AUs) that stand out — do not over-analyze every AU, only mention the most relevant ones, and dont write what facial movement the AU references.
- Do **not** over-analyze or speculate; be very true to what is actually present in the data available. 
- Do not reflect on the emotional bond, synchrony or similar aspects of the interaction.
- Write your description as a single, natural paragraph — do not use bullet points, numbered steps, new lines or section headings.

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

        series = torch.tensor(all_signals, dtype=torch.float32)
        
        # Check for invalid data (from HAR)
        if torch.isnan(series).any() or torch.isinf(series).any():
            print(f"❌ Invalid data detected in Psychotherapy CoT sample")
            print(f"Row keys: {row.keys()}")
            print(f"Series shape: {series.shape}")
            print(f"NaN positions: {torch.isnan(series).nonzero()}")
            print(f"Inf positions: {torch.isinf(series).nonzero()}")
            raise ValueError("Invalid data detected")
        
        # Create prompts (one per AU, properly normalized)
        prompts = []
        au_idx = 0
        
        # Therapist AUs
        for au_name in au_cols:
            mean = all_means[au_idx]
            std = all_stds[au_idx]
            time_series = all_signals[au_idx]
            
            # Normalize: (x - mean) / std
            normalized_series = [(val - mean) / (std + 1e-8) for val in time_series]
            
            text_prompt = f"Facial AU activation for {au_name} (therapist)"
            prompts.append(TextTimeSeriesPrompt(text_prompt, normalized_series))
            au_idx += 1
        
        # Patient AUs
        for au_name in au_cols:
            mean = all_means[au_idx]
            std = all_stds[au_idx]
            time_series = all_signals[au_idx]
            
            # Normalize: (x - mean) / std
            normalized_series = [(val - mean) / (std + 1e-8) for val in time_series]
            
            text_prompt = f"Facial AU activation for {au_name} (patient)"
            prompts.append(TextTimeSeriesPrompt(text_prompt, normalized_series))
            au_idx += 1
        
        return prompts

    def _format_sample(self, row):
        """Format the sample with additional metadata."""
        sample = super()._format_sample(row)
        sample["patient_id"] = row["patient_id"]
        sample["therapist_id"] = row["therapist_id"]
        sample["interview_type"] = row["interview_type"]
        sample["window_start"] = row["window_start"]
        sample["window_end"] = row["window_end"]
        sample["labels"] = row.get("labels", {})
        sample["baseline"] = row.get("baseline", {})
        sample["answer"] = row.get("answer", {})
        return sample


# Test the dataset
if __name__ == "__main__":
    dataset = PsychotherapyCoTQADataset(split="train", EOS_TOKEN=";", max_samples=5)
    dataset_val = PsychotherapyCoTQADataset(split="validation", EOS_TOKEN=";", max_samples=5)
    dataset_test = PsychotherapyCoTQADataset(split="test", EOS_TOKEN=";", max_samples=5)
    
    print(f"Dataset sizes: Train: {len(dataset)}, Validation: {len(dataset_val)}, Test: {len(dataset_test)}")
    
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
        print(sample["time_series_text"])
        #print(sample["time_series"][0])
