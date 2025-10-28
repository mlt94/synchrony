from torch.utils.data import Dataset
from typing import List, Tuple, Literal
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from prompt.text_time_series_prompt import TextTimeSeriesPrompt
from time_series_datasets.QADataset import QADataset
from time_series_datasets.psychotherapy.psychotherapy_loader import load_psychotherapy_cot_splits


class PsychotherapyCoTQADataset(QADataset):
    """
    Own implementation inheriting the QADataset class from the OpenTSLM module.
    """

    def __init__(self, 
                 split: Literal["train", "test", "validation"],
                 EOS_TOKEN: str,
                 format_sample_str: bool = False, time_series_format_function=None,
                 max_samples: int = None):
        self.max_samples = max_samples
        super().__init__(split=split, EOS_TOKEN=EOS_TOKEN, format_sample_str=format_sample_str, time_series_format_function=time_series_format_function)
        

    def _load_splits(self) -> Tuple[List, List, List]:
        """Load train/val/test splits as plain Python lists."""
        train_list, val_list, test_list = load_psychotherapy_cot_splits(
            max_samples=self.max_samples  # Pass it to the loader
        )
        
        return train_list, val_list, test_list

    def _get_answer(self, row) -> str:
        """Get the rationale (answer) for the sample."""
        return ""

    def _get_pre_prompt(self, row) -> str:
        """Generate the pre-prompt instruction."""
        return """You are given facial action unit (AU) time-series for a patient and a therapist during a psychotherapy session. Your task is to analyze and summarize patterns in the time-series, paying special attention to co-occurrence or overlap in facial movements between the two individuals.

Instructions:
- Describe temporal patterns (e.g., synchrony, turn-taking, simultaneous activation).
- Write your reasoning as a single, coherent paragraph without bullet points or section headers.
- Always provide a rationale based on the data.

"""

    def _get_post_prompt(self, row) -> str:
        """Generate the post-prompt."""
        return "Please write your rationale based on the time-series: "

    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        """Create time-series prompts for patient and therapist."""
        prompts = []
        
        au_cols = row.get("au_columns", [])
        
        # Therapist time-series (2D: time × AUs)
        therapist_signal = row["therapist_signal"]  # List of lists
        therapist_mean = row["therapist_mean"]
        therapist_std = row["therapist_std"]
        therapist_text = f"Therapist facial AU time-series (window: {row['window_start']:.2f}s - {row['window_end']:.2f}s, {len(au_cols)} AUs: {', '.join(au_cols)}, sampled at 24 FPS, per-AU normalized)"
        prompts.append(TextTimeSeriesPrompt(therapist_text, therapist_signal))
        
        # Patient time-series (2D: time × AUs)
        patient_signal = row["patient_signal"]
        patient_mean = row["patient_mean"]
        patient_std = row["patient_std"]
        patient_text = f"Patient facial AU time-series (window: {row['window_start']:.2f}s - {row['window_end']:.2f}s, {len(au_cols)} AUs: {', '.join(au_cols)}, sampled at 24 FPS, per-AU normalized)"
        prompts.append(TextTimeSeriesPrompt(patient_text, patient_signal))
        
        return prompts

    def _format_sample(self, row):
        """Format the sample with additional metadata."""
        sample = super()._format_sample(row)
        sample["patient_id"] = row["patient_id"]
        sample["therapist_id"] = row["therapist_id"]
        sample["interview_type"] = row["interview_type"]
        sample["window_start"] = row["window_start"]
        sample["window_end"] = row["window_end"]
        sample["baseline"] = row["baseline"]
        return sample


# Test the dataset
if __name__ == "__main__":
    dataset = PsychotherapyCoTQADataset(split="train", EOS_TOKEN=";", max_samples=5)
    train, val, test = dataset._load_splits()
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    
    # Check first sample
    if len(train) > 0:
        sample = train[0]
        formatted = dataset._format_sample(sample)
        print("\nSample keys:", formatted.keys())
        print("Patient ID:", formatted["patient_id"])
        print("Interview type:", formatted["interview_type"])
        print("Time-series prompts:", len(formatted["time_series_text"]))
