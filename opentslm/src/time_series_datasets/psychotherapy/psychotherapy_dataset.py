from torch.utils.data import Dataset
from typing import List, Tuple, Literal
from prompt.text_time_series_prompt import TextTimeSeriesPrompt
from time_series_datasets.QADataset import QADataset
import yaml

class PsychotherapyQADataset(QADataset):
    """
    Own implementation inheriting the QADataset class from the OpenTSLM module.
    """

    def __init__(self, split: Literal["train", "test", "validation"], EOS_TOKEN: str,
                 format_sample_str: bool = False, time_series_format_function=None,
                 max_samples: int = None):
        super().__init__(split, EOS_TOKEN, format_sample_str, time_series_format_function)

    def _load_splits(self) -> Tuple[Dataset, Dataset, Dataset]:
        """Takes as input therapist identifiers from a config file."""
        config = yaml.safe_load(open('config_opentslm.yaml')) 
        train_splits = config['psychotherapy_splits']['train']
        val_splits = config['psychotherapy_splits']['val']
        test_splits = config['psychotherapy_splits']['test']
        return train_splits, val_splits, test_splits

    def _get_answer(self, row) -> str:
        return row.get("answer", "")

    def _get_pre_prompt(self, row) -> str:
        return f"Question: {row.get('question', '')}"

    def _get_post_prompt(self, row) -> str:
        return "Answer:"

    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        # TODO: Load AU data from CSVs and create prompts text: A descriptive label for the ECG lead (e.g., "ECG Lead I - sampled at ~100Hz, normalized (mean=..., std=...)").
              #time_series: The actual numerical time-series data (a list of normalized signal values for that lead).
        prompts = []
        #prompts.append(TextTimeSeriesPrompt("text, normalized with mean (x) and std(y)", normalized_AU_signal.tolist())
        return prompts