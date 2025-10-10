#
# This source file is part of the OpenTSLM open-source project
#
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

import re
import sys
from typing import Dict, Any

from time_series_datasets.har_cot.HARCoTQADataset import HARCoTQADataset


def extract_label_from_prediction(prediction: str) -> str:
    """
    Extract the label from the model's prediction.
    - If 'Answer:' is present, take everything after the last 'Answer:'
    - Otherwise, take the last word
    - Strips whitespace and punctuation
    """
    pred = prediction.strip()
    # Find the last occurrence of 'Answer:' (case-insensitive)
    match = list(re.finditer(r"answer:\s*", pred, re.IGNORECASE))
    if match:
        # Take everything after the last 'Answer:'
        start = match[-1].end()
        label = pred[start:].strip()
    else:
        # Take the last word
        label = pred.split()[-1] if pred.split() else ""
    # Remove trailing punctuation (e.g., period, comma)
    label = re.sub(r"[\.,;:!?]+$", "", label)
    return label.lower()


def evaluate_har_acc(ground_truth: str, prediction: str) -> Dict[str, Any]:
    """
    <<<<<<< HEAD
    <<<<<<< HEAD
        Evaluate HARAccQADataset predictions against ground truth.
    =======
        Evaluate HARCoTQADataset predictions against ground truth.
    >>>>>>> RealLast/ECG-QA-Integration
    =======
        Evaluate HARCoTQADataset predictions against ground truth.
    >>>>>>> main
        Extracts the label from the end of the model's output and compares to ground truth.
    """
    gt_clean = ground_truth.lower().strip()
    pred_label = extract_label_from_prediction(prediction)
    accuracy = int(gt_clean == pred_label)
    return {"accuracy": accuracy}


def main():
    """Main function to run HAR evaluation."""
    if len(sys.argv) != 2:
        print("Usage: python evaluate_har.py <model_name>")
        print("Example: python evaluate_har.py meta-llama/Llama-3.2-1B")
        sys.exit(1)

    model_name = sys.argv[1]

    dataset_classes = [HARCoTQADataset]
    evaluation_functions = {
        "HARCoTQADataset": evaluate_har_acc,
    }
    evaluator = CommonEvaluator()
    results_df = evaluator.evaluate_multiple_models(
        model_names=[model_name],
        dataset_classes=dataset_classes,
        evaluation_functions=evaluation_functions,
        max_samples=None,  # Limit for faster testing, set to None for full evaluation,
        max_new_tokens=400,
    )
    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)
    print(results_df.to_string(index=False))
    return results_df


if __name__ == "__main__":
    main()
