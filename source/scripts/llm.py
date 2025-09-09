
from transformers import pipeline
import json
import torch
import time

def create_prompt(text: str) -> list:
    """
    Create prompt messages for the LLM labeling task.
    """
    return [
        {"role": "system", "content": "You are an expert psychologist. Please annotate the following german psychotherapy sequences according to one of three categories, negative, neutral or positive. Please only output the annotation category."},
        {"role": "user", "content": text}
    ]

def validate_label(label: str) -> str:
    """
    Ensure label is one of the expected categories.
    """
    valid = {"negative", "neutral", "positive"}
    label = label.strip().lower()
    for v in valid:
        if v in label:
            return v
    return "unknown"

def llm(
    source_path: str = "files/results.json",
    output_path: str = "files/results_labels.json",
    model_name: str = "google/gemma-3-1b-it",
    device: str = "cpu"
) -> None:
    """
    Label transcribed speech turns using an instruction-tuned LLM.
    """
    try:
        pipe = pipeline("text-generation", model=model_name, device=device, torch_dtype=torch.bfloat16)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    try:
        with open(source_path, 'r', encoding='utf-8') as f:
            turns = json.load(f)
    except Exception as e:
        print(f"Error reading input file: {e}")
        return

    labeled_count = 0
    for idx, speech_turn in enumerate(turns):
        messages = create_prompt(speech_turn["text"])
        try:
            outputs = pipe(messages, max_new_tokens=50)
            # Extract label from output
            label = outputs[0]["generated_text"] if "generated_text" in outputs[0] else outputs[0].get("text", "")
            label = validate_label(label)
            turns[idx]["label"] = label
            labeled_count += 1
        except Exception as e:
            print(f"Error labeling turn {idx}: {e}")
            turns[idx]["label"] = "error"

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(turns, f, indent=4, ensure_ascii=False)
        print(f"Labeled {labeled_count} segments. Output saved to {output_path}.")
    except Exception as e:
        print(f"Error writing output file: {e}")


if __name__ == "__main__":
    start = time.time()
    llm()
    print(f"Execution time: {time.time() - start:.2f} seconds")