import json
from itertools import groupby
from typing import List, Dict

def extract_speech_turns_from_file(json_path: str) -> List[Dict]:
    """
    Extracts speech turns from a JSON file, grouping consecutive words by speaker.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    speech_turns = []
    for speaker, group in groupby(results, key=lambda w: w["speaker_id"]):
        group = list(group)
        speech_turns.append({
            "speaker_id": speaker,
            "text": " ".join(w["text"].strip() for w in group),
            "start": group[0]["start"],
            "end": group[-1]["end"]
        })
    return speech_turns
