import json

def extract_speech_turns_from_file(json_path):
    """
    MLT: Code generated with ChatGPT
    Extracts concatenated speech turns from a JSON file of word objects, 
    grouping consecutive words by speaker.

    Args:
        json_path (str): Path to the results.json file.

    Returns:
        List[Dict]: A list of dictionaries, each representing a speech turn with:
            - "speaker_id": ID of the speaker
            - "text": Concatenated words of the speaker's turn
            - "start": Start time of the turn
            - "end": End time of the turn
    """
    # Step 1: Load the JSON file
    with open(json_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    # Step 2: Initialize variables for grouping turns
    speech_turns = []
    current_speaker = None
    current_turn_words = []
    current_start = None

    # Step 3: Iterate through each word entry
    for w in results:
        speaker = w["speaker_id"]
        word = w["word"].strip()
        start = w["start"]
        end = w["end"]

        if speaker != current_speaker:
            # Save the previous turn if words were collected
            if current_turn_words:
                speech_turns.append({
                    "speaker_id": current_speaker,
                    "text": " ".join(current_turn_words),
                    "start": current_start,
                    "end": previous_end
                })
            
            # Start a new turn for the new speaker
            current_speaker = speaker
            current_turn_words = [word]
            current_start = start
        else:
            # Continue with the same speaker's turn
            current_turn_words.append(word)
        
        previous_end = end

    # Append the last collected turn
    if current_turn_words:
        speech_turns.append({
            "speaker_id": current_speaker,
            "text": " ".join(current_turn_words),
            "start": current_start,
            "end": previous_end
        })

    return speech_turns
