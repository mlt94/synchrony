import argparse
import json
import yaml
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import List, Dict, Optional

'''This is the class that enables the full pipeline needed for the language preprocessing, being first speaker diarization and inference of which speaker is the client, second german ASR on the client
It takes as input only the source original file in wav format and outputs a json for the corresponding file with the timesteps for the pyannote chunks'''

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_diarization(input_wav, output_dir, config):
    from pyannote.audio import Pipeline
    from pathlib import Path
    import time

    start = time.time()
    input_wav = Path(input_wav)
    file_subdir = Path(output_dir) / input_wav.stem
    file_subdir.mkdir(exist_ok=True)

    diarization_pipeline = Pipeline.from_pretrained(config.get("diary_model", "pyannote/speaker-diarization-3.1"))
    diarization = diarization_pipeline(str(input_wav), num_speakers=2)
    rttm_path = file_subdir / f"{input_wav.stem}.rttm"
    with open(rttm_path, "w") as rttm_file:
        diarization.write_rttm(rttm_file)
    print(f"Diarization for {input_wav} done in {time.time() - start:.2f}s. RTTM: {rttm_path}")

def merge_segments(segments: List[Dict], max_pause: float, max_duration: Optional[float] = None) -> List[Dict]:
    """Merge consecutive diarization segments for the same speaker if the gap between them
    is less than or equal to max_pause seconds. Optionally enforce a maximum merged duration.

    segments: list of dicts with keys: start, end, speaker_raw
    Returns new list of merged segments with keys: start, end, speaker_raw
    """
    if not segments:
        return segments
    merged = [segments[0].copy()]
    for seg in segments[1:]:
        last = merged[-1]
        gap = seg['start'] - last['end']
        potential_dur = seg['end'] - last['start']
        if seg['speaker_raw'] == last['speaker_raw'] and gap <= max_pause and (max_duration is None or potential_dur <= max_duration):
            last['end'] = seg['end']
        else:
            merged.append(seg.copy())
    return merged

def run_asr(input_wav, output_dir, config):
    from faster_whisper import WhisperModel, BatchedInferencePipeline
    from pydub import AudioSegment
    from pathlib import Path
    import time
    start = time.time()
    input_wav = Path(input_wav)
    file_subdir = Path(output_dir) / input_wav.stem
    rttm_path = file_subdir / f"{input_wav.stem}.rttm"
    if not rttm_path.exists():
        print(f"RTTM file not found for {input_wav}. Run diarization first.")
        return
    # Parse RTTM lines into list of segments
    raw_segments = []  # each: {start, end, speaker_raw, duration}
    speaker_durations = {}
    with open(rttm_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0] != 'SPEAKER':
                continue
            start_t = float(parts[3])
            dur = float(parts[4])
            end_t = start_t + dur
            spk = parts[7]
            raw_segments.append({'start': start_t, 'end': end_t, 'speaker_raw': spk, 'duration': dur})
            speaker_durations[spk] = speaker_durations.get(spk, 0.0) + dur

    if not raw_segments:
        print(f"No segments parsed from {rttm_path}")
        return

    # Determine client vs therapist: longest total duration assumed client
    sorted_spk = sorted(speaker_durations.items(), key=lambda x: x[1], reverse=True)
    client_speaker = sorted_spk[0][0] if sorted_spk else None
    therapist_speaker = sorted_spk[1][0] if len(sorted_spk) > 1 else None

    # Optionally merge segments before mapping
    if config.get('merge_segments', False):
        max_pause = float(config.get('merge_max_pause', 0.75))
        max_dur_val = config.get('merge_max_duration')
        max_dur = float(max_dur_val) if max_dur_val is not None else None
        pre = len(raw_segments)
        raw_segments = merge_segments(sorted(raw_segments, key=lambda s: s['start']), max_pause=max_pause, max_duration=max_dur)
        print(f"Merged segments {pre} -> {len(raw_segments)} for {input_wav.stem}")

    # Map speakers
    def role(spk_raw):
        if spk_raw == client_speaker:
            return 'client'
        if therapist_speaker and spk_raw == therapist_speaker:
            return 'therapist'
        return 'client'

    segments = [
        {'start': s['start'], 'end': s['end'], 'speaker': role(s['speaker_raw'])}
        for s in sorted(raw_segments, key=lambda s: s['start'])
    ]

    model_size = config.get("whisper_model_size", "small")
    device = config.get("device", "cpu")
    whisper_model = WhisperModel(model_size, device=device, compute_type="int8")
    batched_model = BatchedInferencePipeline(model=whisper_model)
    audio_pydub = AudioSegment.from_file(str(input_wav))
    results = []
    export_path = file_subdir / f"{input_wav.stem}_chunk.wav"
    for seg in segments:
        speaker_label = seg['speaker']
        if not config.get("both_speakers", False) and speaker_label == "therapist":
            continue
        start_ms = int(seg['start'] * 1000)
        end_ms = int(seg['end'] * 1000)
        if end_ms - start_ms < 1000:
            continue
        chunk = audio_pydub[start_ms:end_ms]
        chunk.export(export_path, format="wav")
        asr_segments, _ = batched_model.transcribe(str(export_path), beam_size=3, language="de", condition_on_previous_text="False", word_timestamps=False, batch_size=8)
        for t in asr_segments:
            results.append({
                "text": t.text,
                "start": start_ms,
                "end": end_ms,
                "speaker_id": speaker_label
            })

    output_json = file_subdir / f"results_{input_wav.stem}.json"
    with open(output_json, "w") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)
    print(f"ASR for {input_wav} done in {time.time() - start:.2f}s. Output: {output_json}")

def process_file(input_wav, output_dir, config, stage):
    if stage == "diarization":
        run_diarization(input_wav, output_dir, config)
    elif stage == "asr":
        run_asr(input_wav, output_dir, config)
    elif stage == "full":
        run_diarization(input_wav, output_dir, config)
        run_asr(input_wav, output_dir, config)
    else:
        print(f"Unknown stage: {stage}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing .wav files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--parallel", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--stage", type=str, choices=["diarization", "asr", "full"], default="full", help="Pipeline stage to run")
    parser.add_argument("--merge", dest="merge", action="store_true", help="Enable merging of same-speaker segments separated by short pauses (default if not specified: enabled)")
    parser.add_argument("--no-merge", dest="merge", action="store_false", help="Disable merging of segments")
    parser.set_defaults(merge=True)
    parser.add_argument("--merge-max-pause", type=float, default=0.75, help="Maximum pause (s) between same-speaker segments to merge")
    parser.add_argument("--merge-max-duration", type=float, default=None, help="Optional maximum merged segment duration (s)")
    args = parser.parse_args()

    config = load_config(args.config)
    config['merge_segments'] = args.merge
    config['merge_max_pause'] = args.merge_max_pause
    if args.merge_max_duration is not None:
        config['merge_max_duration'] = args.merge_max_duration

    input_dir = Path(args.input_dir)
    wav_files = list(input_dir.glob("*.wav"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"Found {len(wav_files)} .wav files in {input_dir}")
    with ProcessPoolExecutor(max_workers=args.parallel) as executor:
        futures = [executor.submit(process_file, str(f), str(output_dir), config, args.stage) for f in wav_files]
        for fut in futures:
            fut.result()

if __name__ == "__main__":
    main()



