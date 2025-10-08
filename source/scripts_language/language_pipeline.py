import argparse
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from utils import load_config

'''This is the class that enables the full pipeline needed for the language preprocessing, being first speaker diarization and inference of which speaker is the client, second german ASR on the client
It takes as input only the source original file in wav format and outputs a json for the corresponding file with the timesteps for the pyannote chunks'''


def run_diarization(input_wav, output_dir, config):
    from pyannote.audio import Pipeline
    from pathlib import Path
    import time

    start = time.time()
    input_wav = Path(input_wav)
    file_subdir = Path(output_dir) / input_wav.stem
    file_subdir.mkdir(exist_ok=True)

    target_device = config.get("device", "cpu")
    print(f"[language_pipeline] Running diarization on device: {target_device}")

    try:
        diarization_pipeline = Pipeline.from_pretrained(
            config.get("diary_model", "pyannote/speaker-diarization-3.1"),
            use_auth_token=config.get("hf_token"),
        )
        # Move pipeline to the configured device (cuda/cpu) if available
        try:
            diarization_pipeline.to(target_device)
        except Exception:
            # If .to is not supported by the pipeline, continue with default
            pass
    except Exception as e:
        print(
            "Failed to load pyannote pipeline. Ensure you have access and are authenticated. "
            "If using gated repos, set hf_token in config.yaml or the HUGGINGFACE_HUB_TOKEN env var.\n"
            f"Error: {e}"
        )
        return
    diarization = diarization_pipeline(str(input_wav), num_speakers=2)
    rttm_path = file_subdir / f"{input_wav.stem}.rttm"
    with open(rttm_path, "w") as rttm_file:
        diarization.write_rttm(rttm_file)
    print(f"Diarization for {input_wav} done in {time.time() - start:.2f}s. RTTM: {rttm_path}")

def run_asr(input_wav, output_dir, config):
    from pyannote.core import Annotation, Segment
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

    annotation = Annotation()
    with open(rttm_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if parts[0] == "SPEAKER":
                start_t = float(parts[3])
                duration = float(parts[4])
                speaker = parts[7]
                segment = Segment(start_t, start_t + duration)
                annotation[segment] = speaker
    mapping = {
        annotation.argmax(): "client",
        [label for label in annotation.labels() if label != annotation.argmax()][0]: "therapist"
    }

    model_size = config.get("whisper_model_size", "small")
    device = config.get("device", "cuda")
    compute_type = config.get("whisper_compute_type", "int8")
    print(f"[language_pipeline] Running ASR on device: {device} (compute_type={compute_type})")
    whisper_model = WhisperModel(model_size, device=device, compute_type=compute_type)
    batched_model = BatchedInferencePipeline(model=whisper_model)
    audio_pydub = AudioSegment.from_file(str(input_wav))
    results = []
    export_path = file_subdir / f"{input_wav.stem}_chunk.wav"
    for segment in annotation.itersegments():
        speaker_label = mapping[next(iter(annotation.get_labels(segment)))]
        if not config.get("both_speakers", False) and speaker_label == "therapist":
            continue
        start_ms = int(segment.start * 1000)
        end_ms = int(segment.end * 1000)
        if end_ms - start_ms < 1000:
            continue
        chunk = audio_pydub[start_ms:end_ms]
        chunk.export(export_path, format="wav")
        segments, _ = batched_model.transcribe(
            str(export_path),
            beam_size=3,
            language="de",
            condition_on_previous_text=False,
            word_timestamps=False,
            batch_size=8,
        )
        for t in segments:
            results.append({
                "text": t.text,
                "start": start_ms,
                "end": end_ms,
                "speaker_id": speaker_label
            })

    output_json = file_subdir / f"results_{input_wav.stem}.json"
    with open(output_json, "w", encoding="utf-8") as file:
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
    parser.add_argument("--config", type=str, default="config_language.yaml", help="Path to config file")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing .wav files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--parallel", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--stage", type=str, choices=["diarization", "asr", "full"], default="full", help="Pipeline stage to run")
    args = parser.parse_args()

    config = load_config(args.config) or {}

    requested = str(config.get("device", "auto") or "auto").lower()
    cuda_available = False
    if requested in {"auto", ""}:
        try:
            import torch

            cuda_available = torch.cuda.is_available()
        except Exception:
            cuda_available = False
        device = "cuda" if cuda_available else "cpu"
    else:
        device = config.get("device", "cpu")
        try:
            import torch

            cuda_available = torch.cuda.is_available()
        except Exception:
            cuda_available = False

    config["device"] = device
    print(f"[language_pipeline] Selected device: {device} | CUDA available: {cuda_available}")
    input_dir = Path(args.input_dir)
    wav_files = list(input_dir.glob("*.wav"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"Found {len(wav_files)} .wav files in {input_dir}")
    with ProcessPoolExecutor(max_workers=args.parallel) as executor:
        futures = [executor.submit(process_file, str(f), str(output_dir), config, args.stage) for f in wav_files]
        for fut in futures:
            fut.result()  # Optionally handle exceptions

if __name__ == "__main__":
    main()



