from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from faster_whisper import WhisperModel, BatchedInferencePipeline

import torch
from pyannote.core import Annotation
from pyannote.core.annotation import Segment
from pyannote.audio import Pipeline
import numpy as np
from pathlib import Path
from pydub import AudioSegment

from IPython import embed
import time
import json
import time
import warnings
warnings.simplefilter("once", DeprecationWarning)

'''This is the class that enables the full pipeline needed for the language preprocessing, being first speaker diarization and inference of which speaker is the client, second german ASR on the client
It takes as input only the source original file in wav format and outputs a json for the corresponding file with the timesteps for the pyannote chunks'''



class Preprocess:
    """
    Pipeline for speaker diarization and ASR on psychotherapy session audio.
    """
    def __init__(self, source_file: str, output_dir: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.source_file = Path(source_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Initialize diarization
        self.diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")

        # Initialize Faster Whisper
        model_size = "small"
        self.model_faster_whisper = WhisperModel(model_size, device=self.device, compute_type="int8")

    def diarize(self) -> None:
        """
        Run speaker diarization on the source file and save RTTM output.
        """
        diarization = self.diarization_pipeline(self.source_file, num_speakers=2)
        output_path = self.output_dir / f"{self.source_file.stem}.rttm"
        with open(output_path, "w") as rttm_file:
            diarization.write_rttm(rttm_file)
        print(f"RTTM file saved to: {output_path}")

    def asr_faster(self) -> list:
        """
        Run ASR on the chunked, saved audio file and return segments.
        """
        client_audio_file = self.output_dir / f"{self.source_file.stem}.wav"
        batched_model = BatchedInferencePipeline(model=self.model_faster_whisper)
        segments, info = batched_model.transcribe(
            client_audio_file,
            beam_size=3,
            language="de",
            condition_on_previous_text="False",
            word_timestamps=False,
            batch_size=8
        )
        return segments

    def chunk_asr(self, both_speakers: bool = False) -> list:
        """
        Chunk audio by diarized segments and run ASR on each chunk.
        Only runs ASR on client segments if both_speakers is False.
        """
        full_rttm = self.output_dir / f"{self.source_file.stem}.rttm"
        annotation = Annotation()
        with open(full_rttm, "r") as f:
            for line in f:
                parts = line.strip().split()
                if parts[0] == "SPEAKER":
                    start = float(parts[3])
                    duration = float(parts[4])
                    speaker = parts[7]
                    segment = Segment(start, start + duration)
                    annotation[segment] = speaker

        # Map speaker labels: most speaking part is client
        mapping = {
            annotation.argmax(): "client",
            [label for label in annotation.labels() if label != annotation.argmax()][0]: "therapist"
        }

        export_path = self.output_dir / f"{self.source_file.stem}.wav"
        audio_pydub = AudioSegment.from_file(self.source_file)
        results = []
        for segment in annotation.itersegments():
            speaker_label = mapping[next(iter(annotation.get_labels(segment)))]
            if not both_speakers and speaker_label == "therapist":
                continue

            start_ms = int(segment.start * 1000)
            end_ms = int(segment.end * 1000)
            if end_ms - start_ms < 1000:
                continue
            chunk = audio_pydub[start_ms:end_ms]
            chunk.export(export_path, format="wav")
            text_chunk = self.asr_faster()

            for t in text_chunk:
                results.append({
                    "text": t.text,
                    "start": start_ms,
                    "end": end_ms,
                    "speaker_id": speaker_label
                })

        return results


if __name__ == "__main__":
    start = time.time()
    pipeline = Preprocess(
        source_file="/mnt/c/users/mlut/OneDrive - ITU/DESKTOP/sync/synchrony/test.wav",
        output_dir="/mnt/c/users/mlut/OneDrive - ITU/DESKTOP/sync/synchrony/files/"
    )
    pipeline.diarize()
    results = pipeline.chunk_asr()

    output_json = Path("/mnt/c/users/mlut/OneDrive - ITU/DESKTOP/sync/synchrony/files/results.json")
    with open(output_json, "w") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)
    print(f"Execution time: {time.time() - start:.2f} seconds") #outputs 1358 seconds, i.e., 22 minutes for the four and a half-minute video



