from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from faster_whisper import WhisperModel

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

import warnings
warnings.simplefilter("once", DeprecationWarning)

'''This is the class that enables the full pipeline needed for the language preprocessing, being first speaker diarization and inference of which speaker is the client, second german ASR on the client
It takes as input only the source original file in wav format and outputs a json for the corresponding file with the timesteps for the pyannote chunks'''


class preprocess:
    def __init__(self, source_file, output_dir):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.source_file = Path(source_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        #init diarization
        self.diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")

        #init faster-distill whisper
        model_size = "turbo" #tiny.en, tiny, base.en, base, small.en, small, medium.en, medium, large-v1, large-v2, large-v3, large, distil-large-v2, distil-medium.en, distil-small.en, distil-large-v3, large-v3-turbo, turbo
        self.model_faster_whisper = WhisperModel(model_size, device=self.device, compute_type="float32")

    def diarize(self):
        '''Diarizes the full source file'''
        diarization = self.diarization_pipeline(self.source_file, num_speakers=2)
        output = self.output_dir / f"{self.source_file.stem}.rttm"
        with open(output, "w") as rttm:
            diarization.write_rttm(rttm)
        return None
    
    def asr_faster(self):
        '''Executes on the chunked, saved audio file'''
        client_audio_file = self.output_dir / f"{self.source_file.stem}.wav"
        segments, info = self.model_faster_whisper.transcribe(client_audio_file, beam_size=5, language="de", condition_on_previous_text="False", word_timestamps=True)
        return segments
    
    
    def chunk_asr(self, both_speakers=True):
        '''Requires the full, diarized source file'''
        #first, open the full diarized source file and read it into memory in a nice useful format
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

        #Get tranlation mapping for speaker labels, assuming the most speaking part is the client
        mapping = {annotation.argmax(): "client", [label for label in annotation.labels() if label != annotation.argmax()][0]:"therapist"}

        #extract segments and apply ASR
        export_path = self.output_dir / f"{self.source_file.stem}.wav"
        audio_pydumb = AudioSegment.from_file(self.source_file)
        results = []
        for segment in annotation.itersegments():
            if not both_speakers: #activate if both_speakers=False, and we should only run ASR on the client
                if mapping[next(iter(annotation.get_labels(segment)))] == "therapist": #if we have opened a segment belonging to the therapist, then skip
                    continue

            start_ms = int(segment.start * 1000) #convert to milliseconds as expected by pydub
            end_ms = int(segment.end * 1000)
            if end_ms - start_ms < 1000: #many chunks are small, we do not need to transcribe those which lasts less than 1 second, i.e, 1000 ms
                continue
            chunk = audio_pydumb[start_ms:end_ms]
            chunk.export(export_path, format="wav")        
            text_chunk = preprocess.asr_faster()
            
            for t in text_chunk:
                for word in t.words:    
                    results.append({
                        "word" : word.word,
                        "start" : start_ms + int(word.start * 1000),
                        "end": start_ms + int(word.end * 1000),
                        "speaker_id": mapping[next(iter(annotation.get_labels(segment)))]
                    })
        return results

    
if __name__=="__main__":
    preprocess = preprocess(source_file="/mnt/c/users/mlut/OneDrive - ITU/DESKTOP/sync/synchrony/test.wav", output_dir="/mnt/c/users/mlut/OneDrive - ITU/DESKTOP/sync/synchrony/files/")
    #preprocess.diarize()
    results = preprocess.chunk_asr()
    with open('/mnt/c/users/mlut/OneDrive - ITU/DESKTOP/sync/synchrony/files/results.json', "w") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)

    
    

