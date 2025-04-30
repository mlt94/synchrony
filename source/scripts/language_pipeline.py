from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
from pyannote.core import Annotation
from pyannote.core.annotation import Segment
from pyannote.audio import Pipeline
import numpy as np
from pathlib import Path
from pydub import AudioSegment

import time

import warnings
warnings.simplefilter("once", DeprecationWarning)

'''This is the class that enables the full pipeline needed for the language preprocessing, being first speaker diarization and inference of which speaker is the client, second german ASR on the client and third annotations according to PACS with some LLM
It takes as input only the source original file in either mp3 or wav format and outputs an excel file for the corresponding file with the timesteps for the pyannote chunks, the text and the PACS LLM classification'''


class preprocess:
    def __init__(self, source_file, output_dir):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.source_file = Path(source_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        #init diarization
        self.diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")

        #init ASR
        model_id = "openai/whisper-large-v3"
        asr = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, low_cpu_mem_usage=True, use_safetensors=True).to(self.device)
        processor = AutoProcessor.from_pretrained(model_id)
        self.asr_pipe = pipeline(
            "automatic-speech-recognition",
            model=asr,
            tokenizer=processor.tokenizer,
            return_timestamps=True,
            feature_extractor=processor.feature_extractor,
            device=self.device,
        )

    def diarize(self):
        '''Diarizes the full source file'''
        diarization = self.diarization_pipeline(self.source_file)
        output = self.output_dir / f"{self.source_file.stem}.rttm"
        with open(output, "w") as rttm:
            diarization.write_rttm(rttm)
        return None
    
    def chunk_client(self):
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

        #get annotations of dominant speaker, which we will assume is client for most of the time                    
        client_speech_turns = annotation.subset(set([annotation.argmax()])) 


        #extract the chunks belonging to dominant speaker
        export_path = self.output_dir / f"{self.source_file.stem}_client.wav"
        audio_pydumb = AudioSegment.from_file(self.source_file)
        chunks = []
        for segment in client_speech_turns.itersegments():
            start_ms = int(segment.start * 1000) #convert to milliseconds as expected by pydub
            end_ms = int(segment.end * 1000)
            chunk = audio_pydumb[start_ms:end_ms]
            chunks.append(chunk)

            #execute asr here on each chunk
            chunk.export(export_path, format="wav")
            results = preprocess.asr()
            text = results["text"]
            print(f"From {start_ms} to {end_ms}: {text}")
            


        #write the chunks out to rttm format so we can use them again when making the final excel
        output_rttm = self.output_dir / f"{self.source_file.stem}_client.rttm"
        with open(output_rttm, "w") as f:
            client_speech_turns.write_rttm(f)

        #export the chunked audio file
        client_audio = sum(chunks)
        client_audio.export(export_path, format="wav")
       
        return None
    

    def asr(self):
        '''Executes on the chunked, client audio file'''
        client_audio_file = self.output_dir / f"{self.source_file.stem}_client.wav"
        result = self.asr_pipe(str(client_audio_file))
        return result
    
    
if __name__=="__main__":
    preprocess = preprocess(source_file="/mnt/c/users/mlut/OneDrive - ITU/DESKTOP/sync/synchrony/test.wav", output_dir="/mnt/c/users/mlut/OneDrive - ITU/DESKTOP/sync/synchrony/files/")
    #preprocess.diarize()
    preprocess.chunk_client()
    
    

