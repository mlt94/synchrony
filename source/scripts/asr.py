from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
from pydub import AudioSegment
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    device=device,
)

# Load the already downloaded audio file
audio_path = '/home/mlut/synchrony/test.mp3'  # Replace with the path to your audio file
# Use the ASR pipeline to transcribe the audio
result = pipe(audio_path)

# Output the transcription
print(result["text"])
