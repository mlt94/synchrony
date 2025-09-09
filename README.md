# Synchrony investigation
This repo explores how the facial expressions between a client and a therapist can be analyzed by factoring in the audio/text modality. We chunk the facial expressions in terms of who was talking when and for how long, and we also label the speech turn using an LLM. This gives a much richer background for analyzing the facial movements compared to existing tools.

## Features
- 2‑speaker diarization (`pyannote.audio`)
- Heuristic role mapping (most speech = client)
- Faster Whisper ASR (German) on client or both speakers
- Skips very short (<1s) segments
- Optional LLM post‑labeling (sentiment-style: negative / neutral / positive)
- JSON outputs for downstream analysis

## Repo Layout
```
source/scripts/
	language_pipeline.py  # diarization + chunked ASR
	llm.py                # simple labeling of transcribed segments
	utils.py              # helper: merge words into speech turns
files/                  # example audio + generated artifacts
```

