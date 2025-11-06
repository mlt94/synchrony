# Dyadic synchrony in psychotherapy: OpenTSLM with facial movements and transcripts
This repo attempts to solve core problems in psychotherapy research with regards to how a client and a therapist aligns and adjusts their non-verbal behavior over the course of a treatment session. 
The time-series inputs are the action unit estimates of both the client and the therapists facial movements; the text-based input is the transcribed audio from the psychotherapy sessions.
The core questions asked the openTSLM pipeline relates to different measures of relational quality, rappor and synchrony, for which we have client and therapist self-reported labels.

Original [openTSLM paper](https://arxiv.org/abs/2510.02410)

## Repo Layout
```
.
├── README.md
├── opentslm/
│   ├── src/
│   │   ├── time_series_datasets/
│   │   └── ...
│   └── evaluation/
├── source/
│   ├── scripts_language/
│   │   ├── language_pipeline.py
│   │   ├── consistency_check_therapist_client.py
│   │   └── ...
│   ├── scripts_source_data_clean/
│   │   └── ...
│   ├── scripts_face/
│   │   ├── local_dataloader.py
│   │   └── ...
│   └── plotting/



