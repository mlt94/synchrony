# Dyadic synchrony in psychotherapy: OpenTSLM with facial movements and transcripts
This repo attempts to solve core problems in psychotherapy research with regards to how a client and a therapist aligns and adjusts their non-verbal behavior over the course of a treatment session. 
The time-series inputs are the action unit estimates of both the client and the therapists facial movements; the text-based input is the transcribed audio from the psychotherapy sessions.
The core questions asked the openTSLM pipeline relates to different measures of relational quality, rappor and synchrony, for which we have client and therapist self-reported labels.

## The OpenTSLM module
I have forked the OpenTSLM module and are working on custom dataset classes. Open questions relates to generating the rationales; potentially I will follow the workflow of the authors, but I have a feeling that another approach can be utilized for this specific problem-domain.
Having GPT4o generate rationales makes very good sense in a medical domain; they can be easily validated, and a nessesary to ensure a viable chain-of-thought have occured. For the pscychotherapy domain, I am thinking that the rationales are inherently the words uttered by the therapist and the client; they are "labeling" (or providing the rationale) for their non-verbal behavior themselves. This is not specific to psychotherapy; all people do that all the time. 
More thinking is needed on this.

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


