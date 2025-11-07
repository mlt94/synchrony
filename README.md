# Dyadic synchrony in psychotherapy: OpenTSLM with facial movements and transcripts
This repo attempts to solve core problems in psychotherapy research with regards to how a client and a therapist aligns and adjusts their non-verbal behavior over the course of a treatment session. 
The time-series inputs are the action unit estimates of both the client and the therapists facial movements; the text-based input is the transcribed audio from the psychotherapy sessions.
The core questions asked the openTSLM pipeline relates to different measures of relational quality, rappor and synchrony, for which we have client and therapist self-reported labels.

Original [openTSLM paper](https://arxiv.org/abs/2510.02410)


# Description generation (or what is known as rationales in OpenTSLM)
As it is an objective in our research field to go "beyond labels" and gather more nuanced explanatation for how some features relates to an outcome, the openTSLM pipelines is trained to generate rationales supporting the final classification label. As the medical datasets used in the original paper, and also the dataset used this study, do not come with prepared descriptions or rationales underpinning the relationship between the features and the outcome, these must be generated to enable a chain-of-thought style reasoning.
The openTSLM authors use GPT-4o, and gives it either a plot of the time-series data, some context information,the correct label and instructs the model to output a rationale over the relationship.
In the present study, we cannot use GPT-4o as the transcription data is incredibly sensitive and cannot leave the computer on which it is stored. This problem is addressed in the following way:
First, a VLM is prompted to simply describe the pattern observable in the AU estimates for the four selected AUs. This time-series data is visualized in two heatmaps, one for the client and for the patient. Note, that our full pipeline is build to work per client or therapist speech turn. (See `source/opentslm/synchrony_project/generate_time_series_descriptions.py`). This outputs a small text snippet.
Second, an LLM is instructed to combine the previous text snippet with the summaries over the speech turns generated in `source/scripts_language/summarize.py` to one coherent descriptive paragraph resembling the rationales of the original openTSLM. The script to perform this is `opentslm\synchrony_project\combine_transcripts_with_time_series_descriptions.py`.
We intentionally refer to these outputted CoT estimates as descriptions in our study, as the existing theoretical body of knowledge in this particular domain of psychotherapy research, does not offer enough certainty on the relational processes taking place to warrant calling them rationales. We are trying to build that knowledge in the present study. 


## Repo Layout
```
.
├── README.md
├── opentslm/
│   ├── synchrony_project/       # Project-specific code for synchrony analysis
│   │   ├── generate_time_series_rationales.py  # Main pipeline for AU rationale generation
│   │   └── test_simple_generation.py
│   ├── src/                     # OpenTSLM core library
│   │   ├── time_series_datasets/    # Project-specific dataset loaders
│   │   │   └── psychotherapy_dataset.py
│   │   ├── model/
│   │   └── ...
│   └── evaluation/              # Evaluation scripts and results
└── source/                      # Data preprocessing and auxiliary scripts
```



