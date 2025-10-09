'''
Objective: How does the therapist align their facial movements to those of the patient?

Method: Cross attention transformer implementation // multi-dimensional time-series

Therapist --> Q_t, K_t, V_t
Patient --> Q_p, K_p, V_p

Confirmation point 1: Confirm that what we want is softmax(Q_t * K_p / sqrt^dimension) * V_p
8/10 --> normalize AUs (preprocesessing)
8/10 --> should we perform some initial transformation to the AU's similar to tokenization? investigate openTSLM


Confirmation point 2: We will train one encoder-block with the patient embeddings,
one decoder block with the therapist embedding and cross-attention as per above
--> train two encoders, therapist and patient, one decoder for therapist



Data characteristics:
We have 82 patient-therapist dyads.
They have engaged in 3 interviews together, personality, attachment and wonder-question.

Question 3: How do we structure this training pipeline? By the therapists?
--> dont mix patients or therapists across folds!
--> involve as many different therapists as possible in the training (better generalizatoin)

Question 4: The recordings have different fps: solve through linear interpolation in the timestamp?
--> downsample to lowest denominator (24)
--> investigate if the team can re-run the OpenFace algorithm, else find smoothing/filter/downsampling algorithms


'''