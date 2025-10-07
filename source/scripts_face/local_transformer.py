'''
Objective: How does the therapist align their facial movements to those of the patient?

Method: Cross attention transformer implementation
Therapist --> Q_t, K_t, V_t
Patient --> Q_p, K_p, V_p

Confirmation point 1: Confirm that what we want is softmax(Q_t * K_p / dimension) * V_p

Confirmation point 2: We will train one encoder-block with the patient embeddings, one decoder block 
with the therapist embedding and cross-attention as per above

Data characteristics:
We have 82 patient-therapist dyads.
They have engaged in 3 interviews together, personality, attachment and wonder-question.

Question 3: How do we structure this training pipeline? By the therapists?


Question 4: The recordings have different fps.....

'''