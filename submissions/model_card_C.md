---
{}
---
language: en
license: cc-by-4.0
tags:
- text-classification
repo: https://github.com/DennisKiselev/nlu-coursework

---

# Model Card for h61781jp-h37701dk-NLI

<!-- Provide a quick summary of what the model is/does. -->

This is a classification model that was trained to
      detect whether one piece of text (the premise) semantically supports another (the hypothesis).


## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

This model is based upon a ROBERTA model finetuned upon 26K pairs of premises & hypotheses. This model utilised several data augmentation methods including synonym replacement and random word swapping. This model employed LR scheduling to improve performance.

- **Developed by:** Jack Pay and Dennis Kiselev
- **Language(s):** English
- **Model type:** Supervised
- **Model architecture:** Transformers
- **Finetuned from model [optional]:** roberta-base

### Model Resources

<!-- Provide links where applicable. -->

- **Repository:** https://huggingface.co/FacebookAI/roberta-base
- **Paper or documentation:** https://arxiv.org/pdf/1907.11692.pdf

## Training Details

### Training Data

<!-- This is a short stub of information on the training data that was used, and documentation related to data pre-processing or additional filtering (if applicable). -->

26K training premise-hypothesis pairs, and more than 6K validation pairs.

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Training Hyperparameters

<!-- This is a summary of the values of hyperparameters used in training the model. -->


      - learning_rate: 2e-05
      - train_batch_size: 16
      - eval_batch_size: 16
      - optimizer: AdamW
      - num_epochs: 6
      - learning_rate_scheduling: get_linear_schedule_with_warmup
      - num_warmup_steps: 26944 
      - num_training_steps: 161664

#### Speeds, Sizes, Times

<!-- This section provides information about how roughly how long it takes to train the model and the size of the resulting model. -->


      - overall training time: 1 hour
      - duration per training epoch: 20 minutes
      - model size: 475.6MB

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data & Metrics

#### Testing Data

<!-- This should describe any evaluation data used (e.g., the development/validation set provided). -->

6K validation pairs.

#### Metrics

<!-- These are the evaluation metrics being used. -->


      - Accuracy
      - Precision
      - Macro Precision
      - Weighted Macro Precision
      - Recall
      - Macro Recall
      - Weighted Macro Recall
      - F1-Score
      - Macro F1-Score
      - Weighted Macro F1-Score
      - MCC
      - Loss

### Results

The model obtained an accuracy of 88%, weighted macro-precision of 88%, weighted macro-recall of 88%, weighted macro F1-score of 88%, MCC of 0.760, and loss of 1.75

## Technical Specifications

### Hardware


      - RAM: at least 16 GB
      - Storage: at least 0.5GB,
      - GPU: V100

### Software


      - torch: 2.2.1+cu121

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

Any inputs (concatenation of two sequences) longer than
      512 subwords will be truncated by the model.

## Additional Information

<!-- Any other information that would be useful for other people to know. -->

Data augmentation methods included synonym replacement, synonym insertion, random word deletion, and random word swapping.
