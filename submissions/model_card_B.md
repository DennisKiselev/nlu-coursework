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

This model is based on XLNet embeddings, employs a Bi-LSTM architecture with subtractive & multiplicative sentence fusion, attention, dropout and LR scheduling.

- **Developed by:** Jack Pay and Dennis Kiselev
- **Language(s):** English
- **Model type:** Supervised
- **Model architecture:** Non-Transformer DNN
- **Finetuned from model [optional]:** xlnet-base-cased

### Model Resources

<!-- Provide links where applicable. -->

- **Repository:** https://huggingface.co/xlnet/xlnet-base-cased
- **Paper or documentation:** https://arxiv.org/pdf/1906.08237.pdf

## Training Details

### Training Data

<!-- This is a short stub of information on the training data that was used, and documentation related to data pre-processing or additional filtering (if applicable). -->

26K training premise-hypothesis pairs, and more than 6K validation pairs.

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Training Hyperparameters

<!-- This is a summary of the values of hyperparameters used in training the model. -->


      - learning_rate: 1e-3
      - learning_rate_scheduling: ReduceLROnPlateau
      - scheduling_monitor: val_loss
      - scheduling_factor: 0.1
      - scheduling_min_delta: 0.01
      - scheduling_min_lr: 1e-5
      - scheduling_patience: 2
      - train_batch_size: 256
      - val_batch_size: 256
      - num_epochs: 20
      - dropout: 0.5
      - loss_function: categorical_crossentropy
      - optimizer: RMSprop
      - prediction_activation_function: softmax
      - embedding_trainable: False

#### Speeds, Sizes, Times

<!-- This section provides information about how roughly how long it takes to train the model and the size of the resulting model. -->


      - overall training time: 5 minutes
      - duration per training epoch: 16 seconds
      - model size: 162.8MB

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

The model obtained an accuracy of 73%, weighted macro-precision of 73%, weighted macro-recall of 73%, weighted macro F1-score of 73%, MCC of 0.455 and loss of 0.54

## Technical Specifications

### Hardware


      - RAM: at least 8 GB
      - Storage: at least 0.17GB,
      - GPU: V100

### Software


      - keras: 2.15.0,
      - transformers: 4.38.2,
      - tensorflow: 2.15.0,
      - numpy: 1.25.2,
      - nltk: 3.8.1

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

Any premises over 110 tokens, or any hypotheses over 60 tokens will be truncated to these lengths. Sequences below these must be zero-padded.

## Additional Information

<!-- Any other information that would be useful for other people to know. -->

Various experimentation was conducted to find the optimal parameters settings for various hyperparameters, such as the size of certain layers.
