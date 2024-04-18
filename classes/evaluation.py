from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    log_loss,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
import typing


@dataclass(frozen=True)
class GeneralKeys:
    """
    Dataclass for general keys for the process
    """

    PREMISE_KEY: str = "Premise"
    HYPOTHESIS_KEY: str = "Hypothesis"
    LABEL_KEY: str = "Label"
    LOSS_KEY: str = "Loss"
    PREDICTED_KEY: str = "Predicted Label"
    TRUE_KEY: str = "True Label"


@dataclass
class MacroMetric:
    """
    Dataclass for metrics that can be turned into macro & weighted macro
    """

    NORMAL_KEY: str
    MACRO_KEY: str
    WEIGHTED_KEY: str

    def __init__(self, key: str):
        self.NORMAL_KEY: str = key
        self.MACRO_KEY: str = f"Macro {key}"
        self.WEIGHTED_KEY: str = f"Weighted Macro {key}"


@dataclass(frozen=True)
class MetricKeys:
    """
    Dataclass associated with keys for the evaluation metrics
    """

    ACCURACY_KEY: str = "Accuracy"
    PRECISION: MacroMetric = MacroMetric(key="Precision")
    F1: MacroMetric = MacroMetric(key="F1-Score")
    RECALL: MacroMetric = MacroMetric(key="Recall")
    MCC_KEY: str = "MCC"
    LOSS_KEY: str = "Loss"


def evaluate(true_labels: np.array, predicted_logits: np.array) -> pd.DataFrame:
    """
    Uses the true and predicted labels & sklearn to create extensive evaluation metrics. Formats into a dataframe that it returns

    true_labels:        (N) sized array storing the true (0, 1) labels of the data
    predicted_logits:   (N, 2) sized array storing the predicted logits from the model, therefore the predicted probabilities for either class
    """
    loss = log_loss(true_labels, predicted_logits)  # Uses logits for loss

    # Otherwise utilises argmax of the prediction logits, to get the predicted labels
    predicted_labels = np.argmax(predicted_logits, axis=1)

    accuracy = accuracy_score(true_labels, predicted_labels)

    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, predicted_labels
    )

    weighted_precision = np.average(precision, weights=support)
    weighted_recall = np.average(recall, weights=support)
    weighted_f1 = np.average(f1, weights=support)

    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)

    macro_precision = precision_score(true_labels, predicted_labels, average="macro")
    macro_recall = recall_score(true_labels, predicted_labels, average="macro")
    macro_f1 = f1_score(true_labels, predicted_labels, average="macro")

    mcc = matthews_corrcoef(true_labels, predicted_labels)

    # Format into dataframe for easier viewing
    df = pd.DataFrame(
        [
            [
                accuracy,
                precision,
                macro_precision,
                weighted_precision,
                recall,
                macro_recall,
                weighted_recall,
                f1,
                macro_f1,
                weighted_f1,
                mcc,
                loss,
            ]
        ],
        columns=[
            MetricKeys.ACCURACY_KEY,
            MetricKeys.PRECISION.NORMAL_KEY,
            MetricKeys.PRECISION.MACRO_KEY,
            MetricKeys.PRECISION.WEIGHTED_KEY,
            MetricKeys.RECALL.NORMAL_KEY,
            MetricKeys.RECALL.MACRO_KEY,
            MetricKeys.RECALL.WEIGHTED_KEY,
            MetricKeys.F1.NORMAL_KEY,
            MetricKeys.F1.MACRO_KEY,
            MetricKeys.F1.WEIGHTED_KEY,
            MetricKeys.MCC_KEY,
            MetricKeys.LOSS_KEY,
        ],
    )
    return df


@dataclass
class ClassLabels:
    """
    Dataclass for the string class labels. Used in the confusion matrix generation
    """

    ZERO_KEY: str = "Not Entailing"
    ONE_KEY: str = "Entailing"


def draw_confusion_matrix(
    true_labels: np.array,
    predicted_logits: np.array,
    classes: typing.List[str] = [ClassLabels.ZERO_KEY, ClassLabels.ONE_KEY],
) -> np.array:
    """
    Will make a confusion matrix using the predicted and true values & will display this. Returns the confusion matrix as an array
    """
    predicted_labels = np.argmax(predicted_logits, axis=1)

    conf_mat = confusion_matrix(true_labels, predicted_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=classes)
    disp.plot()
    plt.show()
    return conf_mat


def most_confused_samples(
    true_logits: np.array,
    predicted_logits: np.array,
    premises: typing.List[str],
    hypotheses: typing.List[str],
    num: int = 5,
    loss_function: callable = keras.losses.categorical_crossentropy,
) -> pd.DataFrame:
    """
    Will print the num samples with the highest loss

    true_logits:        (N, 2) sized array storing the one hot encoded labels of the data
    predicted_logits:   (N, 2) sized array storing the predicted logits from the model, therefore the predicted probabilities for either class
    premises:           (N) sized array storing the string premises
    hypotheses:         (N) sized array storing the string hypotheses

    num:                Integer number of samples to report about. The top M (or num) samples will be displayed
    loss_function:      Executable function used for the loss calculation. By default this is just categorical cross entropy
    """
    # Gets the samples that have the highest loss
    loss_per_sample = [
        loss.numpy() for loss in loss_function(true_logits, predicted_logits)
    ]
    largest_indices = np.argsort(loss_per_sample)[-num:][::-1]

    predicted_labels = np.argmax(predicted_logits, axis=1)
    true_labels = np.argmax(true_logits, axis=1)

    # Makes the dataframe with the confused samples
    confused_samples = [
        [
            premises[i],
            hypotheses[i],
            loss_per_sample[i],
            predicted_labels[i],
            true_labels[i],
        ]
        for i in largest_indices
    ]
    df = pd.DataFrame(
        confused_samples,
        columns=[
            GeneralKeys.PREMISE_KEY,
            GeneralKeys.HYPOTHESIS_KEY,
            GeneralKeys.LOSS_KEY,
            GeneralKeys.PREDICTED_KEY,
            GeneralKeys.TRUE_KEY,
        ],
    )
    return df
