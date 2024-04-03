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
)


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


def evaluate(true_labels: np.array, predicted_labels: np.array) -> pd.DataFrame:
    """
    Uses the true and predicted labels & sklearn to create extensive evaluation metrics. Formats into a dataframe that it returns
    """
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
        ],
    )
    return df
