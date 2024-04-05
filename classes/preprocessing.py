import typing
from dataclasses import dataclass
import pandas as pd
import os

@dataclass(frozen=True)
class DatasetKeys:
    """
    Dataclass associated with keys for the data csvs
    """

    PREMISE_KEY: str = "premise"
    HYPOTHESIS_KEY: str = "hypothesis"
    LABEL_KEY: str = "label"


@dataclass(frozen=True)
class PathKeys:
    TRAIN_FILEPATH: str = "data/training_data/training_data/NLI"
    TRAIN_DATASET: str = f"{TRAIN_FILEPATH}/train.csv"
    DEV_DATASET: str = f"{TRAIN_FILEPATH}/dev.csv"

    TRIAL_FILEPATH: str = "data/trial_data/trial_data"
    TRIAL_DATASET: str = f"{TRIAL_FILEPATH}/NLI_trial.csv"

def load_data(data_dir: str = "./") -> typing.Tuple[typing.Tuple[typing.List[str], typing.List[str], typing.List[int]], typing.Tuple[typing.List[str], typing.List[str], typing.List[int]]]:
    """
    Will load in both the training & trial data based on the data directory
    
    data_dir:      String directory to the location of the data
    """
    return (load_data_csv(filepath=os.path.join(data_dir, PathKeys.TRAIN_DATASET)), load_data_csv(filepath=os.path.join(data_dir, PathKeys.DEV_DATASET)))


def load_data_csv(
    filepath: str,
) -> typing.Tuple[typing.List[str], typing.List[str], typing.List[int]]:
    """
    Will load in data from the csv filepath specified. Expects the string filepath to a csv file. Returns tuple of the premises, hypotheses and labels
    """
    dataset = pd.read_csv(filepath).to_dict()
    premises = list(map(str, dataset[DatasetKeys.PREMISE_KEY].values()))
    hypotheses = list(map(str, dataset[DatasetKeys.HYPOTHESIS_KEY].values()))
    labels = list(map(int, dataset[DatasetKeys.LABEL_KEY].values()))
    return (premises, hypotheses, labels)
