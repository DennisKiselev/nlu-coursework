import typing
from dataclasses import dataclass
import pandas as pd

@dataclass(frozen=True)
class DatasetKeys:
  """
  Dataclass associated with keys for the data csvs
  """
  PREMISE_KEY: str = "premise"
  HYPOTHESIS_KEY: str = "hypothesis"
  LABEL_KEY: str = "label"

def load_data_csv(filepath: str) -> typing.Tuple[typing.List[str], typing.List[str], typing.List[int]]:
  """
  Will load in data from the csv filepath specified. Expects the string filepath to a csv file. Returns tuple of the premises, hypotheses and labels
  """
  dataset = pd.read_csv(filepath).to_dict()
  premises = list(map(str, dataset[DatasetKeys.PREMISE_KEY].values()))
  hypotheses = list(map(str, dataset[DatasetKeys.HYPOTHESIS_KEY].values()))
  labels = list(map(int, dataset[DatasetKeys.LABEL_KEY].values()))
  return (premises, hypotheses,labels)