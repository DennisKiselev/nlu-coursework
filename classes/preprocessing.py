import typing
from dataclasses import dataclass
import pandas as pd
import os
from dataclasses import dataclass
import random
from nltk.corpus import wordnet, stopwords
from itertools import chain
import nltk
import numpy as np
from evaluation import GeneralKeys

nltk.download("wordnet")
nltk.download("stopwords")

STOP_WORDS = set(stopwords.words("english"))


@dataclass(frozen=True)
class DatasetKeys:
    """
    Dataclass associated with keys for the data csvs
    """

    PREMISE_KEY: str = GeneralKeys.PREMISE_KEY.lower()
    HYPOTHESIS_KEY: str = GeneralKeys.HYPOTHESIS_KEY.lower()
    LABEL_KEY: str = GeneralKeys.LABEL_KEY.lower()


@dataclass(frozen=True)
class PathKeys:
    TRAIN_FILEPATH: str = "data/training_data/training_data/NLI"
    TRAIN_DATASET: str = f"{TRAIN_FILEPATH}/train.csv"
    DEV_DATASET: str = f"{TRAIN_FILEPATH}/dev.csv"

    TRIAL_FILEPATH: str = "data/trial_data/trial_data"
    TRIAL_DATASET: str = f"{TRIAL_FILEPATH}/NLI_trial.csv"


def load_data(
    data_dir: str = "./",
) -> typing.Tuple[
    typing.Tuple[typing.List[str], typing.List[str], typing.List[int]],
    typing.Tuple[typing.List[str], typing.List[str], typing.List[int]],
]:
    """
    Will load in both the training & trial data based on the data directory

    data_dir:      String directory to the location of the data
    """
    return (
        load_data_csv(filepath=os.path.join(data_dir, PathKeys.TRAIN_DATASET)),
        load_data_csv(filepath=os.path.join(data_dir, PathKeys.DEV_DATASET)),
    )


def load_data_csv(
    filepath: str,
) -> typing.Tuple[typing.List[str], typing.List[str], typing.List[int]]:
    """
    Will load in data from the csv filepath specified. Expects the string filepath to a csv file. Returns tuple of the premises, hypotheses and labels
    """
    dataset = pd.read_csv(filepath)
    premises = dataset[DatasetKeys.PREMISE_KEY].astype(str).tolist()
    hypotheses = dataset[DatasetKeys.HYPOTHESIS_KEY].astype(str).tolist()
    labels = dataset[DatasetKeys.LABEL_KEY].astype(str).tolist()
    return (premises, hypotheses, labels)


def get_synonym(word: str) -> str:
    """
    Will return a random synonym for a word
    """
    syns = list(
        set(
            chain.from_iterable(
                [
                    [l.name().replace("_", " ") for l in syn.lemmas()]
                    for syn in wordnet.synsets(word)
                ]
            )
        )
    )
    if syns == []:  # If there are no synonyms then return the normal word
        return word
    return random.choice(syns)


@dataclass(frozen=True)
class TextAugmentations:
    """
    Dataclass associated with keys for data augmentation
    """

    SYNONYM_REPLACEMENT: str = "synonym_replacement"
    SYNONYM_INSERTION: str = "synonym_insertion"
    WORD_DELETION: str = "word_deletion"
    WORD_SWAP: str = "word_swap"


def synonym_replacement(text: str, quantity: int) -> str:
    """
    Will randomly replace quantity number of words within the text with a synonym of the same word
    """
    split_text = text.split(" ")
    random_samples = random.sample(range(0, len(split_text)), quantity)
    for i in random_samples:
        if split_text[i] not in STOP_WORDS:
            split_text[i] = get_synonym(word=split_text[i])
    return " ".join(split_text)


def synonym_insertion(text: str, quantity: int) -> str:
    """
    Will randomly insert an additional quantity number of words within the text which are synonyms of other words
    """
    split_text = text.split(" ")
    random_samples = np.array(random.sample(range(0, len(split_text)), quantity))
    out_text = []
    for i in range(len(split_text)):
        out_text.extend(
            [split_text[i]]
            if i not in random_samples
            else [split_text[i], get_synonym(word=split_text[i])]
        )
    out_text
    return " ".join(out_text)


def word_deletion(text: str, quantity: int) -> str:
    """
    Will randomly insert an additional quantity number of words within the text which are synonyms of other words
    """
    split_text = text.split(" ")
    random_samples = np.array(random.sample(range(0, len(split_text)), quantity))
    return " ".join(
        [split_text[i] for i in range(len(split_text)) if i not in random_samples]
    )


def word_swap(text: str, quantity: int) -> str:
    """
    Will randomly swap quantity number of samples within the sentence
    """
    split_text = text.split(" ")

    num = quantity * 2
    if num > len(split_text):
        return None
    random_samples = np.array(random.sample(range(0, len(split_text)), num))
    count = 0
    while count < len(random_samples):
        temp = split_text[random_samples[count]]
        split_text[random_samples[count]] = split_text[random_samples[count + 1]]
        split_text[random_samples[count + 1]] = temp
        count += 2
    return " ".join(split_text)


def random_augmentation(
    sentence: str,
    augmentations: typing.List[TextAugmentations] = [
        TextAugmentations.SYNONYM_REPLACEMENT,
        TextAugmentations.SYNONYM_INSERTION,
        TextAugmentations.WORD_DELETION,
        TextAugmentations.WORD_SWAP,
    ],
    quantity: int = 1,
) -> str:
    """
    Will randomly apply a data augmentation to a sentence. Number of augmentations is equal to quantity. Returns None if the augmentation was unsuccessful
    """
    if len(sentence.split(" ")) <= quantity:
        return None

    # Randomly select an augmentation method
    augmentation: TextAugmentations = random.choice(augmentations)
    if augmentation == TextAugmentations.SYNONYM_REPLACEMENT:
        augmentation_method = synonym_replacement
    elif augmentation == TextAugmentations.SYNONYM_INSERTION:
        augmentation_method = synonym_insertion
    elif augmentation == TextAugmentations.WORD_DELETION:
        augmentation_method = word_deletion
    elif augmentation == TextAugmentations.WORD_SWAP:
        augmentation_method = word_swap

    # Take sentence to lower for easier synonm generation
    sentence = sentence.lower()
    # Augments
    aug_sentence = augmentation_method(text=sentence, quantity=quantity)
    if (
        aug_sentence == sentence
    ):  # If the sentence didn't change then we return None: We don't want to duplicate sentences
        return None
    return aug_sentence


def augment_data(
    premises: typing.List[str],
    hypotheses: typing.List[str],
    labels: np.array,
    premise_quantity: int = 1,
    hypothesis_quantity: int = 1,
):
    """
    Takes the premises & hypotheses and will augment both of these randomly. Uses different quantities of augmentations for either the premises & hypotheses
    """
    aug_premises, aug_hypotheses, aug_labels = [], [], []
    for premise, hypothesis, label in zip(premises, hypotheses, labels):
        # Does different augmentations on the premise & hypothesis
        aug_premise = random_augmentation(sentence=premise, quantity=premise_quantity)
        aug_hypothesis = random_augmentation(
            sentence=hypothesis, quantity=hypothesis_quantity
        )

        # If either the premise or hypothesis are the same then we don't add
        if aug_premise is None or aug_hypothesis is None:
            continue

        # Otherwise add these new samples
        aug_premises.append(aug_premise)
        aug_hypotheses.append(aug_hypothesis)
        aug_labels.append(label)
    return (
        premises + aug_premises,
        hypotheses + aug_hypotheses,
        np.concatenate((labels, np.array(aug_labels))),
    )
