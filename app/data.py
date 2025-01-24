"""Custom Dataset and data functions."""

import json
from typing import Tuple

from torch.utils.data import Dataset
from transformers import BertTokenizer

from app.constants import (
    CLASSIFICATION_LABEL_ENCODINGS,
    REGRESSION_LABEL_ENCODINGS,
    TrainingObjective,
)


class SentencePairDataset(Dataset):
    """Custom Dataset for encoding sentence pairs."""

    def __init__(
        self,
        sentence_pairs: list[tuple],
        labels: list[str],
        tokenizer: BertTokenizer,
        head: TrainingObjective,
    ):
        """Initialize SentencePairDataset."""
        self.sentence_pairs = sentence_pairs
        self.head = head
        self.labels = self.encode_labels(labels)
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        """Get a sentence pair and label from the dataset."""
        left, right = self.sentence_pairs[idx]

        left = self.tokenizer.encode_plus(
            left,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        right = self.tokenizer.encode_plus(
            right,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        label = self.labels[idx]

        return left, right, label

    def __len__(self):
        """Get the length of the dataset."""
        return len(self.labels)

    def encode_labels(self, labels: list) -> list[list]:
        """Encode dataset labels."""
        encoded_labels = []
        if self.head == TrainingObjective.CLASSIFICATION:
            encoding_dict = CLASSIFICATION_LABEL_ENCODINGS
        else:
            encoding_dict = REGRESSION_LABEL_ENCODINGS

        for label in labels:
            encoded_labels.append(encoding_dict[label])
        return encoded_labels


def load_data(filename) -> Tuple[list[tuple], list[str]]:
    """Loads sentence pair data from jsonl file."""
    with open(filename, "r") as jfile:
        json_list = list(jfile)

    pairs = []
    labels = []
    for json_str in json_list:
        res = json.loads(json_str)
        sentence_1 = res["sentence1"]
        sentence_2 = res["sentence2"]
        label = res["gold_label"]

        if label in CLASSIFICATION_LABEL_ENCODINGS:
            pairs.append((sentence_1, sentence_2))
            labels.append(label)

    return pairs, labels
