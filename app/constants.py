"""Constants and such."""

from enum import Enum

CLASSIFICATION_LABEL_ENCODINGS = {
    "neutral": [0.0, 1.0, 0.0],
    "contradiction": [1.0, 0.0, 0.0],
    "entailment": [0.0, 0.0, 1.0],
}

REGRESSION_LABEL_ENCODINGS = {
    "neutral": 0.0,
    "contradiction": -1.0,
    "entailment": 1.0,
}


class BERTModels(Enum):
    """Bert model enumeration. Each variant requires a huggingface model name
    and the size of the hidden layers. The latter is needed to initialize
    SentenceBERT properly. Copy this scheme if you want to add enum variants!

    """

    TINY_BERT = ("prajjwal1/bert-tiny", 128)
    MINI_BERT = ("prajjwal1/bert-mini", 256)
    SMALL_BERT = ("prajjwal1/bert-small", 512)
    MED_BERT = ("prajjwal1/bert-medium", 512)
    BERT_BASE = ("google-bert/bert-base-uncased", 768)


class TrainingObjective(Enum):
    """Training Objective Enum class."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class ConcatStrategies(Enum):
    """Concatenation Strategies for classification head. Section 6. pg 7."""

    UV = "u,v"
    UsubV = "|u-v|"
    UxV = "u*v"
    UsubVUxV = "|u-v|u*v"
    UVUxV = "u,vu*v"
    UVUsubV = "u,v|u-v|"
    UVUsubVUxV = "u,v|u-v|u*v"
