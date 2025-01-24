"""Constants and such."""

from enum import Enum

LABEL_ENCODINGS = {
    "neutral": [0.0, 1.0, 0.0],
    "contradiction": [1.0, 0.0, 0.0],
    "entailment": [0.0, 0.0, 1.0],
}


class TrainingObjective(Enum):
    """Training Objective Enum class."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"
