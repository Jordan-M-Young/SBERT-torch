"""Training and Evaluation functions."""

import numpy as np
from torch import DoubleTensor, tensor
from torch.nn import CrossEntropyLoss, MSELoss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from app.sbert import SentenceBERT


def train(
    data: DataLoader,
    model: SentenceBERT,
    loss_func: CrossEntropyLoss | MSELoss,
    optimizer: Optimizer,
) -> float:
    """Training loop."""
    model.train()
    batch_loss = 0.0
    for _, batch in enumerate(data):
        left, right, labels = batch
        labels = np.array(labels).transpose()
        labels = tensor(labels)
        output = model(left, right)
        output = output.type(DoubleTensor)
        loss = loss_func(output, labels)
        loss_val = loss.detach().item()
        batch_loss += loss_val
        loss.backward()

        optimizer.step()

    return batch_loss


def evaluate(
    data: DataLoader, model: SentenceBERT, loss_func: CrossEntropyLoss | MSELoss
) -> float:
    """Evaluation Loop."""
    model.eval()
    batch_loss = 0.0
    for _, batch in enumerate(data):
        left, right, labels = batch
        labels = np.array(labels).transpose()
        labels = tensor(labels)
        output = model(left, right)

        loss = loss_func(output, labels)
        loss_val = loss.detach().item()
        batch_loss += loss_val

    return batch_loss
