"""Utility Functions."""


def log_epoch(epoch: int, train_loss: float, test_loss: float)-> None:
    print(f"Epoch {epoch} | Train Loss: {train_loss}  | Test Loss {test_loss} ")
