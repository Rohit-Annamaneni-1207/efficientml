import torch


def accuracy(correct: int, total: int) -> float:
    """
    Computes classification accuracy.
    """

    if total == 0:
        return 0.0

    return correct / total