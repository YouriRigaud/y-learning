"""Metric functions for the library."""

#Author: Youri Rigaud
#License: MIT License

from .types import ArrayLike

def r2_score(y: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Compute the r2 score.

    Parameters:
        y (ArrayLike): A (nb_samples, ) shape ArrayLike representing the true labels of the data.
        y_pred (ArrayLike): A (nb_samples, ) shape ArrayLike representing the predicted labels of the data.

    Returns:
        r2_score (float): The computed r2 score.
    """
    RSS = ((y - y_pred)** 2).sum()
    TSS = ((y - y.mean()) ** 2).sum()
    return 1. - RSS/TSS