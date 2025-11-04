"""Metric functions for the library."""

#Author: Youri Rigaud
#License: MIT License

import numpy as np

from .types import ArrayLike

def r2_score(y: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Compute the r2 score.
    Use for regression task.

    Parameters:
        y (ArrayLike): A (nb_samples, ) shape ArrayLike representing the true target values of the data.
        y_pred (ArrayLike): A (nb_samples, ) shape ArrayLike representing the predicted target values of the data.

    Returns:
        r2_score (float): The computed r2 score.
    """
    RSS = ((y - y_pred)** 2).sum()
    TSS = ((y - y.mean()) ** 2).sum()
    return 1. - RSS/TSS

def accuracy_score(y: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Compute the accuracy score.
    Use for classication task.

    Parameters:
        y (ArrayLike): A (nb_samples, ) shape ArrayLike representing the true target values of the data.
        y_pred (ArrayLike): A (nb_samples, ) shape ArrayLike representing the predicted target values of the data.

    Returns:
        accuracy_score (float): The computed accuracy score.
    """
    accuracy_score = np.mean(y == y_pred)
    return accuracy_score

def MSE(y: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Compute the MSE (Mean Square Error).
    Use for regression task.

    Parameters:
        y (ArrayLike): A (nb_samples, ) shape ArrayLike representing the true target values of the data.
        y_pred (ArrayLike): A (nb_samples, ) shape ArrayLike representing the predicted target values of the data.

    Returns:
        MSE (float): The computed MSE.
    """
    MSE = np.mean((y - y_pred)** 2)
    return MSE