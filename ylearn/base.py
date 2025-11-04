"""Base class for all the estimators."""

# Author: Youri Rigaud
# License: MIT License

from __future__ import annotations
from abc import ABC, abstractmethod

from .types import ArrayLike
from ylearn.metrics import r2_score

class BaseEstimator(ABC):
    """
    An abstract class representing a ML estimator.
    """
    
    @abstractmethod
    def fit(self, X: ArrayLike, y: ArrayLike) -> BaseEstimator:
        """
        Train the estimator on data X to fit target values y.

        Parameters:
            X (ArrayLike): A (nb_samples, nb_features) shape ArrayLike representing the training data.
            y (ArrayLike): A (nb_samples, ) shape ArrayLike representing the target values of the training data.
        
        Returns:
            self (BaseEstimator): Self trained estimator object.
        """
        pass
    
    @abstractmethod
    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        Predict the target values of the data X.

        Parameters:
            X (ArrayLike): A (nb_queries, nb_features) shape ArrayLike representing the queries data.
        
        Returns:
            y_pred (ArrayLike): The target values predicted by the estimator.
        """
        pass
    
    @abstractmethod
    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """
        Score the model on the test data, the scorer depend on the model itself.
        It computes the prediction on test data X and compares it with the true target values y.

        Parameters:
            X (ArrayLike): A (nb_samples, nb_features) shape ArrayLike representing the test data.
            y (ArrayLike): A (nb_samples, ) shape ArrayLike representing the true target values of the test data.

        Returns:
            score (float): The computed score. 
        """
        pass