"""Base class for all the estimators."""

# Author: Youri Rigaud
# License: MIT License

from __future__ import annotations
from abc import ABC, abstractmethod

from .types import ArrayLike
from metrics import r2_score

class BaseEstimator(ABC):
    """
    An abstract class representing a ML estimator.
    """
    
    @abstractmethod
    def fit(self, X: ArrayLike, y: ArrayLike) -> BaseEstimator:
        """
        Train the estimator on data X to fit labels y.

        Parameters:
            X (ArrayLike): A (nb_samples, nb_features) shape ArrayLike representing the training data.
            y (ArrayLike): A (nb_samples, ) shape ArrayLike representing the labels of the training data.
        
        Returns:
            self (BaseEstimator): Self trained estimator object.
        """
        pass
    
    @abstractmethod
    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        Predict the labels of the data X.

        Parameters:
            X (ArrayLike): A (nb_queries, nb_features) shape ArrayLike representing the queries data.
        
        Returns:
            y_pred (ArrayLike): The labels predicted by the estimator.
        """
        pass
    
    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """
        Evaluate the model with the r2 score by default on test data.
        It computes the prediction on test data X and compares it with the true labels y.

        Parameters:
            X (ArrayLike): A (nb_samples, nb_features) shape ArrayLike representing the test data.
            y (ArrayLike): A (nb_samples, ) shape ArrayLike representing the true labels of the test data.

        Returns:
            score (float): The computed r2 score. 
        """
        y_pred = self.predict(X)
        return r2_score(y, y_pred)