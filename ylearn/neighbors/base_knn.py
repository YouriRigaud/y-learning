"""Module to model the base of KNN estimators."""

#Author: Youri Rigaud
#License: MIT License

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
import numpy as np

from ylearn.base import BaseEstimator
from ylearn.types import ArrayLike
from ylearn.utils import euclidean_distance

class BaseKNN(BaseEstimator, ABC):
    """
    An abstract class representing a KNN model.
    """

    def __init__(self, k: int = 3) -> None:
        """
        Initialize the KNN estimator.

        Parameters:
            k (int): The number of desired neighbors of the KNN model.
        """
        super().__init__()
        self.k = k

    def fit(self, X: ArrayLike, y: ArrayLike) -> BaseKNN:
        """
        Train the KNN estimator on data X to fit target values y.

        Parameters:
            X (ArrayLike): A (nb_samples, nb_features) shape ArrayLike representing the training data.
            y (ArrayLike): A (nb_samples, ) shape ArrayLike representing the target values of the training data.
        
        Returns:
            self (KNN): Self trained KNN estimator object.
        """
        self._X_train = X
        self._y_train = y
        return self
    
    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        Predict the target values of the data X.

        Parameters:
            X (ArrayLike): A (nb_queries, nb_features) shape ArrayLike representing the queries data.
        
        Returns:
            y_pred (ArrayLike): The target values predicted by the KNN estimator.
        """
        return np.array([self._predict(x) for x in X])

    @abstractmethod
    def _predict(self, x: ArrayLike) -> Any:
        """
        Predict the target value of the point x.

        Parameters:
            x (ArrayLike): A (nb_features, ) shape ArrayLike representing the x point.
        
        Returns:
            pred (Any): The target value predicted by the KNN estimator.
        """
        pass

    def _compute_k_neighbors(self, x: ArrayLike) -> ArrayLike:
        """
        Find the k nearest neighbors of the point x and return their target values.

        Parameters:
            x (ArrayLike): A (nb_features, ) shape ArrayLike representing a point.
        
        Returns:
            k_nearest_target (ArrayLike): A (k, ) shape ArrayLike representing the target values of the k nearest neighbors.
        """
        # compute the distances with all training points
        distances = np.array([euclidean_distance(x, x_train) for x_train in self._X_train])

        # find the k nearest neighbors and their target values
        k_nearest_indices = np.argsort(distances)[:self.k]
        k_nearest_target = np.array([self._y_train[i] for i in k_nearest_indices])
        return k_nearest_target