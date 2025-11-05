"""Module to model the KNN classification estimator."""

#Author: Youri Rigaud
#License: MIT License

from typing import Any
import numpy as np

from ylearn.neighbors import BaseKNN
from ylearn.types import ArrayLike
from ylearn.metrics import accuracy_score

class KNNClassifier(BaseKNN):
    """
    KNN classifier model.
    For the moment, only integer label are supported.
    """

    def _predict(self, x: ArrayLike) -> int:
        """
        Predict the label of the point x.

        Parameters:
            x (ArrayLike): A (nb_features, ) shape ArrayLike representing the x point.
        
        Returns:
            pred (Any): The target value predicted by the KNN estimator.
        """
        # get the k nearest neighbors labels
        k_nearest_label = self._compute_k_neighbors(x)

        # get the most represented label
        counts = np.bincount(k_nearest_label.astype(int))
        return counts.argmax()
    
    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """
        Score the model on the test data.
        It uses the accuracy score because of the classification task.

        Parameters:
            X (ArrayLike): A (nb_samples, nb_features) shape ArrayLike representing the test data.
            y (ArrayLike): A (nb_samples, ) shape ArrayLike representing the true target values of the test data.

        Returns:
            accuracy_score (float): The computed accuracy score.
        """
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)