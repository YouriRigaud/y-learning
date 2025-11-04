"""Module to model the KNN regression estimator."""

#Author: Youri Rigaud
#License: MIT License

from typing import Any
import numpy as np

from ylearn.neighbors import BaseKNN
from ylearn.types import ArrayLike
from ylearn.metrics import r2_score

class KNNRegressor(BaseKNN):
    """
    KNN regressor model.
    """

    def _predict(self, x: ArrayLike) -> int:
        """
        Predict the target value of the point x.

        Parameters:
            x (ArrayLike): A (nb_features, ) shape ArrayLike representing the x point.
        
        Returns:
            pred (Any): The target value predicted by the KNN estimator.
        """
        # get the k nearest neighbors target values
        k_nearest_target_values = self._compute_k_neighbors(x)

        # return the mean of the target values
        return np.mean(k_nearest_target_values)
    
    def score(self, X, y) -> float:
        """
        Score the model on the test data.
        It uses the r2 score because of the regression task.

        Parameters:
            X (ArrayLike): A (nb_samples, nb_features) shape ArrayLike representing the test data.
            y (ArrayLike): A (nb_samples, ) shape ArrayLike representing the true target values of the test data.

        Returns:
            r2_score (float): The computed r2 score.
        """
        y_pred = self.predict(X)
        return r2_score(y, y_pred)