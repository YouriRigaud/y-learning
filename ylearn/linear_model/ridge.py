"""Module for the ridge linear model estimators."""

#Author: Youri Rigaud
#License: MIT License

from __future__ import annotations

from ylearn.base import BaseEstimator
from ylearn.types import ArrayLike
from ylearn.linear_model import BaseLinearModel
from ylearn.linear_model.solver import LinearSolverFactory

class Ridge(BaseLinearModel):
    """
    Ridge linear model estimators.
    """

    def __init__(self, lmbd: float = 1.0, solver="qr_ridge", fit_intercept = True) -> None:
        """
        Initialize the ridge linear model.

        Parameters:
            lmbd (float): The lambda ridge coefficient.
            solver (str): The name of the solver to use.
            fit_intercept (bool): True by default, fit the linear model with an intercept value stored in intercept_.
        """
        super().__init__(solver, fit_intercept)
        if lmbd <= 0:
            raise ValueError("Lambda value 'lmbd' must be strictly positive. Use OLS for the case lambda = 0.")
        self.lmbd = lmbd

    def fit(self, X_train: ArrayLike, y_train: ArrayLike) -> Ridge:
        """
        Train the linear model on data X to fit responses y.

        Parameters:
            X_train (ArrayLike): A (nb_samples, nb_features) shape ArrayLike representing the training data.
            y_train (ArrayLike): A (nb_samples, ) shape ArrayLike representing the responses of the training data.
        
        Returns:
            self (Ridge): Self trained Ridge estimator object.
        """
        super().fit(X_train, y_train)
        beta = self.solver.solve(self._X_train, self._y_train, lmbd=self.lmbd)
        self._set_coef(beta)
        return self
