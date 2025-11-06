"""Module for the ordinary least squares linear model estimators."""

#Author: Youri Rigaud
#License: MIT License

from __future__ import annotations

from ylearn.base import BaseEstimator
from ylearn.types import ArrayLike
from ylearn.linear_model import BaseLinearModel
from ylearn.linear_model.solver import LinearSolverFactory

class OLS(BaseLinearModel):
    """
    Ordinary least squares linear model estimators.
    """

    def __init__(self, solver="qr", fit_intercept = True) -> None:
        """
        Initialize the ols linear model.

        Parameters:
            solver (str): The name of the solver to use.
            fit_intercept (bool): True by default, fit the linear model with an intercept value stored in intercept_.
        """
        super().__init__(solver, fit_intercept)

    def fit(self, X_train: ArrayLike, y_train: ArrayLike) -> OLS:
        """
        Train the linear model on data X to fit responses y.

        Parameters:
            X_train (ArrayLike): A (nb_samples, nb_features) shape ArrayLike representing the training data.
            y_train (ArrayLike): A (nb_samples, ) shape ArrayLike representing the responses of the training data.
        
        Returns:
            self (OLS): Self trained OLS estimator object.
        """
        super().fit(X_train, y_train)
        beta = self.solver.solve(self._X_train, self._y_train)
        self._set_coef(beta)
        return self
