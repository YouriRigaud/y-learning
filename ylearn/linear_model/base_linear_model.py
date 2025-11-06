"""Base for linear model estimators."""

#Author: Youri Rigaud
#License: MIT License

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

from ylearn.base import BaseEstimator
from ylearn.types import ArrayLike
from ylearn.metrics import r2_score
from ylearn.linear_model.solver import LinearSolverFactory

class BaseLinearModel(BaseEstimator, ABC):
    """
    Abstract class representing a linear model estimator.
    """
    
    def __init__(self, solver: str, fit_intercept: bool = True) -> None:
        """
        Initialize the linear model.

        Parameters:
            fit_intercept (bool): True by default, fit the linear model with an intercept value stored in intercept_.
        """
        self.solver_name = solver
        self.solver = LinearSolverFactory.get(solver)
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = 0.
    
    def fit(self, X_train: ArrayLike, y_train: ArrayLike) -> BaseLinearModel:
        """
        Train the linear model on data X to fit responses y.

        Parameters:
            X_train (ArrayLike): A (nb_samples, nb_features) shape ArrayLike representing the training data.
            y_train (ArrayLike): A (nb_samples, ) shape ArrayLike representing the responses of the training data.
        
        Returns:
            self (KNN): Self trained KNN estimator object.
        """
        self._X_train = self._add_intercept(X_train)
        self._y_train = y_train
    
    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        Predict the target values of the data X.

        Parameters:
            X (ArrayLike): A (nb_queries, nb_features) shape ArrayLike representing the queries data.
        
        Returns:
            y_pred (ArrayLike): The target values predicted by the KNN estimator.
        """
        X = self._add_intercept(X)
        params = np.concatenate(([self.intercept_], self.coef_)) if self.fit_intercept else self.coef_
        return X @ params

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """
        Score the model on the test data.
        It uses the r2 score because of the regression task of linear model.

        Parameters:
            X (ArrayLike): A (nb_samples, nb_features) shape ArrayLike representing the test data.
            y (ArrayLike): A (nb_samples, ) shape ArrayLike representing the true target values of the test data.

        Returns:
            r2_score (float): The computed r2 score.
        """
        y_pred = self.predict(X)
        return r2_score(y, y_pred)

    def _add_intercept(self, X: ArrayLike) -> ArrayLike:
        """
        Add an intercept column on X if fit_intercept is True.

        Parameters:
            X (ArrayLike): A (nb_samples, nb_features) shape ArrayLike representing data to add an intercept column.

        Returns:
            X (ArrayLike): A (nb_samples, nb_features) shape ArrayLike representing data with an intercept at the first column.
        """
        if self.fit_intercept:
            X = np.hstack([np.ones((X.shape[0],1)), X])
        return X
    
    def _set_coef(self, beta: ArrayLike) -> None:
        """
        Set the coef and intercept from beta.

        Parameters:
            beta (ArrayLike): A (nb_features, ) or (nb_features+1, ) shape ArrayLike of the coefficients returned by the solver.
        """
        if self.fit_intercept:
            self.intercept_ = beta[0]
            self.coef_ = beta[1:]
        else:
            self.coef_ = beta
