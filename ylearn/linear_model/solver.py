"""Module for the linear model solvers."""

#Author: Youri Rigaud
#License: MIT License

from abc import ABC, abstractmethod
import numpy as np

from ylearn.types import ArrayLike

class LinearSolver(ABC):
    """
    Abstract class representing a linear solver.
    """

    @abstractmethod
    def solve(self, X: ArrayLike, y: ArrayLike, lmbd: float = 0.0) -> ArrayLike:
        """
        Solve the following linear equation: (X^T*X + lmbd*I)*w = X^T*Y.

        Parameters:
            X (ArrayLike): A (nb_samples, nb_features) shape ArrayLike.
            y (ArrayLike): A (nb_samples, ) shape ArrayLike.
            lmbd (float): A lambda coefficient for tuning the identity (I) term of the equation.

        Returns:
            w (ArrayLike): A (nb_features, ) shape ArrayLike that represents solution of the equation.
        """
        pass

class NormalEquationSolver(LinearSolver):
    """
    Linear solver for OLS and Ridge.
    """

    def solve(self, X: ArrayLike, y: ArrayLike, lmbd: float = 0.0) -> ArrayLike:
        """
        Solve the following linear equation: (X^T*X + lmbd*I)*w = X^T*Y.
        It uses the numpy linalg solver to resolve it.
        X^T*X + lmbd*I should be invertible.
        Raise a numpy.linalg.LinAlgError if X^T*X + lmbd*I is singular or not square,
        so on look for other solvers.

        Parameters:
            X (ArrayLike): A (nb_samples, nb_features) shape ArrayLike.
            y (ArrayLike): A (nb_samples, ) shape ArrayLike.
            lmbd (float): A lambda coefficient for tuning the identity (I) term of the equation.

        Returns:
            w (ArrayLike): A (nb_features, ) shape ArrayLike that represents solution of the equation.
        """
        nb_features = X.shape[1]
        I = np.eye(nb_features)
        return np.linalg.solve(X.T @ X + lmbd * I, X.T @ y)

class QRSolver(LinearSolver):
    """
    Linear solver for OLS only, the lambda coefficient does not impact the solve.
    """

    def solve(self, X: ArrayLike, y: ArrayLike, lmbd: float = 0.0) -> ArrayLike:
        """
        Solve the following linear equation: X*w = Y.
        It uses the numpy linalg qr decomposition and solver to resolve it.
        X can be singular.
        Raise a numpy.linalg.LinAlgError if the qr factorization fails.

        Parameters:
            X (ArrayLike): A (nb_samples, nb_features) shape ArrayLike.
            y (ArrayLike): A (nb_samples, ) shape ArrayLike.
            lmbd (float): A lambda coefficient for tuning the identity (I) term of the equation.

        Returns:
            w (ArrayLike): A (nb_features, ) shape ArrayLike that represents solution of the equation.
        """
        Q, R = np.linalg.qr(X)
        return np.linalg.solve(R, Q.T @ y)

class QRRidgeSolver(LinearSolver):
    """
    Linear solver for Ridge, could work for OLS but see QRSolver for better performances.
    """

    def solve(self, X: ArrayLike, y: ArrayLike, lmbd: float = 0.0) -> ArrayLike:
        """
        Solve the following linear equation: (X^T*X + lmbd*I)*w = X^T*Y.S
        It uses the numpy linalg qr decomposition of X and solver to resolve it.
        X can be singular.
        Raise a numpy.linalg.LinAlgError if the qr factorization fails.

        Parameters:
            X (ArrayLike): A (nb_samples, nb_features) shape ArrayLike.
            y (ArrayLike): A (nb_samples, ) shape ArrayLike.
            lmbd (float): A lambda coefficient for tuning the identity (I) term of the equation.

        Returns:
            w (ArrayLike): A (nb_features, ) shape ArrayLike that represents solution of the equation.
        """
        nb_features = X.shape[1]
        Q, R = np.linalg.qr(X)
        I = np.eye(nb_features)

        # Right term
        b = R.T @ (Q.T @ y)

        # Left term
        A = R.T @ R + lmbd * I

        # Solve A*w = b
        return np.linalg.solve(A, b)

class LinearSolverFactory:
    """
    The factory of all the linear solvers.

    Attributes:
        _solvers (Dict): A dictionary of all the solvers with their constructor.
    """

    _solvers = {
        "normal": NormalEquationSolver(),
        "qr": QRSolver(),
        "qr_ridge": QRRidgeSolver(),
    }

    @classmethod
    def get(cls, name: str) -> LinearSolver:
        """
        Get the right solver constructor.
        
        Parameters:
            name (str): The name of the solver, see _solvers attribut.

        Returns:
            solver (LinearSolver): The constructor of the right solver.
        """
        if name not in cls._solvers:
            raise ValueError(f"Unknown solver '{name}'. Available: {list(cls._solvers.keys())}")
        return cls._solvers[name]