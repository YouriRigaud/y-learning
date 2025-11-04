"""Utils functions for the library."""

# Author: Youri Rigaud
# License: MIT License

import numpy as np

from ylearn.types import ArrayLike

def euclidean_distance(x1: ArrayLike, x2: ArrayLike) -> float:
    """
    Compute the euclidean distance beetween x1 and x2.

        Parameters:
            x1 (ArrayLike): A (nb_features, ) shape ArrayLike representing a point.
            x2 (ArrayLike): A (nb_features, ) shape ArrayLike representing a point.
        
        Returns:
            distance (float): The euclidean distance of x1 and x2.
    """
    distance = np.sqrt(np.sum((x1-x2)**2))
    return distance