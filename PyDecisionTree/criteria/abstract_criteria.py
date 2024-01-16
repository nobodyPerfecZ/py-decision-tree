from abc import ABC, abstractmethod

import numpy as np


class Criteria(ABC):
    """ Abstract class for representing a criteria method to evaluate a given split. """

    @abstractmethod
    def _call(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Computes the specific criteria value for a given split.

        These criteria value can be used to evaluate the split of
        a dataset in combination with the Information Gain.

        Args:
            X (np.ndarray):
                Numpy Array of shape (N, ?)

            y (np.ndarray):
                Numpy Array of shape (N,)

        Returns:
            float:
                The corresponding criteria value
        """
        pass

    def __call__(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Computes the specific criteria value for a given split.

        These criteria value can be used to evaluate the split of
        a dataset in combination with the Information Gain.

        Args:
            X (np.ndarray):
                Numpy Array of shape (N, ?)

            y (np.ndarray):
                Numpy Array of shape (N,)

        Returns:
            float:
                The corresponding criteria value
        """
        assert X.ndim == 2, f"Illegal X {X.ndim}. The dimension of X should be (N, ?)!"
        assert y.ndim == 1, f"Illegal y {y.ndim}. The dimension of y should be (N,)!"
        assert X.shape[0] == y.shape[0], \
            f"Illegal X {X.ndim} or y {y.ndim}. The dimension of X should be (N, ?) and (N,)!"
        return self._call(X, y)
