from abc import ABC, abstractmethod

import numpy as np


class Criteria(ABC):
    """ Abstract class for representing a criteria method to evaluate a given split. """

    @abstractmethod
    def _call(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Computes the specific criteria value for a given split.

        This criteria value can be used to evaluate the split of
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


class Gini(Criteria):
    """
    Represents the gini impurity of a given dataset (X, y).

    The gini impurity is defined as follows:
        - Gini(X, y) := 1 - sum_i^k p_i²
    """

    def _call(self, X: np.ndarray, y: np.ndarray) -> float:
        # Get the unique class indices
        class_indices = np.unique(y)

        # Compute the gini impurity:
        # Gini(D) := 1 - sum_i^k p_i²
        gini_impurity = 1.0
        for class_idx in class_indices:
            count = np.count_nonzero(y == class_idx)
            p_i = count / len(y)
            gini_impurity -= p_i ** 2
        return gini_impurity


class Entropy(Criteria):
    """
    Represents the entropy of a given dataset (X, y).

    The entropy is defined as follows:
        - H(X, y) = - sum_i^k p_i * log(p_i)
    """

    def _call(self, X: np.ndarray, y: np.ndarray) -> float:
        # Get the unique class indices
        class_indices = np.unique(y)

        # Compute the entropy:
        # H(D) := - sum_i^k p_i * log(p_i)
        entropy = 0.0
        for class_idx in class_indices:
            count = np.count_nonzero(y == class_idx)
            p_i = count / len(y)
            entropy -= p_i * np.log(p_i)
        return entropy


class LogLoss(Criteria):
    def _call(self, X: np.ndarray, y: np.ndarray) -> float:
        # TODO: Implement here + Unittests + Documentation
        """
        y_mode = scipy.stats.mode(y)[0]
        count = np.count_nonzero(y == y_mode)
        p_i = count / len(y)
        return - y_mode * np.log()
        """
        raise


class MSELoss(Criteria):
    """
    Represents the mean squared error (MSE) of a given dataset (X, y).

    The MSELoss is defined as follows:
        - MSE(X, y) = 1/N sum_i^N (y_i - y_mean) ** 2
    """

    def _call(self, X: np.ndarray, y: np.ndarray) -> float:
        y_mean = np.mean(y)
        return np.mean((y - y_mean) ** 2)


class MAELoss(Criteria):
    """
    Represents the mean absolute error (MAE) of a given dataset (X, y).

    The MAELoss is defined as follows:
        - MAE(X, y) = 1/N sum_i^N |y_i - y_mean|
    """

    def _call(self, X: np.ndarray, y: np.ndarray) -> float:
        y_mean = np.mean(y)
        return np.mean(np.abs(y - y_mean))
