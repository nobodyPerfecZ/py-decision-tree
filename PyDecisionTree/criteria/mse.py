import numpy as np

from PyDecisionTree.criteria.abstract_criteria import Criteria


class MSE(Criteria):
    """
    Represents the mean squared error (MSE) of a given dataset (X, y).

    The MSELoss is defined as follows:
        - MSE(X, y) = 1/N sum_i^N (y_i - y_mean) ** 2
    """

    def _call(self, X: np.ndarray, y: np.ndarray) -> float:
        y_mean = np.mean(y)
        return np.mean((y - y_mean) ** 2)
