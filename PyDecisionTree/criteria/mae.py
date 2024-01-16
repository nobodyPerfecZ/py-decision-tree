import numpy as np

from PyDecisionTree.criteria.abstract_criteria import Criteria


class MAE(Criteria):
    """
    Represents the mean absolute error (MAE) of a given dataset (X, y).

    The MAELoss is defined as follows:
        - MAE(X, y) = 1/N sum_i^N |y_i - y_mean|
    """

    def _call(self, X: np.ndarray, y: np.ndarray) -> float:
        y_mean = np.mean(y)
        return np.mean(np.abs(y - y_mean))
