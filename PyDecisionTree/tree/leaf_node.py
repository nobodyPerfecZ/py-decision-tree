from typing import Union

import numpy as np
import scipy

from PyDecisionTree.tree.abstract_node import Node
from PyDecisionTree.tree.decision_node import DecisionNode


class LeafNode(Node):
    """
    Represents a leaf node in a decision tree.

    The leaf node stores all the data (X, y) that are assigned
    to that node. With the data we can then predict new data points
    by (...):
        - In regression: Take the mean value of the stored y-values
        - In classification: Take the mode value of the stored y-values

    Args:
        X (np.ndarray):
            The feature matrix of shape (N, N_features)

        y (np.ndarray):
            The true output of shape (N,)
    """

    def __init__(
            self,
            X: np.ndarray,
            y: np.ndarray,
    ):
        assert X.ndim == 2, f"Illegal X {X.shape}. The matrix should have a shape of (N, N_features)!"
        assert y.ndim == 1, f"Illegal y {y.shape}. The matrix should have a shape of (N,)!"
        assert X.shape[0] == y.shape[0], \
            f"Illegal X {X.shape} or y {y.shape}. Both should have the same number of samples N!"

        self._X = X
        self._y = y

    @property
    def X(self) -> np.ndarray:
        return self._X

    @property
    def y(self) -> np.ndarray:
        return self._y

    def predict(self, return_mean: bool = False) -> Union[float, tuple[float, float]]:
        """
        Returns the prediction of the leaf node.
        This corresponds to (...)
            - the most frequent class index for classification trees
            - the mean value across the outputs for regression trees

        Args:
            return_mean (bool, optional):
                Controls if the mean of the y-values should be returned.

        Returns:
            Union[float, tuple[float, float]]:
                mode (float):
                    The most frequently occurring value.
                    This is necessary for classification trees.

                mean (float, optional):
                    The mean across the y-values
                    This is necessary for regression trees.
        """
        mode = scipy.stats.mode(self._y)[0]

        if return_mean:
            # Case: Return the mean (necessary for regression trees)
            mean = np.mean(self._y)
            return mode, mean
        else:
            return mode

    def __eq__(self, other) -> bool:
        if isinstance(other, LeafNode):
            return np.array_equal(self._X, other._X) and np.array_equal(self._y, other._y)
        elif isinstance(other, DecisionNode):
            return False
        raise NotImplementedError

    def __str__(self) -> str:
        return self._y.__str__()

    def __repr__(self) -> str:
        return self.__str__()
