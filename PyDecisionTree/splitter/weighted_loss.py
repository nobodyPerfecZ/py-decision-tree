import numpy as np

from PyDecisionTree.splitter.abstract_splitter import Splitter


class WeightedLoss(Splitter):
    """
    Represents the weighted loss of a split.

    The weighted loss is defined as follows:
        - WeightedLoss(...) = N1/N * H(X_left, y_left) + N2/N * H(X_right, y_right)
    """

    def _call(
            self,
            X_parent: np.ndarray,
            y_parent: np.ndarray,
            X_left: np.ndarray,
            y_left: np.ndarray,
            X_right: np.ndarray,
            y_right: np.ndarray,
    ) -> float:
        N1 = len(y_left)
        N2 = len(y_right)
        E1 = self.criteria(X_left, y_left)
        E2 = self.criteria(X_right, y_right)

        weighted_entropy = ((N1 * E1) + (N2 * E2)) / (N1 + N2)
        return weighted_entropy
