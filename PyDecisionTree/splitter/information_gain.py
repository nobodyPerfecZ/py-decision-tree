import numpy as np

from PyDecisionTree.splitter.abstract_splitter import Splitter


class InformationGain(Splitter):
    """
    Represents the (negative) information gain of a split.

    The negative information gain is defined as follows:
        - IG(...) = -(N * H(X_parent, y_parent) - N1/N * H(X_left, y_left) - N2/N * H(X_right, y_right))
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
        N = len(y_parent)
        N1 = len(y_left)
        N2 = len(y_right)
        E = self.criteria(X_parent, y_parent)
        E1 = self.criteria(X_left, y_left)
        E2 = self.criteria(X_right, y_right)
        return -((N * E - (N1 * E1) - (N2 * E2)) / N)
