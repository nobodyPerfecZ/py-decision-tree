from abc import ABC, abstractmethod

import numpy as np

from PyDecisionTree.criteria import Criteria


class Splitter(ABC):
    """
    Abstract class for representing a splitter method to evaluate a given question.

    We define the splitter as a loss function which quantifies a given question (split).
    This can be used to search for the best split, which minimizes the loss function (splitter).

    Args:
        criteria (Criteria):
            The criteria function (loss type) we want to use
    """

    def __init__(self, criteria: Criteria):
        self.criteria = criteria

    @abstractmethod
    def _call(
            self,
            X_parent: np.ndarray,
            y_parent: np.ndarray,
            X_left: np.ndarray,
            y_left: np.ndarray,
            X_right: np.ndarray,
            y_right: np.ndarray,
    ) -> float:
        """
        Computes the loss of the loss function (splitter), according to the
        parent dataset (X_parent, y_parent) before splitting, the dataset
        of the left child node (X_left, y_left) and the dataset of the
        right child node (X_right, y_right).

        This method should be implemented if you want to implement specific
        loss functions.

        Args:
            X_parent (np.ndarray):
                The feature matrix of shape (N, N_features) before splitting

            y_parent (np.ndarray):
                The true output of shape (N,) before splitting

            X_left (np.ndarray):
                The feature matrix of shape (N1, N_features) of left child node

            y_left (np.ndarray):
                The true output of shape (N1,) of left child node

            X_right (np.ndarray):
                The feature matrix of shape (N2, N_features) of right child node

            y_right (np.ndarray):
                The true output of shape (N2,) of right child node

        Returns:
            float:
                The loss value
        """
        pass

    def __call__(
            self,
            X_parent: np.ndarray,
            y_parent: np.ndarray,
            X_left: np.ndarray,
            y_left: np.ndarray,
            X_right: np.ndarray,
            y_right: np.ndarray,
    ) -> float:
        """
        Computes the loss of the loss function (splitter), according to the
        parent dataset (X_parent, y_parent) before splitting, the dataset
        of the left child node (X_left, y_left) and the dataset of the
        right child node (X_right, y_right).

        Args:
            X_parent (np.ndarray):
                The feature matrix of shape (N, N_features) before splitting

            y_parent (np.ndarray):
                The true output of shape (N,) before splitting

            X_left (np.ndarray):
                The feature matrix of shape (N1, N_features) of left child node

            y_left (np.ndarray):
                The true output of shape (N1,) of left child node

            X_right (np.ndarray):
                The feature matrix of shape (N2, N_features) of right child node

            y_right (np.ndarray):
                The true output of shape (N2,) of right child node

        Returns:
            float:
                The loss value
        """
        return self._call(X_parent, y_parent, X_left, y_left, X_right, y_right)


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
