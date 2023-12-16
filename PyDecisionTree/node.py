from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import scipy

from PyDecisionTree.question import Question


class Node(ABC):
    """ Abstract class of a Node. """

    @abstractmethod
    def __eq__(self, other) -> bool:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass


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


class DecisionNode(Node):
    """
    Represents a decision node in a decision tree.

    The decision node stores the question (splitting criterias),
    where the data gets split. It also stores the references for
    the next questions (decision nodes) or the leaf nodes.

    Because the CART algorithm is designed to create binary decision
    trees, we only have to hold the reference for the left subtree
    (branch_left) and the right subtree (branch_right)

    Args:
        question (Question):
            The question (criteria) for splitting the data

        branch_left (Node):
            The reference for the left subtree, where the question holds true

        branch_right (Node):
            The reference for the right subtree, where the question holds false
    """

    def __init__(
            self,
            question: Question,
            branch_left: Node,
            branch_right: Node,
    ):
        self._question = question
        self._branch_left = branch_left
        self._branch_right = branch_right

    @property
    def question(self) -> Question:
        return self._question

    @property
    def branch_left(self) -> Node:
        return self._branch_left

    @property
    def branch_right(self) -> Node:
        return self._branch_right

    def apply(self, X: np.ndarray) -> Node:
        """
        Returns the corresponding branch depending on if the question holds true or not:
            - If question holds true, then it returns the left subtree
            - If question holds false, then it returns the right subtree

        Args:
            X (np.ndarray):
                The feature vector of shape (N_features,) or (1, N_features)

        Returns:
            Node:
                The left subtree or right subtree
        """
        assert X.ndim == 2 and X.shape[0] == 1, f"Illegal X {X.ndim}. The matrix should have the shape (1, N_features)!"

        if self._question.match(X, return_indices=False)[0]:
            return self._branch_left
        else:
            return self._branch_right

    def __eq__(self, other) -> bool:
        if isinstance(other, DecisionNode):
            return self._question == other._question and self._branch_left == other._branch_left and self._branch_right == other._branch_right
        raise NotImplementedError

    def __str__(self) -> str:
        return self._question.__str__()

    def __repr__(self) -> str:
        return self.__str__()
