import numpy as np

from PyDecisionTree.tree.abstract_node import Node
from PyDecisionTree.tree.question import Question


class DecisionNode(Node):
    """
    Represents a decision node in a decision tree.

    The decision node stores the question (splitting criteria),
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
            return (self._question == other._question and
                    self._branch_left == other._branch_left and
                    self._branch_right == other._branch_right)
        raise NotImplementedError

    def __str__(self) -> str:
        return self._question.__str__()

    def __repr__(self) -> str:
        return self.__str__()
