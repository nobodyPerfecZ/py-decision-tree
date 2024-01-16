import unittest

import numpy as np

from PyDecisionTree.tree import Question, LeafNode, DecisionNode


class TestDecisionNode(unittest.TestCase):
    """
    Tests the class DecisionNode.
    """

    def setUp(self):
        self.X_left = np.array([
            [0, 1, 2],
        ])
        self.y_left = np.array([0])
        self.X_right = np.array([
            [3, 4, 5],
            [6, 7, 8],
        ])
        self.y_right = np.array([1, 2])
        self.X_test = np.array([[0, 0, 6]])
        self.question = Question(2, 5)
        self.branch_left = LeafNode(self.X_left, self.y_left)
        self.branch_right = LeafNode(self.X_right, self.y_right)
        self.node = DecisionNode(self.question, self.branch_left, self.branch_right)

    def test_apply(self):
        """
        Tests the method apply().
        """
        branch = self.node.apply(self.X_test)
        self.assertEqual(self.branch_left, branch)


if __name__ == '__main__':
    unittest.main()
