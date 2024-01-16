import unittest

import numpy as np

from PyDecisionTree.tree import LeafNode


class TestLeafNode(unittest.TestCase):
    """
    Tests the class LeafNode.
    """

    def setUp(self):
        self.X = np.array([
            [0, 1, 2],
        ])
        self.y = np.array([0])
        self.node = LeafNode(self.X, self.y)

    def test_predict(self):
        """
        Tests the method predict()
        """
        y_pred, y_mean = self.node.predict(return_mean=True)
        np.testing.assert_array_equal(0, y_pred)
        np.testing.assert_array_equal(0, y_mean)


if __name__ == '__main__':
    unittest.main()
