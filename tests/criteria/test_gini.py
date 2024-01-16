import unittest

import numpy as np

from PyDecisionTree.criteria import Gini


class TestGini(unittest.TestCase):
    """
    Tests the class Gini.
    """

    def setUp(self):
        self.X = np.array([
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
        ])
        self.y = np.array([0, 1, 2])
        self.criteria = Gini()

    def test_call(self):
        """
        Tests the magic method __call__().
        """
        value = self.criteria(self.X, self.y)
        np.testing.assert_array_almost_equal(0.6666666666666665, value)


if __name__ == '__main__':
    unittest.main()
