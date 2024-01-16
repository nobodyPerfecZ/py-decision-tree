import unittest

import numpy as np

from PyDecisionTree.criteria import MSE


class TestMSE(unittest.TestCase):
    """
    Tests the class MSE.
    """

    def setUp(self):
        self.X = np.array([
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
        ])
        self.y = np.array([0, 1, 2])
        self.criteria = MSE()

    def test_call(self):
        """
        Tests the magic method __call__().
        """
        value = self.criteria(self.X, self.y)
        np.testing.assert_array_almost_equal(0.6666666666666666, value)


if __name__ == '__main__':
    unittest.main()
