import unittest

import numpy as np

from PyDecisionTree.criteria import Entropy
from PyDecisionTree.splitter import InformationGain


class TestInformationGain(unittest.TestCase):
    """
    Tests the class InformationGain.
    """

    def setUp(self):
        self.X = np.array([
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
        ])
        self.y = np.array([0, 1, 2])

        self.X_left = np.array([
            [0, 1, 2],
        ])
        self.y_left = np.array([0])

        self.X_right = np.array([
            [3, 4, 5],
            [6, 7, 8]
        ])
        self.y_right = np.array([1, 2])

        self.splitter = InformationGain(criteria=Entropy())

    def test_call(self):
        """
        Tests the magic method __call__().
        """
        value = self.splitter(self.X, self.y, self.X_left, self.y_left, self.X_right, self.y_right)
        np.testing.assert_array_almost_equal(-0.6365141682948128, value)


if __name__ == '__main__':
    unittest.main()
