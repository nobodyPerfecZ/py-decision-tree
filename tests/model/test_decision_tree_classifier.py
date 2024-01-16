import unittest

import numpy as np

from PyDecisionTree.model import DecisionTreeClassifier


class TestDecisionTreeClassifier(unittest.TestCase):

    def setUp(self):
        self.X = np.array([
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
        ])
        self.y = np.array([0, 1, 2])
        self.X_test = np.array([
            [0, 0, 0],
            [0, 1, 2],
            [9, 10, 11],
        ])
        self.model = DecisionTreeClassifier(
            splitter="weighted_entropy",
            strategy="best",
            max_features="all",
            random_state=0
        )

    def test_fit(self):
        """
        Tests the method fit().
        """
        self.model.fit(self.X, self.y)
        self.assertIsNotNone(self.model._root)

    def test_predict(self):
        """
        Tests the method predict().
        """
        self.model.fit(self.X, self.y)
        y_pred = self.model.predict(self.X_test)
        np.testing.assert_array_equal([0, 0, 2], y_pred)


if __name__ == "__main__":
    unittest.main()
