import unittest

import numpy as np

from PyDecisionTree.question import Question


class TestQuestion(unittest.TestCase):
    """
    Tests the class Question.
    """
    def setUp(self):
        self.X = np.array([
            [0, 1, 2],
            [3, 2, 5],
            [6, 3, 8],
        ])
        self.y = np.array([0, 1, 2])
        self.question_continuous = Question(0, self.X[1, 0])

    def test_match(self):
        """
        Test the method match().
        """
        matches_continuous = self.question_continuous.match(self.X, return_indices=False)

        np.testing.assert_array_equal([False, True, True], matches_continuous)

        matches_continuous = self.question_continuous.match(self.X, return_indices=True)

        np.testing.assert_array_equal([1, 2], matches_continuous)

    def test_partition(self):
        """
        Tests the method partition().
        """
        X_left_continuous, y_left_continuous, X_right_continuous, y_right_continuous = self.question_continuous.partition(self.X, self.y)

        np.testing.assert_array_equal(self.X[1:], X_left_continuous)
        np.testing.assert_array_equal(self.y[1:], y_left_continuous)
        np.testing.assert_array_equal(self.X[:1], X_right_continuous)
        np.testing.assert_array_equal(self.y[:1], y_right_continuous)

    def test_is_numeric(self):
        """
        Tests the method is_numeric().
        """
        self.assertTrue(Question.is_numeric(1))
        self.assertTrue(Question.is_numeric(1.0))
        self.assertFalse(Question.is_numeric("A"))

    def test_is_categorical(self):
        """
        Tests the method is_categorical().
        """
        self.assertFalse(Question.is_categorical(1))
        self.assertFalse(Question.is_categorical(1.0))
        self.assertTrue(Question.is_categorical("A"))


if __name__ == '__main__':
    unittest.main()
