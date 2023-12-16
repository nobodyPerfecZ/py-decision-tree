from typing import Union, Any

import numpy as np

Numeric_like = Union[int, float, np.float_, np.int_]
Continuous_like = Union[str, np.str_]


class Question:
    """
    A Question represents the splitting statement of a decision tree.

    This class just records the column index (e.g. 0 for Color)
    and a value inside the column (e.g. Green). The 'match' method is
    used to compare the feature values of a column stored in the question.

    Args:
        column (int):
            The corresponding feature (column) index
            from a tabular dataset X with shape (N, ?)

        x (Union[Numeric_like, Continuous_like]):
            The corresponding value to form the question
    """

    def __init__(self, column: int, x: Union[Numeric_like, Continuous_like]):
        assert Question.is_numeric(x) or Question.is_categorical(x), \
            f"Illegal x {x}! It should be a numerical or categorical value!"

        self._column = column
        self._x = x
        self._is_numeric = Question.is_numeric(x)
        self._is_categorical = Question.is_categorical(x)

    @property
    def column(self) -> int:
        return self._column

    @property
    def x(self) -> Union[Numeric_like, Continuous_like]:
        return self._x

    def _match(self, x: Any) -> bool:
        """
        Returns True, if the given value is (...)
            - numeric and value >= the question value
            - categorical and value == the question value

        Args:
            value (Any):
                The value to check

        Returns:
            bool:
                True if the value is inside the question value
        """
        if self._is_numeric:
            # Case: Check if given value is higher or equal to x
            return x >= self._x
        else:
            # Case: Check if given value is equal to x
            return x == self._x

    def match(self, X: np.ndarray, return_indices: bool = True) -> np.ndarray:
        """
        Returns the boolean mask/indices for checking on column in dataset X, if (...)
            - continuous: value >= x
            - categorical: value == x

        Args:
            X (np.ndarray):
                The feature vector of shape (N, ?)

            return_indices (bool, optional):
                Controls if the indices should be returned,
                where the condition holds true

        Returns:
            mask (np.ndarray, optional):
                The boolean mask of shape (N,), if the condition holds true

            indices (np.ndarray, optional):
                The corresponding indices where the condition holds true
        """
        assert X.ndim == 2, f"Illegal X {X.ndim}. The dimension of X should be (N, ?)!"

        # Vectorize the function to speedup the execution
        vectorized_match = np.vectorize(pyfunc=self._match)

        # Check which dtype should be used
        if self._is_numeric:
            dtype = float
        else:
            dtype = str

        # Get the boolean mask
        matches = vectorized_match(X[:, self._column].astype(dtype))

        if return_indices:
            # Case: Return the indices of true conditions
            return np.where(matches)[0]
        else:
            # Case: Return the boolean mask
            return matches

    def partition(
            self,
            X: np.ndarray,
            y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Partitions a given dataset, according to the question.

        Args:
            X (np.ndarray):
                The feature matrix of shape (N, ?)

            y (np.ndarray):
                The true outputs of shape (N,)

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                X_left (np.ndarray):
                    The feature matrix of shape (?, ?),
                    where the question holds true

                y_left (np.ndarray):
                    The true outputs of shape (?,),
                    where the question holds true

                X_right (np.ndarray):
                    The feature matrix of shape (?, ?),
                    where the question holds false

                y_right (np.ndarray):
                    The true outputs of shape (?,),
                    where the question holds false
        """
        mask = self.match(X, return_indices=False)

        X_left = X[mask]
        y_left = y[mask]
        X_right = X[~mask]
        y_right = y[~mask]

        return X_left, y_left, X_right, y_right

    @staticmethod
    def is_numeric(x: Any) -> bool:
        """
        Returns True if the given value is numerical.

        Args:
            x (Any):
                The value to check

        Returns:
            bool:
                True if the value is numerical.
        """
        return isinstance(x, (int, float, np.int_, np.float_))

    @staticmethod
    def is_categorical(x: Any) -> bool:
        """
        Returns True if the given value is categorical.

        Args:
            x (Any):
                The value to check

        Returns:
            bool:
                True if the value is categorical
        """
        return isinstance(x, (str, np.str_))

    def __eq__(self, other) -> bool:
        if isinstance(other, Question):
            return (self.column == other.column) and (self.x == other.x)
        raise NotImplementedError

    def __str__(self):
        if Question.is_numeric(self._x):
            condition = ">="
        else:
            condition = "=="

        return f"Is X{self._column} {condition} {self._x}?"

    def __repr__(self):
        return self.__str__()
