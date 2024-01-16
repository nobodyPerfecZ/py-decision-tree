import numpy as np

from PyDecisionTree.criteria.abstract_criteria import Criteria


class Gini(Criteria):
    """
    Represents the gini impurity of a given dataset (X, y).

    The gini impurity is defined as follows:
        - Gini(X, y) := 1 - sum_i^k p_i²
    """

    def _call(self, X: np.ndarray, y: np.ndarray) -> float:
        # Get the unique class indices
        class_indices = np.unique(y)

        # Compute the gini impurity:
        # Gini(D) := 1 - sum_i^k p_i²
        gini_impurity = 1.0
        for class_idx in class_indices:
            count = np.count_nonzero(y == class_idx)
            p_i = count / len(y)
            gini_impurity -= p_i ** 2
        return gini_impurity
