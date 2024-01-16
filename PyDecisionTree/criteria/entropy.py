import numpy as np

from PyDecisionTree.criteria.abstract_criteria import Criteria


class Entropy(Criteria):
    """
    Represents the entropy of a given dataset (X, y).

    The entropy is defined as follows:
        - H(X, y) = - sum_i^k p_i * log(p_i)
    """

    def _call(self, X: np.ndarray, y: np.ndarray) -> float:
        # Get the unique class indices
        class_indices = np.unique(y)

        # Compute the entropy:
        # H(D) := - sum_i^k p_i * log(p_i)
        entropy = 0.0
        for class_idx in class_indices:
            count = np.count_nonzero(y == class_idx)
            p_i = count / len(y)
            entropy -= p_i * np.log(p_i)
        return entropy
