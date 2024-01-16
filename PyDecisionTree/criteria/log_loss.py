import numpy as np

from PyDecisionTree.criteria.abstract_criteria import Criteria


class LogLoss(Criteria):
    def _call(self, X: np.ndarray, y: np.ndarray) -> float:
        # TODO: Implement here + Unittests + Documentation
        """
        y_mode = scipy.stats.mode(y)[0]
        count = np.count_nonzero(y == y_mode)
        p_i = count / len(y)
        return - y_mode * np.log()
        """
        raise
