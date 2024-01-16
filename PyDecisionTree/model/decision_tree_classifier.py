from typing import Optional

from PyDecisionTree.criteria import Gini, Entropy
from PyDecisionTree.model.abstract_decision_tree import DecisionTree
from PyDecisionTree.splitter import Splitter, InformationGain, WeightedLoss


class DecisionTreeClassifier(DecisionTree):
    """
    Implementation of a Classification Decision Tree based on the
    Continuous And Regression Trees (CART) algorithm.

    See more information about CART here:
    https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier

    This class should be directly called if you want a decision tree for classification tasks.

    Args:
        splitter (str):
            The string argument to differentiate between different
            types of loss function, used to evaluate a question (split).

        strategy (str):
            The string argument to differentiate between different strategy
            types, used to select between the questions.

        max_features (str):
            The string argument to differentiate between different number of
            features to look for finding the best splits.

        max_depth (int, optional):
            The maximum depth of the decision tree. It is used for regularization
            and prevents from over fitting.

        min_samples_split (int, optional):
            The minimum number of samples needed to generate a new decision node.
            It is used for regularization and prevents from over fitting.

        min_samples_leaf (int, optional):
            The minimum number of samples needed to generate new child nodes.
            It is used for regularization and prevents from over fitting.

        random_state (int, optional):
            The seed for the random number generator (used for reproducibility).
    """

    def __init__(
            self,
            splitter: str = "weighted_entropy",
            strategy: str = "best",
            max_features: str = "all",
            max_depth: Optional[int] = None,
            min_samples_split: int = 2,
            min_samples_leaf: int = 1,
            random_state: Optional[int] = None,
    ):
        super().__init__(
            "classification",
            splitter,
            strategy,
            max_features,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            random_state
        )

    def _get_splitter(self, splitter: str) -> Splitter:
        """
        Returns the splitter according to the string argument.
        If (...):
            - splitter := "gini_gain" then we use gini-impurity + information gain as loss function
            - splitter := "entropy_gain" then we use entropy + information gain as loss function
            - splitter := "weighted_gini" then we use gini-impurity + weighted loss as loss function
            - splitter := "weighted_entropy" then we use entropy + weighted loss as loss function

        Args:
            splitter (str):
                The string argument for the splitter (loss function)

        Returns:
            Splitter:
                The loss function
        """
        splitter_wrapper = {
            "gini_gain": (Gini, InformationGain),
            "entropy_gain": (Entropy, InformationGain),
            "weighted_gini": (Gini, WeightedLoss),
            "weighted_entropy": (Entropy, WeightedLoss),
            # "weighted_log_loss": (LogLoss, WeightedLoss),
        }
        if splitter not in splitter_wrapper:
            raise ValueError(f"Unknown splitter {splitter}!")
        criteria_cls, splitter_cls = splitter_wrapper[splitter]
        return splitter_cls(criteria=criteria_cls())
