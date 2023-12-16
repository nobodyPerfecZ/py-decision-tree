from abc import ABC, abstractmethod
from typing import Callable, Optional

import numpy as np

from PyDecisionTree.criteria import Gini, Entropy, MSELoss, MAELoss
from PyDecisionTree.node import LeafNode, DecisionNode, Node
from PyDecisionTree.question import Question
from PyDecisionTree.splitter import Splitter, InformationGain, WeightedLoss


class DecisionTree(ABC):
    """
    Implementation of a Decision Tree (either for classification
    or regression) based on the Continuous And Regression Trees
    (CART) algorithm.

    See more information about CART here:
    https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier

    This class should not be directly called, because here you have to
    give the type of decision tree (classification or regression) as argument
    for __init__ to distinguishes between the different type of trees.

    Args:
        tree_type (str):
            The string argument to differentiate between different
            types of decision trees.

        splitter (str):
            The string argument to differentiate between different
            types of loss function, used to evaluate a question (split).

        strategy (str):
            The string argument to differentiate between different strategy
            types, used to select between the questions.

        max_features (str):
            The string argument to differentiate between different number of
            features to look for finding the best splits.

        random_state (int, optional):
            The seed for the random number generator (used for reproducibility).
    """

    def __init__(
            self,
            tree_type: str,
            splitter: str,
            strategy: str,
            max_features: str,
            random_state: Optional[int] = None,
    ):
        self._tree_type = self._get_tree_type(tree_type)
        self._splitter = self._get_splitter(splitter)
        self._strategy = self._get_strategy(strategy)
        self._max_features = self._get_max_features(max_features)
        self._rng = np.random.RandomState(random_state)

        # Attributes that gets initialized after using .fit() method
        self._root = None

    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> tuple[Question, float]:
        """
        Finds the best split, according to the given dataset (X, y) and strategy.
        If (...):
            - strategy := "best" then we search for the best split according to the loss function
            - strategy := "random" then we select the first randomly founded split

        Args:
            X (np.ndarray):
                The feature matrix of shape (N, N_features)

            y (np.ndarray):
                The true outputs of shape (N,)

        Returns:
            tuple[Question, float]:
                question (Question):
                    The best founded splitting criteria

                loss (float):
                    The output of the loss function for the best question
        """
        # Track the best founded split
        best_loss = np.inf
        best_question = None

        n_features = X.shape[1]
        max_features = int(np.ceil(self._max_features(n_features)))
        selected_features = self._rng.choice(n_features, size=max_features, replace=False)
        for column in selected_features:

            # Get the unique values of the row
            unique_values = np.unique(X[:, column])
            unique_values = self._rng.choice(unique_values, size=len(unique_values), replace=False)

            for value in unique_values:

                # Split the data into the given question
                question = Question(column, value)
                X_left, y_left, X_right, y_right = question.partition(X, y)
                if len(X_left) == 0 or len(y_left) == 0 or len(X_right) == 0 or len(y_left) == 0:
                    # Case: Skip this split if it does not divide the dataset
                    continue

                # Calculate the loss of the split
                loss = self._splitter(X, y, X_left, y_left, X_right, y_right)

                if self._strategy == "random":
                    # Case: Return the first founded split
                    best_loss = loss
                    best_question = question
                    return best_question, best_loss
                else:
                    # Case: Store the current best founded split
                    if loss < best_loss:
                        # Case: Founded better split
                        best_loss = loss
                        best_question = question
        return best_question, best_loss

    def _fit(self, X: np.ndarray, y: np.ndarray) -> Node:
        """
        Returns the root of the decision tree.

        Args:
            X (np.ndarray):
                The feature matrix of shape (N, N_features)

            y (np.ndarray):
                The true outputs of shape (N,)

        Returns:
            Node:
                The root node of the decision tree
        """
        # Find the best split of the current data
        question, loss = self._find_best_split(X, y)

        if loss == np.inf:
            # Case: No further splits are possible
            return LeafNode(X=X, y=y)

        # Split the data
        X_left, y_left, X_right, y_right = question.partition(X, y)

        # Recursively build the left branch
        branch_left = self._fit(X_left, y_left)

        # Recursively build the right branch
        branch_right = self._fit(X_right, y_right)

        return DecisionNode(question, branch_left, branch_right)

    def fit(self, X: np.ndarray, y: np.ndarray):
        assert X.ndim == 2, f"Illegal X {X.shape}. The matrix should have a shape of (N, N_features)!"
        assert y.ndim == 1, f"Illegal y {y.shape}. The matrix should have a shape of (N,)!"
        assert X.shape[0] == y.shape[0], \
            f"Illegal X {X.shape} or y {y.shape}. Both should have the same number of samples N!"

        # Build the decision tree with _fit()
        self._root = self._fit(X, y)

    def _predict(self, X: np.ndarray, node: Node) -> float:
        if X.ndim == 1:
            # Case: Change from shape (N,) to shape (1, N)
            X = np.expand_dims(X, axis=0)

        if isinstance(node, LeafNode):
            # Case: Current node is a leaf node
            mode, mean = node.predict(return_mean=True)
            if self._tree_type == "classification":
                # Case: Decision Tree is used for classification
                return mode
            else:
                # Case: Decision Tree is used for regression
                return mean

        # Case: Current node is a decision node
        next_branch = node.apply(X)
        return self._predict(X, next_branch)

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert X.ndim == 1 or (X.ndim == 2 and X.shape[1]), \
            f"Illegal X {X.shape}. The matrix should have a shape of (N_features,) or (1, N_features)!"
        assert self._root is not None, "You should use .fit() before calling this method!"

        return np.apply_along_axis(self._predict, axis=1, arr=X, node=self._root)

    def _get_tree_type(self, tree_type: str) -> str:
        tree_type_wrapper = {
            "regression": "regression",
            "classification": "classification",
        }
        if tree_type not in tree_type_wrapper:
            raise ValueError(f"Unknown tree_type {tree_type}!")
        return tree_type_wrapper[tree_type]

    @abstractmethod
    def _get_splitter(self, splitter: str) -> Splitter:
        """
        Returns the splitter according to the string argument.

        Args:
            splitter (str):
                The string argument for the splitter (loss function)

        Returns:
            Splitter:
                The loss function
        """
        pass

    def _get_strategy(self, strategy: str) -> str:
        """
        Returns the strategy according the string argument.
        If (...):
            - strategy := "best" then for each decision we select the best founded split
            - strategy := "random" then for each decision we select a random split

        Args:
            strategy (str):
                The string argument for the strategy

        Returns:
            str:
                The strategy
        """
        strategy_wrapper = {
            "best": "best",
            "random": "random",
        }
        if strategy not in strategy_wrapper:
            raise ValueError(f"Unknown strategy {strategy}!")
        return strategy_wrapper[strategy]

    def _get_max_features(self, max_features: str) -> Callable[[int], int]:
        """
        Returns the callable function according to the string argument.
        If (...):
            - max_features := "all" then we will search the best split for all features
            - max_features := "sqrt" then we will search the best split for sqrt(...)
            randomly selected features
            - max_features := "log" then we will search the best split for log(...)
            randomly selected features

        Args:
            max_features (str):
                The string argument for the max_features

        Returns:
            Callable[[int], int]:
                The callable function (int) -> int
        """
        max_features_wrapper = {
            "all": lambda x: x,
            "sqrt": np.sqrt,
            "log": np.log,
        }
        if max_features not in max_features_wrapper:
            raise ValueError("Unknown max_features {max_features}!")
        return max_features_wrapper[max_features]

    def print_tree(self, node, spacing=""):
        """World's most elegant tree printing function."""

        # Base case: we've reached a leaf
        if isinstance(node, LeafNode):
            print(spacing + "Predict", node.y)
            return

        # Print the question at this node
        print(spacing + str(node._question))

        # Call this function recursively on the true branch
        print(spacing + '--> True:')
        self.print_tree(node._branch_left, spacing + "  ")

        # Call this function recursively on the false branch
        print(spacing + '--> False:')
        self.print_tree(node._branch_right, spacing + "  ")


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

        random_state (int, optional):
            The seed for the random number generator (used for reproducibility).
    """


    def __init__(
            self,
            splitter: str,
            strategy: str,
            max_features: str,
            random_state: Optional[int] = None,
    ):
        super().__init__("classification", splitter, strategy, max_features, random_state)

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


class DecisionTreeRegressor(DecisionTree):
    """
    Implementation of a Classification Decision Tree based on the
    Continuous And Regression Trees (CART) algorithm.

    See more information about CART here:
    https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier

    This class should be directly called if you want a decision tree for regression tasks.

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

        random_state (int, optional):
            The seed for the random number generator (used for reproducibility).
    """

    def __init__(
            self,
            splitter: str,
            strategy: str,
            max_features: str,
            random_state: Optional[int] = None,
    ):
        super().__init__("regression", splitter, strategy, max_features, random_state)

    def _get_splitter(self, splitter: str) -> Splitter:
        """
        Returns the splitter according to the string argument.
        If (...):
            - splitter := "weighted_mse" then we use MSE + weighted loss as loss function
            - splitter := "weighted_mae" then we use MAE + weighted loss as loss function

        Args:
            splitter (str):
                The string argument for the splitter (loss function)

        Returns:
            Splitter:
                The loss function
        """
        splitter_wrapper = {
            "weighted_mse": (MSELoss, WeightedLoss),
            "weighted_mae": (MAELoss, WeightedLoss),
        }
        if splitter not in splitter_wrapper:
            raise ValueError(f"Unknown splitter {splitter}!")
        criteria_cls, splitter_cls = splitter_wrapper[splitter]
        return splitter_cls(criteria=criteria_cls())
