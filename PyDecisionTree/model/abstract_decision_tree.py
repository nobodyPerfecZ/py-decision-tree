from abc import ABC, abstractmethod
from typing import Optional, Callable

import numpy as np

from PyDecisionTree.splitter import Splitter
from PyDecisionTree.tree import Question, Node, LeafNode, DecisionNode


class DecisionTree(ABC):
    """
    Implementation of a Decision Tree (either for classification or regression) based on the
    Continuous And Regression Tree (CART) algorithm.

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

        max_depth (int, optional):
            The maximum depth of the decision tree. It is used for regularization
            and prevents from overfitting.

        min_samples_split (int, optional):
            The minimum number of samples needed to generate a new decision node.
            It is used for regularization and prevents from overfitting.

        min_samples_leaf (int, optional):
            The minimum number of samples needed to generate new child nodes.
            It is used for regularization and prevents from overfitting.

        random_state (int, optional):
            The seed for the random number generator (used for reproducibility).
    """

    def __init__(
            self,
            tree_type: str,
            splitter: str,
            strategy: str,
            max_features: str,
            max_depth: Optional[int] = None,
            min_samples_split: int = 2,
            min_samples_leaf: int = 1,
            random_state: Optional[int] = None,
    ):
        self._tree_type = self._get_tree_type(tree_type)
        self._splitter = self._get_splitter(splitter)
        self._strategy = self._get_strategy(strategy)
        self._max_features = self._get_max_features(max_features)
        self._max_depth = self._get_max_depth(max_depth)
        self._min_samples_split = self._get_min_samples_split(min_samples_split)
        self._min_samples_leaf = self._get_min_samples_leaf(min_samples_leaf)
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

                if len(X_left) == 0 or len(y_left) == 0 or len(X_right) == 0 or len(y_right) == 0:
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

    def _fit(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
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
        if len(np.unique(y)) == 1:
            # Case: No further splits are necessary:
            return LeafNode(X=X, y=y)

        if self._max_depth is not None and self._max_depth == depth:
            # Case: Maximum depth is reached
            return LeafNode(X=X, y=y)

        if self._min_samples_split > len(y):
            # Case: Has less than minimum number of splits
            return LeafNode(X=X, y=y)

        # Find the best split of the current data
        question, loss = self._find_best_split(X, y)

        if loss == np.inf:
            # Case: No further splits are possible
            return LeafNode(X=X, y=y)

        # Split the data
        X_left, y_left, X_right, y_right = question.partition(X, y)

        if len(X_left) < self._min_samples_leaf or len(y_left) < self._min_samples_leaf or \
                len(X_right) < self._min_samples_leaf or len(y_right) < self._min_samples_leaf:
            return LeafNode(X=X, y=y)

        # Recursively build the left branch
        branch_left = self._fit(X_left, y_left, depth + 1)

        # Recursively build the right branch
        branch_right = self._fit(X_right, y_right, depth + 1)

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

    def _get_max_depth(self, max_depth: Optional[int]) -> Optional[int]:
        """
        Returns the max_depth of the decision tree, according to the given argument.

        Args:
            max_depth (int):
                The argument we want to check

        Returns:
            Optional[int]:
                The max_depth of the decision tree
        """
        assert max_depth is None or max_depth >= 1, \
            f"Illegal max_depth {max_depth}! The argument should be None or >= 1!"
        return max_depth

    def _get_min_samples_split(self, min_samples_split: int) -> int:
        """
        Returns the min_samples_split of the decision tree, according to the given argument.

        Args:
            min_samples_split (int):
                The argument we want to check.

        Returns:
            int:
                The min_samples_split of the decision tree
        """
        assert min_samples_split >= 2, f"Illegal min_samples_split {min_samples_split}! The argument should be >= 2!"
        return min_samples_split

    def _get_min_samples_leaf(self, min_samples_leaf: int) -> int:
        assert min_samples_leaf >= 1, f"Illegal min_samples_leaf {min_samples_leaf}! The argument should be >= 1!"
        return min_samples_leaf

    def _print_tree(self, node: Node, spacing: str = ""):
        """
        Prints the current node and its left and right neighbors if they exist.

        Args:
            node (Node):
                The current node we want to print.

            spacing (str, optional):
                The spaces to separate the depth of the decision tree
        """
        # Base case: we've reached a leaf
        if isinstance(node, LeafNode):
            print(spacing + "Predict", node.y)
            return

        # Print the question at this node
        print(spacing + str(node.question))

        # Call this function recursively on the true branch
        print(spacing + '--> True:')
        self._print_tree(node.branch_left, spacing + "  ")

        # Call this function recursively on the false branch
        print(spacing + '--> False:')
        self._print_tree(node.branch_right, spacing + "  ")

    def print_tree(self):
        """
        Prints the decision tree, starting from the root node.
        """
        self._print_tree(self._root)

