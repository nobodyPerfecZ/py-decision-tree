# py-decision-tree
Simple Python Implementation of a Decision Tree with Numpy.

# PyDecisionTree
PyDecisionTree is a simple Python Framework for using [Decision Trees (DTs)](https://scikit-learn.org/stable/modules/tree.html).
The decision tree is based on the Continuous and Categorical Trees (CART) algorithm.
Currently, the implementation cannot handle categorical data well, because of NumPys implementation of arrays with 
multiple dtypes!

### Using DTs with PyDecisionTree
For the following we want to train a DT Classifier for the [iris dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html#sklearn.datasets.load_iris):

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

random_state = 0

# Load the iris dataset
X, y = load_iris(return_X_y=True)

# Split data into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=random_state)
```

First we initialize our DTs model:

```python
from PyDecisionTree.model import DecisionTreeClassifier

# Initialize the decision tree classifier
model = DecisionTreeClassifier(random_state=random_state)
```

Now we have to use the `.fit()` method to build the decision tree:

```python
# Train the model
model.fit(X_train, y_train)
```

After fitting the model we can now use the DT to make inferences on new unseen data points:

```python
# Predict new values with the test dataset
y_pred = model.predict(X_test)
```

### Future Features
The following list defines features, that are currently on work:
* [ ] Implement a better visualization tool for the trained decision tree
* [ ] Adjust the code to also handle categorical features
* [ ] Implement the log loss as an splitter option for decision tree classifier
* [x] Adjust workflow file to test on windows & linux

