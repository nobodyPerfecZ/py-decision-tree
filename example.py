from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

from PyDecisionTree.model import DecisionTreeClassifier

if __name__ == "__main__":
    random_state = 0

    # Load the iris dataset
    X, y = load_iris(return_X_y=True)

    # Split data into train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=random_state)

    # Initialize the decision tree classifier
    model = DecisionTreeClassifier(random_state=random_state)

    # Train the model
    model.fit(X_train, y_train)

    # Print the resulting decision tree
    # model.print_tree()

    # Predict new values with the test dataset
    y_pred = model.predict(X_test)

    # Calculate the balanced accuracy score
    acc = balanced_accuracy_score(y_test, y_pred)
    print(f"Balanced Accuracy: {acc}")