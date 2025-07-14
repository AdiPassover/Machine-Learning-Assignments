from numpy import random as np_random
import random
from typing import Optional

from decision_tree import DecisionTree

SETOSA = "Iris-setosa"
VERSICOLOR = "Iris-versicolor"
VIRGINICA = "Iris-virginica"


def parse_dataset(filename: str, classes: list[str], columns: list[int]) -> tuple[list[tuple[float, ...]], list[str]]:
    features = []
    labels = []

    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split(' ')
            if len(parts) == 5:
                try:
                    label = parts[-1]
                    if label not in classes:
                        continue
                    # Extract only the specified columns for features
                    feature = tuple(float(parts[i]) for i in columns)
                    features.append(feature)
                    labels.append(label)
                except ValueError:
                    continue  # Skip lines with invalid data

    return features, labels

def train_test_split(features: list[tuple[float,...]], labels: list[str], test_size: float = 0.5, random_state: Optional[int] = 42) -> tuple[list[tuple[float,...]], list[str], list[tuple[float,...]], list[str]]:
    if random_state is None:
        np_random.seed(random.randint(0, 10000))
    else:
        np_random.seed(random_state)
    indices = np_random.permutation(len(features))

    n_test = int(len(features) * test_size)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    X_train = [features[i] for i in train_indices]
    y_train = [labels[i] for i in train_indices]
    X_test = [features[i] for i in test_indices]
    y_test = [labels[i] for i in test_indices]

    return X_train, y_train, X_test, y_test

def calc_error(predictions: list[str], true_labels: list[str]) -> float:
    correct_predictions = sum(1 for pred, true in zip(predictions, true_labels) if pred == true)
    return (1 - correct_predictions / len(true_labels)) * 100  # Return error percentage

def test_decision_tree(classes: list[str], columns: list[int], num_tests: int, max_depth: int, seed:int=None) -> None:
    features, labels = parse_dataset('iris.txt', classes, columns)
    random.seed(seed)

    train_error_sum = 0
    test_error_sum = 0

    best_tree = None
    best_errors = float('inf'), float('inf')

    for _ in range(num_tests):
        X_train, y_train, X_test, y_test = train_test_split(features, labels, random_state=random.randint(0, 10000))

        tree = DecisionTree(max_depth=max_depth)
        tree.train(X_train, y_train)

        train_pred = tree.predict(X_train)
        train_error = calc_error(train_pred, y_train)
        train_error_sum += train_error
        test_pred = tree.predict(X_test)
        test_error = calc_error(test_pred, y_test)
        test_error_sum += test_error

        if test_error < best_errors[1]:
            best_errors = train_error, test_error
            best_tree = tree

    avg_train_error = train_error_sum / num_tests
    avg_test_error = test_error_sum / num_tests

    print(f"Decision Tree (max_depth={max_depth}):")
    print(f"Average Training Error: {avg_train_error:.2f}%")
    print(f"Average Testing Error: {avg_test_error:.2f}%\n")
    if best_tree:
        print(f"Best Tree Structure (train error: {best_errors[0]:.2f}% test error: {best_errors[1]:.2f}%):")
        print(best_tree)



if __name__ == "__main__":

    classes = [VERSICOLOR, VIRGINICA]
    columns = [1, 2]
    num_tests = 50
    max_depth = 3

    test_decision_tree(classes=classes, columns=columns, num_tests=num_tests, max_depth=max_depth, seed=42)


