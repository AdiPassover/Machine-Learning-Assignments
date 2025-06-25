from typing import Callable, Optional
from knn import KNearestNeighbors
import numpy as np
import random

SETOSA = "Iris-setosa"
VERSICOLOR = "Iris-versicolor"
VIRGINICA = "Iris-virginica"


def parse_dataset(filename: str, classes: list[str]) -> tuple[list[tuple[float, ...]], list[str]]:
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
                    feature = tuple([float(x) for x in parts[:-1]])
                    features.append(feature)
                    labels.append(label)
                except ValueError:
                    continue  # Skip lines with invalid data

    return features, labels

def train_test_split(features: list[tuple[float,...]], labels: list[str], test_size: float = 0.5, random_state: Optional[int] = 42) -> tuple[list[tuple[float,...]], list[str], list[tuple[float,...]], list[str]]:
    if random_state is None:
        np.random.seed(random.randint(0, 10000))
    else:
        np.random.seed(random_state)

    n = len(features)
    indices = np.random.permutation(n)

    n_test = int(n * test_size)
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


def test_knn(classes : list[str], num_tests : int, Ks : list[int], distance_functions: list[callable], function_names: Optional[list[str]] = None) -> None:
    features, labels = parse_dataset('iris.txt', classes)

    for k in Ks:
        for i in range(len(distance_functions)):
            distance_function = distance_functions[i]
            function_name = function_names[i] if function_names else f"Distance Function {i+1}"

            train_error_sum = 0
            test_error_sum = 0
            for _ in range(num_tests):
                X_train, y_train, X_test, y_test = train_test_split(features, labels, random_state=None)

                knn = KNearestNeighbors(k, distance_function)
                knn.train(X_train, y_train)

                train_pred = knn.predict(X_train)
                test_pred = knn.predict(X_test)

                train_error_sum += calc_error(train_pred, y_train)
                test_error_sum += calc_error(test_pred, y_test)

            avg_train_error = train_error_sum / num_tests
            avg_test_error = test_error_sum / num_tests
            print(f"K={k}\t Distance Function={function_name}\t avg train error: {avg_train_error:.2f}%\t "
                  f"avg test error: {avg_test_error:.2f}%\t difference: {avg_train_error - avg_test_error:.2f}%")



def Lp_distance(p: float) -> Callable[[tuple[float, ...], tuple[float, ...]], float]:
    if p == float('inf'):
        def distance(x: tuple[float, ...], y: tuple[float, ...]) -> float:
            return max(abs(x_i - y_i) for x_i, y_i in zip(x, y))
        return distance

    def distance(x: tuple[float, ...], y: tuple[float, ...]) -> float:
        return sum(abs(x_i - y_i) ** p for x_i, y_i in zip(x, y)) ** (1 / p)
    return distance


if __name__ == "__main__":
    classes = [VERSICOLOR, VIRGINICA]

    num_tests = 100

    Ks = [1, 3, 5, 7, 9]

    Ps = [1, 2, float('inf')]
    distance_functions = [Lp_distance(p) for p in Ps]
    function_names = ['L_1  ', 'L_2  ', 'L_inf']

    test_knn(classes, num_tests, Ks, distance_functions, function_names)

