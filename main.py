import random
from typing import Tuple, List, Callable

import numpy as np
from numpy import floating
import argparse

from adaboost import adaboost, predict_adaboost


def parse_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse dataset from file format: x1 x2 label
    Convert labels from {0,1} to {-1,+1} for AdaBoost
    """
    data = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                parts = line.split()
                x1, x2, label = float(parts[0]), float(parts[1]), int(parts[2])
                data.append([x1, x2, label])

    data = np.array(data)
    X = data[:, :2]  # Features
    y = data[:, 2]  # Labels

    # Convert labels from {0,1} to {-1,+1}
    y = np.where(y == 0, -1, 1)

    return X, y


def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.3, random_state: int = 42) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split dataset into train and test sets
    """
    np.random.seed(random_state)
    n = len(X)
    indices = np.random.permutation(n)

    n_test = int(n * test_size)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test


def create_line_classifiers(X: np.ndarray) -> List[Callable]:
    """
    Create classifiers based on all possible lines from pairs of points in the dataset.
    Each pair of points creates 2 rules:
    1. Above line -> +1, below line -> -1
    2. Above line -> -1, below line -> +1
    """
    H = []
    n = len(X)

    # print(f"Creating line classifiers from {n} points...")
    # print(f"This will generate {n * (n - 1)} classifiers (2 per pair of points)")

    # For each pair of points
    for i in range(n):
        for j in range(n):
            if i != j:  # Don't use the same point twice
                p1 = X[i]  # Point 1: (x1, y1)
                p2 = X[j]  # Point 2: (x2, y2)

                # Create line equation: ax + by + c = 0
                # Line through (x1,y1) and (x2,y2)
                x1, y1 = p1[0], p1[1]
                x2, y2 = p2[0], p2[1]

                # Handle vertical lines (x2 = x1)
                if abs(x2 - x1) < 1e-10:
                    # Vertical line: x = x1
                    # ax + by + c = 0 becomes: 1*x + 0*y + (-x1) = 0
                    a, b, c = 1.0, 0.0, -x1
                else:
                    # General line: y - y1 = m(x - x1) where m = (y2-y1)/(x2-x1)
                    # Rearranging: (y2-y1)x - (x2-x1)y + (x2-x1)y1 - (y2-y1)x1 = 0
                    a = y2 - y1
                    b = -(x2 - x1)
                    c = (x2 - x1) * y1 - (y2 - y1) * x1

                # Normalize coefficients to avoid numerical issues
                norm = np.sqrt(a * a + b * b)
                if norm > 1e-10:
                    a, b, c = a / norm, b / norm, c / norm

                # Create two classifiers for this line
                # Rule 1: Above line -> +1, below line -> -1
                def make_classifier_above_positive(a_val, b_val, c_val):
                    def classifier(x):
                        # ax + by + c > 0 means above the line
                        value = a_val * x[0] + b_val * x[1] + c_val
                        return 1 if value > 0 else -1

                    return classifier

                # Rule 2: Above line -> -1, below line -> +1
                def make_classifier_above_negative(a_val, b_val, c_val):
                    def classifier(x):
                        # ax + by + c > 0 means above the line
                        value = a_val * x[0] + b_val * x[1] + c_val
                        return -1 if value > 0 else 1

                    return classifier

                H.append(make_classifier_above_positive(a, b, c))
                H.append(make_classifier_above_negative(a, b, c))

    # print(f"Created {len(H)} line classifiers")
    return H


def calculate_error(y_true: np.ndarray, y_pred: np.ndarray) -> floating:
    """Calculate classification error rate"""
    return np.mean(y_true != y_pred)


# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AdaBoost on a dataset of squares.")
    parser.add_argument('--k', '-k', type=int, default=8, help='Number of weak classifiers to select')
    args = parser.parse_args()

    # Parse dataset
    # print("Parsing dataset from squares.txt...")
    try:
        X, y = parse_dataset('squares.txt')
        # print(f"Dataset loaded: {len(X)} samples, {X.shape[1]} features")
        # print(f"Class distribution: {np.sum(y == 1)} positive, {np.sum(y == -1)} negative")
    except FileNotFoundError:
        # print("Error: squares.txt file not found. Please make sure the file exists.")
        exit(1)


    train_errors = []
    test_errors = []
    NUM_RUNS = 50

    for iter in range(NUM_RUNS):
        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=random.randint(0, 10000))
        # print(f"Train set: {len(X_train)} samples")
        # print(f"Test set: {len(X_test)} samples")

        # Create weak classifiers
        # print("\nCreating decision stump classifiers...")
        H = create_line_classifiers(X_train)
        # print(f"Created {len(H)} weak classifiers")

        # Run AdaBoost
        # print("\nRunning AdaBoost...")
        selected_classifiers, classifier_weights = adaboost(X_train, y_train, H, k=args.k)

        # Make predictions
        # print("\nEvaluating performance...")
        train_predictions = predict_adaboost(X_train, selected_classifiers, classifier_weights)
        test_predictions = predict_adaboost(X_test, selected_classifiers, classifier_weights)

        # Calculate errors
        train_error = calculate_error(y_train, train_predictions)
        test_error = calculate_error(y_test, test_predictions)

        # print(f"\n{'=' * 50}")
        # print(f"RESULTS:")
        # print(f"{'=' * 50}")
        # print(f"Train Error: {train_error:.4f} ({train_error * 100:.2f}%)")
        # print(f"Test Error:  {test_error:.4f} ({test_error * 100:.2f}%)")
        # print(f"Number of selected classifiers: {len(selected_classifiers)}")
        # print(f"Classifier weights: {[f'{w:.4f}' for w in classifier_weights]}")

        # Additional statistics
        train_accuracy = 1 - train_error
        test_accuracy = 1 - test_error
        # print(f"\nTrain Accuracy: {train_accuracy:.4f} ({train_accuracy * 100:.2f}%)")
        # print(f"Test Accuracy:  {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
        print(f"Iteration: {iter + 1}/{NUM_RUNS}")

        train_errors.append(train_error)
        test_errors.append(test_error)

    # Print average results
    avg_train_error = np.mean(train_errors)
    avg_test_error = np.mean(test_errors)
    print(f"{'=' * 50}")
    print(f"RESULTS:")
    print(f"{'=' * 50}")
    print(f"Number of selected classifiers: {args.k}")
    print(f"Train Error: {avg_train_error*100:.2f}\\%)")
    print(f"Test Error:  {avg_test_error*100:.2f}\\%)")
