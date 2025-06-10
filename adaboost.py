import numpy as np
from typing import List, Callable, Tuple


def adaboost(X: np.ndarray, y: np.ndarray, H: List[Callable], k: int) -> Tuple[List[Callable], List[float]]:
    """
    AdaBoost algorithm implementation

    Args:
        X: Dataset features, shape (n, d) where n is number of samples
        y: Dataset labels, shape (n,) with values in {-1, +1}
        H: Set of weak classifiers (hypothesis set)
        k: Number of iterations

    Returns:
        selected_h: List of selected classifiers
        alpha: List of classifier weights
    """
    n = len(X)

    # 1. Initialize point weights D_1(x_i) = 1/n
    D = np.ones(n) / n

    selected_h = []
    alpha = []

    # 2. For iteration t=1,...,k
    for t in range(k):
        # 3. Compute weighted error for each h ∈ H:
        # ε_t(h) = Σ_{i=1}^n D_t(x_i)[h(x_i) ≠ y_i]
        best_h = None
        min_error = float('inf')

        for h in H:
            # Get predictions for all samples
            predictions = np.array([h(x) for x in X])

            # Compute weighted error
            error_mask = (predictions != y).astype(int)
            epsilon_t = np.sum(D * error_mask)

            # 4. Select classifier with min weighted error
            if epsilon_t < min_error:
                min_error = epsilon_t
                best_h = h

        h_t = best_h
        epsilon_t = min_error

        # Prevent division by zero and ensure epsilon_t < 0.5
        if epsilon_t >= 0.5:
            print(f"Warning: Error rate {epsilon_t:.4f} >= 0.5 at iteration {t + 1}")
            break

        if epsilon_t == 0:
            epsilon_t = 1e-10  # Small value to prevent log(inf)

        # 5. Set classifier weight α_t based on its error
        # α_t = (1/2) * ln((1-ε_t(h_t))/ε_t(h_t))
        alpha_t = 0.5 * np.log((1 - epsilon_t) / epsilon_t)

        selected_h.append(h_t)
        alpha.append(alpha_t)

        # 6. Update point weights
        # D_{t+1}(x_i) = (1/Z_t) * D_t(x_i) * e^{-α_t * h_t(x_i) * y_i}
        predictions = np.array([h_t(x) for x in X])

        # Update weights
        D = D * np.exp(-alpha_t * predictions * y)

        # Normalize weights so that Σ_i D_{t+1}(x_i) = 1
        Z_t = np.sum(D)  # Normalizing constant
        D = D / Z_t

        # print(f"Iteration {t + 1}: Error = {epsilon_t:.4f}, Alpha = {alpha_t:.4f}")

    return selected_h, alpha


def predict_adaboost(X: np.ndarray, selected_h: List[Callable], alpha: List[float]) -> np.ndarray:
    """
    Make predictions using the trained AdaBoost classifier

    Args:
        X: Test samples
        selected_h: List of selected weak classifiers
        alpha: List of classifier weights

    Returns:
        predictions: Final predictions
    """
    final_predictions = np.zeros(len(X))

    for h, a in zip(selected_h, alpha):
        predictions = np.array([h(x) for x in X])
        final_predictions += a * predictions

    # Return sign of weighted sum
    return np.sign(final_predictions)
