from typing import Callable

class KNearestNeighbors:

    def __init__(self, k: int, distance_function: Callable[[tuple, tuple], float]):
        self.k = k
        self.distance_function = distance_function


    def train(self, X: list[tuple[float,...]], y: list[str]) -> None:
        self.X_train = X
        self.y_train = y


    def predict_query(self, x: tuple[float,...]) -> str:
        distances = [(self.distance_function(x, x_train), label) for x_train, label in zip(self.X_train, self.y_train)]
        distances.sort(key=lambda pair: pair[0])

        nearest_neighbors = distances[:self.k]
        votes = {}

        for _, label in nearest_neighbors:
            votes[label] = votes.get(label, 0) + 1

        return max(votes.items(), key=lambda item: item[1])[0]

    def predict(self, X: list[tuple[float,...]]) -> list[str]:
        return [self.predict_query(x) for x in X]