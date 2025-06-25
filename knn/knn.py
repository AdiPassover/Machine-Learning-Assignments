from typing import Callable

class KNearestNeighbors:

    def __init__(self, k: int, distance_function: Callable[[tuple, tuple], float]):
        self.k = k
        self.distance_function = distance_function

        self.X_train = []
        self.y_train = []


    def train(self, X: list[tuple[float,...]], y: list[str]) -> None:
        self.X_train = X
        self.y_train = y


    def get_data_size(self) -> int:
        return len(self.X_train)


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


class KNearestNeighborsCondensing(KNearestNeighbors):

    def __init__(self, k: int, distance_function: Callable[[tuple, tuple], float]):
        super().__init__(k, distance_function)


    def train(self, X: list[tuple[float,...]], y: list[str]) -> None:
        eps = self._get_epsilon(X, y)

        self.X_train = []
        self.y_train = []

        for i in range(len(X)):
            if self._dist(X[i], self.X_train) > eps:
                self.X_train.append(X[i])
                self.y_train.append(y[i])


    def _get_epsilon(self, X: list[tuple[float,...]], y: list[str]) -> float:
        min_dist = float('inf')

        for i in range(len(X)):
            for j in range(i + 1, len(X)):
                if y[i] != y[j]:
                    min_dist = min(min_dist, self.distance_function(X[i], X[j]))

        return min_dist

    def _dist(self, x: tuple[float,...], T: list[tuple[float,...]]) -> float:
        min_dist = float('inf')
        for t in T:
            min_dist = min(min_dist, self.distance_function(x, t))
        return min_dist