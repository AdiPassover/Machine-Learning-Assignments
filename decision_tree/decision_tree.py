from typing import Optional

class DecisionTree:
    class TreeNode:
        def __init__(self, feature=None, threshold=None, left=None, right=None):
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.label = None  # Only used for leaves

        @staticmethod
        def leaf(label):
            leaf = DecisionTree.TreeNode()
            leaf.label = label
            return leaf

        def is_leaf(self):
            return self.label is not None

        def propagate(self, x):
            if self.is_leaf():
                return self.label
            if x[self.feature] <= self.threshold:
                return self.left.propagate(x)
            else:
                return self.right.propagate(x)

        def __str__(self):
            if self.is_leaf():
                return f"Leaf({self.label})"
            left_preview = self.left.threshold
            right_preview = self.right.threshold
            if self.left.is_leaf():
                left_preview = self.left.label
            if self.right.is_leaf():
                right_preview = self.right.label
            return f"Node(feature={self.feature}, threshold={self.threshold}), left={left_preview}, right={right_preview})"


    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.root : Optional[DecisionTree.TreeNode] = None


    def train(self, X: list[tuple[float,...]], y: list[str]) -> None:
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        # Stopping condition, choose majority label
        if depth == self.max_depth or len(set(y)) == 1:
            majority_label = max(set(y), key=y.count)
            return self.TreeNode.leaf(majority_label)

        # Find best split
        best_feature, best_threshold, best_gain = None, None, -1
        for feature_index in range(len(X[0])):
            thresholds = sorted(set(x[feature_index] for x in X))
            for t in thresholds:
                left_y = [y[i] for i in range(len(X)) if X[i][feature_index] <= t]
                right_y = [y[i] for i in range(len(X)) if X[i][feature_index] > t]
                gain = self._information_gain(y, left_y, right_y)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_index
                    best_threshold = t

        if best_feature is None:
            majority_label = max(set(y), key=y.count)
            return self.TreeNode.leaf(majority_label)

        left_X = [X[i] for i in range(len(X)) if X[i][best_feature] <= best_threshold]
        left_y = [y[i] for i in range(len(X)) if X[i][best_feature] <= best_threshold]
        right_X = [X[i] for i in range(len(X)) if X[i][best_feature] > best_threshold]
        right_y = [y[i] for i in range(len(X)) if X[i][best_feature] > best_threshold]

        left_node = self._build_tree(left_X, left_y, depth + 1)
        right_node = self._build_tree(right_X, right_y, depth + 1)

        return self.TreeNode(feature=best_feature, threshold=best_threshold, left=left_node, right=right_node)

    def _entropy(self, labels):
        from math import log2
        total = len(labels)
        if total == 0:
            return 0
        counts = {label: labels.count(label) for label in set(labels)}
        return -sum((count / total) * log2(count / total) for count in counts.values())

    def _information_gain(self, parent, left, right):
        total = len(parent)
        if total == 0:
            return 0
        return (self._entropy(parent)
                - (len(left) / total) * self._entropy(left)
                - (len(right) / total) * self._entropy(right))


    def predict(self, X):
        return [self.root.propagate(x) for x in X]

    def __str__(self):
        levels = {}
        q = [(self.root, 0)]
        while q:
            node, level = q.pop(0)
            levels[level] = levels.get(level, []) + [node]
            if node.is_leaf():
                continue
            q.append((node.left, level+1))
            q.append((node.right, level+1))

        tree_str = ""
        for level, nodes in levels.items():
            tree_str += f"Level {level}:\n"
            for node in nodes:
                tree_str += f"  {node}\n"
        return tree_str.strip()  # Remove trailing newline



