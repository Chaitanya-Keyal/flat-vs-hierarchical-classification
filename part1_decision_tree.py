"""Part 1: a decision tree split by Gini decrease or information gain, with sample weights.

Trains a flat tree on Species with each criterion, derives the family from the predicted
species, and sweeps max_depth. Parts 2-4 reuse the tree.

    python part1_decision_tree.py      (from the folder holding Frogs_MFCCs.csv)
"""

import numpy as np

from common import CRITERIA, Data, accuracy, header, pct, print_rows

# Gains closer than this count as tied, so rounding noise never decides a tie, and a split
# needs a gain above it to count as positive.
TOL = 1e-12
DEPTHS = (1, 2, 3, 5, 8, 12, None)


def share(class_weights, total):
    """Class weights over their total, so class probabilities. A side whose total is 0 (its
    rows too light to register, as AdaBoost's weights get after many rounds) gets zeros, and
    with its weight W_j = 0 it then adds nothing to the gain."""
    total = total[..., None]
    return np.divide(class_weights, total, out=np.zeros_like(class_weights), where=total > 0)


def impurity(p, criterion):
    """Gini, 1 - sum p_i^2, or entropy in bits, -sum p_i log2 p_i with 0 log2 0 := 0, of the
    class probabilities p along the last axis."""
    if criterion == "gini":
        return 1 - (p * p).sum(axis=-1)
    log_p = np.log2(p, out=np.zeros_like(p), where=p > 0)
    return -(p * log_p).sum(axis=-1)


class DecisionTree:
    """Binary splits x_j <= t, each chosen by the largest gain over class weights.

    Ties on gain go to the lower feature index, then the lower threshold; leaves predict the
    weighted majority class, ties going to the smallest label. With max_features set, every
    node that searches for a split first draws that many candidate features with rng.
    """

    def __init__(
        self, criterion="gini", max_depth=None, min_samples_split=2, max_features=None, rng=None
    ):
        if criterion not in CRITERIA:
            raise ValueError(f"criterion must be 'gini' or 'entropy', not {criterion!r}")
        if max_features is not None and rng is None:
            raise ValueError("max_features needs an rng to draw the features")
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.rng = rng

    def fit(self, X, y, sample_weight=None):
        X = np.ascontiguousarray(X, dtype=float)
        self.classes_, codes = np.unique(y, return_inverse=True)
        n = len(codes)
        w = np.full(n, 1 / n) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        # row i holds w_i in the column of its class, so summing rows gives class weights
        onehot = np.zeros((n, len(self.classes_)))
        onehot[np.arange(n), codes] = w
        self.nodes = []
        self._grow(X, codes, onehot, np.arange(n), 0)
        # the same tree as flat arrays, for a vectorised predict
        for key in ("feature", "threshold", "left", "right", "value"):
            setattr(self, key, np.array([node[key] for node in self.nodes]))
        return self

    def _grow(self, X, codes, onehot, rows, depth):
        """Add the node holding `rows`, then its subtrees, depth first and left first."""
        weights = onehot[rows].sum(axis=0)
        node = {"feature": -1, "threshold": 0.0, "left": -1, "right": -1, "gain": 0.0}
        node.update(value=int(weights.argmax()), n=len(rows), weights=weights, depth=depth)
        index = len(self.nodes)
        self.nodes.append(node)
        if (
            (codes[rows] == codes[rows[0]]).all()
            or len(rows) < self.min_samples_split
            or (self.max_depth is not None and depth >= self.max_depth)
        ):
            return index
        split = self._best_split(X, onehot, rows, weights)
        if split is not None:
            node["feature"], node["threshold"], node["gain"], left, right = split
            node["left"] = self._grow(X, codes, onehot, left, depth + 1)
            node["right"] = self._grow(X, codes, onehot, right, depth + 1)
        return index

    def _best_split(self, X, onehot, rows, weights):
        """(feature, threshold, gain, left rows, right rows) of the best split of the node
        holding `rows`, or None if no split has positive gain. Every threshold of every
        candidate feature is scored at once."""
        d = X.shape[1]
        if self.max_features is None:
            features = np.arange(d)
        else:  # a fresh random subset at every split
            features = np.sort(self.rng.choice(d, self.max_features, replace=False))
        x = X[np.ix_(rows, features)]
        order = np.argsort(x, axis=0, kind="stable")
        x = np.take_along_axis(x, order, axis=0)
        # class weights left of each cut (after the first 1, 2, ..., n-1 sorted rows) and right
        left = np.cumsum(onehot[rows][order], axis=0)[:-1]
        right = np.maximum(weights - left, 0)  # clip rounding below zero
        W, w_left, w_right = weights.sum(), left.sum(axis=2), right.sum(axis=2)
        gain = (
            impurity(weights / W, self.criterion)
            - w_left / W * impurity(share(left, w_left), self.criterion)
            - w_right / W * impurity(share(right, w_right), self.criterion)
        )
        gain[x[1:] == x[:-1]] = -np.inf  # thresholds lie between distinct values only
        best = gain.max()
        if not best > TOL:
            return None
        # the first near-best cut in feature order, then in threshold order
        near = (gain >= best - TOL).T
        f, k = np.unravel_index(np.argmax(near), near.shape)
        a, b = x[k, f], x[k + 1, f]
        t = (a + b) / 2
        if not a <= t < b:  # a and b are adjacent floats
            t = a
        left_rows, right_rows = np.sort(rows[order[: k + 1, f]]), np.sort(rows[order[k + 1 :, f]])
        return int(features[f]), float(t), float(gain[k, f]), left_rows, right_rows

    def apply(self, X):
        """The leaf each row of X lands in; all rows move down the tree together."""
        X = np.asarray(X, dtype=float)
        node = np.zeros(len(X), dtype=int)
        active = np.flatnonzero(self.feature[node] >= 0)
        while active.size:
            at = node[active]
            go_left = X[active, self.feature[at]] <= self.threshold[at]
            node[active] = np.where(go_left, self.left[at], self.right[at])
            active = active[self.feature[node[active]] >= 0]
        return node

    def predict(self, X):
        return self.classes_[self.value[self.apply(X)]]

    @property
    def n_leaves(self):
        return int((self.feature < 0).sum())

    @property
    def depth(self):
        return max(node["depth"] for node in self.nodes)


def sanity_checks():
    """The handout's checks (a)-(d), printed, so that a failure stops the script."""
    print("\nSanity checks")
    p = np.array([3, 3]) / 6
    gini, entropy = impurity(p, "gini"), impurity(p, "entropy")
    assert np.isclose(gini, 0.5) and np.isclose(entropy, 1)
    print(f"  (a) class counts (3, 3): Gini {gini:.4f}, entropy {entropy:.4f} bit")

    X, y = np.array([[1.0], [2.0], [3.0], [4.0]]), np.array([0, 0, 1, 1])
    tree = DecisionTree("entropy").fit(X, y)
    root, acc = tree.nodes[0], accuracy(y, tree.predict(X))
    assert tree.n_leaves == 2 and root["threshold"] == 2.5
    assert np.isclose(root["gain"], 1) and acc == 1
    print(
        f"  (b) X = 1 2 3 4, y = 0 0 1 1: one split, x <= {root['threshold']:.2f}, "
        f"information gain {root['gain']:.4f} bit, train accuracy {acc:.2f}"
    )

    X, y, w = np.zeros((3, 1)), np.array([0, 0, 1]), np.array([0.1, 0.1, 0.8])
    tree = DecisionTree("gini").fit(X, y, sample_weight=w)
    pred = tree.predict(X[:1])[0]
    assert tree.n_leaves == 1 and pred == 1
    print(f"  (c) 3 identical rows, y = 0 0 1, weights 0.1 0.1 0.8: one leaf, predicting {pred}")

    X, y = np.arange(6.0).reshape(3, 2), np.array(["Rhinellagranulosa"] * 3)
    tree = DecisionTree("entropy").fit(X, y)
    assert tree.n_leaves == 1 and (tree.predict(X) == y).all()
    print(f"  (d) a single class: one leaf, predicting {tree.predict(X[:1])[0]}")
    print("  All passed.")


def main():
    data = Data()
    header("Part 1: single decision tree, flat (predicts Species)", data)
    sanity_checks()

    rows, sizes = [], {}
    for criterion in CRITERIA:
        tree = DecisionTree(criterion).fit(data.X_train, data.y_train)
        rows.append(("1. Single Decision Tree", criterion.capitalize(), *data.scores(tree)))
        sizes[criterion] = {"depth": tree.depth, "leaves": tree.n_leaves}
    print_rows(rows)
    print("Family %: the family of each predicted species, by the training rows' species map.")
    print("\nUnconstrained trees")
    for criterion, size in sizes.items():
        print(f"  {criterion.capitalize():<8} depth {size['depth']}, {size['leaves']} leaves")

    print("\nAccuracy against max_depth (not part of the matrix)")
    print(
        f"{'max_depth':<10}"
        + "".join(f"{c.capitalize() + ' train':>16}{'test':>7}" for c in CRITERIA)
    )
    sweep = {c: {} for c in CRITERIA}
    for depth in DEPTHS:
        line = f"{depth!s:<10}"
        for criterion in CRITERIA:
            tree = DecisionTree(criterion, max_depth=depth).fit(data.X_train, data.y_train)
            train = accuracy(data.y_train, tree.predict(data.X_train))
            test = accuracy(data.y_test, tree.predict(data.X_test))
            sweep[criterion][str(depth)] = (train, test)
            line += f"{pct(train):>16}{pct(test):>7}"
        print(line)
    return {"rows": rows, "sizes": sizes, "depth_sweep": sweep}


if __name__ == "__main__":
    main()
