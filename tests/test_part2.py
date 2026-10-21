"""Bagging's bootstraps and vote, and SAMME against the handout's rules and binary AdaBoost."""

import numpy as np

from part1_decision_tree import DecisionTree
from part2_bagging_boosting import AdaBoostSAMME, Bagging


def blobs(seed, n=300, k=3):
    """k overlapping Gaussian classes in 4 dimensions."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, size=n)
    return rng.normal(size=(n, 4)) + y[:, None] * 0.8, y


def nodes(tree):
    return [(nd["feature"], nd["threshold"], nd["value"]) for nd in tree.nodes]


def test_bagging_trees_are_fitted_on_the_handout_bootstraps():
    X, y = blobs(0)
    model = Bagging(5, "entropy").fit(X, y)
    rng = np.random.RandomState(42)
    for tree in model.trees_:
        rows = rng.randint(0, len(y), size=len(y))
        assert nodes(tree) == nodes(DecisionTree("entropy").fit(X[rows], y[rows]))


def test_bagging_refits_identically():
    X, y = blobs(1)
    model = Bagging(7, "gini")
    first = model.fit(X, y).predict(X)
    assert np.array_equal(first, model.fit(X, y).predict(X))


def test_bagging_predicts_the_majority_of_its_trees():
    X, y = blobs(2)
    model = Bagging(6, "gini").fit(X, y)
    votes = np.array([tree.predict(X) for tree in model.trees_])
    for i, label in enumerate(model.predict(X)):
        counts = {c: (votes[:, i] == c).sum() for c in np.unique(y)}
        top = max(counts.values())
        assert label == min(c for c, v in counts.items() if v == top)


def classic_adaboost(X, y, T, max_depth):
    """Binary AdaBoost with labels -1/+1: alpha = 1/2 ln((1 - eps) / eps)."""
    s = np.where(y == 1, 1, -1)
    w = np.full(len(y), 1 / len(y))
    trees, alphas = [], []
    for _ in range(T):
        tree = DecisionTree("gini", max_depth=max_depth).fit(X, y, sample_weight=w)
        h = np.where(tree.predict(X) == 1, 1, -1)
        eps = np.clip(w[h != s].sum(), 1e-10, 1 - 1e-10)
        if eps >= 0.5:
            break
        alpha = 0.5 * np.log((1 - eps) / eps)
        w = w * np.exp(-alpha * s * h)
        w /= w.sum()
        trees.append(tree)
        alphas.append(alpha)

    def predict(Z):
        score = sum(a * np.where(t.predict(Z) == 1, 1, -1) for a, t in zip(alphas, trees))
        return (score > 0).astype(int)

    return predict


def test_samme_with_two_classes_is_classic_adaboost():
    X, y = blobs(3, k=2)
    samme = AdaBoostSAMME(20, "gini", max_depth=1).fit(X, y)
    classic = classic_adaboost(X, y, 20, max_depth=1)
    Z = np.random.default_rng(4).normal(size=(500, 4)) + 0.4
    assert np.array_equal(samme.predict(Z), classic(Z))


def test_samme_keeps_only_learners_better_than_chance():
    X, y = blobs(5, k=4)
    model = AdaBoostSAMME(30, "entropy", max_depth=2).fit(X, y)
    assert len(model.trees_) == len(model.alphas_) == len(model.errors_) > 0
    assert all(e < 1 - 1 / 4 for e in model.errors_)
    assert all(a > 0 for a in model.alphas_)


def test_samme_first_error_is_the_plain_error_rate():
    X, y = blobs(6)
    model = AdaBoostSAMME(1, "gini", max_depth=2).fit(X, y)
    tree = DecisionTree("gini", max_depth=2).fit(X, y)
    assert np.isclose(model.errors_[0], np.mean(tree.predict(X) != y))


def test_samme_stops_when_the_first_learner_is_no_better_than_chance():
    # XOR: no split has positive gain, so the stump is one leaf with error 1/2 = 1 - 1/K
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([0, 1, 1, 0])
    model = AdaBoostSAMME(10, "gini", max_depth=1).fit(X, y)
    assert model.trees_ == []
    assert (model.predict(X) == 0).all()  # every score is 0: the smallest label
