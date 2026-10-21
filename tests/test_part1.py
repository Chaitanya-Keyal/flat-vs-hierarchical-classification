"""The decision tree: the handout's checks, a brute-force reference for the split search,
weights, the tie rules, the stopping rules and the Random Forest hook."""

import math

import numpy as np
import pytest

from common import CRITERIA
from part1_decision_tree import TOL, DecisionTree, impurity, sanity_checks


def test_handout_sanity_checks():
    sanity_checks()


@pytest.mark.parametrize("criterion", CRITERIA)
def test_check_b_with_both_criteria(criterion):
    X, y = np.array([[1.0], [2.0], [3.0], [4.0]]), np.array([0, 0, 1, 1])
    tree = DecisionTree(criterion).fit(X, y)
    assert tree.n_leaves == 2 and tree.nodes[0]["threshold"] == 2.5
    assert tree.nodes[0]["gain"] == pytest.approx(0.5 if criterion == "gini" else 1.0)


# -- a slow, obviously correct reference: loops over every feature and every midpoint --------


def reference_impurity(class_weights, criterion):
    total = sum(class_weights)
    ps = [v / total for v in class_weights]
    if criterion == "gini":
        return 1 - sum(p * p for p in ps)
    return -sum(p * math.log2(p) for p in ps if p > 0)


def reference_split(X, y, w, criterion, features):
    classes = sorted(set(y.tolist()))
    n = len(y)

    def weights_of(rows):
        return [sum(w[i] for i in rows if y[i] == c) for c in classes]

    W = sum(w)
    parent = reference_impurity(weights_of(range(n)), criterion)
    candidates = []
    for j in features:
        values = sorted(set(X[:, j].tolist()))
        for a, b in zip(values, values[1:]):
            t = (a + b) / 2
            left = [i for i in range(n) if X[i, j] <= t]
            right = [i for i in range(n) if X[i, j] > t]
            gain = parent
            for rows in (left, right):
                gain -= (
                    sum(w[i] for i in rows) / W * reference_impurity(weights_of(rows), criterion)
                )
            candidates.append((gain, j, t))
    if not candidates:
        return None
    best = max(c[0] for c in candidates)
    if best <= TOL:
        return None
    return min((c for c in candidates if c[0] >= best - TOL), key=lambda c: (c[1], c[2]))


def reference_tree(X, y, w, criterion, depth=0, max_depth=None):
    """Pre-order list of (feature, threshold) for splits and ('leaf', class) for leaves."""
    classes = sorted(set(y.tolist()))
    weights = [sum(w[i] for i in range(len(y)) if y[i] == c) for c in classes]
    leaf = [("leaf", classes[int(np.argmax(weights))])]
    if len(classes) == 1 or (max_depth is not None and depth >= max_depth):
        return leaf
    split = reference_split(X, y, w, criterion, range(X.shape[1]))
    if split is None:
        return leaf
    _, j, t = split
    go_left = X[:, j] <= t
    return [
        (j, t),
        *reference_tree(X[go_left], y[go_left], w[go_left], criterion, depth + 1, max_depth),
        *reference_tree(X[~go_left], y[~go_left], w[~go_left], criterion, depth + 1, max_depth),
    ]


def as_preorder(tree):
    return [
        ("leaf", tree.classes_[node["value"]])
        if node["feature"] < 0
        else (
            node["feature"],
            node["threshold"],
        )
        for node in tree.nodes
    ]


def random_problem(seed):
    """Small integer features (so ties and repeated values are everywhere), 2-4 classes and
    either uniform or random weights."""
    rng = np.random.default_rng(seed)
    n, d, k = rng.integers(2, 30), rng.integers(1, 5), rng.integers(2, 5)
    X = rng.integers(0, 5, size=(n, d)).astype(float)
    y = rng.integers(0, k, size=n)
    w = rng.uniform(0.1, 3, size=n) if seed % 2 else np.ones(n)
    return X, y, w / w.sum()


@pytest.mark.parametrize("criterion", CRITERIA)
@pytest.mark.parametrize("seed", range(40))
def test_root_split_matches_brute_force(criterion, seed):
    X, y, w = random_problem(seed)
    expected = reference_split(X, y, w, criterion, range(X.shape[1]))
    tree = DecisionTree(criterion, max_depth=1).fit(X, y, sample_weight=w)
    root = tree.nodes[0]
    if expected is None or len(set(y.tolist())) == 1:
        assert tree.n_leaves == 1
    else:
        gain, j, t = expected
        assert (root["feature"], root["threshold"]) == (j, t)
        assert root["gain"] == pytest.approx(gain, abs=1e-12)


@pytest.mark.parametrize("criterion", CRITERIA)
@pytest.mark.parametrize("seed", range(40))
def test_whole_tree_matches_brute_force(criterion, seed):
    X, y, w = random_problem(seed)
    tree = DecisionTree(criterion).fit(X, y, sample_weight=w)
    assert as_preorder(tree) == reference_tree(X, y, w, criterion)


# -- weights ------------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(10))
def test_weight_scale_does_not_change_the_tree(seed):
    X, y, _ = random_problem(seed)
    trees = [
        as_preorder(DecisionTree("entropy").fit(X, y, sample_weight=w))
        for w in (None, np.ones(len(y)), np.full(len(y), 7.3))
    ]
    assert trees[0] == trees[1] == trees[2]


@pytest.mark.parametrize("seed", range(10))
def test_integer_weights_equal_duplicated_rows(seed):
    X, y, _ = random_problem(seed)
    k = np.random.default_rng(seed).integers(1, 4, size=len(y))
    weighted = DecisionTree("gini").fit(X, y, sample_weight=k.astype(float))
    duplicated = DecisionTree("gini").fit(np.repeat(X, k, axis=0), np.repeat(y, k))
    assert as_preorder(weighted) == as_preorder(duplicated)


# -- ties and stopping ----------------------------------------------------------------------------


def test_ties_go_to_the_lower_feature_then_the_lower_threshold():
    # both features separate the classes perfectly: the lower index wins
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    assert DecisionTree("gini").fit(X, np.array([0, 1])).nodes[0]["feature"] == 0
    # labels 0 1 0 along x: the cuts at 0.5 and 1.5 have equal gain, the lower one wins
    X = np.array([[0.0], [1.0], [2.0]])
    assert DecisionTree("gini").fit(X, np.array([0, 1, 0])).nodes[0]["threshold"] == 0.5


def test_leaf_ties_go_to_the_smallest_label():
    tree = DecisionTree("gini").fit(np.zeros((2, 1)), np.array(["b", "a"]))
    assert tree.n_leaves == 1 and tree.predict(np.zeros((1, 1)))[0] == "a"


@pytest.mark.parametrize("criterion", CRITERIA)
def test_xor_has_no_positive_gain(criterion):
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    assert DecisionTree(criterion).fit(X, np.array([0, 1, 1, 0])).n_leaves == 1


@pytest.mark.parametrize("max_depth", [1, 2, 3])
def test_max_depth_is_honoured(max_depth):
    rng = np.random.default_rng(1)
    X, y = rng.normal(size=(300, 4)), rng.integers(0, 5, size=300)
    tree = DecisionTree("gini", max_depth=max_depth).fit(X, y)
    assert tree.depth == max_depth and tree.n_leaves <= 2**max_depth


def test_min_samples_split_is_honoured():
    rng = np.random.default_rng(2)
    X, y = rng.normal(size=(200, 3)), rng.integers(0, 3, size=200)
    tree = DecisionTree("entropy", min_samples_split=25).fit(X, y)
    assert all(node["n"] >= 25 for node in tree.nodes if node["feature"] >= 0)


def test_unconstrained_tree_fits_distinct_rows_exactly():
    rng = np.random.default_rng(3)
    X, y = rng.normal(size=(500, 5)), rng.integers(0, 4, size=500)
    tree = DecisionTree("gini").fit(X, y)
    assert (tree.predict(X) == y).all()


def test_training_rows_reach_the_leaves_they_were_fitted_in():
    rng = np.random.default_rng(4)
    X, y = rng.integers(0, 6, size=(300, 3)).astype(float), rng.integers(0, 4, size=300)
    tree = DecisionTree("entropy").fit(X, y)
    reached = np.bincount(tree.apply(X), minlength=len(tree.nodes))
    for i, node in enumerate(tree.nodes):
        if node["feature"] < 0:
            assert reached[i] == node["n"]


def test_impurity_of_pure_and_uniform_nodes():
    assert impurity(np.array([1.0, 0.0]), "entropy") == 0
    assert impurity(np.full(4, 0.25), "entropy") == pytest.approx(2)
    assert impurity(np.full(4, 0.25), "gini") == pytest.approx(0.75)


def test_unknown_criterion_and_missing_rng_are_rejected():
    with pytest.raises(ValueError):
        DecisionTree("misclassification")
    with pytest.raises(ValueError):
        DecisionTree("gini", max_features=2)


# -- the Random Forest hook ---------------------------------------------------------------------


class FirstFeatures:
    """Stands in for RandomState: counts the draws and always offers features m-1, ..., 0."""

    def __init__(self):
        self.draws = 0

    def choice(self, d, m, replace):
        assert not replace and m <= d
        self.draws += 1
        return np.arange(m)[::-1]


def test_features_are_drawn_at_every_searched_node():
    rng = np.random.default_rng(5)
    X, y = rng.normal(size=(200, 6)), rng.integers(0, 3, size=200)
    stub = FirstFeatures()
    tree = DecisionTree("gini", max_features=2, rng=stub).fit(X, y)
    # only features 0 and 1 are ever offered, so the tree equals one grown on them alone
    assert as_preorder(tree) == as_preorder(DecisionTree("gini").fit(X[:, :2], y))
    searched = [node for node in tree.nodes if (node["weights"] > 0).sum() > 1]
    assert stub.draws == len(searched)


def test_drawing_all_features_equals_no_subsetting():
    rng = np.random.default_rng(6)
    X, y = rng.integers(0, 4, size=(150, 5)).astype(float), rng.integers(0, 3, size=150)
    full = DecisionTree("entropy").fit(X, y)
    drawn = DecisionTree("entropy", max_features=5, rng=np.random.RandomState(0)).fit(X, y)
    assert as_preorder(full) == as_preorder(drawn)
