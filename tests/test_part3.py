"""The forest's one random stream, its per-split feature draws and the pairwise agreement of
its trees."""

import numpy as np
import pytest

from part1_decision_tree import DecisionTree
from part2_bagging_boosting import Bagging
from part3_random_forest import RandomForest, agreement
from tests import blobs, nodes


def test_agreement_matches_a_loop_over_pairs():
    votes = np.random.default_rng(0).integers(0, 4, size=(7, 60))
    pairs = [np.mean(votes[a] == votes[b]) for a in range(7) for b in range(a + 1, 7)]
    assert agreement(votes, 4) == pytest.approx(np.mean(pairs))


def test_identical_trees_agree_everywhere():
    assert agreement(np.tile(np.arange(9) % 3, (5, 1)), 3) == 1


def test_a_forest_drawing_every_feature_starts_like_bagging():
    # same first bootstrap; drawing all d features (sorted) is the full search
    X, y = blobs(1)
    forest = RandomForest(3, "gini", X.shape[1]).fit(X, y)
    assert nodes(forest.trees_[0]) == nodes(Bagging(3, "gini").fit(X, y).trees_[0])


def test_a_forest_replays_one_stream_of_bootstraps_and_draws():
    # one RandomState(42) for the whole forest: tree 1's bootstrap, then its feature draws
    # (depth first, left child first), then tree 2's bootstrap, and so on
    X, y = blobs(4)
    forest = RandomForest(4, "gini", 2).fit(X, y)
    rng = np.random.RandomState(42)
    for tree in forest.trees_:
        rows = rng.randint(0, len(y), size=len(y))
        expected = DecisionTree("gini", max_features=2, rng=rng).fit(X[rows], y[rows])
        assert nodes(tree) == nodes(expected)


def test_a_forest_draws_features_per_split_not_per_tree():
    # with one feature per split, a subset drawn once per tree would leave each tree a single
    # feature; drawn at every split, each tree uses several
    X, y = blobs(2)
    forest = RandomForest(10, "entropy", 1).fit(X, y)
    for tree in forest.trees_:
        assert len({nd["feature"] for nd in tree.nodes if nd["feature"] >= 0}) > 1


def test_a_forest_refits_identically():
    X, y = blobs(3)
    model = RandomForest(5, "gini", 2)
    first = model.fit(X, y).predict(X)
    assert np.array_equal(first, model.fit(X, y).predict(X))
