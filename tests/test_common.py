"""The fixed split, the species -> family map, the scores, accuracy, confusion matrices and
votes."""

from collections import Counter

import numpy as np

from common import Data, accuracy, confusion_matrix, load, stratified_split, tally


def test_split_matches_the_handout():
    _, _, species = load()
    train, test = stratified_split(species)
    assert (len(train), len(test)) == (5756, 1439)
    assert len(np.intersect1d(train, test)) == 0
    assert len(np.union1d(train, test)) == len(species)
    for s in np.unique(species):
        n = (species == s).sum()
        assert (species[test] == s).sum() == round(0.2 * n)


def test_split_is_the_same_on_codes_from_unique():
    _, _, species = load()
    codes = np.unique(species, return_inverse=True)[1]
    assert np.array_equal(stratified_split(codes)[1], stratified_split(species)[1])


def test_species_map_comes_from_training_rows():
    data = Data()
    assert len(data.family_of) == 10
    assert set(data.family_of.values()) == set(data.f_train)
    assert np.array_equal(data.families(data.y_test), data.f_test)


class AlwaysHylaMinuta:
    def predict(self, X):
        return np.full(len(X), "HylaMinuta")


def test_scores_derive_the_family_from_the_predicted_species():
    data = Data()
    train, test, family = data.scores(AlwaysHylaMinuta())
    assert train == np.mean(data.y_train == "HylaMinuta")
    assert test == np.mean(data.y_test == "HylaMinuta")
    assert family == np.mean(data.f_test == "Hylidae")  # HylaMinuta's family


def test_confusion_matrix_counts_pairs():
    rng = np.random.default_rng(0)
    labels = np.array(["a", "b", "c"])
    y_true, y_pred = rng.choice(labels, 200), rng.choice(labels, 200)
    cm = confusion_matrix(y_true, y_pred, labels)
    counts = Counter(zip(y_true, y_pred))
    for i, t in enumerate(labels):
        for j, p in enumerate(labels):
            assert cm[i, j] == counts[(t, p)]
    assert np.trace(cm) / cm.sum() == accuracy(y_true, y_pred)


def test_vote_ties_go_to_the_smallest_label():
    votes = np.array([[0, 2, 1], [1, 2, 2], [1, 0, 0], [0, 1, 0]])
    counts = tally(votes, 3)
    assert counts.tolist() == [[2, 2, 0], [1, 1, 2], [2, 1, 1]]
    assert counts.argmax(axis=1).tolist() == [0, 2, 0]  # column 0 ties 2-2 between 0 and 1


def test_weighted_votes_and_no_models():
    votes = np.array([[0, 1], [1, 1]])
    assert np.allclose(tally(votes, 2, np.array([3.0, 1.0])), [[3, 1], [0, 4]])
    empty = tally(np.zeros((0, 4), dtype=int), 3, np.zeros(0))
    assert empty.shape == (4, 3) and not empty.any()
