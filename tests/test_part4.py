"""The router-and-experts pipeline: routing, experts on true families, and the failure split."""

import numpy as np
import pytest

from part4_hierarchical import HierarchicalClassifier, analyse, fit_hierarchy

SPECIES = np.array(["a1", "a2", "b1", "c1", "c2", "c_with_a_much_longer_name"])
FAMILY = {"a1": "A", "a2": "A", "b1": "B", "c1": "C", "c2": "C", "c_with_a_much_longer_name": "C"}


def toy(seed, n=600):
    """Three families, one of them with a single species; the features overlap a little."""
    rng = np.random.default_rng(seed)
    k = rng.integers(0, len(SPECIES), size=n)
    X = rng.normal(size=(n, 3)) + np.stack([k, k % 2, k // 3], axis=1)
    species = SPECIES[k]
    return X, np.array([FAMILY[s] for s in species]), species


def test_predict_matches_routing_each_row_alone():
    model = fit_hierarchy("gini", *toy(0))
    Z = toy(1)[0]
    one_by_one = [model.experts[model.router.predict(z[None])[0]].predict(z[None])[0] for z in Z]
    assert model.predict(Z).tolist() == one_by_one  # same order, names not truncated


def test_experts_learn_the_species_of_their_true_family():
    model = fit_hierarchy("entropy", *toy(2))
    for name, expert in model.experts.items():
        assert set(expert.classes_) == {s for s in SPECIES if FAMILY[s] == name}


def test_a_single_species_family_gets_a_constant_expert():
    model = fit_hierarchy("gini", *toy(3))
    assert model.experts["B"].n_leaves == 1
    assert set(model.experts["B"].predict(toy(4)[0])) == {"b1"}


def test_failure_fractions_add_up():
    model = fit_hierarchy("entropy", *toy(5))
    res = analyse(model, *toy(6))  # checks the identity exactly, on counts
    assert res["overall"] == pytest.approx(res["router"] - res["p_A"])
    assert res["p_B"] == pytest.approx(1 - res["router"])
    assert res["overall"] <= res["oracle"]


class AlwaysB:
    def predict(self, X):
        return np.full(len(X), "B")


def test_a_wrong_router_makes_every_species_wrong():
    model = fit_hierarchy("gini", *toy(7))
    X, family, species = toy(8)
    keep = family != "B"
    res = analyse(
        HierarchicalClassifier(AlwaysB(), model.experts), X[keep], family[keep], species[keep]
    )
    assert res["router"] == res["overall"] == 0 and res["p_B"] == 1
