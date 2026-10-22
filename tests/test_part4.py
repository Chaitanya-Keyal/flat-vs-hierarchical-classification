"""The router-and-experts pipeline: routing, one call per expert, experts on true families,
the oracle and the failure split."""

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
    # copies of some B rows labelled C: the router cannot fit both, so it misroutes training
    # rows, and an expert trained on the predicted family would learn another family's species
    X, family, species = toy(2)
    b = np.flatnonzero(family == "B")[:40]
    X = np.vstack([X, X[b]])
    family = np.concatenate([family, np.full(len(b), "C")])
    species = np.concatenate([species, np.full(len(b), "c1")])
    model = fit_hierarchy("entropy", X, family, species)
    assert (model.router.predict(X) != family).any()
    for name, expert in model.experts.items():
        assert set(expert.classes_) == {s for s in SPECIES if FAMILY[s] == name}


class Spy:
    """An expert that records how many rows each predict call gets."""

    def __init__(self, expert):
        self.expert, self.calls = expert, []

    def predict(self, X):
        self.calls.append(len(X))
        return self.expert.predict(X)


def test_each_expert_is_called_once_on_the_rows_routed_to_it():
    model = fit_hierarchy("gini", *toy(9))
    Z = toy(10)[0]
    spies = {name: Spy(expert) for name, expert in model.experts.items()}
    HierarchicalClassifier(model.router, spies).predict(Z)
    routed = model.router.predict(Z)
    for name, spy in spies.items():
        n = int((routed == name).sum())
        assert spy.calls == ([n] if n else [])


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
    X, family, species = X[keep], family[keep], species[keep]
    res = analyse(HierarchicalClassifier(AlwaysB(), model.experts), X, family, species)
    assert res["router"] == res["overall"] == 0 and res["p_B"] == 1
    # the oracle ignores the router: every row goes to the expert of its true family
    alone = [model.experts[f].predict(x[None])[0] == s for x, f, s in zip(X, family, species)]
    assert res["oracle"] == pytest.approx(np.mean(alone)) and res["oracle"] > 0.5
