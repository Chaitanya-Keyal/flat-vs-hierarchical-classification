"""Gaussian naive Bayes: the handout's check, the estimates and the vectorised posterior."""

import math

import numpy as np
import pytest

from part5_naive_bayes import GaussianNB, sanity_check
from tests import blobs


def test_handout_sanity_check():
    sanity_check()


def test_estimates_follow_the_handout():
    X, y = blobs(0)
    nb = GaussianNB().fit(X, y)
    eps = 1e-9 * X.var(axis=0).max()
    for k, c in enumerate(nb.classes_):
        rows = X[y == c]
        assert np.allclose(nb.mean_[k], rows.mean(axis=0))
        assert np.allclose(nb.var_[k], ((rows - rows.mean(axis=0)) ** 2).mean(axis=0) + eps)
        assert nb.log_prior_[k] == pytest.approx(math.log(len(rows) / len(y)))


def test_log_posterior_matches_a_loop():
    X, y = blobs(1)
    nb = GaussianNB().fit(X, y)
    Z = X[:25]
    got = nb.log_posterior(Z)
    for n, z in enumerate(Z):
        for k in range(len(nb.classes_)):
            expected = nb.log_prior_[k]
            for j, x in enumerate(z):
                var = nb.var_[k, j]
                expected -= 0.5 * (math.log(2 * math.pi * var) + (x - nb.mean_[k, j]) ** 2 / var)
            assert got[n, k] == pytest.approx(expected)


def test_far_away_points_do_not_underflow():
    X, y = blobs(2)
    nb = GaussianNB().fit(X, y)
    far = np.full((3, 4), 1e3)
    assert np.isfinite(nb.log_posterior(far)).all()
    assert len(nb.predict(far)) == 3
