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
    assert nb.epsilon_ == pytest.approx(eps, rel=1e-12)
    for k, c in enumerate(nb.classes_):
        rows = X[y == c]
        assert np.allclose(nb.mean_[k], rows.mean(axis=0))
        # rtol far below eps / var, so that a missing or wrong eps fails
        raw = ((rows - rows.mean(axis=0)) ** 2).mean(axis=0)
        np.testing.assert_allclose(nb.var_[k], raw + eps, rtol=1e-12, atol=0)
        assert nb.log_prior_[k] == pytest.approx(math.log(len(rows) / len(y)))


def test_a_feature_constant_within_a_class_gets_epsilon_as_its_variance():
    X, y = blobs(0)
    X[y == 0, 2] = 0.5
    nb = GaussianNB().fit(X, y)
    assert nb.var_[0, 2] == nb.epsilon_ > 0
    assert np.isfinite(nb.log_posterior(X)).all()


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


def test_predict_rejects_rows_of_another_width():
    X, y = blobs(3)
    nb = GaussianNB().fit(X, y)
    for bad in (X[:, :1], np.hstack([X, X]), X[0]):  # one column would broadcast silently
        with pytest.raises(ValueError):
            nb.predict(bad)
    assert len(nb.predict(X[:1])) == 1


def test_far_away_points_do_not_underflow():
    X, y = blobs(2)
    nb = GaussianNB().fit(X, y)
    far = np.full((3, 4), 1e3)
    assert np.isfinite(nb.log_posterior(far)).all()
    assert len(nb.predict(far)) == 3
