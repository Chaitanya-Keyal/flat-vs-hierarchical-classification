"""Tests for the five parts. They run from the repo root, where Frogs_MFCCs.csv sits."""

import numpy as np


def blobs(seed, n=300, k=3):
    """k overlapping Gaussian classes in 4 dimensions."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, size=n)
    return rng.normal(size=(n, 4)) + y[:, None] * 0.8, y


def nodes(tree):
    """A fitted tree's structure: (feature, threshold, value) of every node, in order."""
    return [(nd["feature"], nd["threshold"], nd["value"]) for nd in tree.nodes]
