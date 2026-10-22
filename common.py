"""Shared code for the five parts: the data, the fixed train/test split, accuracy, confusion
matrices, votes and the printed rows of the results matrix.

Every part runs from the folder that holds Frogs_MFCCs.csv and imports only numpy, pandas and
the standard library.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_FILE = "Frogs_MFCCs.csv"
N_FEATURES = 22
CRITERIA = ("gini", "entropy")


def stratified_split(labels, test_frac=0.2, seed=42):
    """The handout's split, unchanged, so that every submission tests on the same rows."""
    rng = np.random.RandomState(seed)
    test = []
    for c in np.unique(labels):
        idx = rng.permutation(np.where(labels == c)[0])
        test.extend(idx[: int(round(test_frac * len(idx)))])  # noqa: RUF046 (verbatim)
    test = np.sort(np.array(test))
    train = np.setdiff1d(np.arange(len(labels)), test)
    return train, test


def data_path():
    """Frogs_MFCCs.csv in the current folder, or else next to these scripts."""
    for folder in (Path.cwd(), Path(__file__).resolve().parent):
        if (folder / DATA_FILE).exists():
            return folder / DATA_FILE
    raise SystemExit(f"{DATA_FILE} not found: run the script from the folder that holds it.")


def load():
    """X (the 22 MFCCs, in file order), Family and Species of every row."""
    df = pd.read_csv(data_path())
    # The header spells the first nine "MFCCs_ 1" ... "MFCCs_ 9". Family, Genus, Species and
    # RecordID are labels and never features.
    features = [c for c in df.columns if c.startswith("MFCCs_")]
    assert len(features) == N_FEATURES
    X = np.ascontiguousarray(df[features].to_numpy(dtype=float))
    return X, df["Family"].to_numpy(dtype=str), df["Species"].to_numpy(dtype=str)


class Data:
    """The fixed split: X, Species (y) and Family (f) of the training and test rows."""

    def __init__(self):
        X, family, species = load()
        train, test = stratified_split(species)
        self.X_train, self.X_test = X[train], X[test]
        self.y_train, self.y_test = species[train], species[test]
        self.f_train, self.f_test = family[train], family[test]
        # species -> family, read from the training rows; each species has exactly one family
        self.family_of = dict(zip(self.y_train, self.f_train))
        assert all(self.family_of[s] == f for s, f in zip(self.y_train, self.f_train))

    def families(self, species):
        """The family of each species, by the training map."""
        return np.array([self.family_of[s] for s in species])

    def scores(self, model):
        """Train and test accuracy of a fitted Species model, and its family accuracy on the
        test set, derived from the predicted species."""
        test_pred = model.predict(self.X_test)
        return (
            accuracy(self.y_train, model.predict(self.X_train)),
            accuracy(self.y_test, test_pred),
            accuracy(self.f_test, self.families(test_pred)),
        )


def accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def confusion_matrix(y_true, y_pred, labels):
    """Counts with one row per true and one column per predicted label, in the order of the
    sorted array `labels`."""
    k = len(labels)
    cells = np.searchsorted(labels, y_true) * k + np.searchsorted(labels, y_pred)
    return np.bincount(cells, minlength=k * k).reshape(k, k)


def tally(votes, n_classes, weights=None):
    """Votes per sample and class. `votes` holds class indices, one row per model and one
    column per sample; each model's vote counts once, or with its weight. Taking argmax of the
    result breaks ties towards the smallest class index, i.e. the smallest label."""
    hits = votes[:, :, None] == np.arange(n_classes)
    if weights is None:
        return hits.sum(axis=0)
    return np.tensordot(weights, hits, axes=1)


def print_confusion(title, labels, cm):
    print(f"\n{title} (rows: true, columns: predicted)")
    print(" " * 26 + "".join(f"{j:>5}" for j in range(len(labels))))
    for j, (name, row) in enumerate(zip(labels, cm)):
        print(f"{j:>2} {name:<23}" + "".join(f"{v:>5}" for v in row))


def header(title, data):
    print(title)
    n_species, n_families = len(np.unique(data.y_train)), len(np.unique(data.f_train))
    print(
        f"Data: train {len(data.y_train)} / test {len(data.y_test)} rows, "
        f"{data.X_train.shape[1]} MFCC features, {n_species} species in {n_families} families"
    )


def pct(x):
    """An accuracy as a percentage with two decimals, as the results matrix asks."""
    return f"{100 * x:.2f}"


def points(a, b):
    """a - b in percentage points, taken from the two printed values so that it matches them."""
    hundredths = int(pct(a).replace(".", "")) - int(pct(b).replace(".", ""))
    return f"{hundredths / 100:.2f}"


ROW = "{:<32}{:<10}{:>9}{:>9}{:>10}"


def print_rows(rows, title="Results matrix rows"):
    """Print rows of the results matrix: (method, criterion, train, test, family accuracy)."""
    print(f"\n{title}")
    print(ROW.format("Method", "Criterion", "Train %", "Test %", "Family %"))
    for method, criterion, *accs in rows:
        print(ROW.format(method, criterion, *map(pct, accs)))
