"""Part 5: Gaussian naive Bayes, flat (predicts Species). Not a tree, so it has no criterion.

Also prints the correlation matrix of the 22 training features, since naive Bayes assumes
they are independent within each class.

    python part5_naive_bayes.py      (from the folder holding Frogs_MFCCs.csv)
"""

import numpy as np

from common import Data, confusion_matrix, header, print_confusion, print_rows


class GaussianNB:
    """Features independent and Gaussian within each class, worked in log space. Every class
    variance gets var_smoothing times the largest feature variance of the training set added."""

    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.classes_, codes = np.unique(y, return_inverse=True)
        self.epsilon_ = self.var_smoothing * X.var(axis=0).max()
        groups = [X[codes == k] for k in range(len(self.classes_))]
        self.log_prior_ = np.log([len(g) / len(X) for g in groups])  # pi_c = n_c / N
        self.mean_ = np.array([g.mean(axis=0) for g in groups])
        self.var_ = np.array([g.var(axis=0) for g in groups]) + self.epsilon_  # 1/n_c, + eps
        return self

    def log_posterior(self, X):
        """log pi_c - 1/2 sum_j [log(2 pi var_cj) + (x_j - mean_cj)^2 / var_cj] for every row and
        class, up to a constant: an (n, K, d) array summed over the features."""
        diff = np.asarray(X, dtype=float)[:, None, :] - self.mean_
        terms = np.log(2 * np.pi * self.var_) + diff**2 / self.var_
        return self.log_prior_ - 0.5 * terms.sum(axis=2)

    def predict(self, X):
        return self.classes_[self.log_posterior(X).argmax(axis=1)]


def sanity_check():
    """The handout's check, printed, so that a failure stops the script."""
    X, y = np.array([[0.0], [2.0], [10.0], [12.0]]), np.array([0, 0, 1, 1])
    nb = GaussianNB().fit(X, y)
    pred = nb.predict(np.array([[3.0], [9.0]]))
    assert np.allclose(nb.mean_.ravel(), [1, 11]) and np.allclose(nb.var_.ravel(), 1)
    assert pred.tolist() == [0, 1]
    print("\nSanity check")
    print(
        f"  class 0 = {{0, 2}}, class 1 = {{10, 12}}: means {nb.mean_[0, 0]:g} and "
        f"{nb.mean_[1, 0]:g}, variances 1 + {nb.epsilon_:.1e}"
    )
    print(f"  x = 3 -> class {pred[0]}, x = 9 -> class {pred[1]}. Passed.")


def print_correlations(corr):
    print("\nCorrelation matrix of the 22 training features (MFCC 1 to 22)")
    d = len(corr)
    for start in range(0, d, 11):
        cols = range(start, min(start + 11, d))
        print(" " * 8 + "".join(f"{j + 1:>6}" for j in cols))
        for i in range(d):
            print(f"MFCC {i + 1:<3}" + "".join(f"{corr[i, j]:>6.2f}" for j in cols))
        print()
    i, j = np.triu_indices(d, k=1)
    off = np.abs(corr[i, j])
    above = int(np.sum(off > 0.5))
    print(f"Off-diagonal |r| over the {len(off)} pairs: mean {off.mean():.2f}, max {off.max():.2f}")
    print(f"{above} pairs have |r| above 0.5. The five strongest:")
    for k in np.argsort(-off, kind="stable")[:5]:
        print(f"  MFCC {i[k] + 1} and MFCC {j[k] + 1}: r = {corr[i[k], j[k]]:+.2f}")
    return {"mean_abs": float(off.mean()), "max_abs": float(off.max()), "above_half": above}


def within_class_correlation(X, y):
    """Mean |r| between the features inside each class, the dependence naive Bayes assumes
    away (features constant within the class left out)."""
    mean_abs = {}
    for c in np.unique(y):
        Xc = X[y == c]
        r = np.corrcoef(Xc[:, Xc.std(axis=0) > 0], rowvar=False)
        mean_abs[str(c)] = float(np.abs(r[np.triu_indices(len(r), k=1)]).mean())
    return mean_abs


def main():
    data = Data()
    header("Part 5: Gaussian naive Bayes, flat (predicts Species)", data)
    sanity_check()

    nb = GaussianNB().fit(data.X_train, data.y_train)
    rows = [("5. Gaussian Naive Bayes", "-", *data.scores(nb))]
    print_rows(rows)
    print("Family %: the family of each predicted species, by the training rows' species map.")
    print(f"\nepsilon = 1e-9 * max_j Var_train(x_j) = {nb.epsilon_:.3e}")
    constant = list(zip(*np.nonzero(nb.var_ == nb.epsilon_)))  # zero variance before smoothing
    shared = {}
    for k, j in constant:
        value = nb.mean_[k, j]
        shared[str(nb.classes_[k])] = int(np.sum(data.X_test[:, j] == value))
        print(f"MFCC {j + 1} is constant ({value:g}) within {nb.classes_[k]} in training, so its")
        print(
            f"variance there is epsilon; {shared[str(nb.classes_[k])]} of the {len(data.X_test)} "
            f"test rows share that value"
        )
    cm = confusion_matrix(data.y_test, nb.predict(data.X_test), nb.classes_)
    print_confusion("Confusion matrix, test set", nb.classes_, cm)
    predicted = {str(c): (int(cm[:, k].sum()), int(cm[k, k])) for k, c in enumerate(nb.classes_)}

    corr = np.corrcoef(data.X_train, rowvar=False)
    correlation = print_correlations(corr)
    within = within_class_correlation(data.X_train, data.y_train)
    print("\nMean |r| between the features within each species (naive Bayes assumes 0)")
    for c, r in within.items():
        print(f"  {c:<24}{r:.2f}")
    return {
        "rows": rows,
        "epsilon": nb.epsilon_,
        "constant": [(str(nb.classes_[k]), int(j) + 1) for k, j in constant],
        "shared": shared,  # test rows with the constant value, per such species
        "predicted": predicted,  # per species: (test rows predicted as it, of which right)
        "correlation": correlation,
        "within": within,
    }


if __name__ == "__main__":
    main()
