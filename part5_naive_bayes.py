"""Part 5: Gaussian naive Bayes, flat (predicts Species). Not a tree, so it has no criterion.

Also prints the correlation matrix of the 22 training features, since naive Bayes assumes
they are independent within each class.

    python part5_naive_bayes.py      (from the folder holding Frogs_MFCCs.csv)
"""

import numpy as np

from common import Data, header, print_rows


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


def main():
    data = Data()
    header("Part 5: Gaussian naive Bayes, flat (predicts Species)", data)
    sanity_check()

    nb = GaussianNB().fit(data.X_train, data.y_train)
    rows = [("5. Gaussian Naive Bayes", "-", *data.scores(nb))]
    print_rows(rows)
    print("Family %: the family of each predicted species, by the training rows' species map.")
    print(f"\nepsilon = 1e-9 * max_j Var_train(x_j) = {nb.epsilon_:.3e}")
    flat = nb.var_ == nb.epsilon_  # zero variance before smoothing
    for k, j in zip(*np.nonzero(flat)):
        print(
            f"MFCC {j + 1} is constant within {nb.classes_[k]} in training: its variance is epsilon"
        )

    corr = np.corrcoef(data.X_train, rowvar=False)
    return {"rows": rows, "epsilon": nb.epsilon_, "correlation": print_correlations(corr)}


if __name__ == "__main__":
    main()
