"""Part 2: bagging and multi-class AdaBoost (SAMME) on flat trees (predict Species).

Both ensembles use the Part 1 tree and run with each criterion; AdaBoost with decision stumps
is run as well, for comparison.

    python part2_bagging_boosting.py      (from the folder holding Frogs_MFCCs.csv)
"""

import numpy as np

from common import CRITERIA, Data, accuracy, header, pct, print_rows, tally
from part1_decision_tree import DecisionTree


class Bagging:
    """T unconstrained trees, each fitted on its own bootstrap sample of the N training rows,
    combined by majority vote (ties to the smallest label). All the randomness comes from one
    RandomState(seed), created afresh at every fit."""

    def __init__(self, n_estimators=25, criterion="gini", seed=42):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.seed = seed

    def new_tree(self, rng):
        return DecisionTree(self.criterion)  # all 22 features at every split

    def fit(self, X, y):
        rng = np.random.RandomState(self.seed)
        N = len(y)
        self.classes_ = np.unique(y)
        self.trees_ = []
        for _ in range(self.n_estimators):
            rows = rng.randint(0, N, size=N)  # N rows, drawn with replacement
            self.trees_.append(self.new_tree(rng).fit(X[rows], y[rows]))
        return self

    def votes(self, X):
        """Each tree's predicted class index: one row per tree, one column per sample."""
        return np.array([np.searchsorted(self.classes_, tree.predict(X)) for tree in self.trees_])

    def predict(self, X):
        return self.classes_[tally(self.votes(X), len(self.classes_)).argmax(axis=1)]


class AdaBoostSAMME:
    """Multi-class AdaBoost (SAMME): T rounds of depth-limited trees fitted to reweighted
    training rows, combined by a vote weighted with each tree's amount of say alpha."""

    def __init__(self, n_estimators=50, criterion="gini", max_depth=3):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        K, N = len(self.classes_), len(y)
        w = np.full(N, 1 / N)
        self.trees_, self.alphas_, self.errors_ = [], [], []
        for _ in range(self.n_estimators):
            tree = DecisionTree(self.criterion, max_depth=self.max_depth)
            miss = tree.fit(X, y, sample_weight=w).predict(X) != y
            eps = float(np.clip(w[miss].sum(), 1e-10, 1 - 1e-10))  # weighted error, sum w = 1
            if eps >= 1 - 1 / K:  # no better than random guessing: discard it and stop
                break
            alpha = float(np.log((1 - eps) / eps) + np.log(K - 1))  # amount of say
            w = w * np.exp(alpha * miss)  # up-weight the mistakes
            w /= w.sum()
            self.trees_.append(tree)
            self.alphas_.append(alpha)
            self.errors_.append(eps)
        return self

    def predict(self, X):
        """argmax_k sum_t alpha_t [h_t(x) = k], ties to the smallest label. With no tree kept
        every score is 0, so that is the smallest label too."""
        votes = [np.searchsorted(self.classes_, tree.predict(X)) for tree in self.trees_]
        votes = np.array(votes, dtype=int).reshape(len(self.trees_), len(X))
        scores = tally(votes, len(self.classes_), np.array(self.alphas_))
        return self.classes_[scores.argmax(axis=1)]


def main():
    data = Data()
    header("Part 2: bagging and AdaBoost (SAMME) on flat trees (predict Species)", data)
    X, y = data.X_train, data.y_train
    single = {c: DecisionTree(c).fit(X, y) for c in CRITERIA}
    bagging = {c: Bagging(25, c).fit(X, y) for c in CRITERIA}
    boosting = {c: AdaBoostSAMME(50, c, max_depth=3).fit(X, y) for c in CRITERIA}
    stumps = {c: AdaBoostSAMME(50, c, max_depth=1).fit(X, y) for c in CRITERIA}

    rows = [("2a. Bagging", c.capitalize(), *data.scores(bagging[c])) for c in CRITERIA]
    rows += [("2b. AdaBoost", c.capitalize(), *data.scores(boosting[c])) for c in CRITERIA]
    print_rows(rows)
    print("Family %: the family of each predicted species, by the training rows' species map.")

    def test_acc(model):
        return accuracy(data.y_test, model.predict(data.X_test))

    print("\nTest % against the single tree of Part 1")
    print(f"{'Criterion':<10}{'Single tree':>12}{'Bagging':>10}{'AdaBoost':>10}")
    for c in CRITERIA:
        accs = (test_acc(single[c]), test_acc(bagging[c]), test_acc(boosting[c]))
        print(f"{c.capitalize():<10}{pct(accs[0]):>12}{pct(accs[1]):>10}{pct(accs[2]):>10}")

    print("\nAdaBoost rounds kept (of T = 50) and the weighted error eps of the kept trees")
    print(f"{'Criterion':<10}{'max_depth':>10}{'rounds':>8}{'eps first':>11}{'last':>8}{'max':>8}")
    rounds = {}
    for depth, models in ((3, boosting), (1, stumps)):
        for c in CRITERIA:
            errors = models[c].errors_
            rounds[f"{c}-{depth}"] = {"rounds": len(errors), "errors": errors}
            print(
                f"{c.capitalize():<10}{depth:>10}{len(errors):>8}{errors[0]:>11.4f}"
                f"{errors[-1]:>8.4f}{max(errors):>8.4f}"
            )

    print("\nDecision stumps (max_depth = 1, T = 50) against max_depth = 3 (not in the matrix)")
    print(f"{'Criterion':<10}{'depth 3 test %':>16}{'stumps test %':>15}{'species predicted':>19}")
    stump_test = {}
    for c in CRITERIA:
        predicted = len(np.unique(stumps[c].predict(data.X_test)))
        stump_test[c] = test_acc(stumps[c])
        print(
            f"{c.capitalize():<10}{pct(test_acc(boosting[c])):>16}{pct(stump_test[c]):>15}"
            f"{predicted:>13} of {len(stumps[c].classes_)}"
        )
    return {
        "rows": rows,
        "single_test": {c: test_acc(single[c]) for c in CRITERIA},
        "rounds": rounds,
        "stumps_test": stump_test,
    }


if __name__ == "__main__":
    main()
