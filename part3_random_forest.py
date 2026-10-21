"""Part 3: a random forest on flat trees (predict Species): bagging plus a fresh random subset
of m = floor(sqrt(d)) candidate features at every split.

    python part3_random_forest.py      (from the folder holding Frogs_MFCCs.csv)
"""

import math

import numpy as np

from common import CRITERIA, Data, accuracy, header, pct, print_rows, tally
from part1_decision_tree import DecisionTree
from part2_bagging_boosting import Bagging


class RandomForest(Bagging):
    """Bagging whose trees draw max_features candidate features afresh at every split. The
    bootstraps and the feature draws come from one RandomState(seed), created at every fit."""

    def __init__(self, n_estimators, criterion, max_features, seed=42):
        super().__init__(n_estimators, criterion, seed)
        self.max_features = max_features

    def new_tree(self, rng):
        return DecisionTree(self.criterion, max_features=self.max_features, rng=rng)


def agreement(votes, n_classes):
    """The fraction of samples on which two trees predict the same class, averaged over all
    pairs of trees. On a sample where c_k trees vote for class k, c_k (c_k - 1) / 2 of the
    T (T - 1) / 2 pairs agree on k."""
    T, c = len(votes), tally(votes, n_classes)
    return float(np.mean((c * (c - 1)).sum(axis=1) / (T * (T - 1))))


def main():
    data = Data()
    header("Part 3: random forest on flat trees (predict Species)", data)
    X, y = data.X_train, data.y_train
    m = math.isqrt(X.shape[1])
    print(f"m = floor(sqrt({X.shape[1]})) = {m} candidate features, drawn afresh at every split")
    single = {c: DecisionTree(c).fit(X, y) for c in CRITERIA}
    bagging = {c: Bagging(25, c).fit(X, y) for c in CRITERIA}
    forest = {c: RandomForest(50, c, m).fit(X, y) for c in CRITERIA}

    rows = [("3. Random Forest", c.capitalize(), *data.scores(forest[c])) for c in CRITERIA]
    print_rows(rows)
    print("Family %: the family of each predicted species, by the training rows' species map.")

    def test_acc(model):
        return accuracy(data.y_test, model.predict(data.X_test))

    print("\nTest % against the single tree (Part 1) and bagging (Part 2)")
    line = "{:<10}{:>12}{:>10}{:>15}"
    print(line.format("Criterion", "Single tree", "Bagging", "Random Forest"))
    compare = {}
    for c in CRITERIA:
        compare[c] = [test_acc(models[c]) for models in (single, bagging, forest)]
        print(line.format(c.capitalize(), *map(pct, compare[c])))

    print("\nExtra (not graded): how alike are the trees, on the test set?")
    line = "{:<10}{:<15}{:>6}{:>20}{:>18}"
    print(line.format("Criterion", "Ensemble", "trees", "pairwise agreement", "mean tree test %"))
    alike = {}
    for c in CRITERIA:
        for name, model in (("Bagging", bagging[c]), ("Random Forest", forest[c])):
            votes = model.votes(data.X_test)
            agree = agreement(votes, len(model.classes_))
            tree_acc = float(np.mean([accuracy(data.y_test, model.classes_[v]) for v in votes]))
            alike[f"{c}-{name}"] = {"agreement": agree, "tree_test": tree_acc}
            print(line.format(c.capitalize(), name, len(votes), f"{agree:.4f}", pct(tree_acc)))
    return {"m": m, "rows": rows, "compare_test": compare, "alike": alike}


if __name__ == "__main__":
    main()
