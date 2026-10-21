"""Part 4: a hierarchical decision tree. A router tree predicts the family, then the expert
tree of that family predicts the species. Run once with each criterion and compared with the
flat tree of Part 1.

    python part4_hierarchical.py      (from the folder holding Frogs_MFCCs.csv)
"""

import numpy as np

from common import CRITERIA, Data, confusion_matrix, header, pct, print_rows
from part1_decision_tree import DecisionTree


class HierarchicalClassifier:
    """A fitted router (Family model) and a fitted expert per family (Species models)."""

    def __init__(self, router, experts):
        self.router = router
        self.experts = experts  # {family name: expert}

    def predict_family(self, X):
        return self.router.predict(X)

    def predict(self, X):
        """Route every row, then ask each expert once, for all the rows routed to it."""
        family = self.predict_family(X)
        species = np.empty(len(X), dtype=object)  # object: no truncated names
        for name in np.unique(family):
            routed = family == name
            species[routed] = self.experts[name].predict(X[routed])
        return species.astype(str)


def fit_hierarchy(criterion, X, family, species):
    """The router learns Family on every row; each expert learns Species on the rows of its
    true family only."""
    router = DecisionTree(criterion).fit(X, family)
    experts = {
        name: DecisionTree(criterion).fit(X[family == name], species[family == name])
        for name in np.unique(family)
    }
    return HierarchicalClassifier(router, experts)


def oracle_predict(model, X, family):
    """Species predictions when every row goes to the expert of its true family."""
    species = np.empty(len(X), dtype=object)
    for name, expert in model.experts.items():
        rows = family == name
        species[rows] = expert.predict(X[rows])
    return species.astype(str)


def analyse(model, X, family, species):
    """Router, overall and oracle accuracy and the failure fractions p_A (router right,
    expert wrong) and p_B (router wrong), with the identity checked on counts."""
    router_right = model.predict_family(X) == family
    right = model.predict(X) == species
    n_a, n_b = (router_right & ~right).sum(), (~router_right).sum()
    assert not (right & ~router_right).any()  # a wrong family never gives the right species
    assert right.sum() == router_right.sum() - n_a  # A_overall = A_router - p_A
    N = len(X)
    return {
        "router": router_right.sum() / N,
        "overall": right.sum() / N,
        "oracle": float(np.mean(oracle_predict(model, X, family) == species)),
        "p_A": n_a / N,
        "p_B": n_b / N,
    }


def print_matrix(title, labels, cm):
    print(f"\n{title} (rows: true, columns: predicted)")
    print(" " * 26 + "".join(f"{j:>5}" for j in range(len(labels))))
    for j, (name, row) in enumerate(zip(labels, cm)):
        print(f"{j:>2} {name:<23}" + "".join(f"{v:>5}" for v in row))


def details(criterion, model, train, test, data):
    """The per-criterion printout: accuracies, both confusion matrices, per-family results."""
    families, species = np.unique(data.f_train), np.unique(data.y_train)
    print(f"\n--- {criterion.capitalize()} ---")
    print(
        f"Router accuracy: train {pct(train['router'])}, test {pct(test['router'])}. "
        f"Overall accuracy: train {pct(train['overall'])}, test {pct(test['overall'])}."
    )
    print(
        f"Test: oracle {pct(test['oracle'])}, p_A {pct(test['p_A'])}, p_B {pct(test['p_B'])}; "
        f"A_router - p_A = {pct(test['router'] - test['p_A'])} = A_overall (exact on counts)."
    )
    routed, predicted = model.predict_family(data.X_test), model.predict(data.X_test)
    oracle = oracle_predict(model, data.X_test, data.f_test)
    print_matrix(
        "Router confusion matrix, test set",
        families,
        confusion_matrix(data.f_test, routed, families),
    )
    print_matrix(
        "Species confusion matrix, test set",
        species,
        confusion_matrix(data.y_test, predicted, species),
    )
    print("\nPer family, test set")
    line = "{:<17}{:>6}{:>17}{:>16}{:>9}"
    print(line.format("Family", "rows", "router recall %", "expert alone %", "final %"))
    for name in families:
        rows = data.f_test == name
        truth = data.y_test[rows]
        print(
            line.format(
                name,
                rows.sum(),
                pct(np.mean(routed[rows] == name)),
                pct(np.mean(oracle[rows] == truth)),
                pct(np.mean(predicted[rows] == truth)),
            )
        )


def main():
    data = Data()
    header("Part 4: hierarchical decision tree (router on Family, experts on Species)", data)
    families = np.unique(data.f_train)

    print("\nTraining rows per family and species; each expert learns from its family's rows")
    for name in families:
        species, counts = np.unique(data.y_train[data.f_train == name], return_counts=True)
        for i, (s, n) in enumerate(zip(species, counts)):
            print(f"  {name if i == 0 else '':<17}{s:<24}{n:>5}")

    models, train, test = {}, {}, {}
    for c in CRITERIA:
        models[c] = fit_hierarchy(c, data.X_train, data.f_train, data.y_train)
        train[c] = analyse(models[c], data.X_train, data.f_train, data.y_train)
        test[c] = analyse(models[c], data.X_test, data.f_test, data.y_test)

    print("\nLeaves per tree (an expert of a single-species family is one leaf: a constant)")
    line = "{:<10}{:>8}" + "{:>17}" * len(families)
    print(line.format("Criterion", "router", *families))
    for c in CRITERIA:
        sizes = [models[c].experts[name].n_leaves for name in families]
        print(line.format(c.capitalize(), models[c].router.n_leaves, *sizes))

    method = "4. Hierarchical Decision Tree"
    rows = [
        (method, c.capitalize(), train[c]["overall"], test[c]["overall"], test[c]["router"])
        for c in CRITERIA
    ]
    print_rows(rows)
    print("Train % and Test %: the final species accuracy. Family %: the router's test accuracy.")

    print("\nFailure analysis, test set %: (A) router right, expert wrong; (B) router wrong")
    line = "{:<10}{:>12}{:>12}{:>8}{:>8}"
    print(line.format("Criterion", "Router acc", "Oracle acc", "(A)", "(B)"))
    for c in CRITERIA:
        t = test[c]
        print(
            line.format(c.capitalize(), *map(pct, (t["router"], t["oracle"], t["p_A"], t["p_B"])))
        )

    for c in CRITERIA:
        details(c, models[c], train[c], test[c], data)

    print("\nAgainst the flat tree of Part 1, test %")
    line = "{:<10}{:>14}{:>11}{:>13}{:>8}{:>14}"
    print(
        line.format("Criterion", "flat species", "hierarchy", "flat family", "router", "bottleneck")
    )
    flat = {}
    for c in CRITERIA:
        flat[c] = data.scores(DecisionTree(c).fit(data.X_train, data.y_train))
        t = test[c]
        side = "router" if t["p_B"] > t["p_A"] else "experts"
        accs = (flat[c][1], t["overall"], flat[c][2], t["router"])
        print(line.format(c.capitalize(), *map(pct, accs), side))
    print("A perfect router would add oracle - overall: ", end="")
    print(
        ", ".join(
            f"{c.capitalize()} {pct(test[c]['oracle'] - test[c]['overall'])}" for c in CRITERIA
        )
    )
    return {"rows": rows, "train": train, "test": test, "flat": flat}


if __name__ == "__main__":
    main()
