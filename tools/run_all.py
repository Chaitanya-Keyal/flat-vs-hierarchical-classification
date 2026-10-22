r"""Runs the parts as their scripts do and keeps what they print and return.

    uv run python -m tools.run_all              all five parts (about 90 s)
    uv run python -m tools.run_all part4 part5  some of them; the rest come from results.json

results/partN.txt     what `python partN_*.py` prints, byte for byte (the zip check diffs these)
results/results.json  every number the parts return
results/numbers.tex   the numbers report/results_matrix.tex quotes, as \resdef{key}{value}, all
                      formatted by the same pct() the scripts print with
"""

import contextlib
import io
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import common
import part1_decision_tree
import part2_bagging_boosting
import part3_random_forest
import part4_hierarchical
import part5_naive_bayes
from common import CRITERIA, pct

RESULTS = Path(__file__).resolve().parents[1] / "results"
PARTS = {
    "part1": part1_decision_tree,
    "part2": part2_bagging_boosting,
    "part3": part3_random_forest,
    "part4": part4_hierarchical,
    "part5": part5_naive_bayes,
}
METHODS = {
    "1. Single Decision Tree": "dt",
    "2a. Bagging": "bag",
    "2b. AdaBoost": "ada",
    "3. Random Forest": "rf",
    "4. Hierarchical Decision Tree": "hier",
    "5. Gaussian Naive Bayes": "nb",
}


class Tee(io.StringIO):
    """Keeps everything printed and still shows it on the terminal."""

    def write(self, s):
        sys.__stdout__.write(s)
        return super().write(s)


def run(name):
    out, start = Tee(), time.perf_counter()
    with contextlib.redirect_stdout(out):
        numbers = PARTS[name].main()
    (RESULTS / f"{name}.txt").write_text(out.getvalue())
    print(f"[{name}: {time.perf_counter() - start:.0f} s]\n", file=sys.stderr)
    return numbers


def plain(x):
    """numpy scalars and tuples as JSON values."""
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    return x.item() if isinstance(x, np.generic) else x


def leakage():
    """Recordings with syllables on both sides of the split."""
    df = pd.read_csv(common.data_path())
    train, test = common.stratified_split(df["Species"].to_numpy(dtype=str))
    record = df["RecordID"].to_numpy()
    return len(np.intersect1d(record[train], record[test])), len(np.unique(record))


def points(a, b):
    """common.points, with a typeset minus sign when negative."""
    return common.points(a, b).replace("-", r"\textminus{}")


def macros(r):
    m = {}
    for part in r.values():
        for method, criterion, *accs in part["rows"]:
            key = METHODS[method] + ("" if criterion == "-" else f"-{criterion.lower()}")
            for column, acc in zip(("train", "test", "family"), accs):
                m[f"{key}-{column}"] = pct(acc)

    p1, p2, p3, p4, p5 = (r[f"part{i}"] for i in range(1, 6))
    sweep = []
    for c in CRITERIA:
        for i, split in enumerate(("Train", "Test")):
            accs = [pct(acc[i]) for acc in p1["depth_sweep"][c].values()]
            sweep.append(" & ".join([c.capitalize() if i == 0 else "", split, *accs]) + r" \\")
    m["sweep-header"] = " & ".join(p1["depth_sweep"]["gini"])
    m["sweep-rows"] = " ".join(sweep)
    for c in CRITERIA:
        for depth, (train, test) in p1["depth_sweep"][c].items():
            m[f"sweep-{c}-{depth}-train"], m[f"sweep-{c}-{depth}-test"] = pct(train), pct(test)
    for c in CRITERIA:
        m[f"dt-{c}-depth"] = str(p1["sizes"][c]["depth"])
        m[f"dt-{c}-leaves"] = str(p1["sizes"][c]["leaves"])
        single = p2["single_test"][c]
        bag_test, ada_test = (row[3] for row in p2["rows"] if row[1].lower() == c)
        m[f"bag-{c}-gain"] = points(bag_test, single)
        m[f"ada-{c}-gain"] = points(ada_test, single)
        for depth, key in ((part2_bagging_boosting.BOOSTING_DEPTH, "ada"), (1, "stumps")):
            kept = p2["rounds"][f"{c}-{depth}"]
            m[f"{key}-{c}-rounds"] = str(len(kept["errors"]))
            m[f"{key}-{c}-eps-first"] = f"{kept['errors'][0]:.2f}"
            m[f"{key}-{c}-eps-last"] = f"{kept['errors'][-1]:.2f}"
            m[f"{key}-{c}-alpha-last"] = f"{kept['alphas'][-1]:.2f}"
        m[f"stumps-{c}-test"] = pct(p2["stumps"][c]["test"])
        m[f"stumps-{c}-species"] = str(p2["stumps"][c]["species_predicted"])
        rf_test = p3["compare_test"][c][2]
        m[f"rf-{c}-gain-dt"] = points(rf_test, p3["compare_test"][c][0])
        m[f"rf-{c}-gain-bag"] = points(rf_test, p3["compare_test"][c][1])
        for name, key in (("Bagging", "bag"), ("Random Forest", "rf")):
            alike = p3["alike"][f"{c}-{name}"]
            m[f"agree-{c}-{key}"] = f"{alike['agreement']:.3f}"
            m[f"treeacc-{c}-{key}"] = pct(alike["tree_test"])
        t = p4["test"][c]
        for column in ("router", "oracle", "p_A", "p_B"):
            m[f"t3-{c}-{column.replace('_', '').lower()}"] = pct(t[column])
        m[f"hier-{c}-gap"] = points(t["oracle"], t["overall"])
        m[f"hier-{c}-loss"] = points(p4["flat"][c][1], t["overall"])
        m[f"hier-{c}-router-loss"] = points(p4["flat"][c][2], t["router"])
        for family, f in p4["per_family"][c].items():
            short = family[:4].lower()
            m[f"{short}-rows"] = str(f["rows"])
            m[f"hier-{c}-{short}-recall"] = pct(f["router_recall"])
            m[f"hier-{c}-{short}-expert"] = pct(f["expert_alone"])
            m[f"hier-{c}-{short}-final"] = pct(f["final"])
    family_rows = p4["family_rows"]  # per family and criterion: [router right, flat right]
    for c in CRITERIA:
        for family, counts in family_rows.items():  # test rows the flat tree gets right more
            router_right, flat_right = counts[c]
            key = "deficit" if family == "All" else f"{family[:4].lower()}-deficit"
            m[f"hier-{c}-{key}"] = str(flat_right - router_right).replace("-", r"\textminus{}")
        m[f"flat-{c}-bufo-recall"] = pct(family_rows["Bufonidae"][c][1] / int(m["bufo-rows"]))
    m["rf-m"] = str(p3["m"])
    mantissa, exponent = f"{p5['epsilon']:.2e}".split("e")
    m["nb-eps"] = rf"{mantissa} \times 10^{{{int(exponent)}}}"
    for species, feature in p5["constant"]:  # one such pair on this data
        m["nb-constant-species"], m["nb-constant-feature"] = species, str(feature)
        n_predicted, n_right = p5["predicted"][species]
        m["nb-constant-predicted"] = str(n_predicted)
        m["nb-constant-wrong"] = str(n_predicted - n_right)
        m["nb-constant-shared"] = str(p5["shared"][species])
    within = p5["within"].values()
    m["within-min"], m["within-max"] = f"{min(within):.2f}", f"{max(within):.2f}"
    corr = p5["correlation"]
    m["corr-mean"], m["corr-max"] = f"{corr['mean_abs']:.2f}", f"{corr['max_abs']:.2f}"
    m["corr-above"] = str(corr["above_half"])
    both, total = leakage()
    m["rec-both"], m["rec-total"] = str(both), str(total)
    data = common.Data()
    m["n-train"], m["n-test"] = str(len(data.y_train)), str(len(data.y_test))
    m["python"] = platform.python_version()
    m["numpy"], m["pandas"] = np.__version__, pd.__version__
    return m


def main():
    RESULTS.mkdir(exist_ok=True)
    names = sys.argv[1:] or list(PARTS)
    if unknown := set(names) - set(PARTS):
        raise SystemExit(f"unknown part(s): {', '.join(sorted(unknown))}")
    path = RESULTS / "results.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    for name in names:
        results[name] = plain(run(name))
    path.write_text(json.dumps(results, indent=1) + "\n")
    lines = ["% Generated by python -m tools.run_all from results/results.json; do not edit."]
    lines += [rf"\resdef{{{k}}}{{{v}}}" for k, v in macros(results).items()]
    (RESULTS / "numbers.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
