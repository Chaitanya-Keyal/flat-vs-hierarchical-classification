# Part 4: the hierarchical classifier

## In short

A router tree predicts the family; the expert tree of that family then predicts the species. The
router learns Family on all 5756 training rows, each expert learns Species on the rows of its
*true* family (the two one-species families get one-leaf, constant experts). An expert names only
species of its own family, so a routing error is always a species error, and exactly
A_overall = A_router − p_A. Test accuracy is 93.82 / 94.65% (Gini / entropy), 1.11 / 0.76 points
below the flat tree. The experts are fine: routed by the true family they reach 98.26 / 98.33%
(the oracle) and lose only p_A = 1.25 / 0.97% after correct routing. The router is the
bottleneck: it misroutes p_B = 4.93 / 4.38% (about 4× p_A), and it is 1.18 points below the family
accuracy the flat tree gets for free. A perfect router would add 4.44 / 3.68 points. On this
dataset the hierarchy does not help.

Pairs a / b are Gini / entropy; numbers are on the 1439 test rows unless stated.

## The idea

**Local classifier per parent node.** The labels form a tree: 10 species in 4 families, each
species in exactly one family. A flat classifier ignores the tree and separates all 10 species at
once. The hierarchy follows it top-down, with one multi-class classifier at every non-leaf node
choosing among that node's children (the "local classifier per parent node" of Silla and
Freitas, the handout's reference [2]). Here: the root (the **router**, 4 families) and each family
(its **expert**, its species). Five unconstrained trees per criterion, router and experts with the
same criterion (the handout's Table 1).

Training rows (as `results/part4.txt` prints them) and tree sizes, Gini / entropy:

- Bufonidae, 54 rows, all Rhinellagranulosa: expert 1 leaf.
- Dendrobatidae, 434 rows, all Ameeregatrivittata: expert 1 leaf.
- Hylidae, 1732 rows: HylaMinuta 248, HypsiboasCinerascens 378, HypsiboasCordobae 897,
  OsteocephalusOophagus 91, ScinaxRuber 118. Expert 63 / 60 leaves.
- Leptodactylidae, 3536 rows: AdenomeraAndre 538, AdenomeraHylaedactylus 2782,
  LeptodactylusFuscus 216. Expert 17 / 16 leaves.
- Router, all 5756 rows: 160 / 138 leaves for 4 classes. The flat tree of Part 1 has 211 / 181
  leaves for 10.

**Training.** The router is `DecisionTree(criterion).fit(X, family)` on every training row. The
expert of family f is `DecisionTree(criterion).fit(X[family == f], species[family == f])`, where
`family` holds the true labels.

**Why the true family, not the predicted one.** An expert answers "which species, given that the
row is from family f", and the rows that teach that are the rows of family f. Trained on the
router's predictions, an expert would also learn the foreign species the router sends it, it
would change whenever the router changes, and a wrong family could then give the right species,
which breaks the failure analysis below. The router's mistakes only matter at test time. (In our
setting the two coincide anyway: the unconstrained router is 100% right on the training rows. See
the code changes.)

**Inference.** Route every row with the router, group the rows by predicted family, let each
expert predict its whole group in one call, and return the species in the original row order.

**When can the hierarchy win?**

- *Specialisation.* Each expert solves a smaller problem (5 species for Hylidae, 3 for
  Leptodactylidae, 1 for the other two) with the splits that matter inside its family; the
  router only separates 4 coarse groups.
- *Capacity* (Babbar et al., NIPS 2013, Theorem 1). For a top-down classifier the generalisation
  error is at most the empirical (margin) loss, plus a capacity term proportional to
  Σ |D(v)|(|D(v)| − 1) / √m over the non-leaf nodes v (|D(v)| = the number of children of v),
  plus a confidence term. Flat: one node with 10 children, 10 × 9 = 90. Ours: 4 × 3 (root) +
  5 × 4 (Hylidae) + 3 × 2 (Leptodactylidae) + 0 + 0 = 38. Capacity favours the hierarchy; the
  empirical loss favours flat, because a top-down classifier makes several decisions and an early
  error propagates. They argue the hierarchy should win when the classes are very unbalanced
  (each decision is then less unbalanced) and flat when they are balanced.
- Caveats: the bound is for kernel (linear in a feature space) classifiers on taxonomies with
  thousands of classes; for 10 classes and trees it is intuition only. And our root decision is
  not less unbalanced: families range from 54 to 3536 training rows (65.5×), species from 54 to
  2782 (51.5×).

**When can it lose?**

- *Error propagation.* A row sent to the wrong family can never get the right species.
- *Fewer rows per model.* Each expert sees only its family (Hylidae: 1732 of 5756 rows). For
  trees this matters less than it sounds: a flat tree's deep splits are also chosen from the few
  rows that reach them. The oracle (98.26 / 98.33%) shows the experts are fine.
- *The router's problem is not easier.* Each family is a union of species clusters (Results).
- So the hierarchy wins only if the experts gain more than the router loses. Here they do not.

## Metrics

On the N = 1439 test rows, with true and router-predicted family y^F_n, ŷ^F_n and true and final
species y^S_n, ŷ^S_n:

- A_router = (1/N) Σ_n 1[ŷ^F_n = y^F_n]; A_overall = (1/N) Σ_n 1[ŷ^S_n = y^S_n].
- A_oracle: the overall accuracy when every row goes to the expert of its true family.
- p_A: the fraction with the router right and the expert wrong (failure A).
- p_B: the fraction with the router wrong (failure B), so p_B = 1 − A_router.

**A_overall = 1 − p_A − p_B = A_router − p_A ≤ A_router.** Let R be the rows routed right, C the
rows given the right species, O the rows that the expert of their true family gets right.

1. **C ⊆ R: a routing error is always a species error.** The expert of family f was trained on
   family f's rows only, so its `classes_` are f's species and it can only name one of them. If
   the router picks f and the true family is g ≠ f, the predicted species is in f and the true
   one in g; each species has exactly one family, so they differ.
2. So each row is in exactly one of C, R \ C (failure A) and not-R (failure B):
   N = |C| + |R \ C| + |not R|. Divide by N: 1 = A_overall + p_A + p_B, so
   A_overall = 1 − p_A − p_B = A_router − p_A. Since p_A ≥ 0, A_overall ≤ A_router.

Step 1 needs both facts: experts trained on the true family (otherwise an expert may know foreign
species) and one family per species.

**The oracle.** On a row in R the hierarchy and the oracle run the same expert on the same row (a
tree's answer for a row does not depend on the other rows in its group), so they agree; on a row
not in R the hierarchy is always wrong. Hence C = R ∩ O, and:

- A_overall ≤ A_oracle, since C ⊆ O. Equal only if no misrouted row would have been right.
- A_oracle − A_overall = |O \ R| / N: the misrouted rows that the right expert would have got
  right. That is the price of router errors, and what a perfect router would add. It is at most
  p_B: the other misrouted rows would be lost anyway.
- 1 − A_oracle is not p_A: it is p_A plus the misrouted rows that the right expert also gets
  wrong.

Counts, Gini / entropy: |R| = 1368 / 1376, |C| = 1350 / 1362, |O| = 1414 / 1415. Failure A:
18 / 14 rows; failure B: 71 / 63. Of the misrouted rows, the right expert would get 64 / 53
right (the 4.44 / 3.68-point gap) and 7 / 10 wrong. 1 − A_oracle = 25 / 24 rows, i.e. 18 + 7
and 14 + 10.

## The code (`part4_hierarchical.py`)

It imports `DecisionTree` from Part 1 and the shared helpers from `common.py`; otherwise numpy.

### `HierarchicalClassifier`

```python
def predict(self, X):
    family = self.predict_family(X)
    species = np.empty(len(X), dtype=object)  # object: no truncated names
    for name in np.unique(family):
        routed = family == name
        species[routed] = self.experts[name].predict(X[routed])
    return species.astype(str)
```

- `__init__(router, experts)` stores a fitted router and the dict `{family name: fitted expert}`
  the handout asks for. It only ever calls `.predict(X)` on them, so any model with a `predict`
  can be a router or an expert: a forest, another hierarchy, or the tests' `AlwaysB`.
- `predict_family(X)` is the router's output, one family name per row.
- `predict(X)` routes all rows once, then loops over `np.unique(family)`: the predicted families
  that occur, sorted, at most 4. The loop is over groups, not samples, so each expert is called
  once with all its rows. On the Gini test set the four calls get 12, 103, 433 and 891 rows.
- **Order.** `routed` is a boolean mask. `X[routed]` holds the group's rows in their original
  relative order, and `species[routed] = ...` writes the k-th answer to the k-th `True`
  position, which is the row it belongs to. So the output is in the original order, with no
  sorting and no index bookkeeping.
- **`dtype=object`.** numpy string arrays have a fixed width and silently cut longer strings.
  `np.empty(n, dtype=str)` and `np.full(n, "")` are `<U1` (one letter); `np.empty_like(family)`
  is `<U15` (Leptodactylidae is the longest family name) and turns AdenomeraHylaedactylus into
  AdenomeraHylaed: test accuracy 13.83%, and no error (extra check, not in the submission). An
  object array holds Python strings of any length; `.astype(str)` then returns a normal `<U22`
  array, the same kind `DecisionTree.predict` returns.
- A family the router never predicts gets no call, so no expert sees an empty array.

### `fit_hierarchy(criterion, X, family, species)`

- Router: one `DecisionTree(criterion)` on all rows, label Family.
- Experts: a dict comprehension over `np.unique(family)`, each tree fitted on `X[family == name]`
  and `species[family == name]`. `family` is the true label array (`data.f_train`); the router's
  predictions are never used in training.
- A one-species family needs no special case: its root node is pure, so `_grow` stops at once
  and the tree is one leaf that returns that species for any input (Part 1's sanity check (d)).
- Nothing in it is specific to frogs: it fits "parent label → child label" for any two label
  arrays, which makes a genus level a few lines (code changes).

### `oracle_predict(model, X, family)`

The same grouping as `predict`, but by the true `family` instead of the router. It gives
A_oracle and the "expert alone %" column.

### `analyse(model, X, family, species)`

```python
router_right = model.predict_family(X) == family
right = model.predict(X) == species
n_a, n_b = (router_right & ~right).sum(), (~router_right).sum()
assert not (right & ~router_right).any()  # a wrong family never gives the right species
assert right.sum() == router_right.sum() - n_a  # A_overall = A_router - p_A
```

- One boolean per row; `n_a` counts failure A, `n_b` failure B.
- The first assert is step 1 of the proof (C ⊆ R), checked on the data. The second is the
  handout's numerical check, on integer counts, so it is exact: no tolerance, no rounding. (It
  follows from the first; it is there because the handout asks for it.) The line
  `A_router - p_A = 93.82 = A_overall (exact on counts)` that `details` prints formats the same
  quantities.
- Returns `router`, `overall`, `oracle`, `p_A`, `p_B` as fractions. `main` calls it on the
  training and the test set. On the training set they are 1, 1, 1, 0, 0 (every tree fits it
  exactly), which is why Table 3 is about the test set.
- It runs the router twice (directly, and inside `predict`). That costs milliseconds and keeps
  `predict` self-contained.

### `details(criterion, model, train, test, data)`

Prints, per criterion: router and overall accuracy (train and test); oracle, p_A, p_B and the
identity line; the 4 × 4 router and the 10 × 10 species confusion matrices; the per-family table.
Returns the per-family numbers.

- `common.confusion_matrix(y_true, y_pred, labels)` needs `labels` sorted (`np.unique` sorts).
  `np.searchsorted` turns names into indices, each row's cell is `true * k + predicted`, and
  `np.bincount(cells, minlength=k * k).reshape(k, k)` counts every cell at once. Rows are true,
  columns predicted, both in `labels` order. numpy only, as the handout asks. `print_confusion`
  prints the indices as column headers.
- Per family, over the test rows of that true family: `rows`; `router recall`, the share routed
  to it; `expert alone`, the oracle's accuracy on them; `final`, the hierarchy's accuracy on
  them. final is at most each of the other two.

### `main()`

1. `Data()`: the handout's split, 5756 / 1439 rows. Prints the training rows per family and
   species.
2. For each criterion: `fit_hierarchy`, then `analyse` on the training and the test set.
3. Prints the leaves per tree; the two results-matrix rows (Train % and Test % are the final
   species accuracy, Family % is the router's test accuracy, as the handout's footnote says);
   Table 3 (router, oracle, (A), (B)).
4. `details` for each criterion.
5. Against the flat tree: refits Part 1's tree (each script runs on its own; the fit is
   deterministic, so these are Part 1's numbers) and prints flat species against hierarchy,
   flat derived family against router, and the bottleneck, `"router" if p_B > p_A else
   "experts"`; then the perfect-router gain, oracle − overall. Last, per family, the test rows
   given the right family by the router and by the flat tree (its predicted species' family).
6. Returns everything for `tools/run_all.py` (`results.json`, `numbers.tex`). Run directly, the
   script only prints.

### The handout's three common mistakes

1. *Experts trained on the predicted family.* `fit_hierarchy` slices by the true labels.
   `test_experts_learn_the_species_of_their_true_family` checks that every expert's `classes_`
   are exactly its family's species.
2. *Every row through every expert, then keep one answer.* `predict` routes first, and each
   expert sees only its group. (With trees the answers would even match, but that is four
   passes over all rows, and not routing.)
3. *Grouped order.* The answers go back through the same mask.
   `test_predict_matches_routing_each_row_alone` compares `predict` with a per-row loop on a toy
   set whose names include `c_with_a_much_longer_name`, so it catches truncation too.

The other three tests in `tests/test_part4.py`: a one-species family gets a constant expert; the
failure fractions add up and A_overall ≤ A_oracle; a router that always says "B" makes every
non-B row wrong (A_router = A_overall = 0, p_B = 1).

## Results and what they mean

Test %, from `results/part4.txt`:

| | Gini | Entropy |
|---|---|---|
| Flat tree (Part 1), species | 94.93 | 95.41 |
| Hierarchy, species (A_overall) | 93.82 | 94.65 |
| Flat tree, derived family | 96.25 | 96.80 |
| Router (A_router) | 95.07 | 95.62 |
| Oracle (A_oracle) | 98.26 | 98.33 |
| (A) p_A: router right, expert wrong | 1.25 | 0.97 |
| (B) p_B: router wrong | 4.93 | 4.38 |
| A perfect router would add (A_oracle − A_overall) | 4.44 | 3.68 |

Training accuracy is 100.00% for the router, the hierarchy and the oracle, with both criteria.
Every test number is optimistic for new recordings: 55 of the 60 recordings have syllables on
both sides of the split (the handout's leakage caveat).

**Hierarchy against flat: 1.11 / 0.76 points below (16 / 11 rows).** The exact accounting, in
rows: the router gets 17 fewer families right than the flat tree's derived family (1368 against
1385, 1376 against 1393). Inside the right family the experts lose 18 / 14 rows and the flat
tree 19 / 20 (family right, species wrong). So specialisation wins back 1 / 6 rows:
−17 + 1 = −16 and −17 + 6 = −11. The experts do their own job slightly better than the flat
tree; the routing costs more.

**The router is the bottleneck.** p_B is 3.9 / 4.5 times p_A (71 against 18 rows, 63 against
14). With a perfect router the hierarchy would score the oracle's 98.26 / 98.33%, above every flat
model in the matrix (the best is the random forest, 97.78 / 97.64%).

**Why the router is worse than the flat tree's family (1.18 points, 17 rows, both criteria).** A
plausible reason, not a proof: each family is a union of species clusters (Hylidae five,
Leptodactylidae three), and one tree must carve those unions out from the coarse labels alone. A
split that separates two species but leaves the family mix unchanged scores zero family gain, so
the greedy router cannot see that it sets up a good next split. The flat tree learns from the
finer labels and gets the family for free by mapping its species. Families are not a much
simpler problem for a tree: the router needs 160 / 138 leaves for 4 classes, the flat tree
211 / 181 for 10. The effect is small (17 rows on one split), but forests point the same way
(extra: a 50-tree forest router reaches 98.19 / 98.12%, the flat forest's derived family
98.40 / 98.19%).

**Where routing fails.** The Gini router matrix (rows true, columns predicted):

```
                              0    1    2    3
 0 Bufonidae                 10    0    1    3
 1 Dendrobatidae              1   96    5    6
 2 Hylidae                    1    4  404   24
 3 Leptodactylidae            0    3   23  858
```

- The diagonal is the 1368 rows routed right. Rows sum to the true family sizes (14, 108, 433,
  884); columns to the rows each expert receives (12, 103, 433, 891).
- Hylidae ↔ Leptodactylidae is the main confusion: 24 + 23 = 47 of the 71 routing errors
  (entropy: 18 + 19 = 37 of 63).
- Column 0: the Bufonidae expert receives 12 rows, 10 of them Bufonidae (precision 83.33%;
  entropy 10 of 11, 90.91%).

Per family, Gini / entropy:

| Family | rows | router recall % | expert alone % | final % |
|---|---|---|---|---|
| Bufonidae | 14 | 71.43 / 71.43 | 100.00 / 100.00 | 71.43 / 71.43 |
| Dendrobatidae | 108 | 88.89 / 87.96 | 100.00 / 100.00 | 88.89 / 87.96 |
| Hylidae | 433 | 93.30 / 94.69 | 95.61 / 96.30 | 89.84 / 91.92 |
| Leptodactylidae | 884 | 97.06 / 97.40 | 99.32 / 99.10 | 96.72 / 97.17 |

**Small families: routing is everything, but the deficit is elsewhere.** Bufonidae has 54 training
rows (0.94%) and 14 test rows, of which the router gets 10. Its expert and Dendrobatidae's are
constants (expert alone 100%), so final = router recall: routing is everything for them. Their
misrouted rows land in the big families (Bufonidae, Gini: 3 to Leptodactylidae, 1 to Hylidae;
entropy: all 4 to Leptodactylidae). Nuance for the viva: the flat tree is weak on Bufonidae too (9 /
11 of its 14 rows, derived family recall 64.29 / 78.57%; the last table of the output), so this is a
single tree on a rare class more than the hierarchy itself. In rows, most of the router's 17-row
deficit comes from the big families (Gini: 9 Hylidae + 9 Leptodactylidae, 1 back on Bufonidae;
entropy: 9 Leptodactylidae, 4 Dendrobatidae, 3 Hylidae, 1 Bufonidae). The PDF's discussion says the
same.

**The experts.** All of failure A comes from the two multi-species families: Hylidae 15 / 12
rows, Leptodactylidae 3 / 2. Hylidae is the hard one: five species, 63 / 60 leaves,
95.61 / 96.30% alone.

**Reading the 10 × 10.** Indices are the species in sorted order, so the families interleave:
Leptodactylidae 0, 1, 6; Dendrobatidae 2; Hylidae 3, 4, 5, 7, 9; Bufonidae 8. Off-diagonal cells
inside a family are expert errors (they sum to 18 / 14, failure A); cells across families are
routing errors (71 / 63, failure B). Summing the 10 × 10 over family blocks gives the 4 × 4
exactly, because every predicted species is in the predicted family. Example, Gini: HylaMinuta
(row 3) has 49 right and 13 wrong, 7 + 2 to the two Adenomera species and 4 to
Ameeregatrivittata: all routing errors. The weakest species is OsteocephalusOophagus, 16 of 23
(69.57%) with both criteria.

**Gini against entropy.** Entropy is ahead on the headline numbers (router, overall, oracle, p_A,
p_B), but by a few rows (router 8, overall 12) on one split, and not on every family: Gini routes
Dendrobatidae better (88.89 against 87.96%) and its Leptodactylidae expert is better alone
(99.32 against 99.10%). Not a strong conclusion.

**Does the hierarchy help on this dataset?** No. The experts are good, but the router loses more
than specialisation gains.

## Demo

1. From the repo root (where `Frogs_MFCCs.csv` is): `python part4_hierarchical.py`, or
   `uv run python part4_hierarchical.py`. About 2 s; the output is `results/part4.txt` byte for
   byte.
2. Walk down the output:
   - the training rows per family: two families have one species, so their experts are one leaf
     (the leaves table);
   - the results-matrix rows: 93.82 / 94.65, Family % = router 95.07 / 95.62;
   - Table 3: router, oracle, (A), (B);
   - per criterion: the identity line, the 4 × 4 (Hylidae ↔ Leptodactylidae), the per-family
     table (Bufonidae 71.43%; final = router recall for the constant experts);
   - against the flat tree: bottleneck "router"; a perfect router would add 4.44 / 3.68;
   - the per-family rows, router / flat tree: the 17-row gap sits in Hylidae and
     Leptodactylidae.
3. In the code: `predict` (mask write-back, object array), `fit_hierarchy` (`family == name` on
   the true labels), the two asserts in `analyse`.
4. In `results_matrix.pdf`: the shaded rows of Table 2, Table 3 and the paragraph "Hierarchy
   (Part 4)".
5. `uv run pytest tests/test_part4.py`: 5 tests, under a second.
6. Live checks in `uv run python` from the repo root (all run and checked):

```python
import numpy as np
from common import Data, confusion_matrix
from part4_hierarchical import HierarchicalClassifier, analyse, fit_hierarchy, oracle_predict

d = Data()
m = fit_hierarchy("gini", d.X_train, d.f_train, d.y_train)
pred = m.predict(d.X_test)
np.mean(pred == d.y_test)  # 0.9382, A_overall
np.mean(oracle_predict(m, d.X_test, d.f_test) == d.y_test)  # 0.9826, A_oracle


class TrueFamily:  # a router that knows the answer (for d.X_test only): the oracle
    def predict(self, X):
        return d.f_test


analyse(HierarchicalClassifier(TrueFamily(), m.experts), d.X_test, d.f_test, d.y_test)
# router 1.0, overall = oracle = 0.9826, p_A 0.0174 (= 1 - oracle), p_B 0.0

# original order: a per-row loop gives the same list; shuffled rows carry their answers
one = [m.experts[m.router.predict(x[None])[0]].predict(x[None])[0] for x in d.X_test]
one == pred.tolist()  # True
perm = np.random.default_rng(0).permutation(len(d.X_test))
(m.predict(d.X_test[perm]) == pred[perm]).all()  # True

# one call per expert: wrap the experts and record the group sizes
calls = []


class Spy:
    def __init__(self, expert):
        self.expert = expert

    def predict(self, X):
        calls.append(len(X))
        return self.expert.predict(X)


HierarchicalClassifier(m.router, {f: Spy(e) for f, e in m.experts.items()}).predict(d.X_test)
calls  # [12, 103, 433, 891]

# a wrong family never gives the right species
fam = m.predict_family(d.X_test)
((pred == d.y_test) & (fam != d.f_test)).any()  # False

# the 10 x 10 summed over family blocks is the 4 x 4
S, F = np.unique(d.y_train), np.unique(d.f_train)
G = (d.families(S)[:, None] == F).astype(int)  # species x family, 0/1
cm10 = confusion_matrix(d.y_test, pred, S)
(G.T @ cm10 @ G == confusion_matrix(d.f_test, fam, F)).all()  # True

m.experts["Bufonidae"].n_leaves, m.experts["Bufonidae"].classes_  # 1, ['Rhinellagranulosa']
```

## Likely viva questions

- *What is a "local classifier per parent node"?* One multi-class classifier at every non-leaf
  node of the label tree, choosing among its children: the router at the root (4 families) and
  one expert per family (its species). Five trees per criterion here.
- *Why train the experts on the true family?* An expert models "which species, given family f",
  so it learns from family f's rows. Training on predicted families mixes in foreign species,
  ties the experts to the router and breaks the failure analysis. Here it would not even change
  the numbers: the unconstrained router is 100% right on the training rows.
- *What happens to a test row sent to the wrong family?* Its species is always wrong: the expert
  names only its own family's species, and every species has exactly one family. `analyse`
  asserts it.
- *Prove A_overall = A_router − p_A.* Species-right rows are all router-right, and router-right
  rows are species-right or failure A. So #right = #router-right − #A; divide by N.
- *Why is A_overall ≤ A_oracle?* On correctly routed rows both run the same expert on the same
  row; on misrouted rows the hierarchy is always wrong and the oracle may be right.
- *Is p_A = 1 − A_oracle?* No. 1 − A_oracle (Gini: 1.74%, 25 rows) also counts the misrouted rows
  that the right expert would get wrong too (7); those are in p_B. p_A is 1.25% (18 rows).
- *Router or experts: which is the bottleneck?* The router: p_B = 4.93 / 4.38% against
  p_A = 1.25 / 0.97%. A perfect router would give 98.26 / 98.33%, above every flat model.
- *Does the hierarchy help here?* No: 93.82 / 94.65% against 94.93 / 95.41%. The router gets 17
  fewer families right than the flat tree; the experts win back only 1 / 6 rows.
- *Why is the router below the flat tree's derived family accuracy?* Families are unions of
  species clusters; one tree must carve the unions out from coarse labels, and a split that
  separates species without changing the family mix scores zero gain. The flat tree learns from
  the finer labels and gets the family for free. Plausible, consistent with 160 router leaves
  against 211 flat leaves; not proven.
- *Why is Bufonidae routed so badly?* 54 of 5756 training rows; 4 of its 14 test rows land in
  leaves of the big families. Its expert is a constant, so final = router recall = 71.43%. The
  flat tree is weak there too (64.29 / 78.57%).
- *What do the Bufonidae and Dendrobatidae experts look like?* One leaf each: all their training
  rows share one label, the root is pure and the tree stops (sanity check (d)). They return that
  species for any input.
- *Why is training accuracy 100%?* Unconstrained trees split until every leaf is pure (no two
  identical training rows disagree), so router, experts and hierarchy fit the training set
  exactly; p_A = p_B = 0 there.
- *How does `predict` avoid per-sample loops and keep the order?* One router call; a loop over
  the at most 4 predicted families, one expert call per group through a boolean mask; the answers
  are written back through the same mask.
- *Why `dtype=object`?* numpy fixed-width strings truncate silently: an array shaped like the
  family names (`<U15`) would cut AdenomeraHylaedactylus to AdenomeraHylaed.
- *What is the "Family acc." cell in the hierarchical rows?* The router's test accuracy
  (95.07 / 95.62%), not a derived accuracy.
- *What do Babbar et al. say?* The bound's capacity term grows with Σ |D(v)|(|D(v)| − 1) over
  the non-leaf nodes: 90 flat, 38 for our hierarchy, so capacity favours the hierarchy; the
  empirical error favours flat because errors propagate. Which wins depends on the taxonomy and
  the data; here propagation wins.
- *Gini or entropy?* Entropy is ahead on router, overall and oracle accuracy, but by a few rows
  (router 8, overall 12) on one split, and Gini is ahead on some families: not a strong
  conclusion.
- *Would soft routing help?* Not with these trees: every leaf is pure, so the router's
  "probabilities" are one-hot and top-2 routing returns exactly the hard answer (checked). It
  needs soft scores at both levels, e.g. forests.

## Likely code changes

All in `part4_hierarchical.py`. Numbers marked *extra* come from throwaway runs, not from the
submission. If a change is kept, rerun `uv run python -m tools.run_all` so that `results/`
matches again: `tools/package.py` diffs every script's output against it byte for byte.

**Train the experts on the predicted family** (the handout's first mistake). In `fit_hierarchy`:

```python
routed = router.predict(X)  # used in place of the true family when slicing
experts = {
    name: DecisionTree(criterion).fit(X[routed == name], species[routed == name])
    for name in np.unique(routed)
}
```

With the unconstrained router nothing changes: it is 100% right on the training rows, so
`routed` equals `family` and the experts are identical. To see an effect, cap the router, e.g.
`DecisionTree(criterion, max_depth=5)` (training accuracy 94.89 / 94.61%, extra). Then the Gini
Bufonidae expert also learns Ameeregatrivittata and HylaMinuta, 60 / 58 test rows get the right
species from a wrong family, and the first assert in `analyse` fails (compute the metrics without
it to compare). Accuracy actually rises, from 91.31 / 91.80% with true-family experts to
94.51 / 95.48%, because the experts learn to undo the router's training mistakes. So the point is
not a lower score: the experts are no longer family experts, the model is not the hierarchy being
evaluated, and p_A, p_B and the oracle lose their meaning.

**A different router.** `HierarchicalClassifier` only calls `router.predict`, so only the router
line in `fit_hierarchy` changes:

```python
from part3_random_forest import RandomForest

router = RandomForest(50, criterion, 4).fit(X, family)  # about 4 s per fit
```

Extra: router 98.19 / 98.12%, overall 96.80 / 96.87% (p_B 1.81 / 1.88%), above the single flat
tree but below the flat forest (97.78 / 97.64%). Part 1's species tree as the router, read
through the species → family map, routes at 96.25 / 96.80% and gives 94.86 / 95.55%, level with
the flat tree (94.93 / 95.41%):

```python
class FlatRouter:  # Part 1's species tree, read as a family router
    def __init__(self, tree, data):
        self.tree, self.data = tree, data

    def predict(self, X):
        return self.data.families(self.tree.predict(X))


flat = DecisionTree(c).fit(data.X_train, data.y_train)
model = HierarchicalClassifier(FlatRouter(flat, data), models[c].experts)
```

**Soft routing (top-2 families).** Score species s of family f by P(f | x) · P(s | x, f) over the
two likeliest families and keep the best. A tree's P is the class weights of the row's leaf
(`node["weights"]`); a forest's is its vote fractions:

```python
from common import tally


def proba(model, X):
    """P(class | x), columns in model.classes_ order."""
    if hasattr(model, "votes"):  # a forest: the fraction of its trees voting for each class
        return tally(model.votes(X), len(model.classes_)) / len(model.trees_)
    W = np.array([node["weights"] for node in model.nodes])[model.apply(X)]
    return W / W.sum(axis=1, keepdims=True)  # a tree: the class weights of each row's leaf


def predict_soft(model, X, k=2):
    P = proba(model.router, X)
    top = np.argsort(-P, axis=1, kind="stable")[:, :k]
    best, score = np.empty(len(X), dtype=object), np.full(len(X), -1.0)
    for j, name in enumerate(model.router.classes_):
        rows = np.flatnonzero((top == j).any(axis=1))
        if rows.size == 0:
            continue
        expert = model.experts[name]
        s = P[rows, j][:, None] * proba(expert, X[rows])
        better = s.max(axis=1) > score[rows]
        best[rows[better]] = expert.classes_[s.argmax(axis=1)[better]]
        score[rows[better]] = s.max(axis=1)[better]
    return best.astype(str)
```

With the assignment's unconstrained trees every leaf is pure, so P is one-hot and `predict_soft`
returns exactly `predict` (checked). With 50-tree forests at both levels (extra): 97.29 / 97.57%
soft against 97.57 / 97.50% hard, no clear gain. With soft routing a wrong top family no longer
forces a wrong species, so the first assert in `analyse` would no longer hold.

**A genus level** (Family → Genus → Species). The handout leaves Genus unused, so this is an
extension; Genus may be a label, never a feature. A two-level hierarchy has a `predict`, so it can
be an expert, and `fit_hierarchy` already fits "parent → child" for any labels:

```python
import pandas as pd

from common import data_path, stratified_split

df = pd.read_csv(data_path())
train, test = stratified_split(df["Species"].to_numpy(dtype=str))  # the same split as Data
genus = df["Genus"].to_numpy(dtype=str)
g_train, g_test = genus[train], genus[test]


def fit_three_levels(criterion, X, family, genus, species):
    router = DecisionTree(criterion).fit(X, family)
    experts = {
        f: fit_hierarchy(criterion, X[family == f], genus[family == f], species[family == f])
        for f in np.unique(family)
    }
    return HierarchicalClassifier(router, experts)
```

Do not change the three values `common.load()` returns: every part unpacks them.
`analyse(model, data.X_test, data.f_test, data.y_test)` works unchanged (the oracle then fixes
only the family). Only Adenomera and Hypsiboas have two species; the other six genera have one.
Extra: 92.98 / 94.23%, below the two-level 93.82 / 94.65%: one more decision, one more place for
errors to propagate.

**More per-family numbers.** `details` already prints router recall, expert alone and final per
family. The router's precision (of the rows an expert receives, the share that belong to it) and
per-species recall come from the matrices it already builds:

```python
cm = confusion_matrix(data.f_test, routed, families)
precision = np.diag(cm) / cm.sum(axis=0)  # per column: Bufonidae 83.33 / 90.91%
cm10 = confusion_matrix(data.y_test, predicted, species)
recall = np.diag(cm10) / cm10.sum(axis=1)  # per row: lowest OsteocephalusOophagus 69.57%
```

A family the router never predicts has an empty column (0 / 0: nan and a warning). The flat
tree's per-family counts are already printed (the last table of `main`, from `flat_trees`);
Bufonidae gets 9 / 11 of 14 rows, 64.29 / 78.57%.

**Balanced weights for the small families.** `DecisionTree.fit` takes `sample_weight`; give every
family the same total weight. In `fit_hierarchy`:

```python
_, inv, counts = np.unique(family, return_inverse=True, return_counts=True)
w = 1 / counts[inv]  # the scale does not matter: the gain uses weight ratios only
router = DecisionTree(criterion).fit(X, family, sample_weight=w / w.sum())
```

Extra: router 95.62 / 95.76%, overall 94.30 / 94.58% (from 93.82 / 94.65%), training accuracy
still 100%. It does not help where it was meant to: Bufonidae recall falls to 64.29% with Gini
and stays 71.43% with entropy (Dendrobatidae rises to 93.52% with entropy). An unconstrained
tree's leaves are pure whatever the weights, so the weights only change which splits are chosen.
Weighting the experts' species as well makes it worse (93.68 / 94.51%).
