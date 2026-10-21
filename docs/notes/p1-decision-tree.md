# Part 1: the decision tree

## In short

A classification tree grown greedily from the top: at each node, score every threshold of every
feature, keep the split x_j ≤ t with the largest impurity decrease (Gini or entropy), and recurse
until the node is pure. Impurities use class *weight* sums, so AdaBoost (Part 2) can reweight
rows; the default weights 1/N give the plain count-based tree. One stable sort per feature per
node and a cumulative sum of class weights score all cuts of all 22 features at once, so an
unconstrained tree on the 5756 training rows fits in about 0.45 s. Gains within 1e-12 of the best
count as tied and the handout's rule (lower feature, then lower threshold) decides, so rounding
never picks a split. Both unconstrained trees fit the training set exactly and reach 94.93%
(Gini: depth 17, 211 leaves) and 95.41% (entropy: depth 11, 181 leaves) on the test set, with
derived family accuracy 96.25% and 96.80%; the depth sweep runs from underfitting to overfitting.

## The maths

### Impurity and gain

For the rows of one node, with weights w_n:

- Class weights W_i = Σ w_n 1[y_n = i], node weight W = Σ_i W_i, class probabilities
  p_i = W_i / W. This is the handout's p_i = Σ w_n 1[y_n = i] / Σ w_n.
- Gini: G = 1 − Σ_i p_i², the chance that two rows drawn at random (with replacement, by weight)
  have different labels. 0 for a pure node; at most 1 − 1/C for a uniform one (0.9 for the
  C = 10 species).
- Entropy: H = −Σ_i p_i log2 p_i bits, with 0 log2 0 := 0 (the limit of p log p as p → 0). 0 for
  a pure node; at most log2 C (3.32 bits for 10 classes).
- Gain of a split into children L and R, with W_j the total weight in child j (W_L + W_R = W):
  Gain = Imp(parent) − (W_L/W) Imp(L) − (W_R/W) Imp(R). With G it is the Gini decrease, with H
  the information gain.
- The gain is never negative. Class weights add, so p(parent) = (W_L/W) p(L) + (W_R/W) p(R); G
  and H are concave, so the weighted mean of the children's impurities is at most the parent's
  (Jensen). Both are strictly concave, so the gain is 0 exactly when both children keep the
  parent's class mix.
- Gini gains and information gains have different scales and are never compared. At the root:
  Gini 0.7194 with best gain 0.2337; entropy 2.4296 bits with best gain 0.7756 bit.

### Weights

- AdaBoost (Part 2) fits each round's depth-3 tree to reweighted rows. The handout: replace class
  counts by class weight sums everywhere (impurity, gain, leaf majority).
- With w_n = 1/N: p_i = (count_i/N) / (n/N) = count_i / n and W_j/W = n_j / n, exactly the
  unweighted formulas. Any common factor cancels the same way: only relative weights matter
  (`test_weight_scale_does_not_change_the_tree`).
- An integer weight k acts like k copies of the row (`test_integer_weights_equal_duplicated_rows`).
- Weights do not enter the purity test (labels), `min_samples_split` (rows) or the candidate
  thresholds (feature values).

### Worked examples: the handout's sanity checks

(a) Counts (3, 3): p = (½, ½). G = 1 − ¼ − ¼ = 0.5 and H = −2 · ½ log2 ½ = 1 bit.

(b) X = 1, 2, 3, 4 with y = 0, 0, 1, 1, each weight ¼. Parent: G = 0.5, H = 1 bit. The candidate
thresholds are the midpoints 1.5, 2.5 and 3.5:

| t | left (rows per class) | right | Gini gain | information gain |
|---|---|---|---|---|
| 1.5 | (1, 0) | (1, 2) | 0.5 − ¾ · 4/9 = 1/6 ≈ 0.1667 | 1 − ¾ · 0.9183 = 0.3113 |
| 2.5 | (2, 0) | (0, 2) | 0.5 − 0 = 0.5 | 1 − 0 = 1 |
| 3.5 | (2, 1) | (0, 1) | 1/6 ≈ 0.1667 | 0.3113 |

using G(⅓, ⅔) = 1 − 1/9 − 4/9 = 4/9 and H(⅓, ⅔) = 0.9183 bits. t = 2.5 wins, both children are
pure and stop, so the tree has two leaves and fits the four rows.

(c) Three identical rows, y = 0, 0, 1, weights 0.1, 0.1, 0.8. No threshold separates equal values,
so there is no candidate split and the root is a leaf. Its class weights are (0.2, 0.8), so it
predicts 1 although two of the three rows are class 0: the weighted majority, not the count.

(d) One class only: the root is pure, so it is a leaf without any split search.

XOR, where an impure node has zero gain: (0, 0) → 0, (0, 1) → 1, (1, 0) → 1, (1, 1) → 0. Every
split leaves one row of each class on each side, so every gain is 0 and the root is a leaf,
although two levels would classify all four (`test_xor_has_no_positive_gain`). Greedy growth
looks one split ahead.

## The code: `part1_decision_tree.py`

`impurity`, the `DecisionTree` class, and the script (`sanity_checks`, `main`). Parts 2 to 4
import `DecisionTree`.

### `impurity(p, criterion)`

- Works along the last axis: one call scores one node (p of shape (C,)) or every candidate child
  at once (shape (n − 1, m, C)).
- `np.log2(p, out=np.zeros_like(p), where=p > 0)` takes the log only where p > 0 and leaves 0
  elsewhere, so p · log2 p is 0 there: 0 log2 0 = 0 with no warning and no nan.

### `__init__` and `fit`

- The handout's signature, with defaults `criterion="gini"`, `max_depth=None`,
  `min_samples_split=2`, `max_features=None`, `rng=None`. It rejects an unknown criterion, and
  `max_features` without an `rng`.
- `fit(X, y, sample_weight=None)`: X becomes a C-ordered float array, and
  `np.unique(y, return_inverse=True)` gives `classes_` (the sorted labels; strings are fine) and
  `codes`, each row's class index. `w` defaults to 1/n.
- The one-hot weight matrix, n × C: row i holds w_i in the column of its class and 0 elsewhere.
  Summing any set of rows gives that set's class weights, so `onehot[rows].sum(axis=0)` is a
  node's (W_1, ..., W_C).
- `self.nodes = []`, then `_grow` from the root (all rows, depth 0). Last, the node dicts are
  copied into the flat arrays `feature`, `threshold`, `left`, `right`, `value` for `apply`.

### `_grow(X, codes, onehot, rows, depth)`

- Makes the node: `feature` (−1 marks a leaf), `threshold`, `left` and `right` (child indices,
  −1 for none), `gain`, `value` (weighted majority), `n` (rows), `weights` (class weights),
  `depth`.
- Appends it before growing its children, so nodes are numbered in pre-order: the root is 0 and
  an internal node's left child is the next index.
- Stops (the node stays a leaf) if all its rows have one label, if it has fewer than
  `min_samples_split` rows, if it is at `max_depth`, or if `_best_split` finds no gain above
  1e-12. Otherwise it records the split, grows the whole left subtree (x ≤ t), then the right.

### `_best_split(X, onehot, rows, weights)`: every cut at once

n rows in the node, m candidate features (22, or 4 in the forest), C classes:

```python
x = X[np.ix_(rows, features)]  # (n, m)
order = np.argsort(x, axis=0, kind="stable")  # each column's sort order
x = np.take_along_axis(x, order, axis=0)  # each column sorted
left = np.cumsum(onehot[rows][order], axis=0)[:-1]  # (n - 1, m, C)
right = np.maximum(weights - left, 0)  # (n - 1, m, C)
```

1. `onehot[rows][order]` lines the rows' one-hot weights up in each feature's sorted order:
   (n, m, C). Its cumulative sum down the rows gives, at position k, the class weights of the
   k + 1 smallest rows, i.e. the left child of the cut between sorted positions k and k + 1.
   `[:-1]` drops the last position (all rows left, nothing right). One sort and one cumsum per
   node give all n − 1 cuts of every feature; there is no loop over thresholds.
2. The right child is the node's total minus the left. `np.maximum(..., 0)` clips rounding below
   zero: a class that is entirely on the left gets exactly 0 on the right, not −1e-18, which
   would give a pure child a slightly negative Gini.
3. `gain` is the formula above for the whole (n − 1, m) grid: three calls to `impurity`.
   `share(left, w_left)` turns class weights into probabilities. A side whose total weight is 0
   (rows too light to register, as AdaBoost's weights get after very many rounds) gets zeros,
   and since its W_j is 0 it adds nothing: no 0/0, no nan.
4. `gain[x[1:] == x[:-1]] = -np.inf`: a cut between two equal values is no threshold. Identical
   rows (check (c)) therefore have no candidate at all.
5. `best = gain.max()`, and `if not best > TOL: return None`. Written as `not >` so that −inf
   (no candidate) gives "no split".
6. The tie rule:

   ```python
   near = (gain >= best - TOL).T  # (m, n - 1): one row per feature
   f, k = np.unravel_index(np.argmax(near), near.shape)
   ```

   `argmax` of a boolean array is the first True, and after the transpose "first" means the
   lowest feature, then the lowest cut. Within a feature the cuts are in sorted order, so the
   lowest cut is the lowest threshold.
7. `a, b = x[k, f], x[k + 1, f]` and `t = (a + b) / 2`. If a and b are adjacent floats, the
   midpoint can round up to b, and x ≤ t would then send b left as well. The guard
   `if not a <= t < b: t = a` keeps the partition that was scored (an overflow to inf is caught
   the same way). It never triggers on this data.
8. The children are `np.sort(rows[order[: k + 1, f]])` and the rest: the row ids of the k + 1
   smallest values of feature f, exactly the rows that were scored on each side, back in row
   order. Since a ≤ t < b, `apply` sends every training row to the leaf it was grown in
   (`test_training_rows_reach_the_leaves_they_were_fitted_in`).
9. It returns `int(features[f])` (the real feature index; it differs from f only in the forest),
   t, the gain, and the two row sets.

`kind="stable"` keeps rows with equal values in row order, so every partial sum is added in the
same order on every machine and numpy version. Which partitions are scored does not depend on it
(cuts between equal values are masked), only the last bits of the sums.

Check (b) through the code (one feature, weights ¼): `left` = (¼, 0), (½, 0), (½, ¼); `right` =
(¼, ½), (0, ½), (0, ¼); `w_left` = ¼, ½, ¾. The gains are 0.1667, 0.5, 0.1667 (Gini) or 0.3113,
1, 0.3113 (entropy); the best is at k = 1, so a = 2, b = 3 and t = 2.5.

### The tie tolerance, `TOL = 1e-12`

- Ties are common. In 121 of the Gini tree's 210 splits and 86 of the entropy tree's 180, more
  than one feature reaches the best gain (within 1e-12). Most are small nodes (median 7 and 6
  rows), where several features separate a handful of rows equally well.
- Equal in exact arithmetic is not always equal in floats: two splits with the same gain can be
  computed through different operations (one child from the running sum, the other from
  total − left; classes added in a different order). In 9 Gini and 3 entropy nodes the tied
  gains differ, by 1.4e-17 to 4.4e-16.
- Example, in the Gini tree: a node of 27 rows with class counts 1, 24, 1, 1. MFCC 6 can cut off
  two odd rows on the left, MFCC 2 two other odd rows on the right. Both leave a 2-row child with
  two classes and a 25-row child at 24 : 1, so the gains are equal, but MFCC 2's comes out
  1.4e-17 lower. Plain argmax would take MFCC 6; the handout's rule takes MFCC 2.
- With exact comparison, rounding would pick a different split at 6 nodes of the Gini tree and 1
  of the entropy tree, and multiplying all weights by a constant (no change mathematically)
  would change the tree. With the tolerance it does not: weights 1/N, 1 and 7.3 give the same
  trees here, and `test_weight_scale_does_not_change_the_tree` checks it on small problems.
- Why 1e-12: rounding noise between tied gains is at most 4.4e-16, and the smallest real gap
  between the best gain and the next different one is 9.8e-8 (Gini tree) and 1.0e-5 (entropy
  tree). 1e-12 is far from both. In AdaBoost's reweighted trees rows can carry tiny weights, and
  real gains can come within about 1e-12 of each other; the band then calls them tied, a
  difference far too small to matter.
- The same `TOL` is the "positive gain" threshold: a split needs a gain above 1e-12, so a gain
  that is 0 mathematically but comes out as a few 1e-17 never counts as positive.

### Stopping, depth and leaves

- The root has depth 0 and a node at depth d has children at depth d + 1. `max_depth=1` lets only
  the root split: a stump with 2 leaves. `max_depth=0` would give one leaf, the majority class
  (48.33% of the training rows). `tree.depth` is the largest node depth, i.e. the number of
  edges on the longest path (17 and 11).
- `min_samples_split=2` (the handout's Table 1): a node needs 2 rows to split. A 1-row node is
  pure anyway, so at 2 the rule never binds. It counts rows, not weight.
- No gain above 1e-12: leaf (the handout's fourth stopping rule).
- A leaf predicts `value = int(weights.argmax())`, the weighted majority. `argmax` returns the
  first maximum, i.e. the smallest class index, and `classes_` is sorted, so ties go to the
  smallest label (`test_leaf_ties_go_to_the_smallest_label`). Internal nodes store a `value` too,
  which is handy for `predict_proba` or pruning.
- Growth is depth first, left child first (README.txt records this). The order only sets the
  node numbering; the tree is the same in any order.

### The Random Forest hook: `max_features` and `rng`

```python
features = np.sort(self.rng.choice(d, self.max_features, replace=False))
```

- The draw happens inside `_best_split`, so only nodes that search draw; nodes that `_grow`
  stops first (pure, too small, at `max_depth`) draw nothing. Part 3 shares one
  `RandomState(42)` between the bootstraps and these draws, so this is part of what makes the
  forest reproducible (README.txt records it).
- A fresh subset at every node, as the handout requires; one subset per tree would be the
  "different (and wrong)" algorithm.
- Sorted, so that a lower position in `features` is a lower real index and the tie rule still
  means "lower feature index"; `int(features[f])` maps the position back.
- If none of the m drawn features has a positive gain, the node becomes a leaf; there is no
  second draw (the handout's rule).
- Tests: a stub rng that always offers features 1 and 0 must give the tree grown on those two
  features alone and draw exactly once per searched node; `max_features=d` must give the plain
  tree.

### Flat arrays and the vectorised `apply` / `predict`

```python
node = np.zeros(len(X), dtype=int)  # every row starts at the root
active = np.flatnonzero(self.feature[node] >= 0)  # rows sitting at an internal node
while active.size:
    at = node[active]
    go_left = X[active, self.feature[at]] <= self.threshold[at]
    node[active] = np.where(go_left, self.left[at], self.right[at])
    active = active[self.feature[node[active]] >= 0]
```

- All rows move down one level per pass: one gather, one comparison, one `np.where`. The loop
  runs at most depth times (17), never once per row; the 1439 test rows take well under a
  millisecond.
- `predict(X)` is `classes_[value[apply(X)]]`. `n_leaves` counts `feature < 0`; `depth` is the
  largest node depth.

### Complexity

- Per node with n rows and m features: the sorts cost O(m n log n), the cumulative sums and gains
  O(n m C) in time and memory. At the root that is 5755 × 22 × 10 float64 values, about 10 MB
  per array.
- Nodes at one depth hold disjoint rows, so a level costs at most O(N m (log N + C)) and a tree
  O(D N m (log N + C)) for depth D. Measured: a training row takes part in 11.0 split searches
  on average in the Gini tree and 7.7 in the entropy tree.
- A loop over thresholds that recounts the classes for each cut would cost O(n² m) per node:
  22 × 5755 = 126 610 cuts × 5756 rows at the root alone.
- Fit: about 0.45 s per unconstrained tree; `python part1_decision_tree.py` (16 fits) about 5 s.
  Predict: D numpy steps, O(n_test · D) work.

### `sanity_checks` and `main`

- `sanity_checks` runs the handout's (a) to (d) with `assert`, so a failure stops the script
  before any training, and prints what it found.
- `main` loads the fixed split (`Data` in `common.py`), prints the header and the checks, fits
  the two unconstrained trees and prints their matrix rows through `data.scores` (train, test,
  derived family accuracy), then the tree sizes and the sweep over
  `DEPTHS = (1, 2, 3, 5, 8, 12, None)`. It returns all of it as a dict, which `tools/run_all.py`
  writes to `results/results.json` and `results/numbers.tex` for the PDF.

## Results

Pairs a / b are Gini / entropy. Matrix rows (`results/part1.txt`, %):

| Criterion | Train | Test | Family (test) |
|---|---|---|---|
| Gini | 100.00 | 94.93 | 96.25 |
| Entropy | 100.00 | 95.41 | 96.80 |

- Sizes: depth 17 / 11, 211 / 181 leaves (421 / 361 nodes). Every leaf is pure; 93 / 65 leaves
  hold a single training row, and the largest holds 2456 / 2558 rows of AdenomeraHylaedactylus.
- Roots: Gini splits on MFCC 22 ≤ 0.093605 (feature index 21), entropy on MFCC 13 ≤ 0.149325
  (index 12). Each criterion prefers its own: MFCC 13's split has Gini gain 0.2266 against
  0.2337, and MFCC 22's has information gain 0.7292 against 0.7756 bit. Both send most of
  AdenomeraHylaedactylus (2782 of the 5756 training rows, 48.3%) right: 2708 of the 2939 rows
  there for Gini, 2742 of 3129 for entropy.

Accuracy against `max_depth` (%, not in the matrix):

| max_depth | Gini train | Gini test | Entropy train | Entropy test |
|---|---|---|---|---|
| 1 | 60.98 | 61.01 | 61.78 | 62.40 |
| 2 | 69.13 | 69.28 | 73.38 | 73.94 |
| 3 | 78.51 | 79.22 | 84.40 | 84.50 |
| 5 | 90.10 | 88.60 | 91.80 | 90.90 |
| 8 | 97.85 | 93.95 | 98.68 | 94.86 |
| 12 | 99.81 | 94.86 | 100.00 | 95.41 |
| None | 100.00 | 94.93 | 100.00 | 95.41 |

- Depth 1 to 3, high bias: train ≈ test, both low; the trees are too simple. A tree of depth D
  has at most 2^D leaves, so it can name at most 2, 4 or 8 of the 10 species. At depth 3 the
  Gini tree's 8 leaves name only 5 species and the entropy tree's 7, one reason entropy leads
  early.
- Depth 5 to 8: test accuracy still climbs (Gini 88.60 → 93.95) but a gap opens (1.50 → 3.90
  points).
- Depth 12 and None, high variance: training accuracy reaches 100% while test accuracy levels
  off (Gini 94.86 → 94.93, one test row). The deepest splits isolate single rows: they fit
  details of this training sample, not structure, and would come out differently on another
  sample. That variance is what Parts 2 and 3 average away (bagging 96.66 / 96.87, the forest
  97.78 / 97.64).
- The entropy tree is only 11 deep, so max_depth 12 and None give the same tree.
- Entropy is ahead of Gini at every depth on this data; which criterion wins is an empirical
  question.

Derived family accuracy is higher because the family is read off the predicted species: a
species error inside the right family is still a family hit, so family accuracy can never be
below species accuracy. Gini makes 73 species errors on the 1439 test rows but only 54 family
errors; 19 errors stay inside the family (20 of 66 for entropy), 17 / 16 of them among the five
Hylidae species. Part 4's router (95.07 / 95.62) is measured against these 96.25 / 96.80.

## Demo

1. `python part1_decision_tree.py` (in the repo: `uv run python part1_decision_tree.py`), about
   5 s. Point at the data line (5756 / 1439 rows, 22 MFCCs, 10 species in 4 families), the
   sanity checks (a) to (d), the two matrix rows, the tree sizes, and the depth table: train
   reaching 100% while test levels off.
2. In the code: the one-hot line in `fit`; in `_best_split` the lines from
   `x = X[np.ix_(rows, features)]` to the `gain` expression, the equal-values mask, the two
   tie-rule lines and the midpoint guard; the `while` loop in `apply`.
3. Live check, the root of the Gini tree:

   ```bash
   uv run python -c "
   from common import Data
   from part1_decision_tree import DecisionTree
   d = Data()
   t = DecisionTree('gini').fit(d.X_train, d.y_train)
   print(t.nodes[0]); print(t.depth, t.n_leaves, len(t.nodes))
   "
   ```

   prints (the dict is one long line, wrapped here)

   ```
   {'feature': 21, 'threshold': 0.0936050469982061, 'left': 1, 'right': 342,
    'gain': 0.23370138041251196, 'value': 1, 'n': 5756,
    'weights': array([0.09346769, 0.48332175, ...]), 'depth': 0}
   17 211 421
   ```

   Read it out: feature 21 is MFCC 22 (0-based); the left subtree is nodes 1 to 341 (pre-order),
   so the right child is 342; value 1 is AdenomeraHylaedactylus (`t.classes_[1]`); with weights
   1/N, `weights` are the class shares and sum to 1; 421 = 2 · 211 − 1 nodes, as in any tree
   where every split has two children.
4. A second check: `DecisionTree('entropy', max_depth=1)` gives 2 leaves and 61.78% training
   accuracy; or work check (b) on paper with the table above.
5. Optional: `uv run pytest tests/test_part1.py`, 198 tests in about a second. They include a
   slow brute-force reference (a loop over every feature and midpoint) that the vectorised
   search must match, root and whole tree, on 40 random problems with both criteria.

## Likely viva questions

- *Why sort once per node instead of looping over thresholds?* Sorting the node's rows by a
  feature makes every threshold a prefix of the sorted order, so one cumulative sum gives the
  class weights of every left child: O(n log n) per feature instead of O(n²) for recounting at
  each threshold. Sorting once per tree and partitioning the sorted lists at each split would
  drop the log factor too, at the cost of more bookkeeping; the handout asks for one sort per
  node, and a tree takes about 0.45 s.
- *Why midpoints, and only between consecutive distinct values?* The handout fixes it. Every t
  in [a, b) gives the same training partition, so t matters only for new values between a and
  b, and the midpoint is the farthest from both. Measured: t = a gives 94.65 / 94.72 test, t
  just below b 94.65 / 95.07, the midpoint 94.93 / 95.41. No t separates equal values, and a cut
  between non-consecutive values equals one of the consecutive cuts.
- *Can an impure node have zero gain?* Yes, when every split leaves each child with the parent's
  class mix, as in XOR. The node becomes a leaf, although two splits together would be perfect:
  greedy growth looks one split ahead. (Finding the smallest consistent tree is NP-hard; greedy
  growth is the standard heuristic.)
- *Why do Gini and entropy choose different splits?* They score the same candidates with
  different concave functions. −p log2 p rises infinitely steeply from p = 0, while p(1 − p)
  rises with slope 1, so entropy gives more credit for removing small minorities. Here they
  differ already at the root (MFCC 22 against MFCC 13). Their gains are on different scales (at
  most 0.9 against 3.32 bits for 10 classes) and are never compared.
- *Why is the Gini tree deeper, 17 against 11?* Depth is set by one longest path. In the Gini
  tree a node of 427 rows at depth 5, 403 of them AdenomeraAndre, sheds its 24 other rows a few
  at a time (420, 412, 405, 402, 401, 400, 397, 393, 388, 381 rows) until the last one is
  isolated at depth 17. The leaf counts are closer: 211 against 181.
- *Why 100% training accuracy?* Without `max_depth` a node stops only when it is pure or no
  split has positive gain. All 7195 rows of this data are distinct and no impure node ran out of
  positive gain, so all 211 / 181 leaves are pure. It is not guaranteed: XOR is impure with no
  positive gain, and identical rows with different labels can never be separated.
- *Duplicate rows, or identical rows with different labels?* An exact duplicate acts like one row
  with twice the weight. Identical feature vectors with different labels end in the same leaf
  (every cut between them is masked), which predicts their weighted majority, as in check (c);
  training accuracy then stays below 100%. This dataset has no repeated feature rows.
- *Why a tolerance for ties, and why 1e-12?* Mathematically equal gains can differ by about
  1e-16 in floats, so exact comparison would let rounding, not the handout's rule, pick the
  split, and rescaling the weights would change the tree. In the unconstrained trees the noise
  is at most 4.4e-16 and real gaps are at least 9.8e-8; 1e-12 sits between.
- *How do sample weights enter?* Only through the one-hot weight matrix: class weight sums
  replace counts in p_i, in W_j/W and in the leaf majority. The purity test,
  `min_samples_split` and the thresholds ignore them. Scaling all weights changes nothing; 1/N
  gives the unweighted tree.
- *What does `min_samples_split=2` do here?* Nothing: a 1-row node is pure anyway. It counts
  rows, not weight; with 25 (as in the test) it would stop small nodes early.
- *What does `max_depth` count?* Edges from the root: the root has depth 0, `max_depth=1` is a
  stump with 2 leaves, AdaBoost's `max_depth=3` allows at most 8 leaves, `None` means no limit.
- *How does the forest's feature subsetting fit in?* `max_features` features are drawn afresh,
  without replacement, in every node that searches for a split, then sorted so the tie rule
  keeps its meaning; no positive gain among them means a leaf. Nodes that stop early draw
  nothing.
- *What if a sample weight is 0?* `share` gives a weightless side zero probabilities, so it adds
  nothing to the gain and such a cut is never chosen (its gain is about 0). Without it, 0/0 would
  make one gain nan, `gain.max()` nan, and the node a leaf. The same thing happens without
  zero weights in AdaBoost after very many rounds (from about round 85 at depth 3), when some
  rows weigh less than rounding error. `test_rows_without_weight_give_no_nan` checks it.
- *Time and memory?* Per node O(m n log n + n m C); per tree O(D N m (log N + C)); arrays of
  about 10 MB at the root; about 0.45 s per unconstrained tree; predict in D vectorised steps.
- *How does this differ from sklearn's `DecisionTreeClassifier`?* Same impurities and midpoint
  thresholds. sklearn visits the features in a random order at every node, even with all
  features, and keeps the first best split it meets, so ties go to a random feature and the
  tree depends on `random_state`; ours applies the handout's fixed rule. A cross-check run
  outside the repo (sklearn is prohibited, so it is in neither the code nor the zip) found the
  same root splits, and test accuracy 94.16-94.93 (Gini) and 95.00-95.62 (entropy) over
  random_state 0-9; ours, 94.93 / 95.41, is inside both ranges.
- *Why is the derived family accuracy higher?* A species error inside the right family is still
  a family hit: 19 of Gini's 73 test errors and 20 of entropy's 66 are, mostly among the Hylidae.
- *If the full tree overfits, why does test accuracy not drop?* Overfitting shows as the gap,
  5.07 / 4.59 points at full depth, not as a fall: here the deepest levels add almost nothing on
  the test set (Gini 94.86 at depth 12, 94.93 unconstrained). The leakage caveat likely helps:
  syllables of the same recording sit on both sides of the split.
- *Why does a stump reach only about 61%?* It has 2 leaves, so it predicts at most 2 of the 10
  species; even the best stump could not beat the two largest classes, 63.92% of the training
  rows. It gets 60.98 / 61.78.

## Likely code changes

Keep every new parameter's default equal to today's behaviour, so Parts 2 to 4, which import
`DecisionTree`, print the same. Check with `uv run pytest tests/test_part1.py` and the script;
`results/` changes only through `uv run python -m tools.run_all`, the PDF through
`tools.build_pdf`.

- **Misclassification error as a third criterion.** In `impurity`, before the Gini branch:

  ```python
  if criterion == "error":  # misclassification error
      return 1 - p.max(axis=-1)
  ```

  and in `__init__`, `if criterion not in (*CRITERIA, "error"):`. Do not add it to `CRITERIA` in
  `common.py`: every part loops over that tuple. Result: train 98.91, test 93.88, 165 leaves,
  depth 14, with 7 leaves left impure. Error is concave but not strictly, so a split can improve
  purity at zero gain: parent (70, 30) into (40, 0) and (30, 30) has error gain 0 (the code
  computes −5.6e-17), so the node stays a leaf, while its Gini gain is 0.12 and its information
  gain 0.2813.
- **`min_samples_leaf`.** Add `min_samples_leaf=1` to `__init__` (and store it), then in
  `_best_split`, after the equal-values mask:

  ```python
  n_left = np.arange(1, len(rows))  # rows left of each cut
  small = (n_left < self.min_samples_leaf) | (len(rows) - n_left < self.min_samples_leaf)
  gain[small] = -np.inf  # a 1-D mask over the cuts: blocks them for every feature
  ```

  1 gives today's tree; 5 gives train 97.17 / 97.78, test 94.37 / 94.65, 122 / 118 leaves. It
  counts rows, not weight.
- **A gain threshold.** Add `min_gain=0.0` to `__init__` (and store it), and replace
  `if not best > TOL:` with `if not best > max(TOL, self.min_gain):`. Our gain is relative to
  the node, so a 2-row node with two classes still gains 0.5 (Gini); sklearn's
  `min_impurity_decrease` first multiplies the gain by the node's share of the total weight
  (`best * W / total`, with `total = w.sum()` stored in `fit`). At 0.01: train 98.80 / 99.97,
  test 94.09 / 95.48, 156 / 178 leaves.
- **`max_leaf_nodes`.** Needs best-first growth: keep the candidate splits in a heap, keyed by the
  impurity they remove (gain × node weight), and always expand the best. Add
  `max_leaf_nodes=None` to `__init__`, `import heapq` at the top, and replace `_grow` with:

  ```python
  def _grow(self, X, codes, onehot, rows, depth):
      heap, leaves = [], 1

      def add(rows, depth):  # make a node; queue its best split, if it may have one
          weights = onehot[rows].sum(axis=0)
          node = {"feature": -1, "threshold": 0.0, "left": -1, "right": -1, "gain": 0.0}
          node.update(value=int(weights.argmax()), n=len(rows), weights=weights, depth=depth)
          index = len(self.nodes)
          self.nodes.append(node)
          stop = (
              (codes[rows] == codes[rows[0]]).all()
              or len(rows) < self.min_samples_split
              or (self.max_depth is not None and depth >= self.max_depth)
          )
          split = None if stop else self._best_split(X, onehot, rows, weights)
          if split is not None:  # most impurity removed first; ties to the older node
              heapq.heappush(heap, (-split[2] * weights.sum(), index, split))
          return index

      add(rows, depth)
      while heap and (self.max_leaf_nodes is None or leaves < self.max_leaf_nodes):
          _, index, (f, t, g, left, right) = heapq.heappop(heap)
          node = self.nodes[index]
          node.update(feature=f, threshold=t, gain=g)
          node["left"] = add(left, node["depth"] + 1)
          node["right"] = add(right, node["depth"] + 1)
          leaves += 1
      return 0
  ```

  Unlimited, it grows the same tree (same predictions; the nodes are numbered differently, which
  `apply` does not care about). 50 leaves: test 93.33 / 92.98; 100 leaves: 94.02 / 95.00. With
  `max_features`, nodes now draw their features in best-first order instead of pre-order, so a
  forest built on it would change even without a limit.
- **`predict_proba`.** In `fit`, add `"weights"` to the keys copied into flat arrays, then:

  ```python
  def predict_proba(self, X):
      w = self.weights[self.apply(X)]  # class weights of each row's leaf
      return w / w.sum(axis=1, keepdims=True)  # columns in the order of classes_
  ```

  Its argmax equals `predict` (both take the first maximum). For the unconstrained trees every
  leaf is pure, so every probability is 0 or 1; it says more with `max_depth`.
- **Feature importances** (mean decrease in impurity). Set `self.n_features_ = X.shape[1]` in
  `fit`, then:

  ```python
  def feature_importances(self):
      imp = np.zeros(self.n_features_)
      total = self.nodes[0]["weights"].sum()
      for node in self.nodes:
          if node["feature"] >= 0:  # weight share of the node times its gain
              imp[node["feature"]] += node["weights"].sum() / total * node["gain"]
      return imp / imp.sum()
  ```

  Before normalising, the sum is the root's impurity minus the leaves' weighted impurity, i.e.
  the root's impurity here (0.7194 and 2.4296 bits): a quick check. Gini: MFCC 22 0.343, 13
  0.106, 12 0.099, 17 0.071, 14 0.054; MFCC 16 is never used. Entropy: MFCC 13 0.342, 12 0.128,
  22 0.102, 10 0.079, 2 0.063. Counting splits instead would rank MFCC 2, 1 and 3 first (46, 29
  and 21 splits in the Gini tree), mostly in small, deep nodes.
- **Printing the tree.**

  ```python
  def show(self, max_depth=3, i=0):
      node = self.nodes[i]
      pad = "  " * node["depth"]
      if node["feature"] < 0 or node["depth"] >= max_depth:
          print(f"{pad}-> {self.classes_[node['value']]}  (n={node['n']})")
          return
      print(
          f"{pad}MFCC {node['feature'] + 1} <= {node['threshold']:.6f}"
          f"  (n={node['n']}, gain {node['gain']:.4f})"
      )
      self.show(max_depth, node["left"])
      self.show(max_depth, node["right"])
  ```

  `DecisionTree("gini").fit(d.X_train, d.y_train).show(2)` prints

  ```
  MFCC 22 <= 0.093605  (n=5756, gain 0.2337)
    MFCC 13 <= 0.014246  (n=2817, gain 0.1238)
      -> AdenomeraAndre  (n=1467)
      -> HypsiboasCordobae  (n=1350)
    MFCC 17 <= 0.050693  (n=2939, gain 0.0609)
      -> HypsiboasCordobae  (n=186)
      -> AdenomeraHylaedactylus  (n=2753)
  ```

  Under each test the x ≤ t branch comes first; `->` is a leaf, or the node's majority where
  printing stops.
- **Exact tie-breaking, no tolerance.** In `_best_split`, `near = (gain == best).T` instead of
  `(gain >= best - TOL).T`. Then: Gini test 94.86 instead of 94.93 and entropy 95.34 instead of
  95.41, one test row each; the split changes at 6 nodes of the Gini tree and 1 of the entropy
  tree, on gain differences of 1.4e-17 to 2.2e-16; with all weights 1 instead of 1/N the trees
  change again (Gini test 95.00); and 16 of the 198 tests fail: 11 whole-tree brute-force
  comparisons (the reference keeps the 1e-12 rule) and 5 of the 10 weight-scale tests. Dropping
  TOL from the positivity check as well (`best > 0`) gives the same trees here.
