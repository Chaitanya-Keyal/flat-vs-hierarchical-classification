# Part 3: the random forest

## In short

A random forest is bagging plus feature subsetting: each of T = 50 unconstrained trees grows on
its own bootstrap sample, and every node that searches for a split draws m = ⌊√22⌋ = 4 of the 22
MFCCs afresh and searches only those. Bagged trees all split near the root on the same strong
features (MFCCs_22 is the root of 23 of the 25 bagged Gini trees), so they make the same mistakes
and the vote cancels fewer of them; the forest's 50 roots use 11 different features. In code it
is `Bagging` with another `new_tree`, and one RandomState(42) feeds both the bootstraps and the
node draws. It is the best flat model: test 97.78 / 97.64 (Gini / entropy) against
96.66 / 96.87 for bagging and 94.93 / 95.41 for one tree; family 98.40 / 98.19; train 100.00.
Its trees are weaker (mean test 92.94 / 93.07 against bagging's 93.67 / 94.31) but agree less
(0.8991 / 0.9026 against 0.9218 / 0.9320), and the vote gains more than the trees lose.

## The idea

Pairs a / b are Gini / entropy. Numbers marked *not part of the submission* come from throwaway
runs with other settings, seeds or extra code (seed 42 unless stated); the others are printed by
the scripts or read off the very models they fit (root features, leaf counts, vote margins).

### Bagging plus a feature draw at every split

- **Bagging** (Part 2): T trees, each fitted on N = 5756 rows drawn with replacement from the
  training set, combined by majority vote. Each tree is unconstrained: low bias, high variance.
  The vote keeps the low bias and averages part of the variance away.
- **Random forest** (handout Table 1: T = 50, bootstrap of size N, m = 4 drawn afresh at every
  split, unconstrained, majority vote): the same, plus one change inside the tree. Every node that
  searches for a split draws m of the d = 22 features without replacement and looks for the best
  split among those only. Its children draw again.
- **Why bagged trees are alike.** Two bootstraps share about 40% of the training rows (each holds
  1 − 1/e ≈ 63.2% of them, and 0.632² ≈ 0.40), and the greedy search takes the strongest feature
  whenever it is on offer; with all 22 on offer, it always is. So bagged trees copy the single
  tree's top. The Part 1 Gini tree splits on MFCCs_22, then MFCCs_13 and MFCCs_17; 23 of the 25
  bagged Gini trees also put MFCCs_22 at the root (the other two, MFCCs_13). All 25 entropy trees
  put MFCCs_13 at the root, as the Part 1 entropy tree does. The top splits decide how the rows
  are divided for everything below, so trees with the same top tend to make the same mistakes.
- **What the draw changes.** A given feature is on offer at a node with probability
  m/d = 4/22 ≈ 0.18, so the best feature is missing at about 82% of nodes, the root included. In
  the Gini forest, MFCCs_22 was on offer at 9 of the 50 roots and won all 9; 31 roots had neither
  MFCCs_22 nor MFCCs_13 on offer. The 50 roots use 11 different features (entropy: 11 too). Lower
  down every feature is still available, because each node draws again: each forest uses all 22.

### Averaging correlated trees

Take T trees whose outputs at a point x each have variance σ² and pairwise correlation ρ, over
the random draws (bootstrap, features). Their average has T variance terms and T(T − 1)
covariance terms of ρσ² each (ESL, ch. 15):

```
Var( (1/T) Σ_t f_t(x) ) = (1/T²) (T σ² + T(T − 1) ρσ²) = ρσ² + (1 − ρ) σ²/T
```

- σ²: how much one tree's output at x changes with its draws. Large for unconstrained trees.
- ρ: how much two trees' outputs move together. Shared rows and a shared top make it large in
  bagging.
- (1 − ρ)σ²/T: the part that averaging removes. It falls as 1/T and is small well before T = 50.
- ρσ²: the floor. No number of trees removes it; only a smaller ρ does, and that is what the
  feature draw is for. The price: each tree is a little worse, because the best feature is often
  not on offer (a little more bias). It pays as long as the drop in ρ outweighs that.
- The average has the bias of one tree: averaging removes variance, never bias. That is why the
  members are grown unconstrained (low bias).
- What it looks like here (test % of the first k trees of one fit, Gini; not part of the
  submission): bagging 93.61, 96.04, 96.32, 96.66 at T = 1, 5, 10, 25; the forest 92.77, 96.25,
  97.22, 97.92 at T = 1, 5, 10, 20, then flat (97.78 at 50, 97.71 at 200). The quick rise is the
  1/T term going; the plateau is the floor; the forest's plateau is higher because its ρ is lower.
- The formula is for averaging numbers (a regression output, a class probability). A vote over
  species is not literally an average, so the formula explains the mechanism rather than giving a
  number to check. The closest measurement (test set, not part of the submission): the
  correlation between two trees' 0/1 error indicators, averaged over pairs, is 0.469 / 0.497 for
  bagging and 0.374 / 0.381 for the forest. The handout's agreement is a simpler proxy for it.

### Why m = ⌊√d⌋

- m trades strength (accurate trees) against correlation (alike trees). m = d: every split sees
  every feature, which is bagging. m = 1: each split's feature is random and only its threshold is
  optimised: the most diverse and the weakest trees.
- ⌊√d⌋ is the standard default for classification (ESL ch. 15, sklearn); for regression it is
  ⌊d/3⌋. It is a convention, not a derivation. Breiman's 2001 paper tried 1 and ⌊log₂ d + 1⌋
  (5 here) and found the error not very sensitive to the choice.
- `math.isqrt(22)` = 4 is the exact integer floor of √22 ≈ 4.69, with no float rounding.
  ⌊log₂ 22⌋ = ⌊4.46⌋ = 4 as well (sklearn's `"log2"`), so that rule gives the identical forest.
- Sweep (T = 50, seed 42; not part of the submission):

| m | Test % | Agreement | Mean tree test % | Fit time, Gini |
|---|---|---|---|---|
| 1 | 97.36 / 96.94 | 0.8502 / 0.8559 | 89.73 / 90.06 | 2.0 s |
| 2 | 97.50 / 97.78 | 0.8839 / 0.8822 | 91.93 / 91.78 | 2.5 s |
| 3 | 97.57 / 97.71 | 0.8941 / 0.8945 | 92.58 / 92.65 | 3.5 s |
| 4 | 97.78 / 97.64 | 0.8991 / 0.9026 | 92.94 / 93.07 | 4.3 s |
| 5 | 97.78 / 97.78 | 0.9019 / 0.9046 | 93.09 / 93.21 | 6.8 s |
| 6 | 97.29 / 97.71 | 0.9025 / 0.9077 | 93.07 / 93.36 | 8.3 s |
| 8 | 97.36 / 97.85 | 0.9073 / 0.9126 | 93.38 / 93.65 | 10.0 s |
| 11 | 97.29 / 97.71 | 0.9122 / 0.9165 | 93.49 / 93.80 | 11.9 s |
| 16 | 97.22 / 97.08 | 0.9160 / 0.9234 | 93.61 / 94.11 | 18.3 s |
| 22 | 96.46 / 96.46 | 0.9231 / 0.9337 | 93.51 / 94.27 | 20.8 s |

- Agreement and mean tree accuracy rise with m: stronger but more alike trees. Test accuracy is
  flat, within about 0.5 points, from m = 2 to 11, and falls at 16 and 22; m = 22 is bagging with
  50 trees. Even m = 1 (97.36 / 96.94) does at least as well as bagging (96.66 / 96.87). Inside
  the flat region the differences are noise: the forest moves by up to 0.5 points between seeds
  (see Results).
- Smaller m gives bigger trees (Gini: 338 leaves on average at m = 1, 221 at m = 4, 163 at m = 22;
  worse splits purify less, so more are needed) but faster fits, since each node sorts only m
  columns.

### Per split, not per tree

- Drawing 4 features once per tree and growing the whole tree on them is Ho's random subspace
  method. The handout calls it "a different (and wrong) algorithm": it is not a random forest.
- Why it is worse at m = 4: every split of a tree must use the same 4 of the 22 MFCCs. If the
  strong ones are not among them, the tree is weak everywhere, and 4 MFCCs may not be enough to
  tell 10 species apart well (such trees need about 440 leaves, twice the forest's). They disagree
  a lot, but mostly because many are poor: high bias, and a vote cannot remove bias. Per split,
  every node draws again, so each tree can use every feature somewhere and stays strong; the
  randomness only changes which feature is used where, above all near the root, which is where
  bagged trees were alike.
- Checked (T = 50, m = 4; not part of the submission): per tree, the mean tree test accuracy is
  84.63 / 85.04 % (the worst tree 66.16 / 67.41), agreement 0.7743 / 0.7796, and the vote only
  95.62 / 95.69, below bagging. Low agreement is not enough; the trees must also be strong.
- With 11 features per tree (half, a common random-subspace choice) it reaches 97.71 / 97.92: the
  method is not useless, but 4 features per tree starve each tree.

## The code

### `RandomForest` is `Bagging` with another tree

`part3_random_forest.py`:

```python
class RandomForest(Bagging):
    def __init__(self, n_estimators, criterion, max_features, seed=42):
        super().__init__(n_estimators, criterion, seed)
        self.max_features = max_features

    def new_tree(self, rng):
        return DecisionTree(self.criterion, max_features=self.max_features, rng=rng)
```

- `Bagging.new_tree(rng)` returns `DecisionTree(self.criterion)`: all 22 features at every split.
  It takes `rng` only so that a subclass can use it. Apart from storing `max_features`, the forest
  overrides only this method; `fit`, `votes` and `predict` are Bagging's, so the whole difference
  between Part 2 and Part 3 is which tree gets built.
- `Bagging.fit` (`part2_bagging_boosting.py`):

  ```python
  rng = np.random.RandomState(self.seed)
  ...
  for _ in range(self.n_estimators):
      rows = rng.randint(0, N, size=N)  # N rows, drawn with replacement
      self.trees_.append(self.new_tree(rng).fit(X[rows], y[rows]))
  ```

  The RandomState is created inside `fit`, so refitting gives the same forest
  (`test_a_forest_refits_identically`). The same `rng` object goes into every tree, so the node
  draws continue the forest's single stream. The bootstrap indexes arrays; no loop over rows.
- `votes(X)`: a T × n array of class indices, one row per tree: each tree's species mapped into
  the forest's sorted `classes_` with `np.searchsorted` (fine even if a bootstrap missed a
  species).
- `predict(X)`: `tally(votes, 10)` counts the votes per row and species; `argmax` takes the first
  maximum, i.e. the smallest label, the handout's tie rule. Ties do happen: 3 test rows (Gini) and
  1 (entropy) have a tied top vote in the forest.
- `main`: `m = math.isqrt(X.shape[1])`, then `RandomForest(50, c, m)` for each criterion. It also
  refits the single tree and `Bagging(25, c)`, which builds its own fresh RandomState(42), so the
  comparison columns are exactly Part 1's and Part 2's numbers. `data.scores` gives train, test
  and family accuracy (each predicted species mapped to its family). The returned dict feeds
  `tools/run_all.py`; the PDF quotes the agreements to three decimals (0.899 / 0.903 against
  0.922 / 0.932).

### The draw in `DecisionTree._best_split`

`part1_decision_tree.py`; the handout asked for the `max_features` and `rng` arguments in Part 1:

```python
d = X.shape[1]
if self.max_features is None:
    features = np.arange(d)
else:  # a fresh random subset at every split
    features = np.sort(self.rng.choice(d, self.max_features, replace=False))
x = X[np.ix_(rows, features)]
```

- `rng.choice(22, 4, replace=False)`: 4 distinct feature indices, the handout's call. The search
  runs on the node's rows and those 4 columns only, and `int(features[f])` maps the winning column
  back to its feature index, so nodes store real indices and `predict` needs no change.
- Only nodes that search draw. `_grow` first returns if the node is pure, has fewer than
  `min_samples_split` rows or sits at `max_depth`; only then does it call `_best_split`. Here every
  searching node found a split, so the 50 Gini trees made one draw per internal node: 10 989.
- **Why sorted.** The tie rule (lower feature index, then lower threshold) is implemented as "the
  first near-best cut in column order": `near = (gain >= best - TOL).T` and `np.argmax(near)` scan
  feature by feature. `choice` returns the indices in random order; sorting makes column order
  equal index order. Ties are common deep in a tree: at a node with two rows of different species,
  every feature that separates them gives the same gain. Without the sort, three tests fail
  (checked): `test_drawing_all_features_equals_no_subsetting` and
  `test_features_are_drawn_at_every_searched_node` in `tests/test_part1.py`, and
  `test_a_forest_drawing_every_feature_starts_like_bagging` in `tests/test_part3.py`.
- **No positive gain: a leaf.** `_best_split` returns None when `not best > TOL` (TOL = 1e-12),
  and `_grow` leaves the node a leaf that predicts its majority. No second draw, as the handout
  says; sklearn would keep searching other features. Gini and entropy are strictly concave, so a
  split has zero gain only if both children have exactly the parent's class mix; in practice that
  takes drawn features that are constant on the node's rows. At m = 4 it never happened: every
  leaf of every tree in both forests is pure. At m = 1 (not part of the submission) it happens
  408 / 375 times, each time because the one drawn feature was constant on the node, mostly
  MFCCs_1 (194 distinct values in the training set).
- `DecisionTree("gini", max_features=2)` with no `rng` raises ValueError (tested).
- The tests swap the RNG for a stub. `FirstFeatures` (`tests/test_part1.py`) counts its calls and
  always offers features m − 1, ..., 0, reversed so that the sort matters. With m = 2 the tree must
  equal one grown on columns 0 and 1 alone, and the number of draws must equal the number of
  impure nodes (the ones that search). `test_drawing_all_features_equals_no_subsetting`: m = d with
  a real RandomState must give the full-search tree.
  `test_no_gain_among_the_drawn_features_makes_a_leaf`: offered only a constant feature, the node
  stays a leaf after exactly one draw.
- In `tests/test_part3.py`, `test_a_forest_replays_one_stream_of_bootstraps_and_draws` rebuilds
  every tree from one RandomState(42) in the order below and must match node for node, and
  `test_a_forest_draws_features_per_split_not_per_tree` checks that each tree of an m = 1 forest
  uses several features (a subset drawn once per tree would give it one).

### One random stream, in a fixed order

- The handout's rule: one fresh `np.random.RandomState(42)` per ensemble, used for all its
  randomness. The order of the calls: tree 1's bootstrap (`randint(0, N, size=N)`), then one
  `choice` per searching node in the order `_grow` visits them (depth first, left child first),
  then tree 2's bootstrap, and so on. README.txt states this order.
- Every draw depends on all the draws before it: tree t's bootstrap depends on how many nodes
  trees 1 to t − 1 searched. Anything that changes which nodes search or their order (a
  `max_depth`, drawing at pure nodes too, breadth-first growth, a RandomState per tree) changes
  every later tree, and the printed numbers move.
- `choice(22, m, replace=False)` returns the first m entries of `permutation(22)`, so it uses the
  stream even when m = 22. `RandomForest(50, c, 22)` is therefore bagging as an algorithm, and its
  first tree equals `Bagging`'s first tree
  (`test_a_forest_drawing_every_feature_starts_like_bagging`), but from the second tree on the
  bootstraps differ: test 96.46 / 96.46 against 96.59 / 97.08 for `Bagging(50, c)` (not part of
  the submission). That gap is pure chance, a useful sense of scale.
- The first k trees of a T-tree forest are exactly the k-tree forest, since nothing in tree k
  depends on later trees. So one fit gives accuracy against T: vote with `votes[:k]`.
- Part 3's bagging numbers equal Part 2's (96.66 / 96.87) because `Bagging(25, c)` starts its own
  RandomState(42) at `fit`.

### `agreement` and the mean tree accuracy

```python
def agreement(votes, n_classes):
    T, c = len(votes), tally(votes, n_classes)
    return float(np.mean((c * (c - 1)).sum(axis=1) / (T * (T - 1))))
```

- The handout's quantity: for each pair of trees, the fraction of test rows on which they
  predict the same species, averaged over the P = T(T − 1)/2 pairs.
- Derivation. Fix a test row on which c_k trees vote for species k (Σ_k c_k = T). Two trees agree
  on it exactly when they voted for the same species, so Σ_k c_k(c_k − 1)/2 pairs agree, and the
  fraction of all P pairs that agree on this row is a = Σ_k c_k(c_k − 1) / (T(T − 1)) (the halves
  cancel). The handout averages the indicator [s and t agree on row i] over the n rows, then over
  the pairs; the order of two finite sums does not matter:

  ```
  (1/P) Σ_pairs (1/n) Σ_rows [agree] = (1/n) Σ_rows (1/P) Σ_pairs [agree] = (1/n) Σ_rows a
  ```

  and the right-hand side is the code's `np.mean` over rows.
- Example: four trees vote A, A, A, B. c = (3, 1), so (3·2 + 1·0)/(4·3) = 0.5; indeed 3 of the 6
  pairs (the three A–A pairs) agree.
- One tally instead of a loop over the 1225 pairs of 50 trees.
  `test_agreement_matches_a_loop_over_pairs` checks it against that loop, and
  `test_identical_trees_agree_everywhere` that identical trees give 1.
- Mean tree accuracy: `np.mean([accuracy(data.y_test, model.classes_[v]) for v in votes])`. Each
  row v of `votes` is one tree's predictions as class indices, `classes_[v]` turns them back into
  species, and the mean over the T trees is the accuracy of a typical member: the strength.
- Scale. Two trees right on fractions p_s and p_t of the rows are both right, and so agree, on at
  least p_s + p_t − 1 of them. Averaged over pairs: agreement ≥ 2p − 1, with p the mean tree
  accuracy, i.e. 0.8735 for the bagged Gini trees and 0.8589 for the forest's. So agreement lives
  between about 0.86 and 1 here, and a drop of 0.02 to 0.03 is not small. Part of it is just the
  weaker trees (the floor drops too); the error correlation above separates the two effects.

## Results and what they mean

### Against the single tree and bagging

| Test % | Single tree (Part 1) | Bagging, 25 trees | Forest, 50 trees |
|---|---|---|---|
| Gini | 94.93 | 96.66 | 97.78 |
| Entropy | 95.41 | 96.87 | 97.64 |

- The forest is the best flat model: 1.12 / 0.77 points above bagging (16 / 11 more of the 1439
  test rows right) and 2.85 / 2.23 above the single tree (41 / 32 rows). Family accuracy is
  98.40 / 98.19 (bagging 97.50 / 97.50, single tree 96.25 / 96.80): of the 32 / 34 species
  errors, 9 / 8 still land in the right family. Train 100.00 / 100.00.
- Not because of 50 trees against 25 (not part of the submission): `Bagging(50, c)` gets
  96.59 / 97.08, and the forest's first 25 trees alone get 97.85 / 97.36.
- Not luck of the seed (not part of the submission): over seeds 0 to 4 our forest gets
  97.29-97.64 (Gini) and 97.29-97.78 (entropy), bagging 95.76-96.73 and 96.66-97.15. Every forest
  beats every bagging run. The forest's Gini/entropy gap (0.14 points) is far inside that spread:
  neither criterion wins.
- A cross-check run outside the repo (never committed or submitted): sklearn's
  `RandomForestClassifier` with 50 trees and `max_features=4`, over `random_state` 0 to 4, gets
  97.36-97.78 (Gini) and 97.43-97.92 (entropy). Ours is inside both ranges.

### Strength against correlation: the handout's extra

Not graded; from `results/part3.txt`:

| Test set | Pairwise agreement | Mean tree % | Vote % |
|---|---|---|---|
| Gini, bagging (25 trees) | 0.9218 | 93.67 | 96.66 |
| Gini, forest (50 trees) | 0.8991 | 92.94 | 97.78 |
| Entropy, bagging (25 trees) | 0.9320 | 94.31 | 96.87 |
| Entropy, forest (50 trees) | 0.9026 | 93.07 | 97.64 |

- The handout asks: is the forest's agreement lower? Yes, for both criteria. Each forest tree is
  weaker (about 0.7 / 1.2 points), because many of its splits cannot use the best feature, but the
  trees go wrong on different rows and the vote turns that into accuracy: the forest's vote is
  about 4.8 / 4.6 points above its average tree, bagging's only 3.0 / 2.6.
- This is Breiman's (2001) strength against correlation. He bounds the forest's error by
  ρ(1 − s²)/s², with ρ the mean correlation between the trees and s the strength (the expected
  margin by which the vote picks the right class). The feature draw costs a little s and buys a
  lower ρ.
- The forest's trees are bigger: 220.8 leaves on average against 165.3 for the bagged Gini trees
  (203.0 against 142.6 for entropy), at similar depth (14.6 against 14.4; 12.4 against 11.3).

### Training accuracy is 100%

- Each tree is unconstrained and the 5756 training rows are all distinct, so each tree fits its
  own bootstrap exactly (every leaf is pure; checked).
- A training row is in a given bootstrap with probability 1 − (1 − 1/N)^N ≈ 1 − 1/e = 0.632, so
  about 31.6 of the 50 trees saw it (between 19 and 44 here), and all of them vote its species.
  The other ~18.4 vote as on unseen rows and are right about 93% of the time (each tree's accuracy
  on the rows it did not see: 93.07 / 93.01 % on average). So the true species wins every vote:
  over all 5756 rows the smallest margin, its votes minus those of the strongest other species,
  is 4, for both criteria.
- Likely, not guaranteed. Bagging's 25 trees have less slack (each row is in 15.7 bootstraps on
  average, in as few as 8): two rows (Gini) and one (entropy) were in only 10 of them and lose 10
  votes to 14 or 15 for one wrong species, hence 99.97 / 99.98. In the m sweep, 6 of the 20
  forests lose one or two training rows, outvoted or tied and given to a smaller label (99.97 to
  99.98; not part of the submission).
- So 100% on the training set says nothing about overfitting; the test accuracy, or the
  out-of-bag accuracy (see the questions), does.

## Demo

1. Run `python part3_random_forest.py` from the folder with `Frogs_MFCCs.csv`
   (`uv run python part3_random_forest.py` in the repo). About 35 s. Most of it is the bagging
   refit, about 11 s per criterion; each forest takes about 5 s despite twice the trees, because
   its nodes sort 4 columns, not 22. The output equals `results/part3.txt` byte for byte.
2. Point at: line 3 (m = 4); the two matrix rows; the comparison table (single tree < bagging <
   forest, both criteria); the agreement table (lower agreement, weaker trees, better vote).
3. Code, five stops: `RandomForest.new_tree` (part3) → `Bagging.fit` (part2: one RandomState, the
   bootstrap, `new_tree(rng)`) → `_grow` (part1: the stopping checks come before the search, so
   stopped nodes draw nothing) → `_best_split` (part1: the draw and the sort) → `agreement`
   (part3).
4. Tests: `uv run pytest tests/test_part3.py tests/test_part1.py -q`.
5. The notebook's Part 3 section (`notebooks/demo.ipynb`) shows the rows and a plot of agreement
   and mean tree accuracy, bagging against forest. Its "Try it" cell fits m = 1, 4 and 22 with
   T = 25, not 50 (Gini), and prints m = 1: 97.08%, agreement 0.847, mean tree 89.55%; m = 4:
   97.85%, 0.899, 93.00% (the first 25 trees of the matrix forest); m = 22: 96.53%, 0.923,
   93.54%.

Live checks in `uv run python`, started in the repo folder (nothing is edited):

```python
import numpy as np
from common import Data, accuracy, tally
from part2_bagging_boosting import Bagging
from part3_random_forest import RandomForest, agreement

data = Data()
X, y, Xt, yt = data.X_train, data.y_train, data.X_test, data.y_test
for m in (1, 4, 22):  # m = 22 takes about 20 s
    model = RandomForest(50, "gini", m).fit(X, y)
    print(m, accuracy(yt, model.predict(Xt)), agreement(model.votes(Xt), 10))
```

Expect, rounded: m = 1: 0.9736, 0.8502; m = 4: 0.9778, 0.8991; m = 22: 0.9646, 0.9231. Agreement
rises with m as the trees become alike; m = 22 is bagging with 50 trees.

More, after `forest = RandomForest(50, "gini", 4).fit(X, y)`:

- Roots (feature j is MFCCs_{j+1}): `np.unique([t.nodes[0]["feature"] for t in forest.trees_])`
  lists 11 features. With `Bagging(25, "gini").fit(X, y).trees_` in place of `forest.trees_` and
  `return_counts=True`: 12 twice and 21 (MFCCs_22) 23 times.
- Accuracy against T from one fit: `v = forest.votes(Xt)`, then
  `[accuracy(yt, forest.classes_[tally(v[:k], 10).argmax(axis=1)]) for k in (1, 5, 10, 25, 50)]`
  gives about 0.9277, 0.9625, 0.9722, 0.9785, 0.9778.
- A refit is identical: `(forest.predict(Xt) == forest.fit(X, y).predict(Xt)).all()` is True.
- Another seed: `RandomForest(50, "gini", 4, seed=0)` gets about 0.9757.

## Likely viva questions

- *What is a random forest, in one line?* Bagging of unconstrained trees plus a fresh random
  subset of m = 4 candidate features at every split; majority vote over T = 50 trees.
- *Why do random feature subsets reduce correlation?* Bagged trees build the same top from the
  strongest features (MFCCs_22 is the root of 23 of the 25 bagged Gini trees), and the top decides
  everything below, so they err alike. With 4 of 22 on offer, the best feature is missing at about
  82% of nodes (31 of the forest's 50 roots had neither MFCCs_22 nor MFCCs_13): different tops,
  different mistakes.
- *Why does lower correlation help?* Var = ρσ² + (1 − ρ)σ²/T. More trees shrink only the second
  term; the floor ρσ² falls only with ρ.
- *What does m = d give?* Bagging, here with 50 trees: 96.46 / 96.46 test, agreement
  0.9231 / 0.9337, close to the bagging rows. Only its first tree equals `Bagging`'s first tree,
  because the draws still use the random stream.
- *What does m = 1 give?* A random feature at every split, with only the threshold chosen: the
  most diverse and weakest trees (agreement 0.8502 / 0.8559, mean tree 89.73 / 90.06 %), yet the
  vote still gets 97.36 / 96.94, no worse than bagging. It is also where the no-gain rule fires.
- *Why ⌊√d⌋ = 4?* A conventional compromise between strength and correlation. Our sweep is flat
  within about 0.5 points from m = 2 to 11, so the exact value matters little here.
- *Why per split, not per tree?* Per tree, each tree is stuck with 4 MFCCs for all its splits and
  is weak (84.63 / 85.04 % on average, the worst 66.16 / 67.41 %), so the vote gets only
  95.62 / 95.69. Per split, every tree can still use every feature and stays strong.
- *Why unconstrained trees; don't they overfit?* Each tree overfits, but differently, and the vote
  cancels that variance; it cannot cancel bias, so the members should have low bias, i.e. be deep.
  Depth-limited forests do worse (not part of the submission): 87.21 / 87.84 at depth 3,
  96.87 / 97.15 at depth 8, 97.50 / 97.85 at depth 12, against 97.78 / 97.64 unconstrained.
- *Can more trees hurt?* No: as T grows the vote converges (Breiman 2001), so the test error
  levels off rather than rising; more trees only cost time. Ours wobbles by a few rows after about
  20 trees (Gini: 97.92 at 20, 97.78 at 50, 97.64 at 100, 97.71 at 200).
- *Why is training accuracy 100% but bagging's 99.97 / 99.98?* About 63% of the trees saw each
  training row and vote for it; with 50 trees that majority is safe here (smallest margin 4
  votes). With 25 trees, a row seen by only 10 can be outvoted.
- *What is the OOB error?* Each tree misses about 1/e ≈ 36.8% of the rows. Predicting each
  training row with only the trees that did not see it gives a test-like estimate without a test
  set. Ours: 97.88 / 97.72, against test 97.78 / 97.64 (not part of the submission).
- *How would you get feature importances?* Mean decrease in impurity: each split's gain times its
  node's share of the training weight, summed per feature, normalised per tree, averaged. The
  forest (Gini) spreads it: MFCCs_13 0.111, MFCCs_15 0.100, MFCCs_22 0.098; bagging gives
  MFCCs_22 0.323 (not part of the submission). MDI is measured on training rows and favours
  features with many distinct values; permutation importance (shuffle one feature on held-out
  rows, measure the drop in accuracy) avoids both.
- *Why not exactly sklearn's numbers?* Different random streams (sklearn seeds every tree
  separately and draws features one at a time); when no drawn feature gives a valid split sklearn
  keeps searching more features, we make a leaf; sklearn breaks ties by its random feature order;
  it averages class probabilities instead of counting votes (the same with pure leaves). Exact
  equality is impossible; ours is inside its five-seed range.
- *Why does the order of the random draws matter?* There is one stream, so every draw depends on
  all before it: tree t's bootstrap depends on how many nodes the earlier trees searched. Drawing
  at pure nodes, growing breadth first, or a RandomState per tree would change every later tree.
  The fixed order (README.txt) makes the run reproducible and comparable.
- *What if none of the 4 drawn features gives a positive gain?* The node becomes a leaf; no
  redraw. It never happened in our forests (every leaf is pure); it needs the drawn features to be
  constant on the node.
- *Why sort the drawn features?* Ties on gain go to the lower feature index, and the code takes
  the first near-best column; sorting makes column order equal index order. Without it three
  tests fail.
- *Is the forest's edge over bagging real?* 16 / 11 test rows. It survives 50 bagged trees
  (96.59 / 97.08) and five other seeds (every forest beats every bagging run). The forest's
  Gini/entropy gap (0.14 points) is noise.
- *Why is the forest faster than bagging, with twice the trees?* Each node sorts and scans 4
  columns instead of 22: about 5 s per criterion against about 11 s, even though its trees are
  bigger.

## Likely code changes

Anything that changes what `part3_random_forest.py` prints breaks the zip check of
`tools/package.py`, which compares the output with `results/part3.txt`. To keep such a change,
rerun `uv run python -m tools.run_all part3` (and `uv run python -m tools.build_pdf` if a matrix
number moved).

- **Per-tree feature subsets**, to compare with per split. In `part3_random_forest.py` (import
  `N_FEATURES` from `common`): a stand-in for the RandomState that offers the same features at
  every node, the trick the `FirstFeatures` stub uses in the tests, and a forest that draws once
  per tree, right after the bootstrap:

  ```python
  class SameFeatures:
      """Offers one fixed feature subset at every node of a tree."""

      def __init__(self, features):
          self.features = features

      def choice(self, d, m, replace=False):
          return self.features


  class PerTreeForest(RandomForest):
      def new_tree(self, rng):
          features = rng.choice(N_FEATURES, self.max_features, replace=False)
          same = SameFeatures(features)
          return DecisionTree(self.criterion, max_features=self.max_features, rng=same)
  ```

  `_best_split` still sorts, nodes still store real feature indices and `predict` is unchanged.
  Expect 95.62 / 95.69 test at m = 4.
- **max_features as a sweep.** At the end of `main`, before the `return` (`test_acc` is defined
  by then):

  ```python
  print("\nmax_features sweep (Gini, not part of the matrix)")
  for mf in (1, 2, 4, 8, 11, 22):
      model = RandomForest(50, "gini", mf).fit(X, y)
      agree = agreement(model.votes(data.X_test), len(model.classes_))
      print(f"m = {mf:>2}: test {pct(test_acc(model))}, agreement {agree:.4f}")
  ```

  About a minute (m = 22 alone takes about 20 s); the values are the sweep table above.
- **OOB accuracy.** In `Bagging.fit` (`part2_bagging_boosting.py`), record the rows each tree did
  not see. It makes no extra random draws, so every printed number stays the same:

  ```python
  self.trees_, self.oob_ = [], []
  for _ in range(self.n_estimators):
      rows = rng.randint(0, N, size=N)  # N rows, drawn with replacement
      self.oob_.append(np.bincount(rows, minlength=N) == 0)  # rows this tree never saw
      self.trees_.append(self.new_tree(rng).fit(X[rows], y[rows]))
  ```

  and in `part3_random_forest.py`:

  ```python
  def oob_accuracy(model, X, y):
      """Each training row voted on only by the trees whose bootstrap missed it."""
      votes = np.where(model.oob_, model.votes(X), -1)  # -1 matches no class: no vote
      counts = tally(votes, len(model.classes_))
      has = counts.sum(axis=1) > 0  # rows with at least one out-of-bag tree (all, here)
      return accuracy(y[has], model.classes_[counts[has].argmax(axis=1)])
  ```

  `pct(oob_accuracy(forest[c], X, y))` gives 97.88 / 97.72 (bagging: 96.77 / 96.59).
- **Feature importances (MDI).** In `part3_random_forest.py` (import `N_FEATURES`):

  ```python
  def importances(model):
      """Mean decrease in impurity per feature, normalised per tree, averaged over trees."""
      imp = np.zeros((len(model.trees_), N_FEATURES))
      for t, tree in enumerate(model.trees_):
          for node in tree.nodes:
              if node["feature"] >= 0:  # a split: its gain times its share of the weight
                  imp[t, node["feature"]] += node["weights"].sum() * node["gain"]
      imp /= imp.sum(axis=1, keepdims=True)
      return imp.mean(axis=0)
  ```

  Each tree is fitted with weight 1/N per bootstrap row, so a node's `weights.sum()` is its share
  of the tree's rows, and `gain` is the impurity drop per unit of the node's weight; the product
  is the drop in the tree's total impurity. Feature j is `MFCCs_{j + 1}`. Print the top five
  with `imp = importances(forest[c])`, then
  `for j in np.argsort(-imp)[:5]: print(f"MFCCs_{j + 1}", f"{imp[j]:.3f}")`.
- **A different m rule.** Line 40, `m = math.isqrt(X.shape[1])`, and the message on line 41.
  log₂: `int(math.log2(X.shape[1]))` is 4 too, so nothing changes. d/3: `X.shape[1] // 3` = 7.
  ⌈√d⌉: `math.isqrt(X.shape[1] - 1) + 1` = 5. Any m from 2 to 11 lands within about 0.5 points
  (sweep table).
- **Limiting depth.** Give the forest a `max_depth` and pass it on:

  ```python
  def __init__(self, n_estimators, criterion, max_features, seed=42, max_depth=None):
      super().__init__(n_estimators, criterion, seed)
      self.max_features = max_features
      self.max_depth = max_depth


  def new_tree(self, rng):
      return DecisionTree(
          self.criterion, max_depth=self.max_depth, max_features=self.max_features, rng=rng
      )
  ```

  A depth limit stops nodes from searching, so they draw nothing and every later draw shifts.
  Even the first tree is not the unconstrained one cut at the limit: after the first node that
  stops there, its later nodes, above the limit too, get other draws (with max_depth=8 the first
  difference is a node at depth 6), and every later tree gets another bootstrap. Test % (T = 50,
  m = 4; not part of the submission): depth 3: 87.21 / 87.84; 5: 92.49 / 93.75; 8: 96.87 / 97.15;
  12: 97.50 / 97.85; unconstrained 97.78 / 97.64.
- **Accuracy against T**, without refitting, from the prefix property (after the forests are
  fitted in `main`):

  ```python
  for c in CRITERIA:
      votes = forest[c].votes(data.X_test)
      for k in (1, 5, 10, 25, 50):
          pred = forest[c].classes_[tally(votes[:k], len(forest[c].classes_)).argmax(axis=1)]
          print(c, k, pct(accuracy(data.y_test, pred)))
  ```
