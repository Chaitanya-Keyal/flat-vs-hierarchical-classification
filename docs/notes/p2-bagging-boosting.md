# Part 2: bagging and AdaBoost (SAMME)

## In short

Bagging fits 25 unconstrained trees, each on its own bootstrap sample (N rows drawn with
replacement, about 63% of them distinct), and takes a majority vote. Averaging deep, different
trees cuts variance: test accuracy rises from 94.93 / 95.41 (single tree, Gini / entropy) to
96.66 / 96.87. AdaBoost (SAMME) fits 50 depth-3 trees in sequence, each on weights that stress the
rows the earlier trees got wrong, and lets each tree vote with its say
α = ln((1 − ε)/ε) + ln(K − 1). The ln 9 keeps α positive for any weighted error ε below 0.9, the
error of guessing among 10 classes. Boosting cuts bias: one depth-3 tree gets 79.22 / 84.50, the
50-tree vote 95.34 / 94.16, +0.41 / −1.25 points against the single full tree. All 50 rounds are
kept; ε rises from 0.21 / 0.16 to 0.60 / 0.49 as the weight piles onto the hard rows. Stumps name
only two species per round and reach 83.32 / 84.16.

Pairs a / b are Gini / entropy throughout. `part2_bagging_boosting.py` prints the matrix rows,
the single tree's test accuracy, ε first / last / max and the stump results (all kept in
`results/part2.txt`); Parts 1 and 3 print the single-tree, depth-sweep and tree-agreement
numbers. Every other number here comes from throwaway runs of the same classes and is not part of
the submission (some are marked as a reminder).

## Bagging (2a)

### Bootstrap samples

- `rows = rng.randint(0, N, size=N)`: N row indices drawn uniformly with replacement, the
  handout's own call. `X[rows], y[rows]` copies the rows by fancy indexing, with no Python loop.
- One draw misses a given row with probability 1 − 1/N, so all N draws miss it with probability
  (1 − 1/N)^N ≈ e⁻¹ = 0.368. A bootstrap therefore holds 1 − (1 − 1/N)^N = 63.2% of the distinct
  rows (N = 5756); the rest of its N slots are repeats. Our 25 bootstraps hold 61.7% to 64.0%
  (mean 63.0%).
- Each training row is in 15.7 of the 25 bootstraps on average, and every row is in at least one.
- A row drawn k times counts like one row of weight k, so a bootstrap tree is the tree on the
  rows drawn at least once, weighted by their counts (checked: identical). A weight of 0 is not
  the same as leaving the row out: such rows still sit between the drawn values and move the
  thresholds. On the full data with count weights, the first Gini bootstrap tree gets 93.12% test
  against 93.61%. The rows a tree never saw are its out-of-bag (OOB) rows: a free validation set
  for that tree.

### Why the vote helps

- A full tree has low bias and high variance: 100% train, 94.93 / 95.41 test (Part 1). Grown on
  another sample, its deep splits, and so its mistakes, would be different.
- T predictors with variance σ² each and pairwise correlation ρ average to variance
  ρσ² + (1 − ρ)σ²/T. More trees remove the second term; the first stays. A vote is not literally
  an average, but it behaves the same way (the details are in `p3-random-forest.md`).
- Each bagged tree alone is worse than the Part 1 tree (mean test 93.67 / 94.31: it sees only 63%
  of the distinct rows), yet the vote of 25 beats it: 96.66 / 96.87. Their mistakes differ, so
  they partly cancel.
- The trees are still strongly correlated: two bagged trees agree on 92.2% / 93.2% of the test
  rows, on average over all pairs (Part 3). That caps the gain; Part 3's random forest lowers ρ
  by sampling features.
- Bagging does nothing for bias: where every tree is wrong the same way, so is the vote. Hence
  deep, low-bias trees.
- More trees do not overfit; the accuracy levels off. Test with 1, 5, 10, 25, 50 trees: 93.61,
  96.04, 96.32, 96.66, 96.59 (Gini); 94.58, 96.18, 96.66, 96.87, 97.08 (entropy) (*not part of
  the submission*, except at 25).

### The vote and its ties

- `votes(X)` is a (T, n) array of class indices; `tally` turns it into (n, K) vote counts and
  `argmax(axis=1)` takes the first maximum. `classes_` comes from `np.unique`, so it is sorted and
  a tie goes to the smallest label, as the handout says.
- Ties happen even with an odd T, because ten classes allow 12-12-1. 5 / 4 test rows are tied;
  the true species is among the tied ones each time, and the smallest-label rule gets 2 / 1 of
  them right (the largest label would get 3 / 3). The rule is the handout's, not ours to tune.

### One RandomState(42) per model, made in fit

- `fit` starts with `rng = np.random.RandomState(self.seed)`. So:
  - Gini and entropy bagging draw the same 25 bootstraps (a bagged tree draws nothing itself),
    which makes the criterion comparison paired: same samples, only the criterion differs;
  - a refit gives the same model (`test_bagging_refits_identically`). Made in `__init__`, a
    second `fit` would carry on along the stream and draw new bootstraps;
  - Part 3 fits `Bagging(25, c)` again and prints the same 96.66 / 96.87;
  - `test_bagging_trees_are_fitted_on_the_handout_bootstraps` replays the stream and rebuilds
    every tree.
- Another seed moves the result by a few tenths: seed 0 gives 96.73 / 96.94, seed 1 gives
  96.53 / 96.94 (*not part of the submission*).
- `new_tree(rng)` ignores `rng` here. It is the hook that Part 3's `RandomForest` overrides to
  pass `max_features` and `rng` to the tree. The forest's trees then draw from the same stream
  between the bootstraps, so only its first bootstrap equals bagging's.

### Why train accuracy is 99.97 / 99.98, not 100

- Each bagged tree fits its own bootstrap perfectly (checked), so the trees that saw a row all
  vote for its species. Only the trees that did not see it can be wrong.
- A row is lost when it sits in few bootstraps and the trees that missed it agree on one wrong
  species. That happens to 2 / 1 of the 5756 training rows (5754/5756 = 99.97%,
  5755/5756 = 99.98%):
  - training row 5500, OsteocephalusOophagus (both criteria): in 10 of the 25 bootstraps; 10
    votes for it, 15 for HypsiboasCinerascens;
  - training row 4000, HylaMinuta (Gini only): in 10 bootstraps; 10 votes for it, 14 for
    AdenomeraAndre, 1 for Ameeregatrivittata.
- With 50 trees both reach 100.00 (*not part of the submission*).

## AdaBoost (2b): SAMME

### Why the binary update does not work for K = 10

- Binary AdaBoost codes the labels as ±1 and updates w ← w·exp(−α·y·h(x)), where y·h(x) is +1 for
  a hit and −1 for a miss. Ten species have no ±1 coding.
- Its α = ½ ln((1 − ε)/ε) turns negative once ε > ½, so it stops at ε ≥ ½. With ten classes a
  useful tree often has ε > ½: our Gini depth-3 trees do in 19 of the 50 rounds, first in round 6
  (entropy: 5 rounds, first in round 8). The binary rule would have stopped boosting there.
- SAMME (Zhu, Zou, Rosset and Hastie, *Multi-class AdaBoost*, 2009) is the K-class version: the
  update uses the indicator 1[h(x) ≠ y], and α gets an extra ln(K − 1). Both come from fitting a
  K-class exponential loss one tree at a time (forward stagewise), as AdaBoost does for two
  classes.

### The algorithm (handout page 3), with its code

1. wᵢ = 1/N: `w = np.full(N, 1 / N)`.
2. For t = 1, ..., 50, fit a `max_depth=3` tree with sample weights w:
   `tree.fit(X, y, sample_weight=w)`.
3. ε = Σᵢ wᵢ·1[h(xᵢ) ≠ yᵢ], the weight of the misclassified rows; Σ w = 1, so it is a weighted
   error rate. Clip it to [10⁻¹⁰, 1 − 10⁻¹⁰]. If ε ≥ 1 − 1/K = 0.9, discard the tree and stop.
4. α = ln((1 − ε)/ε) + ln(K − 1), the tree's amount of say.
5. wᵢ ← wᵢ·exp(α·1[h(xᵢ) ≠ yᵢ]), then divide every wᵢ by Σ w.
6. H(x) = argmaxₖ Σₜ αₜ·1[hₜ(x) = k].

### Why 1 − 1/K is "random guessing"

- Picking one of K labels uniformly at random is right with probability 1/K whatever the
  weights, so its expected weighted error is 1 − 1/K: 0.9 for K = 10, ½ for K = 2.
- α > 0 ⇔ (K − 1)(1 − ε) > ε ⇔ ε < 1 − 1/K. The rule stops exactly when a tree would get no
  positive say.
- Our trees can never trip it. A leaf predicts its weighted majority, which holds at least 1/K of
  the leaf's weight, so ε ≤ 1 − 1/K, with equality only if every leaf has exactly equal class
  weights. All four runs keep 50 of 50 rounds. The rule does fire on symmetric toy data (XOR, in
  the tests).

### α, the clip and ε = 0

- Small ε, large α. The first trees get α = 3.49 / 3.89 (ε = 0.2149 / 0.1560), the last
  1.77 / 2.25, the smallest 1.66 / 2.02.
- ln(K − 1) = ln 9 = 2.197 is added to every α. It keeps α positive for ε between ½ and 0.9.
  Without it a tree with ε = 0.6 would get α < 0: it would vote against its own predictions, and
  the update would shrink the weights of its mistakes.
- A perfect tree has ε = 0, and (1 − ε)/ε divides by zero. `eps` is a Python float, so `fit`
  would raise ZeroDivisionError; with a numpy float, α = inf, and inf · 0 = nan in
  `alpha * miss` would turn every weight into nan. Clipped, ε = 10⁻¹⁰ gives α = 25.22: huge but
  finite.
- Then no row is missed, the weights do not change, and the next tree is the same tree. With
  `max_depth=None` that is exactly what happens: ε = 10⁻¹⁰ in every round, 50 copies of the
  Part 1 tree, test 94.93 (*not part of the submission*). Boosting needs a weak learner.
- The upper clip keeps the log finite at ε = 1; the stopping rule acts long before that.

### The update

- Misses are multiplied by e^α, hits by e⁰ = 1, and then everything is divided by the total.
  Classic AdaBoost's "hits × e^(−α'), misses × e^(α')", with its own α' = α/2 (see K = 2 below),
  differs only by a common factor, which the normalisation removes.
- On the new weights, the tree just added has weighted error exactly 1 − 1/K. Its misses weigh
  ε·e^α = ε·(1 − ε)(K − 1)/ε = (1 − ε)(K − 1) and its hits 1 − ε, so the misses' share is
  (K − 1)/K. (Checked: 0.9 to within 4 × 10⁻¹⁶ after every update of all four runs.) The next
  tree sees weights on which the last tree is no better than guessing, so it has to find
  something new.

### The weighted vote

- Each tree adds its α to the score of the species it predicts; the largest score wins; a tie
  goes to the smallest label. In the code: `tally(votes, K, np.array(self.alphas_))`, then
  `argmax`.
- After two rounds the vote is still the first tree (79.22 / 84.50): α₁ > α₂, so the second tree
  alone cannot outvote the first. Later trees overrule an early one only together.

### K = 2: classic AdaBoost with α doubled

- Classic AdaBoost, labels and predictions in {−1, +1}: α' = ½ ln((1 − ε)/ε),
  w ← w·exp(−α'·y·h(x)), H(x) = sign Σ α'·h(x), stop at ε ≥ ½.
- SAMME with K = 2: ln(K − 1) = ln 1 = 0, so α = ln((1 − ε)/ε) = 2α'. The stopping rule
  1 − 1/K = ½ is the same.
- Weights: classic multiplies hits by e^(−α') and misses by e^(α'). Multiplying every weight by
  e^(α') changes nothing after normalisation, and gives hits × 1, misses × e^(2α') = e^α: SAMME's
  update. Same weights, so the same next tree and the same ε, round after round.
- Vote: with s₊ = Σ α·1[h = +1] and s₋ = Σ α·1[h = −1],
  s₊ − s₋ = Σ α·h(x) = 2 Σ α'·h(x). SAMME picks +1 exactly when classic's sum is positive: the
  same classifier, since doubling every α does not change a sign. A zero sum goes to the smaller
  label in both (the test's classic version uses `score > 0`).
- `test_samme_with_two_classes_is_classic_adaboost` runs both with 20 stumps and compares their
  predictions on 500 points.

### How the tree honours sample_weight

- `DecisionTree.fit` builds `onehot`, an n × K array holding wᵢ in the column of row i's class.
  A node's class weights are the column sums over its rows; the impurity uses
  pₖ = class weight / node weight; the gain weights each child by W_child / W; a leaf predicts
  the class with the largest weight. Class counts never appear.
- Only `min_samples_split` (a row count), the purity test (labels) and the candidate thresholds
  (feature values) ignore the weights.
- Scaling every weight by the same factor gives the same tree, and w = 1/N gives the unweighted
  tree (both checked). So round 1's ε is the plain training error of one depth-3 tree:
  0.2149 = 1 − 0.7851, and Part 1 prints 78.51% train at depth 3
  (`test_samme_first_error_is_the_plain_error_rate`).
- Part 1's sanity check (c): three identical rows, labels 0 0 1, weights 0.1 0.1 0.8, give one
  leaf predicting 1.

### When no tree is kept

- If the first tree already has ε ≥ 0.9, `trees_` stays empty. `predict` reshapes the empty vote
  list to (0, n) (that is what the `.reshape` is for), `tally` returns all-zero scores and
  `argmax` picks index 0: the smallest label.
- `test_samme_stops_when_the_first_learner_is_no_better_than_chance` builds this with XOR: no
  split has positive gain, so the stump is one leaf with weights ½ / ½, ε = ½ = 1 − 1/K, nothing
  is kept and every prediction is 0.

## The code

### `Bagging`

- `__init__(n_estimators=25, criterion="gini", seed=42)` only stores the settings.
- `new_tree(rng)` returns `DecisionTree(self.criterion)`: `max_depth=None`,
  `min_samples_split=2`, all 22 features at every split. `rng` is unused here (the Part 3 hook).
- `fit(X, y)`: a fresh `RandomState(seed)`; `classes_ = np.unique(y)`, sorted, which fixes the
  class indices and the tie order; then 25 times: draw `rows`, fit a tree on `X[rows], y[rows]`,
  keep it. It returns `self`, so `Bagging(25, c).fit(X, y)` is the fitted model.
- `votes(X)`: one row per tree, one column per sample, holding class indices.
  `np.searchsorted(self.classes_, tree.predict(X))` turns each label into its index in the sorted
  `classes_`. A tree's own `classes_` holds only the species in its bootstrap; going through the
  ensemble's `classes_` keeps the indices right even if a bootstrap missed a species (it never
  does here: the rarest species has 54 training rows). Part 3 reuses `votes`.
- `predict(X)`: `tally(votes, K)` gives the (n, K) counts, `argmax(axis=1)` the winning index,
  `classes_[...]` the label.

### `AdaBoostSAMME`

- `__init__(n_estimators=50, criterion="gini", max_depth=3)` only stores the settings.
- `fit(X, y)`, line by line:
  - `w = np.full(N, 1 / N)` and three empty lists, `trees_`, `alphas_`, `errors_`.
  - `tree = DecisionTree(self.criterion, max_depth=self.max_depth)`: a fresh tree every round.
  - `miss = tree.fit(X, y, sample_weight=w).predict(X) != y`: a boolean mask of the training
    rows this tree gets wrong. Every tree sees all N rows; only the weights change.
  - `eps = float(np.clip(w[miss].sum(), 1e-10, 1 - 1e-10))`: the clipped weighted error, as a
    plain float (it ends up in `results.json`).
  - `if eps >= 1 - 1 / K: break` comes before the appends, so the failing tree is discarded.
  - `alpha = float(np.log((1 - eps) / eps) + np.log(K - 1))`.
  - `w = w * np.exp(alpha * miss)`: True counts as 1 and False as 0, so misses × e^α, hits × 1.
    Then `w /= w.sum()`.
  - Only then are the tree, α and ε appended. `errors_` is kept for the printout.
- `predict(X)`: each tree's predictions as class indices, a (T, n) int array (`dtype=int` and the
  `reshape` matter only when no tree was kept); `tally` with `np.array(self.alphas_)` as weights
  gives the (n, K) scores Σₜ αₜ·1[hₜ(x) = k]; `argmax`; the label.

### `common.tally(votes, n_classes, weights=None)`

- `hits = votes[:, :, None] == np.arange(n_classes)`: a (T, n, K) boolean array;
  `hits[t, i, k]` says whether model t votes k on sample i.
- Without weights, `hits.sum(axis=0)`: the (n, K) vote counts. With weights,
  `np.tensordot(weights, hits, axes=1)` = Σₜ weights[t]·hits[t]: the (n, K) scores.
- The caller's `argmax(axis=1)` takes the first maximum, so the smallest class index, so the
  smallest label. Bagging, AdaBoost and the random forest all vote through it.
- Size: T·n·K booleans, 50 × 5756 × 10 ≈ 2.9 MB.

### `main()`

- First `binary_check`, the handout's "Check: for K = 2 this is the usual AdaBoost with α
  doubled". On the training rows of the two most common species (AdenomeraHylaedactylus and
  HypsiboasCordobae, 3679 rows) it runs 10 rounds of stumps twice: binary AdaBoost written out
  (labels −1 / +1, α' = ½ ln((1 − ε)/ε), weights × e^(∓α')) and `AdaBoostSAMME`. It asserts that
  every SAMME α is exactly 2α' and that both predict the same species on every row, then prints
  "Passed."; a failure stops the script.
- `Data()`: the handout's split. For each criterion it fits the Part 1 tree,
  `Bagging(BAGGING_T, c)`, `AdaBoostSAMME(BOOSTING_T, c, BOOSTING_DEPTH)` and the stumps,
  `AdaBoostSAMME(BOOSTING_T, c, max_depth=1)`, with the constants (25, 50, 3) at the top of the
  file.
- It prints the four matrix rows (`data.scores`: train, test, and the family accuracy derived
  from the predicted species through the training rows' species → family map); test % against
  the single tree; the rounds kept, with ε first / last / max; and stumps against depth 3, with
  the number of species the stumps predict on the test set.
- It returns a dict, which `tools/run_all.py` saves to `results/results.json` and
  `results/numbers.tex` for the report.
- About 45 s in all. Per criterion: bagging 10 to 11 s, AdaBoost 7 to 9 s, stumps about 3 s.

## Results and what they mean

| Model                           | Train %         | Test %        | Family %      |
|---------------------------------|-----------------|---------------|---------------|
| Single tree (Part 1)            | 100.00 / 100.00 | 94.93 / 95.41 | 96.25 / 96.80 |
| Bagging, 25 full trees          | 99.97 / 99.98   | 96.66 / 96.87 | 97.50 / 97.50 |
| AdaBoost, 50 depth-3 trees      | 97.52 / 96.35   | 95.34 / 94.16 | 96.73 / 95.00 |
| One depth-3 tree (Part 1 sweep) | 78.51 / 84.40   | 79.22 / 84.50 |               |
| AdaBoost, 50 stumps             | 84.12 / 83.79   | 83.32 / 84.16 |               |
| One stump (Part 1 sweep)        | 60.98 / 61.78   | 61.01 / 62.40 |               |

The stumps' train accuracy is *not part of the submission*; everything else is printed.

### Bagging against the single tree

- +1.73 / +1.46 points on test, and family accuracy 97.50 / 97.50 against 96.25 / 96.80: the
  variance of single trees averaged away. The equal family numbers are a coincidence: 1403 of the
  1439 rows each time, but not the same rows.
- Out-of-bag accuracy is 96.77 / 96.59 (*not part of the submission*), within 0.3 points of the
  test accuracy, without touching the test set.

### AdaBoost against the single tree

- Boosting starts from a weak tree (79.22 / 84.50) and ends near the full tree: 95.34 / 94.16,
  +0.41 / −1.25 points, so a little above it with Gini and below it with entropy. Train is
  97.52 / 96.35, a train-test gap of 2.2 points against 5.07 / 4.59 for the full tree. After 50
  rounds the ensemble still underfits a little: its remaining errors are mostly bias, the
  opposite of the full tree's.
- Test after 1, 5, 10, 25, 50 rounds: 79.22, 85.68, 90.62, 93.54, 95.34 (Gini); 84.50, 90.62,
  92.08, 93.40, 94.16 (entropy) (*not part of the submission*, except at 50).
- The curve wobbles. Over rounds 30 to 50 the test accuracy moves between 93.54 and 95.41 (Gini)
  and between 93.95 and 95.41 (entropy), and train accuracy drops in 15 / 19 of the 49 steps:
  each late tree has a say of about 2 and flips many rows. So the Gini-entropy gap (95.34 against
  94.16) is within the wobble, even though the entropy depth-3 tree alone is far better (84.50
  against 79.22).
- The handout fixes T = 50. Picking the best round by its test accuracy would be tuning on the
  test set.

### Why ε rises over the rounds

- ε is measured on the current weights, which stress the hard rows more every round. The update
  leaves the last tree at exactly 0.9 error on the new weights (above), and weight piles onto the
  rows the trees keep missing: after 50 rounds, 1% of the rows (57) hold 60% / 66% of the weight
  and 10% hold 98% / 99%; the largest weight is about 10¹³ times the smallest.
- Those rows are the ones a depth-3 tree cannot carve out, so each new tree misses more of the
  weight.
- The rise is not monotone: 0.2149, 0.3796, 0.4452, 0.4122, 0.4351, ... (Gini), with the
  maximum, 0.6306, in round 26 (entropy: 0.5447 in round 43).
- A rising ε does not mean a worse ensemble: its training accuracy climbs from 78.51 to 97.52
  (Gini). ε stays below 0.9, so all 50 rounds are kept, and α falls with it but stays above
  1.66 / 2.02.

### Why stumps struggle on ten classes

- A stump has one split and two leaves, so it names at most two species per round (each of ours
  names exactly two). Every row of the other eight species is a miss.
- So ε ≥ 1 − (the weight of the two heaviest classes). In round 1 those are
  AdenomeraHylaedactylus (2782 of the 5756 rows) and HypsiboasCordobae (897): a floor of 0.36,
  and ε = 0.39 / 0.38. Boosting then spreads the weight over all ten classes (6% to 16% each by
  the end), the floor rises to about 0.7 (0.71 / 0.70 averaged over rounds 11 to 50), and ε
  averages 0.75 / 0.74 there and ends at 0.78 / 0.80, not far below the 0.9 limit.
- Near the limit the say is small: α = 0.93 / 0.82 for the last stumps, against 1.77 / 2.25 for
  the last depth-3 trees. Each round moves the vote only a little.
- A species wins at x only if the stumps that name it there outweigh every other species. Rare
  species lose: the 50-stump vote never predicts Rhinellagranulosa (54 training rows) on the
  test set, and the Gini one never predicts OsteocephalusOophagus (91) either, although some
  stump names every species. Hence 8 / 9 of 10 species, and 83.32 / 84.16 test against
  95.34 / 94.16 for depth 3.
- More rounds help but do not close the gap: T = 100 gives 86.59 / 86.38 (all ten species),
  T = 500 gives 88.46 / 80.19 (*not part of the submission*).
- With K = 2 a stump can name both classes, which is why stumps are the classic weak learner of
  binary AdaBoost. A depth-3 tree has up to 8 leaves; ours have 7 or 8 and name 5 to 8 species.

## Demo

1. Start `uv run python part2_bagging_boosting.py` first: it takes about 45 s. It prints exactly
   `results/part2.txt`, so that file can stand in if time is short.
2. While it runs, walk through `Bagging.fit` (the `RandomState` at the top, the handout's
   `randint` call), then `AdaBoostSAMME.fit` line by line against the handout's six steps, then
   `tally`.
3. In the output, point at: the K = 2 check ("Passed."); the four matrix rows; bagging
   +1.73 / +1.46 over the single tree and
   AdaBoost +0.41 / −1.25; the rounds table (50 of 50 kept, ε rising but below 0.9); stumps
   83.32 / 84.16 with 8 / 9 of 10 species.
4. In `notebooks/demo.ipynb`, the Part 2 plot: the test accuracy of the vote after each round,
   and each round's ε against the 0.9 line, depth 3 against stumps. Its "Try it" cell,
   `AdaBoostSAMME(20, "gini", max_depth=5)`, prints 20 rounds and 97.08% test.
5. `uv run pytest tests/test_part2.py`: 9 tests in under a second (the handout's bootstraps,
   identical refits, the majority vote, K = 2 is classic AdaBoost, the printed K = 2 check passes,
   K = 4 keeps all 30 rounds with
   errors above ½ and the exact α, a perfect learner's ε clipped to 10⁻¹⁰, the first ε is the
   plain error rate, the stop on XOR).
6. Live checks in `uv run python`, from the repo root:

```python
import numpy as np
from common import Data, tally
from part2_bagging_boosting import AdaBoostSAMME, Bagging

data = Data()
X, y = data.X_train, data.y_train
ada = AdaBoostSAMME(50, "gini", max_depth=3).fit(X, y)  # about 7 s
np.round(ada.errors_[:5], 5)  # 0.21491 0.37963 0.44524 0.4122 0.43514
np.round(ada.alphas_[:5], 5)  # 3.49283 2.68835 2.41715 2.5521 2.45813
np.log((1 - ada.errors_[0]) / ada.errors_[0]) + np.log(9)  # 3.4928..., = alphas_[0]
len(ada.trees_), round(max(ada.errors_), 4)  # (50, 0.6306)
bag = Bagging(25, "gini").fit(X, y)  # about 10 s
wrong = np.flatnonzero(bag.predict(X) != y)  # [4000, 5500]
tally(bag.votes(X[wrong]), 10)  # [14 0 1 10 0 0 0 0 0 0] and [0 0 0 0 15 0 0 10 0 0]
rows = np.random.RandomState(42).randint(0, len(y), size=len(y))
len(np.unique(rows)) / len(y)  # 0.617: the first bootstrap's distinct rows
```

## Likely viva questions

- *Bagging against boosting?* Bagging: independent trees on bootstrap samples, equal votes; it
  cuts variance, so it wants low-bias, high-variance learners. Boosting: trees in sequence, each
  on weights that stress the earlier mistakes, votes weighted by α; it cuts bias, so it wants
  weak learners. Here bagging lifts full trees from 94.93 to 96.66 (Gini), and boosting lifts
  depth-3 trees from 79.22 to 95.34.
- *Why full trees for bagging but depth 3 for boosting?* Averaging cannot remove bias, so bagging
  gets trees with almost none. Boosting needs a learner that errs: a full tree fits the training
  set, ε = 0 is clipped to 10⁻¹⁰, the weights never change, and you get 50 copies of one tree
  (checked: 94.93, the Part 1 tree). Depth 3 is the handout's Table 1 setting.
- *Why does a bootstrap hold about 63.2% of the rows?* A row is missed by all N draws with
  probability (1 − 1/N)^N → e⁻¹ = 0.368.
- *Why is bagging's train accuracy not 100%?* A row in few bootstraps can be outvoted by the
  trees that never saw it: row 5500 is in 10 and loses 10 votes to 15.
- *What is the out-of-bag error?* Predict each training row with only the trees whose bootstrap
  missed it (about 9 of 25): a validation estimate for free. Ours: 96.77 / 96.59, against test
  96.66 / 96.87.
- *Why 1 − 1/K, and what if ε ≥ 1 − 1/K?* It is the error of guessing uniformly among K labels,
  and exactly where α stops being positive. The tree is discarded and boosting stops. A
  weighted-majority tree reaches it only if every leaf has equal class weights.
- *What does ln(K − 1) do?* It adds ln 9 = 2.197 to every α, so α stays positive up to ε = 0.9
  rather than ½. It vanishes for K = 2. Without it the 19 Gini rounds with ε > ½ would get α < 0.
- *What if ε = 0? Why clip?* Unclipped, (1 − ε)/ε divides by zero: a ZeroDivisionError here, inf
  and then nan weights with numpy floats. Clipped, α = 25.22, no weight changes and the same tree
  repeats, so the vote equals that tree.
- *Why does ε rise over the rounds? Is that bad?* The weight moves onto the hard rows, and the
  update leaves the last tree at exactly 0.9 error on the new weights. ε measures a harder
  problem each round; the ensemble's own training accuracy still climbs (78.51 to 97.52, Gini).
- *Why reweight rather than resample?* The handout asks for `sample_weight`; our tree uses the
  weights exactly, needs no random numbers, and every tree sees every row. Resampling (draw N
  rows with probability w) suits learners that cannot take weights, and adds noise; tried on the
  side, it gives 96.18 / 92.49.
- *Show that K = 2 gives classic AdaBoost.* As derived above: α = 2α', the updates agree up to a
  common factor, and the vote difference is 2 Σ α'·h(x), which has classic's sign.
- *What does T do?* Bagging: more trees never overfit, the accuracy levels off (96.32 / 96.66 at
  10 trees, 96.59 / 97.08 at 50). AdaBoost: every round lowers the exponential loss on the
  training rows (bias), but the training error itself wobbles (entropy: 96.92% train at round 41,
  96.35% at 50, 95.26% at 75, 97.69% at 100), and test follows for a while (Gini at T = 100:
  95.55). Boosting can overfit eventually, slowly. T is fixed by the handout.
- *What is a learning rate?* α ← ν·α with 0 < ν ≤ 1: smaller steps, more rounds needed, often
  better generalisation. Ours is ν = 1, the handout's formula. At T = 50, ν = 0.5 gives
  94.79 / 95.00 and ν = 0.1 gives 94.37 / 93.54: too few rounds for such small steps.
- *How are ties broken?* Splits: gains within 10⁻¹² count as tied, then the lower feature index
  and the lower threshold win. Leaves and votes: the smallest label, through `argmax` on the
  sorted `classes_`. Bagging has 5 / 4 tied test rows; AdaBoost's scores (sums of real α) never
  tie here, apart from the all-zero scores when no tree is kept.
- *Why do stumps struggle on ten classes?* Two leaves name two species per round, so ε stays
  near 0.75 once the weight is spread out, α is small, and rare species never win the vote.
- *Why is entropy AdaBoost below the single tree?* After 50 rounds it still underfits (96.35%
  train), and the late rounds wobble by more than a point: it was at 95.41 in round 33. The
  handout fixes T = 50, so we report 94.16.
- *How do you know the SAMME code is right?* The tests above, plus a check outside the repo,
  never submitted (scikit-learn is prohibited in the submission): scikit-learn's
  `AdaBoostClassifier` (SAMME, depth-3 trees) gives the same first five ε and α (0.21491 and
  3.49283, ...) and the same test accuracy, 95.34 / 94.16. Its trees on our exact bootstraps give
  bagging 96.53 / 97.08 against our 96.66 / 96.87, because it breaks ties between equally good
  splits at random.
- *Is AdaBoost robust to label noise?* No. A mislabelled row is missed in every round and its
  weight grows by e^α each time; here 1% of the rows end up with 60% / 66% of the weight. Bagging
  is far more robust.

## Likely code changes

- **A learning rate (shrinkage).** In `AdaBoostSAMME.__init__` add `learning_rate=1.0` and store
  it as `self.learning_rate`; in `fit` scale the say:

  ```python
  alpha = self.learning_rate * float(np.log((1 - eps) / eps) + np.log(K - 1))
  ```

  The shrunk α is used both in the update and in the vote, as in scikit-learn. ν = 1 reproduces
  every number; at T = 50, ν = 0.5 gives 94.79 / 95.00.
- **Staged accuracy per round.** Add to `AdaBoostSAMME`:

  ```python
  def staged_predict(self, X):
      """The vote's prediction after 1, 2, ..., T rounds."""
      scores = np.zeros((len(X), len(self.classes_)))
      for tree, alpha in zip(self.trees_, self.alphas_):
          k = np.searchsorted(self.classes_, tree.predict(X))
          scores[np.arange(len(X)), k] += alpha
          yield self.classes_[scores.argmax(axis=1)]
  ```

  In `main`: `[accuracy(data.y_test, p) for p in boosting[c].staged_predict(data.X_test)]`. The
  last value equals `predict`'s (95.34 / 94.16); the notebook's `staged_test_accuracy` does the
  same with a cumulative sum. For bagging, vote with the first t trees:
  `tally(model.votes(X)[:t], K)`.
- **Out-of-bag accuracy for bagging.** In `Bagging.fit`, record which rows each tree saw. There
  are no extra random draws, so every printed number stays the same. The loop becomes:

  ```python
  self.inbag_ = np.zeros((self.n_estimators, N), dtype=bool)
  for t in range(self.n_estimators):
      rows = rng.randint(0, N, size=N)
      self.inbag_[t, rows] = True
      self.trees_.append(self.new_tree(rng).fit(X[rows], y[rows]))
  ```

  and a new method (`accuracy` is already imported):

  ```python
  def oob_accuracy(self, X, y):
      """Accuracy on the training rows, each voted on only by the trees that never saw it."""
      votes = np.where(self.inbag_, -1, self.votes(X))  # -1 matches no class
      counts = tally(votes, len(self.classes_))
      seen = counts.sum(axis=1) > 0  # rows with at least one out-of-bag tree
      return accuracy(y[seen], self.classes_[counts.argmax(axis=1)][seen])
  ```

  `bagging[c].oob_accuracy(X, y)` gives 96.77 / 96.59. `RandomForest` inherits it.
- **Another T or max_depth.** Change `BAGGING_T`, `BOOSTING_T` or `BOOSTING_DEPTH` at the top
  of `part2_bagging_boosting.py`: `main` builds the models and prints its labels from them (the
  stumps keep `max_depth=1`). Do not run `tools/run_all.py` after a demo change: it rewrites
  `results/`. The notebook's "Try it" cell is the quickest route.
  - A deeper weak learner needs fewer rounds: depth 5 with T = 20 gives 97.08% (Gini).
  - Long runs of depth-3 trees push some weights below rounding error: from round 85
    (entropy) or 111 (Gini) the largest weight is 10¹⁸ to 10²¹ times the smallest. At a cut
    whose right-hand rows weigh less than rounding error, `weights - left` comes out as 0.
    `share` in `part1_decision_tree.py` then gives that side zero probabilities, so it adds
    nothing to the gain and the run carries on without warnings. Without it, 0/0 would make
    the node's best gain nan and silently turn the node into a leaf. At T = 50 the case never
    arises.
- **Resampling instead of reweighting.** In `AdaBoostSAMME.fit`, create
  `rng = np.random.RandomState(42)` at the top and replace the two lines that build and fit the
  tree:

  ```python
  rows = rng.choice(N, size=N, p=w)  # N rows drawn with probability w
  tree = DecisionTree(self.criterion, max_depth=self.max_depth).fit(X[rows], y[rows])
  miss = tree.predict(X) != y  # still the weighted error over all N rows
  ```

  The rest stays. Test 96.18 / 92.49 (*not part of the submission*).
- **Another bagging seed.** `Bagging(25, c, seed=0)` gives 96.73 / 96.94, seed 1 gives
  96.53 / 96.94. Keep 42: the handout asks for it.
- **SAMME.R** is not asked for. It is the "real" variant, which adds each tree's class
  log-probabilities instead of hard votes, and would need class probabilities from the tree.
