# Part 5: Gaussian naive Bayes

## In short

Gaussian naive Bayes models each species as 22 independent one-dimensional Gaussians, one per
MFCC, fitted by maximum likelihood (prior n_c/N, mean, population variance) plus a variance floor
ε = 10⁻⁹ · max_j Var_train(x_j) = 7.027 × 10⁻¹¹. It predicts the class with the largest log prior
plus 22 log densities, for all rows and classes at once through an (n, K, d) broadcast. It is the
weakest model in the matrix: 91.47% train, 90.34% test, 91.94% derived family, against 93.82% to
97.78% test for the tree-based models. Two causes. The MFCCs are far from independent within a
species (mean |r| 0.23 to 0.50, single pairs up to 0.96), so correlated features count the same
evidence several times. And MFCC 1 is 1.0 in all 434 training rows of Ameeregatrivittata, so its
variance there is ε and every row with MFCC 1 = 1.0 gets 6.7 to 9.8 nats towards that species:
NB predicts it for 187 test rows, 79 of them wrongly.

## The model

### Bayes' rule and the naive assumption

- Bayes: p(c | x) = π_c p(x | c) / p(x). The evidence p(x) = Σ_c π_c p(x | c) is the same for
  every class, so argmax_c p(c | x) = argmax_c π_c p(x | c). Classifying never needs p(x).
- Naive: the features are independent *given the class*, p(x | c) = Π_j p(x_j | c). Not
  independent overall: two MFCCs can be correlated over the whole data set because the species
  differ in both, and still be independent inside each species. That is why the script prints
  the within-species correlations as well as the overall matrix.
- Why assume it: a full 22-dimensional Gaussian per class needs a 22 × 22 covariance (253
  numbers). Independence leaves 22 one-dimensional fits per class, 2 numbers each.
- Gaussian: p(x_j | c) = N(x_j; μ_cj, σ²_cj) = exp(−(x_j − μ_cj)² / (2σ²_cj)) / √(2π σ²_cj).
  The MFCCs are continuous, and a Gaussian needs only a mean and a variance, both in closed form.
  The handout prescribes it.

### The estimates (maximum likelihood)

- The handout's three formulas: π_c = n_c / N, μ_cj = (1/n_c) Σ_{n: y_n = c} x_nj and
  σ²_cj = (1/n_c) Σ_{n: y_n = c} (x_nj − μ_cj)² + ε.
- Why these: the training log-likelihood Σ_n [log π_{y_n} + Σ_j log N(x_nj; μ_{y_n j}, σ²_{y_n j})]
  splits into a prior term and one term per class and feature. A Gaussian's likelihood is
  maximised by the sample mean and the 1/n variance; Σ_c n_c log π_c subject to Σ_c π_c = 1
  (Lagrange) by π_c = n_c/N.
- The 1/n_c variance (ddof 0) is the MLE, biased low by (n_c − 1)/n_c. The handout's formula uses
  it, and its sanity check pins it: the variance of {0, 2} is 1 with ddof 0, 2 with ddof 1.
- 449 free parameters: 10 × 22 means, 10 × 22 variances, 9 priors (they sum to 1).
- The training classes, in `classes_` order (sorted by `np.unique`; k is the row in every array):

| k | Species | n_c | π_c | log π_c |
|---|---|---|---|---|
| 0 | AdenomeraAndre | 538 | 0.0935 | −2.37 |
| 1 | AdenomeraHylaedactylus | 2782 | 0.4833 | −0.73 |
| 2 | Ameeregatrivittata | 434 | 0.0754 | −2.58 |
| 3 | HylaMinuta | 248 | 0.0431 | −3.14 |
| 4 | HypsiboasCinerascens | 378 | 0.0657 | −2.72 |
| 5 | HypsiboasCordobae | 897 | 0.1558 | −1.86 |
| 6 | LeptodactylusFuscus | 216 | 0.0375 | −3.28 |
| 7 | OsteocephalusOophagus | 91 | 0.0158 | −4.15 |
| 8 | Rhinellagranulosa | 54 | 0.0094 | −4.67 |
| 9 | ScinaxRuber | 118 | 0.0205 | −3.89 |

### Variance smoothing ε

- ε = 10⁻⁹ · max_j Var_train(x_j): the largest of the 22 feature variances over all 5756
  training rows, classes ignored, times 10⁻⁹. Population variance (ddof 0), like the formula's
  Var_train and the sanity check. The largest is MFCC 3's, 0.07027, so ε = 7.027 × 10⁻¹¹. It is
  added to every class variance. Same rule and default as scikit-learn's `var_smoothing`.
- Why it exists: a variance of 0 (a feature constant within a class, or a class with one row)
  makes log σ² = −∞ and (x − μ)²/σ² either ∞ or 0/0. MFCC 1 is constant in Ameeregatrivittata.
  With `var_smoothing=0` (experiment, not part of the submission) that class's score is nan on all
  1439 test rows, `np.argmax` returns the first nan, and every row is predicted
  Ameeregatrivittata: 7.51% test.
- Why relative to the data: an absolute 10⁻⁹ would mean different things in different units.
  Scaled by a variance of the data, ε scales with it: multiply every feature by a, and every
  variance and ε grow by a², so the predictions do not change (checked with a = 1000).
- Why the largest variance: one number for all features, and never 0 unless every feature is
  constant. Why 10⁻⁹: it should change nothing except at an exact zero. The smallest nonzero class
  variance here (HylaMinuta, MFCC 1: 4.67 × 10⁻⁵) is 6.6 × 10⁵ times ε.
- What it does at a zero: the density becomes a spike of height 1/√(2πε) ≈ 47 591. A row exactly
  at the constant value gets −½ log(2πε) = +10.77 nats (natural-log units) from that feature; any
  other value pays (x − μ)²/(2ε), at least 1.75 × 10⁶ nats on our test set. See "The MFCC 1
  effect" below.

### Log space and the decision rule

The handout's rule, which the code implements term for term:

    ŷ = argmax_c [ log π_c − ½ Σ_j ( log(2π σ²_cj) + (x_j − μ_cj)² / σ²_cj ) ]

- Why logs: the product π_c Π_j p(x_j | c) of 22 densities can underflow. No double is below
  about 5 × 10⁻³²⁴, so a log below about −745 becomes exactly 0. If that happens for every class,
  argmax of ten zeros returns class 0 whatever the data. In log space the product is a sum, and
  −2.8 × 10⁹ (the most negative score on our test set) is an ordinary number.
- Honest detail: on this data the direct product would have worked for the test set. MFCC
  variances are small (~10⁻³), so densities are often above 1, and every test row's best class
  has a log score of at least −129.55. Under 0.6% of the 14 390 (row, class) products underflow
  (the Ameeregatrivittata column for the 55 test rows with MFCC 1 ≠ 1.0, and 24 HylaMinuta
  entries), never a whole row, and the predictions are identical (experiment). The test
  `test_far_away_points_do_not_underflow` builds a case where it fails: at x = 1000 all three
  products are 0 (argmax: class 0), while the log scores still rank the classes (class 2, the
  nearest). The handout requires log space anyway.
- Constants: the code keeps −½ log 2π (it is in the formula), so `log_posterior` returns exactly
  log π_c + log p(x | c) = log p(x, c), the joint log-likelihood. What is left out is log p(x),
  the same for every class, so the argmax equals that of log p(c | x). −(d/2) log 2π could be
  dropped too. Not droppable: −½ log σ²_cj (it differs by class: a narrow Gaussian is taller) and
  log π_c.
- Ties: `argmax` returns the first maximum and `classes_` is sorted, so a tie goes to the smallest
  label, the handout's rule. Exact ties are rare with continuous features; the sanity check's
  x = 6 is one (both classes score −14.1121) and goes to class 0.

## The code

`part5_naive_bayes.py`. The class first:

```python
def fit(self, X, y):
    X = np.asarray(X, dtype=float)
    self.classes_, codes = np.unique(y, return_inverse=True)
    self.epsilon_ = self.var_smoothing * X.var(axis=0).max()
    groups = [X[codes == k] for k in range(len(self.classes_))]
    self.log_prior_ = np.log([len(g) / len(X) for g in groups])  # pi_c = n_c / N
    self.mean_ = np.array([g.mean(axis=0) for g in groups])
    self.var_ = np.array([g.var(axis=0) for g in groups]) + self.epsilon_  # 1/n_c, + eps
    return self
```

- `np.unique(y, return_inverse=True)`: the sorted distinct labels, and for every row the index of
  its label among them (`classes_[codes]` is `y`). So `codes == k` selects class k.
- `X.var(axis=0)`: the 22 feature variances over all training rows; numpy's default is ddof 0.
  `.max()` is MFCC 3's 0.07027.
- `groups`: the rows of each class. A loop over the 10 classes, not over samples.
- `mean_` and `var_` are (K, d) = (10, 22); `g.var(axis=0)` is again ddof 0, the handout's 1/n_c.
  `log_prior_` is (10,). Fitting takes about 2 ms.

```python
def log_posterior(self, X):
    """log pi_c - 1/2 sum_j [log(2 pi var_cj) + (x_j - mean_cj)^2 / var_cj] for every row and
    class, up to a constant: an (n, K, d) array summed over the features."""
    diff = np.asarray(X, dtype=float)[:, None, :] - self.mean_
    terms = np.log(2 * np.pi * self.var_) + diff**2 / self.var_
    return self.log_prior_ - 0.5 * terms.sum(axis=2)
```

- `X[:, None, :]` is (n, 1, d); `mean_` (K, d) broadcasts against it to (n, K, d), so
  `diff[i, k, j] = x_ij − μ_kj`.
- `np.log(2 * np.pi * self.var_)` is (K, d), computed once and broadcast over the rows;
  `diff**2 / self.var_` is (n, K, d). Summing over axis 2, the features, gives (n, K), and
  `log_prior_` (K,) broadcasts over the rows.
- Sizes: 1439 × 10 × 22 = 316 580 doubles (2.5 MB) for the test set, 1 266 320 (10 MB) for the
  training set. This is the broadcasting the handout asks for, with no loop over samples.
- Despite the name it returns the joint log-likelihood log p(x, c), which differs from
  log p(c | x) only by log p(x).

`predict` is `self.classes_[self.log_posterior(X).argmax(axis=1)]`: the label of the best score in
each row.

### `sanity_check`

The handout's check. `main` runs it first; it asserts, so a failure stops the script before any
result is printed. X = [[0], [2], [10], [12]], y = [0, 0, 1, 1]:

- ε = 10⁻⁹ × Var{0, 2, 10, 12} = 10⁻⁹ × 26 = 2.6 × 10⁻⁸ (mean 6; squared deviations 36, 16, 16,
  36; mean 26).
- Class 0: mean 1, variance ((0 − 1)² + (2 − 1)²)/2 = 1, plus ε. Class 1: mean 11, variance
  1 + ε. Priors ½ each: log ½ = −0.6931. And ln 2π = 1.8379.
- x = 3: class 0 scores −0.6931 − ½(1.8379 + (3 − 1)²) = −0.6931 − 2.9189 = −3.6121; class 1
  scores −0.6931 − ½(1.8379 + (3 − 11)²) = −0.6931 − 32.9189 = −33.6121. Class 0, by 30 nats.
- x = 9: the mirror image, (9 − 1)² = 64 against (9 − 11)² = 4. Class 1, by 30 nats.
- Equal priors and variances make it "nearest mean": the log-odds of class 0 is −10x + 60, zero
  at x = 6. ε is far too small to matter.
- The asserts: means [1, 11], variances 1 (`np.allclose` tolerates the 2.6 × 10⁻⁸), predictions
  [0, 1]. The printout "variances 1 + 2.6e-08" shows ε.

### `print_correlations` and `within_class_correlation`

- `np.corrcoef(data.X_train, rowvar=False)`: the 22 × 22 Pearson correlation matrix of the
  training features (`rowvar=False`: columns are the variables). Training rows only, as the
  handout says; the test set is only for evaluation.
- `print_correlations` prints it in two blocks of 11 columns, then takes the 231 pairs above the
  diagonal (`np.triu_indices(d, k=1)`) and prints their mean and max |r|, how many are above 0.5,
  and the five strongest (`argsort(-off, kind="stable")`, a fixed order for equal values).
- `within_class_correlation` does the same per species, on that species' training rows: the
  dependence NB actually assumes away. It first drops features with zero standard deviation
  (MFCC 1 in Ameeregatrivittata, whose correlations would be 0/0 = nan), so that species' mean is
  over 210 pairs, not 231.

### `main`

1. `Data()`: the handout's split, 5756 / 1439 rows. `header` prints the sizes; `sanity_check()`.
2. Fits on the training rows. `data.scores(nb)` (`common.py`) gives train accuracy, test accuracy
   and the derived family accuracy (each predicted species mapped to its family by the training
   map); `print_rows` prints the matrix row.
3. Prints ε, then the constant-feature report: `np.nonzero(nb.var_ == nb.epsilon_)` finds the
   (class, feature) pairs whose smoothed variance is exactly ε, i.e. whose raw variance was 0.
   There is one, (2, 0): Ameeregatrivittata, MFCC 1. It counts the test rows with exactly that
   value (`nb.mean_[k, j]`, the constant): 1384 of 1439.
4. The 10 × 10 test confusion matrix (`common.confusion_matrix`: rows true, columns predicted, in
   `classes_` order). `predicted` stores, per species, (column sum, diagonal) = (test rows
   predicted as it, of which right).
5. The correlation matrix and the within-species means.
6. Returns a dict, which `tools/run_all.py` saves in `results/results.json` and turns into
   `results/numbers.tex`, the source of the report's numbers (79 = 187 − 108 comes from there).

### Tests

`tests/test_part5.py`, 4 tests: the handout's sanity check; the estimates against the formulas
written out (means, 1/n variances plus ε, log priors) on random blobs; the vectorised
`log_posterior` against a triple loop over rows, classes and features; far-away points (x = 1000)
giving finite scores.

## Results and what they mean

### Against the trees

| Model | Test %, Gini / entropy |
|---|---|
| Random forest | 97.78 / 97.64 |
| Bagging | 96.66 / 96.87 |
| AdaBoost | 95.34 / 94.16 |
| Single tree | 94.93 / 95.41 |
| Hierarchy | 93.82 / 94.65 |
| Gaussian NB | 90.34 |

- NB is last: 3.48 points below the weakest tree-based row (hierarchy, Gini) and 7.44 below the
  best (forest, Gini).
- Train 91.47% against test 90.34%, a gap of 1.13 points. NB has 449 parameters for 5756 rows and
  cannot memorise them (a full tree gets 100% train). High bias, low variance: its errors come
  from its assumptions, not from overfitting, and more data would not remove them.
- Its posteriors are overconfident (experiment, posteriors by log-sum-exp): the mean largest
  posterior on the test set is 0.9845 while the accuracy is 0.9034, and 71 of the 139 wrong
  predictions give the wrong class more than 0.99.

### Correlation: the independence assumption fails

- Over all training rows, off-diagonal |r| has mean 0.30 and max 0.91 (MFCC 13 and 15,
  r = −0.91); 43 of the 231 pairs are above 0.5. The five strongest pairs are all two apart and
  negative (13–15, 15–17, 11–13, 9–11, 12–14): MFCC 9, 11, 13, 15 and 17 move as one block (9 and
  13: +0.67, 11 and 15: +0.72, 13 and 17: +0.78).
- NB assumes independence within each species, which the overall matrix does not test. Within a
  species the mean |r| goes from 0.23 (HypsiboasCordobae) to 0.50 (Rhinellagranulosa), and the
  strongest pair in each species from 0.80 (HypsiboasCordobae) to 0.96 (Rhinellagranulosa, MFCC 5
  and 13). The assumption is clearly false.
- Effect, double counting: if two features carry the same information, NB adds its log-likelihood
  ratio twice, as if it were two independent pieces of evidence. A block of five related MFCCs
  casts about five votes for one fact and can outvote a single informative feature. The
  posteriors become overconfident (above).
- Trees are not misled this way. A split tests one feature; after a split on one of two redundant
  features, the other adds little gain. Later splits are conditional on earlier ones, so a tree
  models interactions instead of adding up votes.
- What it costs here (experiment, not part of the submission): a Gaussian per class with a full
  covariance matrix (QDA: the same priors, means and ε, correlations kept) reaches 97.64% test,
  level with the forest's entropy row. The only difference from NB is the off-diagonal
  covariances, so on this data the independence assumption costs about 7 points. One shared full
  covariance (LDA) gets 94.30%.

### The MFCC 1 effect

- MFCC 1 is never above 1.0 and equals 1.0 in 6947 of the 7195 rows. In training it is 1.0 in
  all 434 Ameeregatrivittata rows and in 74.7% to 99.2% of the rows of every other species: on
  its own it says little about the species.
- NB turns that into strong evidence. For Ameeregatrivittata the variance is ε, so at x = 1.0
  the MFCC 1 term is −½ log(2πε) = +10.77 nats. The other species have ordinary variances
  (4.67 × 10⁻⁵ to 2.37 × 10⁻²) and get +0.93 (Rhinellagranulosa) to +4.06 (HylaMinuta). So
  every row with MFCC 1 = 1.0 (1384 of the 1439 test rows) gets 6.7 to 9.8 nats towards
  Ameeregatrivittata from this one feature, a likelihood ratio of about 800 to 19 000. A row with
  any other value pays at least 1.75 × 10⁶ nats and is never predicted Ameeregatrivittata.
- The root cause is the Gaussian shape: MFCC 1 is a point mass at 1.0 plus a tail below it. As a
  discrete probability, P(MFCC 1 = 1.0 | c) is 1.000 for Ameeregatrivittata against 0.987 for
  AdenomeraAndre, almost no evidence; the fitted densities turn it into e^8.62 ≈ 5500.
- Result: NB predicts Ameeregatrivittata for 187 test rows, 79 of them wrongly (44 AdenomeraAndre,
  29 HylaMinuta, 4 LeptodactylusFuscus, 2 ScinaxRuber); its recall is 108/108 but its precision
  108/187. In 59 of the 79 errors the margin over the true species is smaller than MFCC 1's
  contribution, so that feature alone decided against the true species.
- Experiment, not part of the submission: without MFCC 1, NB gets 92.70% test (93.92% train,
  94.72% family). Ameeregatrivittata is then predicted 103 times, 95 right, and 50 of the 79
  errors become right. A larger `var_smoothing` also softens the spike: each factor of 10 takes
  ½ ln 10 = 1.15 nats off the bonus (table under "Likely code changes"). Table 1 fixes 10⁻⁹, so
  the submission does neither.

### The confusion matrix

- Column 2 (Ameeregatrivittata) holds 79 of the 139 errors. HylaMinuta (recall 27/62 = 43.5%)
  and AdenomeraAndre (86/134 = 64.2%) suffer most, both mainly through that column.
- The largest species, AdenomeraHylaedactylus, is 688/696 = 98.9% right.
- The other 60 errors are scattered. The largest cells: 9 HypsiboasCordobae predicted as
  OsteocephalusOophagus, and 5 each of HylaMinuta → AdenomeraHylaedactylus and
  OsteocephalusOophagus → HypsiboasCinerascens.
- 116 of the 139 errors cross a family boundary (the 79 into Ameeregatrivittata, the only
  Dendrobatidae species, among them), so the derived family accuracy, 91.94% (1323 of 1439), is
  only 1.60 points above the species accuracy.

### Correctness

Besides the tests, a one-off comparison outside the repo, never committed or submitted
(scikit-learn is prohibited in the submission): scikit-learn's `GaussianNB(var_smoothing=1e-9)`
gives the same prediction on every test row, the same ε, means and variances (max difference 0),
and joint log-likelihoods equal to within 2 × 10⁻⁶.

## Demo

1. From the repo root, `uv run python part5_naive_bayes.py` (the grader runs plain
   `python part5_naive_bayes.py`); about 0.4 s. It prints exactly `results/part5.txt`.
2. Point at, in order: "Sanity check ... Passed."; the row 91.47 / 90.34 / 91.94; ε = 7.027e-11;
   the MFCC 1 line (constant within Ameeregatrivittata, 1384 of 1439 test rows share 1.0);
   column 2 of the confusion matrix (44, 29, 4 and 2 around the 108); the correlation summary
   (mean 0.30, max 0.91, 43 pairs); the within-species list (0.23 to 0.50).
3. Show `fit` (one line per formula of the handout) and the three lines of `log_posterior` (the
   (n, K, d) broadcast).
4. `uv run pytest tests/test_part5.py`: 4 passed, under a second.

Live checks, in `uv run python` from the repo root:

```python
import numpy as np
from common import Data
from part5_naive_bayes import GaussianNB

data = Data()
nb = GaussianNB().fit(data.X_train, data.y_train)
nb.classes_[2], nb.var_[2, 0], nb.epsilon_  # Ameeregatrivittata, 7.027e-11 twice
nb.var_.shape, np.exp(nb.log_prior_).sum()  # (10, 22), 1.0
nb.log_posterior(data.X_test[:1]).round(2)  # test row 0: 15.66 at k = 0, 22.0 at k = 2
data.y_test[0], nb.predict(data.X_test[:1])  # AdenomeraAndre, predicted Ameeregatrivittata
x = data.X_test[0]  # one score by hand: 15.66
(
    nb.log_prior_[0]
    - 0.5 * (np.log(2 * np.pi * nb.var_[0]) + (x - nb.mean_[0]) ** 2 / nb.var_[0]).sum()
)
```

- Test row 0 is one of the 79. Its MFCC 1 is 1.0, worth 10.77 for Ameeregatrivittata and 2.15
  for AdenomeraAndre: 8.62 nats, more than the margin of 22.00 − 15.66 = 6.34. Without MFCC 1 it
  would be AdenomeraAndre (13.50 against 11.23). Its posterior for the wrong class is 0.9982.
- Another setting: `GaussianNB(var_smoothing=1e-6)` gives 92.01% test (not the handout's value).

## Likely viva questions

- *Why "naive"?* It assumes the features are independent given the class,
  p(x | c) = Π_j p(x_j | c). False for MFCCs (within-species |r| up to 0.96), but it reduces each
  class to 22 one-dimensional fits.
- *Independent or conditionally independent?* Given the class. The overall correlation matrix
  mixes in differences between species means; the within-species correlations test the
  assumption itself.
- *Why Gaussian?* Continuous features, two parameters per class and feature, closed-form MLE; the
  handout prescribes it. A histogram or kernel density per feature would have handled MFCC 1's
  point mass at 1.0 better.
- *Derive the estimates.* The log-likelihood splits per class and feature; a Gaussian's MLE is the
  sample mean and the 1/n_c variance; maximising Σ n_c log π_c with Σ π_c = 1 gives n_c/N.
- *Why ε, and why relative to the largest variance?* A zero variance gives log 0 and division by
  0 (without ε every test row becomes Ameeregatrivittata, 7.51%). Relative, so it scales with the
  data; the largest, so one number serves every feature and is never 0; 10⁻⁹, so it changes
  nothing but the zeros.
- *ddof 0 or 1?* 0: the handout's 1/n_c, the MLE, and what the sanity check needs (variance of
  {0, 2} is 1; with ddof 1 it is 2 and the assert fails). numpy's default is 0. On our data ddof 1
  also gives 90.34% test (experiment), but nan for a class with one row.
- *Why log space?* 22 densities multiplied can underflow to 0 for every class (any log below
  about −745), and argmax of zeros is class 0. A sum of logs does not underflow. Here the direct
  product would have survived on the test set; the far-away test shows where it fails.
- *What can be left out of the score?* Anything equal for every class: log p(x) (never computed)
  and −(d/2) log 2π (kept, as in the formula). Not −½ log σ²_cj: it differs by class.
- *How do the priors enter?* As log π_c, once, against 22 likelihood terms: −4.67
  (Rhinellagranulosa) to −0.73 (AdenomeraHylaedactylus), a spread of 3.94 nats, so they only
  decide close calls. With uniform priors 18 test predictions change, 7 right before and 7 after:
  still 90.34% (experiment).
- *Ties?* `argmax` takes the first maximum and `classes_` is sorted: the smallest label, as the
  handout says. The sanity check's x = 6 is an exact tie and goes to class 0.
- *What if a class has one sample?* Its means are that row, its variances 0 + ε: a spike. It
  scores 22 × 10.77 ≈ 237 nats at its own point and an enormous penalty elsewhere, so it wins
  only for (near-)copies of that row; added as an experiment, such a class was predicted for its
  own test row only. With ddof 1 its variances would be 0/0 = nan, and since `argmax` returns the
  first nan, every row would go to it.
- *Does correlation always hurt NB's accuracy?* No. Classification only needs the right class to
  score highest, not calibrated posteriors. Copy one feature d times: with equal priors every
  score is d times the right log-likelihood, the argmax is unchanged, the posteriors are wildly
  overconfident. It hurts when a correlated block outvotes other evidence, as here: QDA, the same
  model with correlations, gets 97.64% (experiment).
- *NB against LDA and QDA?* All three model each class as a Gaussian. QDA: a full covariance per
  class, quadratic boundaries with cross terms x_j x_k (2759 parameters). LDA: one shared
  covariance, linear boundaries (482). Gaussian NB: a diagonal covariance per class, i.e. QDA with
  the off-diagonals forced to 0; quadratic boundaries without cross terms (449). Experiments: QDA
  97.64%, LDA 94.30%.
- *Why do the trees do better?* No independence or Gaussian assumption; each split uses one
  feature, so redundancy does not add up, and a threshold handles the point mass at 1.0. NB's
  errors are bias: even its train accuracy is only 91.47%.
- *Generative or discriminative?* Generative: it models p(x | c) and π_c, then applies Bayes'
  rule. Logistic regression is the discriminative counterpart.
- *Recall 108/108 but precision 108/187 for Ameeregatrivittata: why?* MFCC 1 is constant in its
  training rows, so variance ε and +10.77 nats at 1.0. All 108 of its test rows have 1.0, and so
  do 1276 test rows of other species.
- *Does feature scaling matter?* Not for NB itself: rescaling a feature shifts every class's
  score by the same amount, and scaling all features together leaves ε's effect unchanged too.
  Scaling features differently changes ε relative to each feature, which matters only where a
  class variance is (near) 0: standardising first gives 89.72% (experiment), since MFCC 1 has the
  smallest variance of the 22 and its spike gets sharper.
- *How do you know it is right?* The sanity check runs and asserts first; the tests compare with
  the formulas and with a plain loop; a one-off scikit-learn comparison outside the submission
  matched every prediction.

## Likely code changes

All are experiments outside the submission: Table 1 fixes the model, and choosing a setting by its
test accuracy would be tuning on the test set, which the handout forbids.

- **predict_proba** (log-sum-exp), a new method of `GaussianNB` after `log_posterior`:

  ```python
  def predict_proba(self, X):
      jll = self.log_posterior(X)
      p = np.exp(jll - jll.max(axis=1, keepdims=True))  # the best class gets exp(0) = 1
      return p / p.sum(axis=1, keepdims=True)
  ```

  Subtracting the row maximum cancels in the normalisation, but keeps `exp` from overflowing and
  makes the sum at least 1, so no 0/0. log p(c | x) = jll − max − log Σ_c exp(jll − max). Test row
  0 gives 0.9982 for Ameeregatrivittata.
- **Uniform priors**: in `fit`, `self.log_prior_ = np.full(len(groups), -np.log(len(groups)))`
  (any constant works: it shifts every class equally). 90.34% test, 18 predictions changed.
- **ddof 1**: `g.var(axis=0, ddof=1)` in the `var_` line. The sanity check's assert then fails
  (variances 2) and stops the script, and `test_estimates_follow_the_handout` fails. With the
  check skipped: 91.45% train, 90.34% test.
- **Another var_smoothing**: pass it in `main`, `GaussianNB(var_smoothing=1e-6)`, and leave the
  default at 1e-9. `sanity_check` uses the default, and from 1e-6 up its `np.allclose(var, 1)`
  fails (ε = 2.6 × 10⁻⁵ on its data). Test % and test rows predicted Ameeregatrivittata:

  | var_smoothing | 1e-12 | 1e-10 | 1e-9 | 1e-8 | 1e-6 | 1e-4 | 1e-3 | 1e-2 | 1e-1 | 1 |
  |---|---|---|---|---|---|---|---|---|---|---|
  | Test % | 88.53 | 89.78 | 90.34 | 91.04 | 92.01 | 92.77 | 92.98 | 92.84 | 90.62 | 83.46 |
  | Predicted Am. | 217 | 196 | 187 | 173 | 147 | 124 | 118 | 113 | 140 | 170 |

  A smaller ε sharpens the MFCC 1 spike; a larger one softens it, until from about 10⁻² (ε ≈
  7 × 10⁻⁴) it swamps real class variances and accuracy falls.
- **Drop a feature**: the score is a sum over features, so dropping one deletes its term.

  ```python
  from common import accuracy

  keep = np.arange(1, 22)  # every column but MFCC 1 (column 0)
  nb2 = GaussianNB().fit(data.X_train[:, keep], data.y_train)
  accuracy(data.y_test, nb2.predict(data.X_test[:, keep]))  # 0.9270
  ```

  `data.scores(nb2)` would not work as is: it predicts on all 22 columns. To make it permanent,
  slice `data.X_train` and `data.X_test` right after `Data()` in `main`; the MFCC numbers that
  `main` and `print_correlations` print (`j + 1`) then shift by one.
- **Decorrelate first (PCA by eigendecomposition)**, fitted on the training rows only:

  ```python
  mu = data.X_train.mean(axis=0)
  w, V = np.linalg.eigh(np.cov(data.X_train - mu, rowvar=False))  # ascending eigenvalues
  V = V[:, ::-1]  # largest first; keep the top q with V[:, :q]
  Z_train, Z_test = (data.X_train - mu) @ V, (data.X_test - mu) @ V
  nb_pca = GaussianNB().fit(Z_train, data.y_train)
  ```

  90.55% test with all 22 components, 91.31% with the top 15. Barely better: the rotated features
  are uncorrelated over the pooled training set (max |r| ≈ 2 × 10⁻¹⁴) but not within a species
  (mean |r| 0.10 to 0.36), and within is what NB needs. One rotation cannot diagonalise ten
  different class covariances. It does remove the MFCC 1 spike (no rotated feature is constant
  within a species): Ameeregatrivittata is predicted 127 times.
- **Full covariance per class (QDA)**, numpy only:

  ```python
  class GaussianQDA(GaussianNB):
      def fit(self, X, y):
          super().fit(X, y)  # classes_, epsilon_, log_prior_, mean_
          X, y = np.asarray(X, dtype=float), np.asarray(y)
          self.eig_ = []
          for k, c in enumerate(self.classes_):
              g = X[y == c] - self.mean_[k]
              S = g.T @ g / len(g) + self.epsilon_ * np.eye(X.shape[1])  # 1/n_c, + eps
              self.eig_.append(np.linalg.eigh(S))  # S = V diag(w) V^T
          return self

      def log_posterior(self, X):
          X = np.asarray(X, dtype=float)
          out = np.empty((len(X), len(self.classes_)))
          for k, (w, V) in enumerate(self.eig_):
              z = (X - self.mean_[k]) @ V  # coordinates along S's eigenvectors
              out[:, k] = self.log_prior_[k] - 0.5 * (
                  np.log(2 * np.pi * w).sum() + (z**2 / w).sum(axis=1)
              )
          return out
  ```

  `predict` is inherited. 98.51% train, 97.64% test, 98.54% family. In each class's eigenvector
  coordinates z the components are independent with variances w, so QDA is NB in each class's own
  rotated axes, and NB is QDA with every S diagonal. The loop is over the 10 classes, not samples.
  LDA: one pooled S, the sum of all classes' centred scatter g.T @ g divided by N, plus ε on the
  diagonal, shared by all classes: 94.30%.
