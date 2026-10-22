# The assignment: what is asked and where it lives

## In short

Five parts on one dataset, the UCI Anuran Calls: 7,195 frog-call syllables, 22 MFCCs each. They
are a decision tree (Part 1), bagging and AdaBoost built on it (Part 2), a random forest
(Part 3), a router-and-experts hierarchy (Part 4) and Gaussian naive Bayes (Part 5), all from
scratch in numpy. Flat models predict the species directly; the hierarchy predicts the family
first. The handout fixes every hyperparameter, the split and the seeds, so there is nothing to
tune: the marks come from implementing the spec exactly and explaining it. We submit one zip,
`2023B4A70727H_A1.zip`, holding the five scripts, `common.py`, `README.txt` and
`results_matrix.pdf`. It is due at 23:55 on 26 October, and each group then has a 20-minute
demo and viva on 27-29 October.

## Rules that apply everywhere

- **Prohibited:** every machine-learning or deep-learning library and every library helper for
  splits, metrics or confusion matrices. A prohibited import earns zero for that part.
  **Allowed:** Python 3, numpy, pandas and the standard library, plus matplotlib for optional
  plots. We use matplotlib only in the notebook, which is not submitted.
- **Features:** only the 22 MFCC columns. Family, Genus, Species and RecordID are never features.
  `common.load` keeps the columns that start with `MFCCs_`, in file order; the header spells the
  first nine `MFCCs_ 1` ... `MFCCs_ 9`, with a space. So feature j is MFCC j+1, which matters for
  the tie rule.
- **The split:** the handout's `stratified_split`, copied verbatim into `common.py` and run on the
  Species strings. Seed 42, 5,756 train and 1,439 test rows. Codes from `np.unique` give the same
  split; `pd.factorize` codes would not, since they number the species in a different order. The
  test set is used only for the final numbers: the species-to-family map, the NB smoothing and the
  correlations all come from the training rows.
- **Fixed settings (the handout's Table 1):**

  | Model | Settings |
  |---|---|
  | All trees | `min_samples_split=2` |
  | Single tree | unconstrained |
  | Bagging | T=25 unconstrained trees, all 22 features at every split |
  | AdaBoost | T=50 rounds of depth-3 trees, SAMME |
  | Random forest | T=50 unconstrained trees, 4 features drawn afresh at every split |
  | Hierarchy | unconstrained router and experts, all with the same criterion |
  | Naive Bayes | variance smoothing 1e-9 times the largest training variance |

- **Randomness:** one fresh `np.random.RandomState(42)` per ensemble model, used for all of its
  bootstraps and feature draws. Bagging and the forest create theirs at the start of `fit`;
  AdaBoost draws no random numbers. The stream runs: bootstrap 1, the forest's feature draws in
  tree 1, bootstrap 2, and so on (README.txt says so).
- **Ties:** on gain, the lower feature index wins, then the lower threshold. In leaves and votes,
  the smallest label in sorted order wins. Candidate thresholds are midpoints between consecutive
  distinct values.
- **Leakage caveat:** 55 of the 60 recordings have syllables in both train and test, so every
  accuracy is optimistic for new recordings. The handout calls this a property of the split, not
  a bug.

## The data

| Family | Species | Rows | Test rows |
|---|---|---:|---:|
| Bufonidae | Rhinellagranulosa | 68 | 14 |
| Dendrobatidae | Ameeregatrivittata | 542 | 108 |
| Hylidae | HylaMinuta, HypsiboasCinerascens, HypsiboasCordobae, OsteocephalusOophagus, ScinaxRuber | 2,165 | 433 |
| Leptodactylidae | AdenomeraAndre, AdenomeraHylaedactylus, LeptodactylusFuscus | 4,420 | 884 |

- Two families have one species each, so their experts are a single leaf.
- The classes are very unbalanced: AdenomeraHylaedactylus alone has 3,478 rows.
- Every row was scaled by its largest absolute MFCC, so the values lie in [-1, 1]. MFCC 1 is
  exactly 1.0 in 6,947 of the 7,195 rows.
- There are no missing values and no duplicate feature rows, which is why unconstrained trees can
  reach 100% on the training set.

## Where each item lives

The "Results matrix rows" block in each script's output holds its rows of Table 2; Part 4 prints
Table 3 under "Failure analysis". `results/partN.txt` is that output, verbatim.

| Handout item | Code | Printed as |
|---|---|---|
| P1 `DecisionTree(criterion, max_depth, min_samples_split, max_features, rng)` | `part1_decision_tree.py` | |
| P1 sanity checks (a)-(d) | `sanity_checks` | "Sanity checks" |
| P1.1 flat tree, Gini and entropy, train and test | `main` | rows "1. Single Decision Tree" |
| P1.2 derived family accuracy | `common.Data.scores`, `families` | the Family % column |
| P1.3 accuracy against max_depth | `main` | "Accuracy against max_depth" |
| P2 bagging, T=25 | `part2_bagging_boosting.py`, `Bagging` | rows "2a. Bagging" |
| P2 AdaBoost SAMME, T=50, depth 3 | `AdaBoostSAMME` | rows "2b. AdaBoost" |
| P2.2 against the single tree | `main` | "Test % against the single tree" |
| P2.3 stumps against depth 3 | `main` | "Decision stumps" |
| P3 random forest, m=4 per split | `part3_random_forest.py`, `RandomForest` | rows "3. Random Forest" |
| P3.2 against the tree and bagging | `main` | "Test % against the single tree ..." |
| P3.3 tree agreement (extra) | `agreement` | "Extra (not graded)" |
| P4 router, experts, `HierarchicalClassifier` | `part4_hierarchical.py` | rows "4. Hierarchical ...", Table 3 |
| P4 accuracies, oracle, p_A, p_B, matrices, per family, identity | `analyse`, `details` | the per-criterion blocks |
| P4 against the flat tree | `main` | "Against the flat tree of Part 1" |
| P5 `GaussianNB(var_smoothing=1e-9)` and its check | `part5_naive_bayes.py` | "Sanity check", row "5. ..." |
| P5.2 correlation matrix of the training features | `print_correlations` | "Correlation matrix ..." |
| README.txt | `README.txt` | |
| results_matrix.pdf | `report/results_matrix.tex`, built by `tools/build_pdf.py` | |

## Results at a glance

Test accuracy (%), Gini / entropy:

| Model | Species | Family |
|---|---|---|
| Single tree | 94.93 / 95.41 | 96.25 / 96.80 |
| Bagging | 96.66 / 96.87 | 97.50 / 97.50 |
| AdaBoost | 95.34 / 94.16 | 96.73 / 95.00 |
| Random forest | 97.78 / 97.64 | 98.40 / 98.19 |
| Hierarchy | 93.82 / 94.65 | router 95.07 / 95.62 |
| Naive Bayes | 90.34 | 91.94 |

The forest is best, and the hierarchy loses to the flat tree because the router is the
bottleneck. Naive Bayes is last: the MFCCs are far from independent within a species. Page 2 of
`results_matrix.pdf` gives the reasons, with numbers.

A difference in points, here and in the PDF, subtracts the two printed values, so it matches the
tables (`common.points`). Counted in test rows it is exact: bagging's 1.73 points over the single
tree is 25 of the 1439 rows, 1.737 points.

## How we know the numbers are right

- **234 tests** (`uv run pytest`):
  - the handout's sanity checks;
  - a slow brute-force split search that must pick the same feature, threshold and gain as the
    vectorised one on 40 random problems full of ties, each with both criteria, and a slow whole
    tree compared node by node;
  - the same tree whatever the scale of the weights, and integer weights equal to duplicated rows;
  - SAMME with two classes against classic AdaBoost, and with four classes its 1 − 1/K limit,
    its exact α and the clipped ε of a perfect learner;
  - the forest's single random stream, replayed tree by tree, and its draws per split;
  - the routing order, one call per expert, experts on their true family, the oracle and the
    failure identity;
  - naive Bayes against a plain loop, with ε checked exactly;
  - and more besides.
- **Mutation checks**, run outside the repo: the code was broken on purpose in 14 ways the
  handout warns about (binary AdaBoost's stopping rule, one feature subset per tree, experts on
  the predicted family, no ε, ...), and the tests fail on every one.
- **An independent prototype**, written separately from the handout alone and thrown away, gave
  the same numbers in every cell of the matrix.
- **A scikit-learn cross-check**, run outside the repo and never part of the submission:
  - naive Bayes and the confusion matrices are identical;
  - AdaBoost has the same weighted error and alpha in every round and the same test accuracy;
  - the trees choose the same root splits, and our test accuracies sit inside the spread sklearn
    gets across random seeds, since sklearn breaks ties at random and the handout's rule is
    deterministic.
- **A clean-room run of the zip** (`uv run python -m tools.package`): unzipped into an empty
  folder with only the CSV added, each script prints byte for byte what `results/` holds. That
  held on Python 3.12 with numpy 2.5 and pandas 3.0, and on Python 3.9 with numpy 1.26 and
  pandas 2.2.

## Logistics

- **Submission:** one zip named `<StudentID>_A1.zip`, ours `2023B4A70727H_A1.zip`. Build it with
  `uv run python -m tools.package`, or `uv run python -m tools.package ID1 ID2 ID3` for one zip
  per member. The CSV is not in the zip; the grader puts `Frogs_MFCCs.csv` next to the scripts.
- **Demo and viva:** 20 minutes per group on 27-29 October, at a time agreed with the TAs
  (f20231091@hyderabad.bits-pilani.ac.in, f20230390@hyderabad.bits-pilani.ac.in). See
  [demo.md](demo.md).
- **Similarity check:** submissions are checked for similarity and the code must be our own.
  Every member should be able to explain any line, so read all five part notes.
