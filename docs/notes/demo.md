# The demo: a 20-minute run sheet

## In short

Twenty minutes for three people. Spend about two minutes on the setup and the rules, two to
three minutes per part, and leave the rest for questions and live changes. Run the scripts
themselves, since they print exactly what the matrix says. Use the notebook for the plots and
the "Try it" cells, and keep `results_matrix.pdf` open beside them. Parts 2 and 3 take about 40
and 30 seconds, so start them first and talk while they run. Anyone may be asked anything, so
everyone should have read every part's notes.

## Before the slot

- Charge the laptop and close everything else; Parts 2 and 3 are CPU-bound.
- In the repo root, run `uv sync --group notebook`. Open `notebooks/demo.ipynb` in JupyterLab
  (`uv run --group notebook jupyter lab notebooks/demo.ipynb`) and run all its cells once, about
  90 s, so the plots are already there.
- Open `results_matrix.pdf`, the handout (`assignment.pdf`) and the five scripts in an editor.
- Have a terminal in the repo root with `uv run python part1_decision_tree.py` typed and ready.

## Run sheet

- **0:00-2:00: setup.** The layout: five scripts, `common.py`, the matrix PDF.
  Imports are numpy and pandas only. `unzip -l dist/2023B4A70727H_A1.zip` lists the eight files;
  mention that `tools/package.py` re-runs them in a clean folder (about 90 s, so not live).
- **2:00-6:00: Part 1.** Run the script: the sanity checks, the two rows, the depth
  sweep. In the code, `_best_split`: one sort per node, cumulative class weights, every cut at
  once, midpoints, the tie rule. The depth plot in the notebook.
- **6:00-9:00: Part 2.** Start `part2_bagging_boosting.py`; while it runs, walk through
  `Bagging.fit`, then `AdaBoostSAMME.fit` line by line against the handout's six steps. The
  round-by-round plot: the error climbing towards 0.9, stumps falling behind.
- **9:00-11:00: Part 3.** `RandomForest` is `Bagging` with `new_tree` overridden; the
  draw happens in `DecisionTree._best_split`. The agreement plot: weaker, less alike trees make
  a better vote.
- **11:00-14:00: Part 4.** Run the script: Table 3, the identity check, the
  confusion matrices, the per-family tables. In the code, `predict` groups rows by predicted
  family and calls each expert once. The router is the bottleneck. Its 17-row deficit against
  the flat tree is mostly in the two large families: Gini 9 Hylidae and 9 Leptodactylidae (one
  row back on Bufonidae), entropy 3 and 9, plus 4 in Dendrobatidae and 1 in Bufonidae.
- **14:00-16:00: Part 5.** Run the script: the sanity check, the row, the
  correlation matrix. The Ameeregatrivittata column of the confusion matrix, and why its MFCC 1
  variance is epsilon.
- **16:00-20:00: questions.** Questions and live changes.

## Live changes to have ready

Each one is a small edit or a notebook cell. The part notes give the code and where it goes.

- **Part 1:**
  - another criterion (misclassification error) in `impurity`;
  - `min_samples_leaf`;
  - printing the top of the tree from `tree.nodes`;
  - feature importances: the sum of n × gain per feature.
- **Part 2:**
  - a learning rate that scales alpha;
  - staged accuracy per round (in the notebook);
  - out-of-bag accuracy for bagging;
  - another T or another depth.
- **Part 3:**
  - m = 1 or 22 (the notebook's "Try it");
  - one feature subset per tree instead of per split, to show why the handout calls it wrong.
- **Part 4:**
  - a weaker router (the notebook's "Try it");
  - experts trained on the predicted family. Nothing changes here, because the router is 100%
    right on the training rows; explain why.
  - the forest as the router.
- **Part 5:**
  - dropping MFCC 1 (92.70% test, as an experiment);
  - uniform priors;
  - ddof=1;
  - `predict_proba` with log-sum-exp.

## If something goes wrong

- **The CSV is not found.** Run the scripts from the repo root, or from the folder that holds
  `Frogs_MFCCs.csv`.
- **A script seems slow.** Part 2 takes about 40 s and Part 3 about 30 s on this laptop. That is
  expected: they fit about 250 and 150 trees.
- **Asked for numbers.** Every number in the PDF is in `results/partN.txt`; a rerun reproduces
  them exactly.
