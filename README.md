# Flat vs. Hierarchical Classification: Trees, Ensembles and Naive Bayes

Species classification of frog calls (UCI Anuran Calls, 22 MFCCs per call) with a decision tree,
bagging, AdaBoost, a random forest, a family-then-species hierarchy and Gaussian naive Bayes, all
built from scratch in NumPy.

| Part | What it does |
|------|--------------|
| 1 | A decision tree split by Gini decrease or information gain, with sample weights; depth sweep |
| 2 | Bagging (25 trees) and multi-class AdaBoost (SAMME, 50 depth-3 trees), and AdaBoost with stumps |
| 3 | A random forest: bagging, plus 4 of the 22 features drawn afresh at every split |
| 4 | A router tree on Family, then one expert tree per family on Species, against the flat tree |
| 5 | Gaussian naive Bayes, and the correlation between MFCCs that it assumes away |

## 1. Layout

```
assignment.pdf               the handout
Frogs_MFCCs.csv              the UCI data, as downloaded
common.py                    loading, the handout's split, accuracy, confusion matrices, votes
part1_decision_tree.py       one script per part; each prints its rows of the results matrix
  ... part5_naive_bayes.py
README.txt                   the submission's readme: versions, commands, choices the handout leaves open
results_matrix.pdf           the filled-in matrix, and a page on what the numbers show
results/                     part1.txt ... part5.txt (what each script prints), results.json, numbers.tex
report/results_matrix.tex    the source of results_matrix.pdf; its numbers come from results/numbers.tex
tools/                       run_all.py (results/), build_pdf.py (results_matrix.pdf), package.py (the zip)
tests/                       the handout's sanity checks, brute-force references, hand-built cases
notebooks/demo.ipynb         the demo: each part's numbers and plots, by calling the part scripts
docs/notes/                  study notes: the maths, the code, the demo, likely viva questions
pyproject.toml, uv.lock      dependencies; .python-version pins Python 3.12
```

The repo root holds the files the submission zip is made of, so running a script here is the same
as running it in the unzipped submission. Each model lives in its part's script: `DecisionTree`
in Part 1, which Parts 2 to 4 import.

## 2. Environment

```bash
uv sync                      # numpy and pandas, plus pytest and ruff (the dev group)
uv sync --group notebook     # adds JupyterLab and Matplotlib, for the demo notebook
```

`uv.lock` pins every version. Tests, then formatting and lint:

```bash
uv run pytest
uv run ruff format . && uv run ruff check .
```

The submitted scripts import only numpy, pandas and the standard library, and keep to Python 3.9
syntax (ruff targets py39), so they run on whatever Python the grader has.

## 3. Data

`Frogs_MFCCs.csv` is the file in UCI's
[Anuran Calls (MFCCs)](https://archive.ics.uci.edu/dataset/406/anuran+calls+mfccs) download,
unchanged (sha256 `8b161e8d8535b871043cce7ecb7a3496e4b8baef377f1fe9bd4bc91b6effca38`): 7,195
syllables from 60 recordings, 22 MFCC columns (the header spells the first nine `MFCCs_ 1` to
`MFCCs_ 9`) and four label columns. Only the MFCCs are features; Family (4 classes) and Species
(10) are the labels, while Genus and RecordID are never used.

| Family | Species (rows) |
|---|---|
| Bufonidae | Rhinellagranulosa (68) |
| Dendrobatidae | Ameeregatrivittata (542) |
| Hylidae | HylaMinuta (310), HypsiboasCinerascens (472), HypsiboasCordobae (1121), OsteocephalusOophagus (114), ScinaxRuber (148) |
| Leptodactylidae | AdenomeraAndre (672), AdenomeraHylaedactylus (3478), LeptodactylusFuscus (270) |

The handout's `stratified_split` (seed 42, stratified on Species) gives 5,756 training and 1,439
test rows. 55 of the 60 recordings have syllables on both sides, so test accuracy is optimistic
for new recordings.

## 4. Running

```bash
uv run python part1_decision_tree.py     # one part, exactly as the submission runs it
uv run python -m tools.run_all           # all five into results/, about 90 s
uv run python -m tools.build_pdf         # results_matrix.pdf, with latexmk
```

Every random draw comes from a `RandomState(42)`: the handout's `stratified_split` makes one for
the split, and bagging and the forest each make their own when fitted (AdaBoost draws none). The
scripts print nothing that varies between runs, so a rerun reproduces `results/` byte for byte;
the PDF carries no build date, so it does too.

## 5. Results

Test accuracy (%), from `results/`; the full matrix, the hierarchy's failure analysis and a page
of discussion are in `results_matrix.pdf`.

| Model | Gini | Entropy |
|---|---|---|
| Single decision tree | 94.93 | 95.41 |
| Bagging (25 trees) | 96.66 | 96.87 |
| AdaBoost (SAMME, 50 depth-3 trees) | 95.34 | 94.16 |
| Random forest (50 trees, 4 features per split) | 97.78 | 97.64 |
| Hierarchical (router on Family, experts on Species) | 93.82 | 94.65 |

Gaussian naive Bayes has no splitting criterion: 90.34%.

## 6. Demo

```bash
uv run --group notebook jupyter lab notebooks/demo.ipynb
```

The models and metrics all come from the part scripts. The notebook prints the matrix rows and
adds plots (accuracy against depth, AdaBoost round by round, how alike the trees are, confusion
matrices, the MFCC correlations), with a "Try it" cell per part for live changes. It reloads the
scripts before every cell, so an edit to a script reaches the next cell run. It is committed
without outputs; running every cell takes about 90 s.

## 7. Submission

```bash
uv run python -m tools.package                  # dist/2023B4A70727H_A1.zip
uv run python -m tools.package ID [ID ...]      # one zip per student ID
```

Each zip holds the eight files the handout names, dated by the last commit that changed one of
them, so the same commit of a full clone always gives the same bytes. The script first rebuilds
the PDF in an empty folder and checks that it matches `results_matrix.pdf` byte for byte (skipped,
with a warning, where latexmk is missing; the scan of the PDF's text below needs pdftotext, from
poppler-utils, and is skipped the same way). It then unzips each zip into an empty folder, adds the CSV,
runs every part with a plain `python partN_*.py`, and checks the output against `results/` byte
for byte. It also checks the imports against an allow-list and that no prohibited library is
named anywhere. `--python "CMD"` runs the scripts under another interpreter instead, e.g.
`uv run --no-project --python 3.9 --with numpy==1.26.4 --with pandas==2.2.3 python`.
