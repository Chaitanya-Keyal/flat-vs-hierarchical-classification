BITS F464 Machine Learning, Assignment 1: Flat vs. Hierarchical Classification From Scratch

Group
  Chaitanya Keyal   2023B4A70727H
  Vishesh Agarwal   2023B4A70611H
  TBD               TBD

Versions used
  Python 3.12.10, numpy 2.5.4, pandas 3.0.6. The code imports only numpy, pandas and the Python
  standard library, and prints exactly the same on Python 3.9 with numpy 1.26 and pandas 2.2.

How to run
  Put Frogs_MFCCs.csv in this folder, then run each script from here with one command:

    python part1_decision_tree.py      about 6 s   Single Decision Tree rows; sanity checks
                                                   (a)-(d); accuracy against max_depth
    python part2_bagging_boosting.py   about 45 s  Bagging and AdaBoost rows; against the
                                                   single tree; AdaBoost rounds; stumps
    python part3_random_forest.py      about 35 s  Random Forest rows; against the single tree
                                                   and bagging; agreement between the trees
    python part4_hierarchical.py       about 2 s   Hierarchical rows and Table 3; confusion
                                                   matrices; per-family accuracy; against the
                                                   flat tree
    python part5_naive_bayes.py        about 1 s   Gaussian Naive Bayes row; sanity check;
                                                   correlation matrix of the training features

  Each script prints its rows of results_matrix.pdf under "Results matrix rows", in %, with two
  decimals, then the other numbers its part asks for. common.py holds the shared code: loading,
  the handout's stratified_split, accuracy, confusion matrices, votes and the printed rows.

Choices the handout leaves open
  - Ties on gain: gains within 1e-12 of the best count as tied, and then the lower feature index,
    then the lower threshold, wins. A split needs a gain above 1e-12 to count as positive. This
    keeps floating-point rounding from deciding a tie.
  - Trees grow depth first, the left (x <= t) child first. The root has depth 0, so max_depth=1
    is a stump.
  - Every ensemble model creates its own np.random.RandomState(42) when it is fitted. Bagging and
    the forest draw each tree's bootstrap with rng.randint(0, N, size=N). The forest then draws
    np.sort(rng.choice(22, 4, replace=False)) at every node that searches for a split; nodes that
    stop first (pure, at max_depth, or too small) draw nothing.
  - Ties in leaves and in votes go to the smallest label in sorted order.
