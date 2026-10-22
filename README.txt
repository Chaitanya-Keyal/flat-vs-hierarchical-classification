BITS F464 Machine Learning, Assignment 1: Flat vs. Hierarchical Classification From Scratch

Group
  Chaitanya Keyal   2023B4A70727H
  Vishesh Agarwal   2023B4A70611H
  Deeksha Thakur    2023B4A71257H

Versions used
  Python 3.12.10, numpy 2.5.4, pandas 3.0.6. The code imports only numpy, pandas and the Python
  standard library, and prints exactly the same on Python 3.9 with numpy 1.26 and pandas 2.2.

How to run
  Put Frogs_MFCCs.csv in this folder, then run each script from here with one command:

    python part1_decision_tree.py      about 5 s   Single Decision Tree rows; sanity checks
                                                   (a)-(d); accuracy against max_depth
    python part2_bagging_boosting.py   about 45 s  Bagging and AdaBoost rows; against the
                                                   single tree; AdaBoost rounds; stumps
    python part3_random_forest.py      about 35 s  Random Forest rows; against the single tree
                                                   and bagging; agreement between the trees
    python part4_hierarchical.py       about 3 s   Hierarchical rows and Table 3; confusion
                                                   matrices; per-family accuracy; against the
                                                   flat tree
    python part5_naive_bayes.py        about 1 s   Gaussian Naive Bayes row; sanity check;
                                                   correlation matrix of the training features

  Each script prints its rows of results_matrix.pdf under "Results matrix rows" (Part 4 prints
  Table 3 under "Failure analysis"), in %, with two decimals, along with the other numbers its
  part asks for. common.py holds the shared code: loading, the handout's stratified_split,
  accuracy, confusion matrices, votes and the printed rows.

Choices the handout leaves open
  - Ties on gain: gains within 1e-12 of the best count as tied, and then the lower feature index,
    then the lower threshold, wins. A split needs a gain above 1e-12 to count as positive. This
    keeps floating-point rounding from deciding a tie.
  - Trees grow depth first, the left (x <= t) child first. The root has depth 0, so max_depth=1
    is a stump.
  - Bagging and the forest each create their own np.random.RandomState(42) when fitted; AdaBoost
    draws no random numbers. Each tree's bootstrap, rng.randint(0, N, size=N), is drawn just
    before that tree is grown. While it grows, the forest draws
    np.sort(rng.choice(22, 4, replace=False)) at every node that searches for a split, in growth
    order; nodes that stop first (pure, at max_depth, or too small) draw nothing. So the stream
    runs: bootstrap 1, the draws of tree 1, bootstrap 2, the draws of tree 2, and so on.
  - Ties in leaves and in votes go to the smallest label in sorted order.
