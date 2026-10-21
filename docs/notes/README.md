# Study notes

The ideas behind each part, how the code implements them, and what to show and say in the demo
and viva. Not part of the submission. Any member may be asked about any part, so read all of them,
in this order:

- [assignment.md](assignment.md): what is asked, the rules that apply everywhere, where each item
  lives in the code and the output, and how we know the numbers are right
- [p1-decision-tree.md](p1-decision-tree.md): impurity and gain with sample weights, the
  vectorised split search, the tie rule, stopping; Parts 2 to 4 reuse this tree
- [p2-bagging-boosting.md](p2-bagging-boosting.md): bootstraps and the vote, multi-class AdaBoost
  (SAMME), and why stumps struggle with ten classes
- [p3-random-forest.md](p3-random-forest.md): a fresh feature subset at every split, and why
  weaker but less alike trees make a better vote
- [p4-hierarchical.md](p4-hierarchical.md): the router and the experts, the failure analysis,
  and why the hierarchy loses here
- [p5-naive-bayes.md](p5-naive-bayes.md): Gaussian naive Bayes in log space, correlated MFCCs,
  and the one feature with no variance
- [demo.md](demo.md): a 20-minute run sheet, part by part

Every part file opens with an In short, then follows the same order: the ideas and the maths,
the code, the results and what they mean, the demo, likely viva questions with answers, and the
code changes most likely to be asked for, with where to make them. Numbers are quoted from
`results/`; anything marked as an experiment is not part of the submission.
