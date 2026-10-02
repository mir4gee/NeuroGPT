# Seed ensemble of the plain recipe (5,000 steps), 9-fold leave-one-subject-out

Authors' encoder-only recipe, no alignment, no augmentation, 5,000 steps (on the validation plateau; see
`accuracy_selection.md`), seeds 1, 2, 3. Ensemble = mean of per-seed softmax probabilities. Metric: held-out
accuracy on the unseen subject.

| Fold | Subject | s1 | s2 | s3 | mean | ensemble |
|---|---|---|---|---|---|---|
| 0 | A01 | 0.658 | 0.660 | 0.639 | 0.652 | 0.653 |
| 1 | A02 | 0.446 | 0.443 | 0.458 | 0.449 | 0.446 |
| 2 | A03 | 0.736 | 0.773 | 0.743 | 0.751 | 0.760 |
| 3 | A04 | 0.507 | 0.507 | 0.514 | 0.509 | 0.524 |
| 4 | A05 | 0.543 | 0.523 | 0.509 | 0.525 | 0.530 |
| 5 | A06 | 0.500 | 0.474 | 0.500 | 0.491 | 0.500 |
| 6 | A07 | 0.660 | 0.648 | 0.660 | 0.656 | 0.660 |
| 7 | A08 | 0.740 | 0.743 | 0.734 | 0.739 | 0.743 |
| 8 | A09 | 0.609 | 0.627 | 0.618 | 0.618 | 0.620 |

- Single run mean 0.599 +/- 0.110; three-seed ensemble 0.604 +/- 0.110; gain +0.005, higher on 8 of 9 subjects
  (paired t p = 0.022, Wilcoxon p = 0.020). Consistent but small. A third seed adds nothing over two (0.604 both).
- Same as the 10,000-step control (0.600), so the shorter schedule costs nothing.
- Paper (single model): 0.645 +/- 0.104. The ensemble uses three models, so it would not be a like-for-like comparison
  even if it reached that number.
- Seeds share the pre-trained weights and the data, so their errors are highly correlated; that limits the gain.

Summary of levers tried (all 9 folds unless noted): augmentation hurts (validation only), per-session alignment +0.022
(not significant), seed ensembling +0.005.
