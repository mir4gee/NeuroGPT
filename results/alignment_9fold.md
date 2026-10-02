# Per-session Euclidean Alignment on NeuroGPT: 9-fold leave-one-subject-out, seed 1234

Authors' encoder-only recipe (`--ft-only-encoder=True`, 10,000 steps, batch 32, lr 1e-4, pre-trained weights),
BCI-IV-2a. Both arms were run on the same machine (RTX 3050, torch 2.6, transformers 4.38.1) with the same seed
and split, so the comparison is controlled. `none` = vendored input pipeline, `ea` = each recording session's
trials aligned so their mean covariance is the identity (`src/alignment/`). Metric: held-out accuracy at the
final step on the unseen subject.

| Fold | Subject | none | ea | ea - none |
|---|---|---|---|---|
| 0 | A01 | 0.634 | 0.733 | +0.099 |
| 1 | A02 | 0.434 | 0.424 | -0.010 |
| 2 | A03 | 0.753 | 0.769 | +0.016 |
| 3 | A04 | 0.517 | 0.510 | -0.007 |
| 4 | A05 | 0.557 | 0.540 | -0.017 |
| 5 | A06 | 0.483 | 0.503 | +0.020 |
| 6 | A07 | 0.649 | 0.705 | +0.056 |
| 7 | A08 | 0.760 | 0.750 | -0.010 |
| 8 | A09 | 0.616 | 0.668 | +0.052 |
| **Mean +/- std** | | **0.600 +/- 0.113** | **0.622 +/- 0.128** | **+0.022** |

- Aligned is higher on 5 of 9 subjects. Paired t-test p = 0.131, Wilcoxon p = 0.191: **not significant**.
- Two subjects (A01, A07) plus A09 carry the gain (+0.099, +0.056, +0.052); the other six are within +/-0.02.
- Control on this machine (0.600) is consistent with the Colab baseline (0.607, `baseline_9fold.md`).
- For reference, the paper reports 0.645 +/- 0.104 for the same recipe. Neither arm reaches it.

## Caveats
- One seed, 9 subjects. The spread across subjects (about 0.11 to 0.13) is much larger than the +0.022 mean difference.
- Alignment of the held-out subject uses that subject's own unlabeled trials (transductive); the paper's protocol does not.
- Euclidean Alignment is an existing technique (He & Wu, IEEE TBME 2020, cited from memory); the contribution here is
  applying it to a pre-trained EEG foundation model under a controlled protocol.
- No from-scratch control was run, so this does not test the value of pre-training.
