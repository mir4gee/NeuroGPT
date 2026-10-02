# Session-to-session calibration (competition protocol, NOT the paper's cross-subject protocol)

For each subject: train on session T (288 labeled trials), test on session E (288 trials, recorded on another day).
Settings declared before any result: 1,000 steps, lr 5e-5, final-step accuracy; seeds 1-3 averaged per subject
(single-run accuracy, not an ensemble). All three columns are scored on the same data (session E).

- loso   : cross-subject model (trained on the other 8 subjects), no calibration
- within : authors' pre-trained Neuro-GPT fine-tuned on the subject's session T only
- calib  : the cross-subject model further fine-tuned on the subject's session T

| Subject | loso | within | calib |
|---|---|---|---|
| A01 | 0.663 | 0.657 | 0.810 |
| A02 | 0.431 | 0.441 | 0.568 |
| A03 | 0.778 | 0.715 | 0.911 |
| A04 | 0.514 | 0.573 | 0.723 |
| A05 | 0.499 | 0.602 | 0.325 |
| A06 | 0.472 | 0.554 | 0.627 |
| A07 | 0.649 | 0.645 | 0.825 |
| A08 | 0.712 | 0.727 | 0.852 |
| A09 | 0.572 | 0.685 | 0.754 |
| **Mean +/- std** | **0.588 +/- 0.119** | **0.622 +/- 0.091** | **0.711 +/- 0.181** |

- calib > loso on 8/9 subjects (Wilcoxon p = 0.055); calib > within on 8/9 (p = 0.129). With 9 subjects neither
  reaches p < 0.05.
- A05 is a consistent failure of calib (0.29 to 0.35 in all three seeds) while within reaches about 0.60 on the same
  data. It is kept in the mean; replacing it after seeing test results would be selection on the test set.
- Not comparable to the paper's 0.645, which never uses labels from the test subject.
