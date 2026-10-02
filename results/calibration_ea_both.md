# Exploratory: per-session alignment in both stages + calibration (competition protocol)

**Exploratory.** This variant was motivated by test-session results (alignment did best for A05 on session E in
`gated_calibration.md`), so it is partly fitted to the test data and must not be presented as a clean, pre-declared result.

Setup: cross-subject (LOSO) models trained with per-session Euclidean alignment (5,000 steps, seeds 1-3), then
calibrated on the subject's session T with alignment (1,000 steps, lr 5e-5, same as `calibration.md`), tested on
session E. All columns on the same session-E trials; mean of 3 seeds per subject.

| Subject | LOSO | LOSO + EA | calib | **calib + EA (both stages)** |
|---|---|---|---|---|
| A01 | 0.663 | 0.725 | 0.810 | 0.831 |
| A02 | 0.431 | 0.400 | 0.568 | 0.583 |
| A03 | 0.778 | 0.788 | 0.911 | 0.904 |
| A04 | 0.514 | 0.551 | 0.723 | 0.737 |
| A05 | 0.499 | 0.574 | 0.325 | 0.717 |
| A06 | 0.472 | 0.462 | 0.627 | 0.628 |
| A07 | 0.649 | 0.699 | 0.825 | 0.823 |
| A08 | 0.712 | 0.714 | 0.852 | 0.798 |
| A09 | 0.572 | 0.640 | 0.754 | 0.761 |
| **Mean +/- std** | 0.588 +/- 0.119 | 0.617 +/- 0.130 | 0.711 +/- 0.181 | **0.754 +/- 0.101** |

- vs no calibration (LOSO): +0.166, better on 9/9 subjects, Wilcoxon p = 0.004.
- vs plain calibration: +0.043, better on 6/9, paired t p = 0.36, Wilcoxon p = 0.31 (not significant). The gain is almost
  entirely A05 (0.325 -> 0.717, all three seeds 0.70-0.75); A08 drops 0.054.
- Interpretation (hypothesis): aligning each session removes the day-to-day shift that made A05's calibrated model fail
  on session E; it also lowers the spread across subjects (std 0.181 -> 0.101).
- Not comparable to the paper's cross-subject 0.645 (uses the target user's session-T labels).
