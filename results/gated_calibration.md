# Validation-gated calibration (competition protocol), declared before results

Candidates train on a fixed 80% of the subject's session T, are validated on the other 20% of T, and predict
session E. Per subject the candidate with the best mean validation accuracy (3 seeds) is chosen (ties -> calib);
reported E accuracy is the 3-seed ensemble of the chosen candidate. 1,000 steps, lr 5e-5.

| Subject | calib val / E | within val / E | within_ea val / E | chosen | gated E |
|---|---|---|---|---|---|
| A01 | 0.804 / 0.792 | 0.609 / 0.594 | 0.661 / 0.656 | calib | 0.792 |
| A02 | 0.632 / 0.594 | 0.414 / 0.410 | 0.466 / 0.458 | calib | 0.594 |
| A03 | 0.862 / 0.896 | 0.735 / 0.691 | 0.586 / 0.660 | calib | 0.896 |
| A04 | 0.724 / 0.722 | 0.649 / 0.559 | 0.494 / 0.580 | calib | 0.722 |
| A05 | 0.701 / 0.288 | 0.655 / 0.549 | 0.569 / 0.660 | calib | 0.288 |
| A06 | 0.626 / 0.569 | 0.511 / 0.535 | 0.661 / 0.569 | within_ea | 0.569 |
| A07 | 0.868 / 0.830 | 0.701 / 0.642 | 0.764 / 0.646 | calib | 0.830 |
| A08 | 0.925 / 0.851 | 0.747 / 0.674 | 0.764 / 0.677 | calib | 0.851 |
| A09 | 0.856 / 0.719 | 0.804 / 0.615 | 0.667 / 0.649 | calib | 0.719 |

Session-E mean +/- std: calib 0.696 +/- 0.189, within 0.585 +/- 0.086, within_ea 0.617 +/- 0.070,
**gated 0.696 +/- 0.189** (gating picked calib for 8 of 9 subjects).

Finding: gating does not catch A05. Its calibrated model validates well inside session T (0.701) but fails on session
E (0.288), so the failure is a session-to-session shift that a same-session validation split cannot detect. Training on
80% of T lowers calib from 0.711 (full T, `calibration.md`) to 0.696. The 0.75 target was not reached.
