# Accuracy-improvement attempts: selection on training subjects only

Protocol: for outer fold 0 (held-out subject A01), A01 is excluded entirely. Validation subjects are A02+A03,
training subjects A04..A09 (12 sessions). Same recipe, seed 1234, 10,000 steps. Nothing here used any outer
test subject. Decision rule declared in advance: take the best validation setting, but keep plain unless a setting
beats it by at least 0.01.

| Setting | Validation accuracy (final step) |
|---|---|
| K0 plain (authors' recipe) | **0.579** |
| K1 random-crop margin 125 samples | 0.570 |
| K2 random-crop margin 250 samples | 0.478 |
| K3 margin 250 + noise 0.1 + channel-dropout 0.1 + amplitude jitter 0.1 | 0.440 |
| K4 per-session Euclidean Alignment | 0.571 |

Outcome: none beats plain, so none was run on the outer folds. Augmentation hurts (the widened window pulls in the
pre-cue and post-imagery seconds, which is an untested explanation). Validation accuracy of plain training plateaus
by about step 3,000 to 3,500 and stays flat to 10,000 (0.58 to 0.60), so training length is not a lever.
Per-session alignment on all 9 outer folds gave +0.022, not significant (`alignment_9fold.md`).

Remaining lever tried next: averaging the predictions of several seeds (variance reduction), with a 5,000-step
schedule chosen from the plateau above.
