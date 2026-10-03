# Novelty (class-conditional soft prompt-tuning): results

## Final: all 9 folds (local re-run, 2026-10-04, `experiments/run_prompt_tuning_local.sh`)
Same settings as below (readout=skip, frozen backbone in eval mode, lr 3e-4, 10,000 steps, seed 1234), held-out
accuracy at the final step:

| Fold | Subject | 4 prompt tokens/class | 0 tokens (ablation) |
|---|---|---|---|
| 0 | A01 | 0.351 | 0.352 |
| 1 | A02 | 0.328 | 0.328 |
| 2 | A03 | 0.401 | 0.375 |
| 3 | A04 | 0.380 | 0.344 |
| 4 | A05 | 0.368 | 0.351 |
| 5 | A06 | 0.359 | 0.358 |
| 6 | A07 | 0.406 | 0.424 |
| 7 | A08 | 0.401 | 0.380 |
| 8 | A09 | 0.358 | 0.356 |
| **Mean +/- std** | | **0.372 +/- 0.027** | **0.363 +/- 0.028** |

- Prompts vs no prompts: +0.009, higher on 6/9 subjects, paired t p = 0.133, Wilcoxon p = 0.117 -> **no significant effect**.
- Both are 0.23 below the fine-tuned baseline on the same machine (0.600), and below the paper's frozen linear probe (0.443).
- These supersede the partial Colab numbers below, which agree (folds 0-3: 0.359 vs 0.357).

---
# Earlier status (Colab, incomplete)

**Status: incomplete.** The Colab runtime became unresponsive at about 07:50 IST while the final 27 runs were
in progress, so the 9-fold novelty comparison was not finished. Numbers below are what had been logged before
that. Per-fold results and logs are written to Google Drive (`MyDrive/neurogpt_results/nov_*`) by the run itself.

## What was run

Frozen Neuro-GPT (encoder, embedder, GPT-2), trainable: prompt bank + pooler + decoding head (1,336,740 of
78,708,916 parameters, 1.70%). Same split, steps (10,000), batch (32) and metric as the baseline. Metric:
held-out accuracy at the final step on the unseen subject (leave-one-subject-out). Code: `src/prompt_tuning/`.

## Findings that are solid (single fold-0 diagnostics, all on training data or an eval-mode probe)

1. **Original spec (readout = last token of the frozen GPT) does not learn.** Training loss stays at ln 4 = 1.39
   (chance) and held-out accuracy at about 0.25. With the frozen backbone later kept in eval mode it is still
   1.36 at 2,000 steps (lr 3e-4 and 1e-3).
2. **Where information is lost (fold 0, logistic-regression probe, eval mode):**

   | Frozen features | Held-out accuracy | Mean cosine between samples |
   |---|---|---|
   | EEG-encoder output | 0.493 | 0.887 |
   | GPT last-token state | 0.356 | 0.972 |

   The class information is in the encoder output; the frozen GPT's last-token state is nearly constant.
3. **Implementation bug found and fixed:** a "frozen" backbone still applied dropout (p = 0.5 in the EEG encoder)
   and BatchNorm updates in training mode, so the head trained on corrupted features. The frozen backbone is now
   kept in eval mode while training (`PromptTunedModel.train()`, unit-tested).
4. **Fix kept inside the method:** `--readout=skip` adds a trainable linear skip from the frozen encoder features
   and a parameter-free BatchNorm on the pooler input. The default `--readout=gpt` is the original spec.
5. **Learning rate chosen from training loss only (no test subject involved).** Fold-0 training data, 2,000 steps:
   `skip` reaches 1.02 (lr 3e-4) and 1.01 (lr 1e-3); `gpt` stays at 1.36. Neither `skip` value met the declared bar
   of "under 1.0", so the smaller, 3e-4, was taken. All novelty runs use 3e-4; the baseline keeps the paper's 1e-4.

## Partial 9-fold results (final-step held-out accuracy; folds finished before the stall)

| Fold | Subject | Baseline (fine-tuned encoder) | skip, k=4 (prompts) | skip, k=0 (no prompts) |
|---|---|---|---|---|
| 0 | A01 | 0.648 | 0.349 | 0.373 |
| 1 | A02 | 0.438 | 0.326 | 0.319 |
| 2 | A03 | 0.743 | 0.399 | 0.384 |
| 3 | A04 | 0.507 | 0.363 | 0.352 |
| 4 | A05 | 0.562 | not finished | 0.337 |

Mean over folds 0 to 3 (the only folds finished for all three): baseline 0.584, skip k=4 **0.359**, skip k=0 **0.357**.
Paired difference k=4 minus k=0: **+0.002** over 4 folds, i.e. no measurable effect of the prompt tokens so far.
With n = 4 this is indicative only. Not finished: `skip_k4` folds 4 to 8, `skip_k0` folds 5 to 8, and all of `gpt_k4`.

## Reading

- Frozen-backbone + head (skip, k=0) reaches about 0.36, below the paper's frozen linear probe (0.443) and well below
  the fine-tuned baseline (0.607 over 9 folds). Our own logistic-regression probe on fold 0 reached 0.493 on the same
  frozen encoder features, so the trained head is also weaker than a properly fitted linear probe.
- Prompt tokens added nothing measurable on the folds finished. A plausible explanation is that the pre-trained GPT is
  small relative to what prompt tuning needs and discards the class information before the prompts can matter; this
  was not tested.
- Not yet done: the remaining folds, a from-scratch control, and a token-count sweep.
