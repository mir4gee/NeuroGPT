# Hardware path: 8-bit Neuro-GPT on a microcontroller and an FPGA accelerator

Deployed model = the encoder-only Neuro-GPT recipe: EEG Conformer encoder + classifier head,
**717,973 parameters** (the head 2160->256->32->4 holds 561k of them; the encoder alone is 156k).

## 1. Quantization (post-training, no retraining)
BatchNorm folded into the spatial conv; weights symmetric per output channel; inputs of every conv, linear layer
and attention product quantized per tensor with ranges from 256 *training* trials (99.99th percentile).
LayerNorm, softmax, GELU, ELU, pooling and residual additions stay in higher precision.
**Fusion:** the temporal conv (1x25) and spatial conv (22x1) have no nonlinearity between them, so they are merged
offline into one 22x25 conv (exact in float): W[o,c,k] = sum_f W2[o,f,c] W1[f,k].

Mean accuracy over 9 subjects x 3 seeds (`src/hw/eval_quant.py`, `quant_results.json`); in brackets the fraction
of test trials with the same prediction as fp32:

| Precision | Cross-subject (LOSO) | Calibrated | Calibrated + alignment (exploratory) |
|---|---|---|---|
| fp32 | 0.599 | 0.711 | 0.754 |
| fp32, fused front end | 0.599 (1.000) | 0.711 (1.000) | 0.754 (1.000) |
| W8A8 | 0.599 (0.982) | 0.711 (0.988) | 0.754 (0.989) |
| **W8A8, fused** | **0.599 (0.987)** | **0.711 (0.992)** | **0.753 (0.992)** |
| W6A8 | 0.602 | 0.707 | 0.752 |
| W4A8 | 0.550 | 0.636 | 0.696 |
| W4A4 | 0.539 | 0.619 | 0.677 |

8-bit is lossless, 6-bit weights nearly so, 4-bit loses 5-8 points.

## 2. Portable C implementation (int8 MACs, float nonlinearities) - `hw/c/`
Front end computed one time step at a time (temporal/fused conv -> ELU -> running average-pool), so the
40x22x476 intermediate tensor is never stored. Verified against PyTorch W8A8 on all test trials of two models
(cross-subject A01: 576 trials; calibrated+aligned A05: 288 trials): **100% identical predictions**, accuracy
identical, max |log-prob difference| <= 0.11.

| | Unfused | **Fused** |
|---|---|---|
| int8 weights | 714 KB | **700 KB** |
| Integer MACs / trial | 62.0 M | **28.5 M** |
| Host latency, 1 core (Ryzen 7 7435HS, -O2) | 33 ms | **13 ms** |

Fused MAC breakdown per trial: front-end conv 20.94 M (73.5%), transformer 6.92 M (24.3%), head 0.56 M (2.0%),
projection 0.09 M (0.3%).
Footprint (x86 build, ARM will differ slightly): code 13 KB, constants 724 KB (flash), static RAM 20 KB,
largest stack frame 26 KB, so about 48 KB RAM. Fits an STM32F4-class MCU (1 MB flash, 192 KB RAM).
MCU latency is **not measured** (no board/toolchain here). Estimate for a 168 MHz Cortex-M4: ~85 ms/trial at
2 MAC/cycle (SIMD, CMSIS-NN-style kernels) to ~0.7 s/trial at 4 cycles/MAC (plain C); either is below the 4-s trial.

## 3. FPGA accelerator for the fused front end - `hw/rtl/frontend_accel.v`
40 parallel int8 MAC lanes (one per filter) share one input sample per cycle; weights in ROM; 550 cycles per time step.
- **Bit-exact in simulation** (Icarus Verilog): 0 mismatches over all 19,040 int32 outputs of one chunk
  (vectors from `src/hw/gen_rtl_vectors.py`), **261,845 cycles per 2-s chunk**.
- **Synthesis** (yosys 0.33 `synth_xilinx -family xc7`, before place-and-route; `fpga_yosys_stat.txt`):

| Resource | Used | Zynq-7020 | Artix-7 35T |
|---|---|---|---|
| LUT | 2,049 | 3.9% | 9.9% |
| Flip-flop | 2,643 | 2.5% | 6.4% |
| DSP48E1 | 43 | 19.5% | 48% |
| RAMB36 | 31 | 22% | 62% |

- Not verified: clock frequency (needs Vivado place-and-route). **At an assumed 100 MHz**: 2.6 ms per chunk, 5.2 ms per
  trial for 73.5% of the work; the transformer and head (7.6 M MACs) would run on the Zynq's ARM cores.

## 4. Lightweight hardware baseline: Hersche, Benini & Rahimi, AICAS 2020
Their code (github.com/MHersche/HDembedding-BCI) unmodified except our data loader and a scikit-learn compatibility
attribute (`src/hw/run_hd_baseline.py`). Protocol: train session T, test session E, artifact trials removed;
multiscale Riemannian features (43 bands, 10,879 features), HD dimension 10,000, sparsity 0.9.

| Subject | binarized SVM | binarized LDA | HD binary | **ours calib, W8A8 fused** | ours calib + alignment, W8A8 fused (exploratory) |
|---|---|---|---|---|---|
| A01 | 0.776 | 0.801 | 0.886 | 0.810 | 0.828 |
| A02 | 0.428 | 0.424 | 0.537 | 0.566 | 0.583 |
| A03 | 0.678 | 0.714 | 0.784 | 0.911 | 0.904 |
| A04 | 0.583 | 0.579 | 0.605 | 0.723 | 0.740 |
| A05 | 0.420 | 0.409 | 0.634 | 0.328 | 0.719 |
| A06 | 0.479 | 0.493 | 0.544 | 0.624 | 0.630 |
| A07 | 0.700 | 0.737 | 0.805 | 0.830 | 0.825 |
| A08 | 0.734 | 0.764 | 0.767 | 0.850 | 0.793 |
| A09 | 0.689 | 0.742 | 0.799 | 0.759 | 0.758 |
| **Mean +/- std** | 0.610 +/- 0.137 | 0.629 +/- 0.154 | **0.707 +/- 0.128** | **0.711 +/- 0.181** | **0.753 +/- 0.100** |

- Ours (calibrated, 8-bit) vs HD: +0.004, 6/9 subjects, Wilcoxon p = 0.50 -> **equal**.
- Ours (calibrated + alignment, 8-bit, exploratory) vs HD: +0.046, 7/9, p = 0.098 -> not significant.
- Caveats: the baseline drops artifact trials and uses only the subject's own data; our test set is all 288 session-E
  trials and our model also learned from the other 8 subjects. Not an identical comparison.
- Model size: the binarized SVM is 4 x 10,879 bits = 5.4 KB and HD class vectors are tiny, but their front end
  (43 band-pass filters, 22x22 covariances, matrix logarithms per band) is float-heavy and was not costed here.

## 5. Quantization-aware training (QAT) for 4-bit - `src/hw/qat.py`, `qat_results.json`
Fine-tune each saved model with fake quantization in the loop (straight-through estimator) on its own training data.
Fixed in advance: fused front end, lr 1e-5, AdamW without weight decay, batch 32, dropout on, activation ranges from
256 training trials frozen; 1,000 steps (LOSO) / 300 steps (calibrated). Seed 1, 9 subjects per row.

| Setting | fp32 | W4A8 PTQ | **W4A8 QAT** | W4A4 PTQ | **W4A4 QAT** |
|---|---|---|---|---|---|
| Cross-subject | 0.600 | 0.579 | **0.595** | 0.558 | **0.588** |
| Calibrated | 0.708 | 0.655 | **0.707** | 0.633 | **0.709** |
| Calibrated + alignment | 0.756 | 0.727 | **0.745** | 0.708 | **0.734** |

QAT vs PTQ, Wilcoxon over subjects: LOSO W4A8 +0.016 (p=0.098), W4A4 +0.030 (8/9, p=0.008); calibrated W4A8 +0.052
(8/9, p=0.008), W4A4 +0.076 (9/9, p=0.004); aligned W4A8 +0.018 (p=0.148), W4A4 +0.026 (p=0.078).
4-bit weights: ~350 KB (~2.9 Mbit), versus 140 x 36 Kbit = 5.0 Mbit of block RAM on a Zynq-7020, so the whole model can stay on chip (not implemented here).
