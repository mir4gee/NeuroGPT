#!/usr/bin/env bash
# Our novelty: class-conditional soft prompt-tuning (see PROMPT_TUNING.md).
# Same BCI-IV-2a fold, same base hyperparameters as run_actual_baseline.sh,
# so the two runs are directly comparable — the only real difference is the
# adaptation mechanism (learned soft-prompt menu + frozen backbone vs. the
# vendor's own ft-only-encoder fine-tuning).
#
# Requires: the released pretrained checkpoint at ../pretrained_model/pytorch_model.bin
# and BCI-IV-2a data at ../../bci2a_egg_npz/ (see PROMPT_TUNING.md).
#
# --num-prompt-tokens-per-condition=0 reduces this to the frozen linear-probe
# baseline (no prompt tokens at all) for a free ablation.
set -euo pipefail
python3 ../src/prompt_tuning/train_prompt_tuning.py \
    --training-style='decoding' \
    --num-decoding-classes=4 \
    --training-steps=10000 \
    --eval_every_n_steps=500 \
    --log-every-n-steps=1000 \
    --num_chunks=2 \
    --per-device-training-batch-size=32 \
    --per-device-validation-batch-size=32 \
    --chunk_len=500 \
    --chunk_ovlp=0 \
    --run-name='novelty_prompt_tuning' \
    --fold_i=0 \
    --num-encoder-layers=6 \
    --num-hidden-layers=6 \
    --learning-rate=1e-4 \
    --use-encoder='True' \
    --embedding-dim=1024 \
    --pretrained-model='../pretrained_model/pytorch_model.bin' \
    --dst-data-path="../../bci2a_egg_npz/" \
    --num-prompt-tokens-per-condition=4 \
    --freeze-backbone='True'
