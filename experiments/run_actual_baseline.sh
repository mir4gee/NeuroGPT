#!/usr/bin/env bash
# The "actual" published NeuroGPT fine-tuning approach — unmodified vendored
# code, no novelty. This is scripts/finetune.sh's own configuration,
# reproduced here (with paths relative to experiments/, same depth as
# scripts/) so it sits next to run_novelty_prompt_tuning.sh for a direct,
# apples-to-apples comparison run on the same BCI-IV-2a fold.
#
# Requires: the released pretrained checkpoint at ../pretrained_model/pytorch_model.bin
# and BCI-IV-2a data at ../../bci2a_egg_npz/ (see PROMPT_TUNING.md).
set -euo pipefail
python3 ../src/train_gpt.py \
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
    --run-name='actual_baseline' \
    --ft-only-encoder='True' \
    --fold_i=0 \
    --num-encoder-layers=6 \
    --num-hidden-layers=6 \
    --learning-rate=1e-4 \
    --use-encoder='True' \
    --embedding-dim=1024 \
    --pretrained-model='../pretrained_model/pytorch_model.bin' \
    --dst-data-path="../../bci2a_egg_npz/"
