#!/usr/bin/env bash
# Prompt-tuning novelty (frozen backbone) on all 9 LOSO folds, locally. Same settings as the Colab runs in
# results/novelty_partial.md: readout=skip, lr 3e-4 (chosen on training loss only), 10,000 steps, batch 32.
# Resumable: folds with a finished log are skipped.
# usage: PYTHON=... experiments/run_prompt_tuning_local.sh TOKENS SEED FOLD [FOLD ...]   (TOKENS per class: 4 or 0)
set -euo pipefail
K=$1; SEED=$2; shift 2
PY=${PYTHON:-python3}
HERE="$(cd "$(dirname "$0")" && pwd)"
DATA=${DATA:-"$HERE/../../neurogpt_data/bci2a_egg_npz/"}
OUT=${OUT:-"$HERE/../../neurogpt_data/prompt"}
mkdir -p "$OUT"
cd "$HERE/../scripts"
RUN="skip_k${K}_s${SEED}"
for F in "$@"; do
  LOG="$OUT/${RUN}_fold${F}.log"
  if [ -f "$OUT/${RUN}-${F}/test_metrics.csv" ]; then echo "skip $RUN fold $F"; continue; fi
  "$PY" ../src/prompt_tuning/train_prompt_tuning.py \
    --training-style=decoding --num-decoding-classes=4 \
    --training-steps=10000 --eval_every_n_steps=500 --log-every-n-steps=1000 \
    --num_chunks=2 --chunk_len=500 --chunk_ovlp=0 \
    --per-device-training-batch-size=32 --per-device-validation-batch-size=32 \
    --use-encoder=True --num-encoder-layers=6 --num-hidden-layers=6 --embedding-dim=1024 \
    --learning-rate=3e-4 --freeze-backbone=True --readout=skip --num-prompt-tokens-per-condition="$K" \
    --pretrained-model=../pretrained_model/pytorch_model.bin \
    --dst-data-path="$DATA" --log-dir="$OUT" --seed="$SEED" --fold_i="$F" --run-name="$RUN" > "$LOG" 2>&1
  echo "done $RUN fold $F"
done
