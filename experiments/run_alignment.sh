#!/usr/bin/env bash
# Fine-tune NeuroGPT (authors' encoder-only recipe) on BCI-IV-2a with or without
# per-session Euclidean Alignment. Resumable: folds that already have
# heldout_metrics.json are skipped.
#
# usage: PYTHON=/path/to/python experiments/run_alignment.sh ALIGN SEED FOLD [FOLD ...]
#   ALIGN = none | ea        SEED = integer        FOLD = 0..8
# env:   DATA (converted npz dir), OUT (results dir), STEPS (default 10000), EVAL_EVERY (default 500)
set -euo pipefail
ALIGN=$1; SEED=$2; shift 2
PY=${PYTHON:-python3}
HERE="$(cd "$(dirname "$0")" && pwd)"
DATA=${DATA:-"$HERE/../../neurogpt_data/bci2a_egg_npz/"}
OUT=${OUT:-"$HERE/../../neurogpt_data/runs"}
STEPS=${STEPS:-10000}
EVAL_EVERY=${EVAL_EVERY:-500}
mkdir -p "$OUT"
cd "$HERE/../scripts"
RUN="${ALIGN}_s${SEED}"
for F in "$@"; do
  if [ -f "$OUT/${RUN}-${F}/heldout_metrics.json" ]; then echo "skip $RUN fold $F"; continue; fi
  "$PY" ../src/alignment/train_aligned.py \
    --training-style=decoding --num-decoding-classes=4 \
    --training-steps="$STEPS" --eval_every_n_steps="$EVAL_EVERY" --log-every-n-steps=1000 \
    --num_chunks=2 --per-device-training-batch-size=32 --per-device-validation-batch-size=32 \
    --chunk_len=500 --chunk_ovlp=0 --ft-only-encoder=True --use-encoder=True \
    --num-encoder-layers=6 --num-hidden-layers=6 --embedding-dim=1024 --learning-rate=1e-4 \
    --pretrained-model=../pretrained_model/pytorch_model.bin \
    --dst-data-path="$DATA" --log-dir="$OUT" \
    --seed="$SEED" --align="$ALIGN" --run-name="$RUN" --fold_i="$F" \
    > "$OUT/${RUN}_fold${F}.log" 2>&1
  echo "done $RUN fold $F"
done
