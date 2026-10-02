#!/usr/bin/env bash
# Session-to-session calibration on BCI-IV-2a (competition protocol, NOT the paper's cross-subject one).
# For subject s: train on its session T (288 labeled trials), test on its session E (288 trials, another day).
#   ARM=calib  : start from the cross-subject (LOSO) model of the same fold and seed (runs5k/none_s<seed>-<fold>)
#   ARM=within : start from the authors' pre-trained Neuro-GPT only
# Settings declared before any result: 1,000 steps, lr 5e-5, final-step accuracy.
# usage: PYTHON=... experiments/run_calibration.sh ARM SEED SUBJECT [SUBJECT ...]   (subjects 1..9)
set -euo pipefail
ARM=$1; SEED=$2; shift 2
PY=${PYTHON:-python3}
HERE="$(cd "$(dirname "$0")" && pwd)"
DATA=${DATA:-"$HERE/../../neurogpt_data/bci2a_egg_npz/"}
LOSO=${LOSO:-"$HERE/../../neurogpt_data/runs5k"}
OUT=${OUT:-"$HERE/../../neurogpt_data/calib"}
mkdir -p "$OUT"
cd "$HERE/../scripts"
for S in "$@"; do
  F=$((S - 1)); RUN="${ARM}_s${SEED}_subj${S}"
  if [ -f "$OUT/${RUN}-${F}/heldout_metrics.json" ]; then echo "skip $RUN"; continue; fi
  INIT=""
  if [ "$ARM" = calib ]; then INIT="--init-from=$LOSO/none_s${SEED}-${F}/model_final/model.safetensors"; fi
  "$PY" ../src/alignment/train_aligned.py \
    --training-style=decoding --num-decoding-classes=4 \
    --training-steps=1000 --eval_every_n_steps=100 --log-every-n-steps=100 \
    --num_chunks=2 --per-device-training-batch-size=32 --per-device-validation-batch-size=32 \
    --chunk_len=500 --chunk_ovlp=0 --ft-only-encoder=True --use-encoder=True \
    --num-encoder-layers=6 --num-hidden-layers=6 --embedding-dim=1024 --learning-rate=5e-5 \
    --pretrained-model=../pretrained_model/pytorch_model.bin \
    --dst-data-path="$DATA" --log-dir="$OUT" --seed="$SEED" --fold_i="$F" \
    --calib-subject="$S" $INIT --run-name="$RUN" > "$OUT/${RUN}.log" 2>&1
  echo "done $RUN"
done
