#!/usr/bin/env bash
# Validation-gated calibration (competition protocol: labels from the subject's session T are used).
# Each candidate trains on 80% of session T, is validated on the fixed other 20% of T, and predicts session E.
# The choice between candidates is made later from the 20% T validation only (results/gated_eval.py).
#   ARM=calib     : start from the cross-subject (LOSO) model of the same fold/seed
#   ARM=within    : start from the authors' pre-trained Neuro-GPT
#   ARM=within_ea : as within, with per-session Euclidean Alignment
# Declared before any result: 1,000 steps, lr 5e-5, val fraction 0.2, final-step metrics.
# usage: PYTHON=... experiments/run_gated_calibration.sh ARM SEED SUBJECT [SUBJECT ...]
set -euo pipefail
ARM=$1; SEED=$2; shift 2
PY=${PYTHON:-python3}
HERE="$(cd "$(dirname "$0")" && pwd)"
DATA=${DATA:-"$HERE/../../neurogpt_data/bci2a_egg_npz/"}
LOSO=${LOSO:-"$HERE/../../neurogpt_data/runs5k"}
OUT=${OUT:-"$HERE/../../neurogpt_data/gated"}
mkdir -p "$OUT"
cd "$HERE/../scripts"
for S in "$@"; do
  F=$((S - 1)); RUN="${ARM}_s${SEED}_subj${S}"
  if [ -f "$OUT/${RUN}-${F}/test_metrics.json" ]; then echo "skip $RUN"; continue; fi
  EXTRA="--align=none"
  case "$ARM" in
    calib) EXTRA="--align=none --init-from=$LOSO/none_s${SEED}-${F}/model_final/model.safetensors" ;;
    within) EXTRA="--align=none" ;;
    within_ea) EXTRA="--align=ea" ;;
    *) echo "unknown arm $ARM"; exit 1 ;;
  esac
  "$PY" ../src/alignment/train_aligned.py \
    --training-style=decoding --num-decoding-classes=4 \
    --training-steps=1000 --eval_every_n_steps=100 --log-every-n-steps=100 \
    --num_chunks=2 --per-device-training-batch-size=32 --per-device-validation-batch-size=32 \
    --chunk_len=500 --chunk_ovlp=0 --ft-only-encoder=True --use-encoder=True \
    --num-encoder-layers=6 --num-hidden-layers=6 --embedding-dim=1024 --learning-rate=5e-5 \
    --pretrained-model=../pretrained_model/pytorch_model.bin \
    --dst-data-path="$DATA" --log-dir="$OUT" --seed="$SEED" --fold_i="$F" \
    --calib-subject="$S" --calib-val-frac=0.2 $EXTRA --run-name="$RUN" > "$OUT/${RUN}.log" 2>&1
  echo "done $RUN"
done
