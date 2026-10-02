#!/usr/bin/env bash
# Choose augmentation settings WITHOUT touching any outer test subject.
# Outer fold 0 holds out subject A01, so here A01 is excluded from training and from
# validation entirely; A02+A03 are the validation subjects and A04..A09 train.
# usage: PYTHON=... experiments/run_inner_selection.sh NAME MARGIN NOISE CHDROP AMP [ALIGN]
set -euo pipefail
NAME=$1; MARGIN=$2; NOISE=$3; CHDROP=$4; AMP=$5; ALIGN=${6:-none}
PY=${PYTHON:-python3}
HERE="$(cd "$(dirname "$0")" && pwd)"
DATA=${DATA:-"$HERE/../../neurogpt_data/bci2a_egg_npz/"}
OUT=${OUT:-"$HERE/../../neurogpt_data/inner"}
mkdir -p "$OUT"
cd "$HERE/../scripts"
"$PY" ../src/alignment/train_aligned.py \
  --training-style=decoding --num-decoding-classes=4 \
  --training-steps=10000 --eval_every_n_steps=500 --log-every-n-steps=1000 \
  --num_chunks=2 --per-device-training-batch-size=32 --per-device-validation-batch-size=32 \
  --chunk_len=500 --chunk_ovlp=0 --ft-only-encoder=True --use-encoder=True \
  --num-encoder-layers=6 --num-hidden-layers=6 --embedding-dim=1024 --learning-rate=1e-4 \
  --pretrained-model=../pretrained_model/pytorch_model.bin \
  --dst-data-path="$DATA" --log-dir="$OUT" --seed=1234 --fold_i=0 \
  --val-subjects=2,3 --exclude-subjects=1 \
  --align="$ALIGN" --margin="$MARGIN" --noise-std="$NOISE" --ch-drop="$CHDROP" --amp-scale="$AMP" \
  --run-name="$NAME" > "$OUT/${NAME}.log" 2>&1
echo "done $NAME"
