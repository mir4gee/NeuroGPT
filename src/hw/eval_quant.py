#!/usr/bin/env python3
"""
Accuracy of the deployed model under post-training quantization, for every saved model:
  loso      : cross-subject models (runs5k/none_s<seed>-<fold>), tested on the held-out subject (both sessions)
  calib     : calibrated models    (calib/calib_s<seed>_subj<S>-<fold>), tested on session E
  calib_ea  : exploratory aligned calibrated models (calib/calib_ea_s...), aligned data, session E
Activation ranges are calibrated on 256 trials of each model's own TRAINING data (never the test data).
usage: python src/hw/eval_quant.py DATA_DIR RUNS_ROOT OUT_JSON
"""
import json
import os
import sys

import numpy as np
import torch

_SRC = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, _SRC)

import train_gpt  # noqa: E402
from alignment.ea_dataset import AlignedMotorImageryDataset  # noqa: E402
from hw.quantize import calibrate, quantize  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

DATA, ROOT, OUT = (os.path.abspath(p) for p in sys.argv[1:4])
CONFIGS = {'fp32': (None, None, False), 'fused_fp32': (None, None, True), 'W8A8': (8, 8, False),
           'fused_W8A8': (8, 8, True), 'W6A8': (6, 8, False), 'W4A8': (4, 8, False), 'W4A4': (4, 4, False)}
SEEDS = [1, 2, 3]
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)
os.chdir(os.path.join(os.path.dirname(_SRC), 'scripts'))  # dataset loads ../inputs/tMatrix_value.npy

FILES = sorted(os.listdir(DATA))[:18]
KW = dict(sample_keys=['inputs', 'attention_mask'], chunk_len=500, num_chunks=2, ovlp=0, root_path=DATA + '/', gpt_only=False)


def make_encoder():
    from encoder.conformer_braindecode import EEGConformer
    return EEGConformer(n_outputs=4, n_chans=22, n_times=500, ch_pos=None, is_decoding_mode=True)


def load(path):
    enc = make_encoder()
    sd = load_file(path)
    enc.load_state_dict({k[len('encoder.'):]: v for k, v in sd.items() if k.startswith('encoder.')})
    return enc.to(dev).eval()


def tensors(ds):
    x = torch.stack([ds[i]['inputs'] for i in range(len(ds))])
    y = torch.as_tensor(np.asarray(ds.labels)).long()
    return x, y


@torch.no_grad()
def accuracy(model, x, y):
    pred = torch.cat([model(x[i:i + 256].to(dev)).argmax(1).cpu() for i in range(0, len(x), 256)])
    return float((pred == y).float().mean()), pred


def calib_batches(x):
    idx = torch.randperm(len(x), generator=torch.Generator().manual_seed(0))[:256]
    return [x[idx[i:i + 64]].to(dev) for i in range(0, 256, 64)]


def run(model_path, train_x, test_x, test_y):
    base = load(model_path)
    res, ref = {}, None
    for name, (wb, ab, fuse) in CONFIGS.items():
        m = quantize(base, wb, ab, fuse).to(dev)
        if ab is not None:
            calibrate(m, calib_batches(train_x))
        acc, pred = accuracy(m, test_x, test_y)
        if ref is None:
            ref = pred
        res[name] = {'acc': acc, 'agree_fp32': float((pred == ref).float().mean())}
    return res


results = {}
# cross-subject (LOSO)
for f in range(9):
    tr = [x for i, x in enumerate(FILES) if i // 2 != f]
    te = FILES[2 * f: 2 * f + 2]
    trx, _ = tensors(AlignedMotorImageryDataset(tr, align='none', **KW))
    tex, tey = tensors(AlignedMotorImageryDataset(te, align='none', **KW))
    for s in SEEDS:
        key = f'loso/s{s}/A0{f + 1}'
        results[key] = run(f'{ROOT}/runs5k/none_s{s}-{f}/model_final/model.safetensors', trx, tex, tey)
        print(key, {k: round(v['acc'], 3) for k, v in results[key].items()}, flush=True)
    del trx
# calibration (session T -> E), plain and aligned
for arm, align in (('calib', 'none'), ('calib_ea', 'ea')):
    for S in range(1, 10):
        f = S - 1
        trx, _ = tensors(AlignedMotorImageryDataset([FILES[2 * f + 1]], align=align, **KW))
        tex, tey = tensors(AlignedMotorImageryDataset([FILES[2 * f]], align=align, **KW))
        for s in SEEDS:
            key = f'{arm}/s{s}/A0{S}'
            results[key] = run(f'{ROOT}/calib/{arm}_s{s}_subj{S}-{f}/model_final/model.safetensors', trx, tex, tey)
            print(key, {k: round(v['acc'], 3) for k, v in results[key].items()}, flush=True)

json.dump(results, open(OUT, 'w'), indent=1)
print('\nsummary (mean accuracy over subjects and seeds; agreement with fp32 predictions):')
for arm in ('loso', 'calib', 'calib_ea'):
    keys = [k for k in results if k.startswith(arm + '/')]
    print(arm, '  '.join(f"{c}: {np.mean([results[k][c]['acc'] for k in keys]):.3f} "
                         f"({np.mean([results[k][c]['agree_fp32'] for k in keys]):.3f})" for c in CONFIGS))
