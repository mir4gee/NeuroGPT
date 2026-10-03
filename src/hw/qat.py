#!/usr/bin/env python3
"""
Quantization-aware training (QAT) to recover 4-bit accuracy. Starts from every saved model, fine-tunes it with
fake quantization in the loop (straight-through estimator) on that model's OWN training data, and reports
held-out accuracy at the final step next to post-training quantization (PTQ) of the same model.

Settings fixed before any result: fused front end; W4A8 and W4A4; activation ranges calibrated once on 256
training trials and then frozen; AdamW lr 1e-5, no weight decay, batch 32, dropout active;
1,000 steps for cross-subject (LOSO) models, 300 steps for calibrated models (their own data is one session).
usage: python src/hw/qat.py DATA_DIR RUNS_ROOT OUT_JSON
"""
import json
import os
import sys

import numpy as np
import torch

_SRC = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, _SRC)

from alignment.ea_dataset import AlignedMotorImageryDataset  # noqa: E402
from encoder.conformer_braindecode import EEGConformer  # noqa: E402
from hw.quantize import QLayer, calibrate, quantize  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

DATA, ROOT, OUT = (os.path.abspath(p) for p in sys.argv[1:4])
CONFIGS = {'W4A8': (4, 8), 'W4A4': (4, 4)}
SEEDS = [int(s) for s in os.environ.get('QAT_SEEDS', '1,2,3').split(',')]
STEPS = {'loso': 1000, 'calib': 300, 'calib_ea': 300}
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
os.chdir(os.path.join(os.path.dirname(_SRC), 'scripts'))
FILES = sorted(os.listdir(DATA))[:18]
KW = dict(sample_keys=['inputs', 'attention_mask'], chunk_len=500, num_chunks=2, ovlp=0, root_path=DATA + '/',
          gpt_only=False)


def tensors(files, align):
    ds = AlignedMotorImageryDataset(files, align=align, **KW)
    return torch.stack([ds[i]['inputs'] for i in range(len(ds))]), torch.as_tensor(np.asarray(ds.labels)).long()


def load(path):
    enc = EEGConformer(n_outputs=4, n_chans=22, n_times=500, ch_pos=None, is_decoding_mode=True)
    sd = load_file(path)
    enc.load_state_dict({k[len('encoder.'):]: v for k, v in sd.items() if k.startswith('encoder.')})
    return enc.eval()


@torch.no_grad()
def accuracy(m, x, y):
    m.eval()
    pred = torch.cat([m(x[i:i + 256].to(dev)).argmax(1).cpu() for i in range(0, len(x), 256)])
    return float((pred == y).float().mean())


def calib_batches(x):
    idx = torch.randperm(len(x), generator=torch.Generator().manual_seed(0))[:256]
    return [x[idx[i:i + 64]].to(dev) for i in range(0, 256, 64)]


def run(path, trx, try_, tex, tey, steps, seed):
    base = load(path)
    out = {}
    for name, (wb, ab) in CONFIGS.items():
        ptq = quantize(base, wb, ab, fuse=True).to(dev)
        calibrate(ptq, calib_batches(trx))
        out[f'ptq_{name}'] = accuracy(ptq, tex, tey)

        m = quantize(base, wb, ab, fuse=True, qat=True).to(dev)
        calibrate(m, calib_batches(trx))  # activation ranges from training data, frozen from here on
        opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-5, weight_decay=0.0)
        g = torch.Generator().manual_seed(seed)
        m.train()
        for _ in range(steps):
            idx = torch.randint(0, len(trx), (32,), generator=g)
            loss = torch.nn.functional.nll_loss(m(trx[idx].to(dev)), try_[idx].to(dev))  # model ends in LogSoftmax
            opt.zero_grad()
            loss.backward()
            opt.step()
        for mod in m.modules():
            if isinstance(mod, QLayer):
                mod.freeze()
        out[f'qat_{name}'] = accuracy(m, tex, tey)
    return out


results = {}
jobs = []
for f in range(9):
    jobs.append(('loso', f, 'none', [x for i, x in enumerate(FILES) if i // 2 != f], FILES[2 * f:2 * f + 2]))
for arm, align in (('calib', 'none'), ('calib_ea', 'ea')):
    for f in range(9):
        jobs.append((arm, f, align, [FILES[2 * f + 1]], [FILES[2 * f]]))
for arm, f, align, trf, tef in jobs:
    trx, try_ = tensors(trf, align)
    tex, tey = tensors(tef, align)
    for s in SEEDS:
        path = (f'{ROOT}/runs5k/none_s{s}-{f}/model_final/model.safetensors' if arm == 'loso'
                else f'{ROOT}/calib/{arm}_s{s}_subj{f + 1}-{f}/model_final/model.safetensors')
        torch.manual_seed(s)
        key = f'{arm}/s{s}/A0{f + 1}'
        results[key] = run(path, trx, try_, tex, tey, STEPS[arm], s)
        print(key, {k: round(v, 3) for k, v in results[key].items()}, flush=True)
        json.dump(results, open(OUT, 'w'), indent=1)

print('\nsummary (mean accuracy over subjects and seeds):')
for arm in ('loso', 'calib', 'calib_ea'):
    keys = [k for k in results if k.startswith(arm + '/')]
    cols = sorted(results[keys[0]])
    print(arm, '  '.join(f'{c}: {np.mean([results[k][c] for k in keys]):.3f}' for c in cols))
