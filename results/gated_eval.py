#!/usr/bin/env python3
"""
Validation-gated calibration. For each subject, average each candidate's validation accuracy (20% of session T,
fixed split) over seeds, pick the best candidate (ties -> calib), and report that candidate's session-E accuracy
from the seed ensemble (mean softmax of the 3 seeds). Session E is never used for the choice.
usage: python results/gated_eval.py GATED_DIR
"""
import json
import os
import sys

import numpy as np

root = sys.argv[1]
ARMS = ['calib', 'within', 'within_ea']  # order = tie-break priority
SEEDS = [1, 2, 3]


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def load(arm, seed, s):
    d = os.path.join(root, f'{arm}_s{seed}_subj{s}-{s - 1}')
    if not os.path.exists(os.path.join(d, 'test_metrics.json')):
        return None
    val = json.load(open(os.path.join(d, 'heldout_metrics.json')))['eval_accuracy']
    return val, softmax(np.load(os.path.join(d, 'test_logits.npy'))), np.load(os.path.join(d, 'test_labels.npy'))


rows, header = [], 'subj  ' + '  '.join(f'{a:>9}(val/E)' for a in ARMS) + '   chosen     gated-E'
print(header)
for s in range(1, 10):
    per = {}
    for a in ARMS:
        runs = [load(a, k, s) for k in SEEDS]
        if any(r is None for r in runs):
            break
        val = np.mean([r[0] for r in runs])
        ens = float((np.mean([r[1] for r in runs], 0).argmax(1) == runs[0][2]).mean())
        per[a] = (val, ens)
    if len(per) < len(ARMS):
        print(f'A0{s}  (incomplete)')
        continue
    best = max(ARMS, key=lambda a: (round(per[a][0], 6), -ARMS.index(a)))
    rows.append([per[a][1] for a in ARMS] + [per[best][1]])
    print(f'A0{s}  ' + '  '.join(f'{per[a][0]:.3f}/{per[a][1]:.3f}    ' for a in ARMS) + f'   {best:9s}  {per[best][1]:.3f}')

if rows:
    r = np.array(rows)
    names = ARMS + ['GATED']
    print('\nseed-ensemble session-E accuracy, mean +/- std over', len(r), 'subjects:')
    for i, n in enumerate(names):
        print(f'  {n:10s} {r[:, i].mean():.3f} +/- {r[:, i].std(ddof=1) if len(r) > 1 else 0:.3f}')
