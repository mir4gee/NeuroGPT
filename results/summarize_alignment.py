#!/usr/bin/env python3
"""
Summarise held-out accuracy (final step, unseen subject) for the alignment runs.
usage: python results/summarize_alignment.py RUNS_DIR [SEED ...]
Reads RUNS_DIR/<align>_s<seed>-<fold>/heldout_metrics.json written by train_aligned.py.
"""
import json
import os
import sys

import numpy as np

runs = sys.argv[1]
seeds = sys.argv[2:] or ['1234']
ARMS = ['none', 'ea']


def acc(arm, seed, fold):
    p = os.path.join(runs, f'{arm}_s{seed}-{fold}', 'heldout_metrics.json')
    return json.load(open(p))['eval_accuracy'] if os.path.exists(p) else None


for seed in seeds:
    print(f'--- seed {seed} ---')
    print('fold subj   none     ea   ea-none')
    rows = []
    for f in range(9):
        a, b = acc('none', seed, f), acc('ea', seed, f)
        sa = f'{a:.3f}' if a is not None else '  -  '
        sb = f'{b:.3f}' if b is not None else '  -  '
        d = f'{b - a:+.3f}' if (a is not None and b is not None) else '   -  '
        print(f'{f:4d}  A0{f+1}  {sa}  {sb}  {d}')
        if a is not None and b is not None:
            rows.append((a, b))
    if rows:
        r = np.array(rows)
        d = r[:, 1] - r[:, 0]
        print(f'n={len(r)}  none {r[:,0].mean():.3f}+/-{r[:,0].std(ddof=1) if len(r)>1 else 0:.3f}  '
              f'ea {r[:,1].mean():.3f}+/-{r[:,1].std(ddof=1) if len(r)>1 else 0:.3f}  '
              f'mean diff {d.mean():+.3f}  folds ea>none: {(d > 0).sum()}/{len(r)}')
        if len(r) >= 5:
            from scipy.stats import ttest_rel, wilcoxon
            print(f'paired t p={ttest_rel(r[:,1], r[:,0]).pvalue:.4f}  Wilcoxon p={wilcoxon(r[:,1], r[:,0]).pvalue:.4f}')
