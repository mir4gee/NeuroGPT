#!/usr/bin/env python3
"""
Score single runs and the seed-ensemble (mean of per-seed softmax probabilities) per fold.
usage: python results/ensemble_eval.py RUNS_DIR ALIGN SEED [SEED ...]
Reads RUNS_DIR/<align>_s<seed>-<fold>/heldout_logits.npy + heldout_labels.npy.
"""
import os
import sys

import numpy as np

runs, align, seeds = sys.argv[1], sys.argv[2], sys.argv[3:]


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


rows = []
print('fold subj ' + ' '.join(f's{s:>6}' for s in seeds) + '   mean  ensemble  ens-mean')
for f in range(9):
    probs, labels, accs = [], None, []
    for s in seeds:
        d = os.path.join(runs, f'{align}_s{s}-{f}')
        lp, yp = os.path.join(d, 'heldout_logits.npy'), os.path.join(d, 'heldout_labels.npy')
        if not (os.path.exists(lp) and os.path.exists(yp)):
            continue
        p, y = softmax(np.load(lp)), np.load(yp)
        probs.append(p)
        labels = y
        accs.append(float((p.argmax(1) == y).mean()))
    if len(probs) < len(seeds):
        continue
    ens = float((np.mean(probs, axis=0).argmax(1) == labels).mean())
    rows.append((accs, ens))
    print(f'{f:4d}  A0{f+1} ' + ' '.join(f'{a:7.3f}' for a in accs) + f'   {np.mean(accs):.3f}    {ens:.3f}    {ens - np.mean(accs):+.3f}')

if rows:
    single = np.array([np.mean(r[0]) for r in rows])
    ens = np.array([r[1] for r in rows])
    print(f'\nn_folds={len(rows)}  single-run mean {single.mean():.3f} +/- {single.std(ddof=1) if len(rows) > 1 else 0:.3f}'
          f'  ensemble mean {ens.mean():.3f} +/- {ens.std(ddof=1) if len(rows) > 1 else 0:.3f}'
          f'  gain {ens.mean() - single.mean():+.3f}  folds ensemble>single-mean: {(ens > single).sum()}/{len(rows)}')
    if len(rows) >= 5:
        from scipy.stats import ttest_rel, wilcoxon
        print(f'paired t p={ttest_rel(ens, single).pvalue:.4f}  Wilcoxon p={wilcoxon(ens, single).pvalue:.4f}')
