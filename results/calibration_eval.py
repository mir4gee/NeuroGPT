#!/usr/bin/env python3
"""
Session-E accuracy per subject for three settings, all on the SAME test data (each subject's session E):
  loso   : cross-subject model, no calibration (from runs5k heldout logits; E is the first 288 trials)
  within : authors' pre-trained model fine-tuned on the subject's session T only
  calib  : cross-subject model further fine-tuned on the subject's session T
Seeds 1..3 are averaged per subject (single-run accuracy, not an ensemble).
usage: python results/calibration_eval.py LOSO_DIR CALIB_DIR
"""
import json
import os
import sys

import numpy as np

loso_dir, calib_dir = sys.argv[1], sys.argv[2]
SEEDS = [1, 2, 3]


def loso_e(seed, f):
    d = os.path.join(loso_dir, f'none_s{seed}-{f}')
    z, y = np.load(os.path.join(d, 'heldout_logits.npy')), np.load(os.path.join(d, 'heldout_labels.npy'))
    n_e = len(y) // 2  # sorted test files are [A0xE, A0xT], 288 trials each
    return float((z[:n_e].argmax(1) == y[:n_e]).mean())


def run_acc(arm, seed, s):
    p = os.path.join(calib_dir, f'{arm}_s{seed}_subj{s}-{s - 1}', 'heldout_metrics.json')
    return json.load(open(p))['eval_accuracy'] if os.path.exists(p) else np.nan


rows = []
print('subj   loso   within  calib')
for s in range(1, 10):
    r = [np.mean([loso_e(k, s - 1) for k in SEEDS]),
         np.nanmean([run_acc('within', k, s) for k in SEEDS]),
         np.nanmean([run_acc('calib', k, s) for k in SEEDS])]
    rows.append(r)
    print(f'A0{s}  ' + '  '.join(f'{v:.3f}' for v in r))
a = np.array(rows)
print('mean   ' + '  '.join(f'{v:.3f}' for v in np.nanmean(a, 0)))
print('std    ' + '  '.join(f'{v:.3f}' for v in np.nanstd(a, 0, ddof=1)))
ok = ~np.isnan(a).any(1)
if ok.sum() >= 5:
    from scipy.stats import wilcoxon
    print(f'calib > loso on {(a[ok,2] > a[ok,0]).sum()}/{ok.sum()} subjects, Wilcoxon p={wilcoxon(a[ok,2], a[ok,0]).pvalue:.4f}')
    print(f'calib > within on {(a[ok,2] > a[ok,1]).sum()}/{ok.sum()} subjects, Wilcoxon p={wilcoxon(a[ok,2], a[ok,1]).pvalue:.4f}')
