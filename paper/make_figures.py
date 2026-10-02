#!/usr/bin/env python3
"""Regenerate the paper figures from the numbers in results/*.md (copied here for reproducibility)."""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'figures')
os.makedirs(OUT, exist_ok=True)
SUBJ = [f'A0{i}' for i in range(1, 10)]

# results/alignment_9fold.md (cross-subject, LOSO, same machine, seed 1234)
loso_none = [0.634, 0.434, 0.753, 0.517, 0.557, 0.483, 0.649, 0.760, 0.616]
loso_ea = [0.733, 0.424, 0.769, 0.510, 0.540, 0.503, 0.705, 0.750, 0.668]
# results/calibration.md (session E accuracy, mean of 3 seeds)
cal_loso = [0.663, 0.431, 0.778, 0.514, 0.499, 0.472, 0.649, 0.712, 0.572]
cal_within = [0.657, 0.441, 0.715, 0.573, 0.602, 0.554, 0.645, 0.727, 0.685]
cal_calib = [0.810, 0.568, 0.911, 0.723, 0.325, 0.627, 0.825, 0.852, 0.754]
cal_ea_both = [0.831, 0.583, 0.904, 0.737, 0.717, 0.628, 0.823, 0.798, 0.761]  # exploratory, results/calibration_ea_both.md

x = np.arange(9)

fig, ax = plt.subplots(figsize=(7, 2.9))
w = 0.38
ax.bar(x - w / 2, loso_none, w, label=f'authors\' recipe (mean {np.mean(loso_none):.3f})')
ax.bar(x + w / 2, loso_ea, w, label=f'+ per-session EA (mean {np.mean(loso_ea):.3f})')
ax.axhline(0.645, ls='--', c='k', lw=0.8, label='paper mean 0.645')
ax.axhline(0.25, ls=':', c='gray', lw=0.8, label='chance')
ax.set_xticks(x, SUBJ); ax.set_ylim(0, 1); ax.set_ylabel('held-out accuracy')
ax.legend(fontsize=7, ncol=2, loc='lower center', bbox_to_anchor=(0.5, 1.0), frameon=False); plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig_cross_subject.pdf')); plt.savefig(os.path.join(OUT, 'fig_cross_subject.png'), dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(7, 3.1))
w = 0.2
ax.bar(x - 1.5 * w, cal_loso, w, label=f'cross-subject only ({np.mean(cal_loso):.3f})')
ax.bar(x - 0.5 * w, cal_within, w, label=f'pre-trained + own session T ({np.mean(cal_within):.3f})')
ax.bar(x + 0.5 * w, cal_calib, w, label=f'cross-subject + own session T ({np.mean(cal_calib):.3f})')
ax.bar(x + 1.5 * w, cal_ea_both, w, label=f'+ alignment in both stages, exploratory ({np.mean(cal_ea_both):.3f})')
ax.axhline(0.25, ls=':', c='gray', lw=0.8)
ax.set_xticks(x, SUBJ); ax.set_ylim(0, 1); ax.set_ylabel('session-E accuracy')
ax.legend(fontsize=7, ncol=2, loc='lower center', bbox_to_anchor=(0.5, 1.0), frameon=False); plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig_calibration.pdf')); plt.savefig(os.path.join(OUT, 'fig_calibration.png'), dpi=150)
plt.close()
print('figures written to', OUT)
