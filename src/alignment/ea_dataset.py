#!/usr/bin/env python3
"""
Per-session Euclidean Alignment (EA) for NeuroGPT's BCI-IV-2a pipeline.

Motivation (see results/): the fine-tuned baseline reaches ~0.89 accuracy on its
training subjects but ~0.61 on an unseen subject, i.e. the loss is subject-to-
subject shift, not capacity. EA (He & Wu, "Transfer Learning for Brain-Computer
Interfaces: A Euclidean Space Data Alignment Approach", IEEE TBME 2020; cited
from memory) re-centres each recording so its mean trial covariance is the
identity, which removes much of the subject-specific spatial mixing before the
pre-trained encoder sees the data. It adds no trainable parameters and uses no
labels. For the held-out subject it uses only that subject's own *unlabeled*
trials (transductive), which must be disclosed when reporting results.

Each .npz file (one session of one subject) is aligned independently. This is
inserted after trial extraction and before the repo's own 22x22 channel mapping
and global normalisation, so the vendored pipeline is otherwise unchanged.
"""
import os
import sys

import numpy as np

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from batcher.downstream_dataset import MotorImageryDataset  # noqa: E402


def euclidean_align(trials: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    trials: (n_trials, n_channels, n_times). Returns trials aligned so that
    the mean per-timepoint covariance across trials is the identity.
    """
    trials = np.asarray(trials, dtype=np.float64)
    n, _, t = trials.shape
    ref = np.einsum('nct,ndt->cd', trials, trials) / (n * t)
    ref = (ref + ref.T) / 2.0
    vals, vecs = np.linalg.eigh(ref)
    floor = eps * max(float(vals.max()), 1e-12)
    inv_sqrt = (vecs * (1.0 / np.sqrt(np.maximum(vals, floor)))) @ vecs.T
    return np.einsum('cd,ndt->nct', inv_sqrt, trials)


class AlignedMotorImageryDataset(MotorImageryDataset):
    """
    MotorImageryDataset with optional per-file Euclidean Alignment.
    align='none' reproduces the vendored behaviour exactly.
    """

    def __init__(self, *args, align: str = 'none', **kwargs):
        if align is None:  # train_gpt.get_config() turns the string 'none' into None
            align = 'none'
        if align not in ('none', 'ea'):
            raise ValueError(f"align must be 'none' or 'ea', got {align!r}")
        self.align = align  # must exist before the parent constructor calls get_trials_all()
        super().__init__(*args, **kwargs)

    def get_trials_all(self):
        trials_all, labels_all, total_num = [], [], []
        for sub_id in range(len(self.data_all)):
            trials, labels = self.get_trials_from_single_subj(sub_id)
            total_num.append(len(trials))
            arr = np.array(trials)
            if self.align == 'ea':
                arr = euclidean_align(arr)
            trials_all.append(arr)
            labels_all.append(np.array(labels))
        trials_all_arr = np.vstack(trials_all)
        trials_all_arr = self.map2pret(trials_all_arr)
        return self.normalize(trials_all_arr), np.array(labels_all).flatten(), total_num
