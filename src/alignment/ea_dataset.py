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
import torch

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


def augment_chunks(x: torch.Tensor, noise_std: float = 0.0, ch_drop: float = 0.0, amp_scale: float = 0.0,
                   generator=None) -> torch.Tensor:
    """
    Light training-time augmentation on a (chunks, channels, time) tensor.
    noise_std: std of additive Gaussian noise (data are per-channel z-scored, so ~unit scale);
    ch_drop:   probability of zeroing a whole channel (same channels for every chunk of the trial);
    amp_scale: multiplies the trial by U(1-amp_scale, 1+amp_scale).
    """
    if amp_scale > 0:
        x = x * (1.0 + amp_scale * (2 * torch.rand(1, generator=generator) - 1))
    if ch_drop > 0:
        keep = (torch.rand(x.shape[1], generator=generator) >= ch_drop).to(x.dtype)
        x = x * keep[None, :, None]
    if noise_std > 0:
        x = x + noise_std * torch.randn(x.shape, generator=generator, dtype=x.dtype)
    return x


class AlignedMotorImageryDataset(MotorImageryDataset):
    """
    MotorImageryDataset with optional per-file Euclidean Alignment.
    align='none' reproduces the vendored behaviour exactly.
    """

    def __init__(self, *args, align: str = 'none', margin: int = 0, augment: bool = False,
                 noise_std: float = 0.0, ch_drop: float = 0.0, amp_scale: float = 0.0,
                 subset: str = None, subset_frac: float = 0.0, **kwargs):
        if align is None:  # train_gpt.get_config() turns the string 'none' into None
            align = 'none'
        if align not in ('none', 'ea'):
            raise ValueError(f"align must be 'none' or 'ea', got {align!r}")
        # these must exist before the parent constructor calls get_trials_all()
        self.align = align
        self.margin = int(margin)
        self.augment = augment
        self.noise_std, self.ch_drop, self.amp_scale = noise_std, ch_drop, amp_scale
        super().__init__(*args, **kwargs)
        # margin > 0 makes each trial longer than the two 2-s chunks, so the vendored split_chunks()
        # picks a random start (random-crop augmentation) when start_samp_pnt == -1. Evaluation data
        # is pinned to the centre crop, i.e. exactly the original t = 2..6 s window.
        self.start_samp_pnt = -1 if (augment and self.margin > 0) else self.margin
        # Optional fixed split of the trials (used to hold out part of a calibration session for validation).
        # The permutation uses a fixed seed (0) so every arm and every training seed sees the same split.
        if subset is not None:
            if subset not in ('train', 'val') or not 0 < subset_frac < 1:
                raise ValueError('subset must be train/val with 0 < subset_frac < 1')
            n = len(self.labels)
            perm = np.random.default_rng(0).permutation(n)
            n_val = int(round(subset_frac * n))
            keep = np.sort(perm[:n_val] if subset == 'val' else perm[n_val:])
            self.trials, self.labels = self.trials[keep], self.labels[keep]

    def __len__(self):
        return len(self.labels)

    def get_trials_from_single_subj(self, sub_id):
        # Same as the vendored method, except the window is widened by `margin` samples on both sides.
        raw = self.data_all[sub_id]['s'].T
        events_type = self.data_all[sub_id]['etyp'].T
        events_position = self.data_all[sub_id]['epos'].T
        events_duration = self.data_all[sub_id]['edur'].T
        idxs = [i for i, x in enumerate((events_type == 768)[0]) if x]
        trial_labels = self.get_labels(sub_id)
        trials, classes = [], []
        for j, index in enumerate(idxs):
            try:
                classes.append(trial_labels[j])
                start = events_position[0, index]
                stop = start + events_duration[0, index]
                trials.append(raw[:22, start + 500 - self.margin: stop - 375 + self.margin])
            except Exception:
                continue
        return trials, classes

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        if self.augment and (self.noise_std > 0 or self.ch_drop > 0 or self.amp_scale > 0):
            item['inputs'] = augment_chunks(item['inputs'], self.noise_std, self.ch_drop, self.amp_scale)
        return item

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
