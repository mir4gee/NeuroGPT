#!/usr/bin/env python3
"""
Tests for per-session Euclidean Alignment. Run with: python tests/test_alignment.py
Synthetic data only; checks the maths and that two "subjects" with different
spatial mixing become comparable after alignment.
"""
import os
import sys

import numpy as np

_SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'src')
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from alignment.ea_dataset import euclidean_align  # noqa: E402

RNG = np.random.default_rng(0)


def _mean_cov(x):
    n, _, t = x.shape
    return np.einsum('nct,ndt->cd', x, x) / (n * t)


def _subject(mixing, n=40, c=6, t=200):
    return np.einsum('cd,ndt->nct', mixing, RNG.standard_normal((n, c, t)))


def test_mean_covariance_becomes_identity():
    x = _subject(RNG.standard_normal((6, 6)) * 5.0)
    aligned = euclidean_align(x)
    assert aligned.shape == x.shape
    assert np.allclose(_mean_cov(aligned), np.eye(6), atol=1e-6)


def test_different_subject_mixings_become_comparable():
    a = _subject(RNG.standard_normal((6, 6)) * 3.0)
    b = _subject(RNG.standard_normal((6, 6)) * 0.2)
    before = np.linalg.norm(_mean_cov(a) - _mean_cov(b))
    after = np.linalg.norm(_mean_cov(euclidean_align(a)) - _mean_cov(euclidean_align(b)))
    assert after < 1e-3 * before


def test_rank_deficient_input_stays_finite():
    x = _subject(RNG.standard_normal((6, 6)))
    x[:, 5, :] = x[:, 4, :]  # duplicate channel, covariance is singular
    assert np.isfinite(euclidean_align(x)).all()


if __name__ == '__main__':
    test_mean_covariance_becomes_identity()
    test_different_subject_mixings_become_comparable()
    test_rank_deficient_input_stays_finite()
    print('All alignment tests passed.')
