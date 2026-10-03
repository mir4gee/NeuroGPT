#!/usr/bin/env python3
"""
Tests for the hardware path (src/hw). Run with: python tests/test_hw.py
Random weights only: checks that BatchNorm folding and front-end fusion are exact in float, and that the
exported integer weights reproduce the fake-quantized weights.
"""
import os
import sys

import torch

_SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'src')
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from encoder.conformer_braindecode import EEGConformer  # noqa: E402
from hw.quantize import QLayer, fold_batchnorm, fuse_frontend, quantize  # noqa: E402

torch.manual_seed(0)


def _model():
    m = EEGConformer(n_outputs=4, n_chans=22, n_times=500, ch_pos=None, is_decoding_mode=True).eval()
    bn = m.patch_embedding.shallownet[2]
    bn.running_mean.uniform_(-1, 1)
    bn.running_var.uniform_(0.5, 2)
    bn.weight.data.uniform_(0.5, 2)
    bn.bias.data.uniform_(-1, 1)
    return m


def test_fold_and_fuse_are_exact():
    m, x = _model(), torch.randn(3, 2, 22, 500)
    with torch.no_grad():
        ref = m(x)
        folded = fold_batchnorm(m)
        assert torch.allclose(folded(x), ref, atol=1e-4)
        assert torch.allclose(fuse_frontend(folded)(x), ref, atol=1e-4)


def test_integer_weights_match_fake_quant():
    q = quantize(_model(), 8, None, fuse=True)
    layers = [mod for mod in q.modules() if isinstance(mod, QLayer)]
    assert len(layers) == 2 + 6 * 6 + 3  # fused front conv + projection, 6 blocks x (q,k,v,o,ff1,ff2), 3 head layers
    for ql in layers:
        assert ql.w_int.dtype == torch.int8 and ql.w_int.abs().max() <= 127
        shape = (-1,) + (1,) * (ql.w_int.dim() - 1)
        assert torch.allclose(ql.w_int.float() * ql.w_scale.reshape(shape), ql.layer.weight.data, atol=1e-7)


if __name__ == '__main__':
    test_fold_and_fuse_are_exact()
    test_integer_weights_match_fake_quant()
    print('All hardware tests passed.')
