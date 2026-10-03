#!/usr/bin/env python3
"""
Post-training quantization (PTQ) of the deployed Neuro-GPT encoder-only model (EEG Conformer + FC head).

What is quantized (simulated with "fake quantization", i.e. round to the integer grid and scale back):
  - BatchNorm is first folded into the spatial convolution (what a hardware implementation does).
  - Weights of every Conv2d / Linear: symmetric, per output channel, `w_bits`.
  - Inputs of every Conv2d / Linear and the attention operands (Q, K, V, attention probabilities):
    per tensor, `a_bits`, with ranges calibrated on training data only (99.99th percentile of |x|).
What stays in higher precision: LayerNorm, softmax, GELU, ELU, average pooling, residual additions
(in hardware these become fixed-point/LUT units; they are a small fraction of the compute).
"""
import copy
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

_SRC = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from encoder.conformer_braindecode import _MultiHeadAttention  # noqa: E402


def _ste_round(v):
    """round() in the forward pass, identity gradient in the backward pass (straight-through estimator)."""
    return v + (torch.round(v) - v).detach()


def fake_quant(x, scale, bits, signed=True):
    lo, hi = (-(2 ** (bits - 1) - 1), 2 ** (bits - 1) - 1) if signed else (0, 2 ** bits - 1)
    return torch.clamp(_ste_round(x / scale), lo, hi) * scale


class ActQuant(nn.Module):
    """Per-tensor activation quantizer; collects |x| percentiles while `calibrating`."""

    def __init__(self, bits, signed=True, pct=99.99):
        super().__init__()
        self.bits, self.signed, self.pct = bits, signed, pct
        self.calibrating, self.samples, self.scale = False, [], None

    def forward(self, x):
        if self.bits is None:
            return x
        if self.calibrating:
            v = x.detach().abs().flatten().float()
            if v.numel() > 200_000:
                v = v[torch.randint(0, v.numel(), (200_000,), device=v.device)]
            self.samples.append(torch.quantile(v, self.pct / 100.0).item())
            return x
        return fake_quant(x, self.scale, self.bits, self.signed)

    def finish(self):
        if self.bits is None:
            return
        rng = max(sum(self.samples) / len(self.samples), 1e-8)
        self.scale = rng / ((2 ** (self.bits - 1) - 1) if self.signed else (2 ** self.bits - 1))
        self.samples = []


def _quant_weight_(module, bits):
    """Quantize in place; return (integer weights, per-output-channel scale) for export."""
    w = module.weight.data
    flat = w.reshape(w.shape[0], -1)
    q = 2 ** (bits - 1) - 1
    scale = flat.abs().amax(dim=1).clamp(min=1e-12) / q
    shape = (-1,) + (1,) * (w.dim() - 1)
    w_int = torch.clamp(torch.round(w / scale.reshape(shape)), -q, q)
    module.weight.data = w_int * scale.reshape(shape)
    return w_int.to(torch.int8), scale


class QLayer(nn.Module):
    """Wraps a Conv2d/Linear: quantize its input, run it with quantized weights."""

    def __init__(self, layer, w_bits, a_bits, qat=False):
        super().__init__()
        self.layer = copy.deepcopy(layer)
        self.w_bits, self.qat = w_bits, qat
        self.w_int = self.w_scale = None
        if w_bits is not None and not qat:
            self.w_int, self.w_scale = _quant_weight_(self.layer, w_bits)
        self.aq = ActQuant(a_bits)

    def forward(self, x):
        x = self.aq(x)
        if not (self.qat and self.w_bits is not None):
            return self.layer(x)
        # QAT: keep float master weights, quantize them on the fly (per output channel, STE)
        w = self.layer.weight
        q = 2 ** (self.w_bits - 1) - 1
        scale = w.detach().reshape(w.shape[0], -1).abs().amax(dim=1).clamp(min=1e-12) / q
        wq = fake_quant(w, scale.reshape((-1,) + (1,) * (w.dim() - 1)), self.w_bits)
        if isinstance(self.layer, nn.Conv2d):
            return F.conv2d(x, wq, self.layer.bias, self.layer.stride, self.layer.padding)
        return F.linear(x, wq, self.layer.bias)

    def freeze(self):
        """End of QAT: bake the quantized weights in and record the integer weights for export."""
        if self.qat and self.w_bits is not None:
            self.w_int, self.w_scale = _quant_weight_(self.layer, self.w_bits)
            self.qat = False


class QAttention(nn.Module):
    """Same maths as _MultiHeadAttention (eval mode), with quantized Q, K, V and attention probabilities."""

    def __init__(self, att, w_bits, a_bits, qat=False):
        super().__init__()
        self.emb_size, self.num_heads = att.emb_size, att.num_heads
        self.att_drop = att.att_drop
        self.queries, self.keys, self.values = (QLayer(att.queries, w_bits, a_bits, qat),
                                                QLayer(att.keys, w_bits, a_bits, qat),
                                                QLayer(att.values, w_bits, a_bits, qat))
        self.projection = QLayer(att.projection, w_bits, a_bits, qat)
        self.q_aq, self.k_aq, self.v_aq = ActQuant(a_bits), ActQuant(a_bits), ActQuant(a_bits)
        self.p_aq = ActQuant(a_bits, signed=False)

    def forward(self, x, mask=None):
        q = rearrange(self.q_aq(self.queries(x)), "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(self.k_aq(self.keys(x)), "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(self.v_aq(self.values(x)), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum("bhqd, bhkd -> bhqk", q, k)
        att = self.att_drop(self.p_aq(F.softmax(energy / self.emb_size ** 0.5, dim=-1)))
        out = torch.einsum("bhal, bhlv -> bhav ", att, v)
        return self.projection(rearrange(out, "b h n d -> b n (h d)"))


def fold_batchnorm(encoder):
    """Fold the BatchNorm after the spatial conv into that conv (exact in eval mode). Returns a copy."""
    enc = copy.deepcopy(encoder).eval()
    seq = enc.patch_embedding.shallownet
    conv, bn = seq[1], seq[2]
    s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    conv.weight.data = conv.weight.data * s.reshape(-1, 1, 1, 1)
    conv.bias.data = (conv.bias.data - bn.running_mean) * s + bn.bias
    seq[2] = nn.Identity()
    return enc


def fuse_frontend(enc):
    """Merge the temporal conv (1->40, 1x25) and the spatial conv (40->40, 22x1) into one 22x25 conv.
    Exact in float: there is no nonlinearity between them (BatchNorm must be folded first).
    W[o,c,k] = sum_f W2[o,f,c] * W1[f,k];  b[o] = b2[o] + sum_{f,c} W2[o,f,c] * b1[f].
    Cuts the front-end multiply-accumulates from 40*25*22 + 40*40*22 to 40*22*25 per output time step."""
    seq = enc.patch_embedding.shallownet
    c1, c2 = seq[0], seq[1]
    w1 = c1.weight.data[:, 0, 0, :]          # [f, k]
    w2 = c2.weight.data[:, :, :, 0]          # [o, f, c]
    fused = nn.Conv2d(1, w2.shape[0], (w2.shape[2], w1.shape[1]))
    fused.weight.data = torch.einsum('ofc,fk->ock', w2, w1).unsqueeze(1)
    fused.bias.data = c2.bias.data + torch.einsum('ofc,f->o', w2, c1.bias.data)
    seq[0], seq[1] = fused, nn.Identity()
    return enc


def quantize(encoder, w_bits, a_bits, fuse=False, qat=False):
    """encoder: the trained EEGConformer (decoding mode). w_bits/a_bits None = keep float."""
    enc = fold_batchnorm(encoder)
    if fuse:
        enc = fuse_frontend(enc)
    if w_bits is None and a_bits is None:
        return enc

    def swap(parent):
        for name, child in parent.named_children():
            if isinstance(child, _MultiHeadAttention):
                setattr(parent, name, QAttention(child, w_bits, a_bits, qat))
            elif isinstance(child, (nn.Conv2d, nn.Linear)):
                setattr(parent, name, QLayer(child, w_bits, a_bits, qat))
            else:
                swap(child)
    swap(enc)
    return enc.eval()


def act_quantizers(model):
    return [m for m in model.modules() if isinstance(m, ActQuant)]


@torch.no_grad()
def calibrate(model, batches):
    qs = act_quantizers(model)
    for q in qs:
        q.calibrating = True
    for x in batches:
        model(x)
    for q in qs:
        q.calibrating = False
        q.finish()
