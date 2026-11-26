"""Joint evaluation of the two frozen MP-0 Small radial prefixes in FP64.

The ordinary schedule differentiates grouped matrix products. The tangent
schedule propagates the single distance derivative alongside each radial value
and contracts both incoming adjoints against those tangents.
"""
import math
from contextlib import ExitStack, contextmanager
from unittest.mock import patch

import torch
from torch.nn import functional as F
from torch.autograd.function import once_differentiable
import triton
import triton.language as tl


@triton.jit
def _bessel_tangent(R, FREQ, OUT, N, CUTOFF: tl.constexpr, P: tl.constexpr,
                    PREF: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    col = tl.arange(0, 16)
    r = tl.load(R + row, row < N, 1.0)
    w = tl.load(FREQ + col, col < 10, 0.)
    u = r / CUTOFF
    power = tl.full((BLOCK,), 1., tl.float64)
    for _ in tl.static_range(0, P - 1):
        power *= u
    # power = u**(p-1). The rearranged derivative is proportional to
    # u**(p-1) (1-u)**2, so both value and slope vanish at the cutoff.
    envelope = (
        1. - (P + 1) * (P + 2) * 0.5 * power * u
        + P * (P + 2) * power * u * u
        - P * (P + 1) * 0.5 * power * u * u * u
    )
    derivative = -0.5 * P * (P + 1) * (P + 2) / CUTOFF * power * (1. - u) * (1. - u)
    inside = r < CUTOFF
    env = tl.where(inside, envelope, 0.)
    denv = tl.where(inside, derivative, 0.)
    safe = tl.where(r == 0, 1., r)
    phase = r[:, None] * w[None, :]
    b = tl.where(r[:, None] == 0, w[None, :], tl.sin(phase) / safe[:, None])
    db = tl.where(r[:, None] == 0, 0., (w[None, :] * tl.cos(phase) - b) / safe[:, None])
    addr = row[:, None] * 10 + col[None, :]
    mask = (row[:, None] < N) & (col[None, :] < 10)
    tl.store(OUT + addr, PREF * b * env[:, None], mask)
    tl.store(OUT + N * 10 + addr, PREF * (db * env[:, None] + b * denv[:, None]), mask)


@triton.jit
def _silu_and_tangent(X, N, GAIN: tl.constexpr, BLOCK: tl.constexpr):
    ix = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    start = tl.program_id(1) * 2 * N
    x = tl.load(X + start + ix, ix < N, 0.)
    dx = tl.load(X + start + N + ix, ix < N, 0.)
    sig = 1. / (1. + tl.exp(-x))
    tl.store(X + start + ix, GAIN * x * sig, ix < N)
    tl.store(X + start + N + ix, GAIN * dx * sig * (1. + x * (1. - sig)), ix < N)


@triton.jit
def _two_adjoints(G0, G1, D0, D1, GR, N, ROWS: tl.constexpr):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    ix = row[:, None] * 64 + tl.arange(0, 64)[None, :]
    mask = row[:, None] < N
    prod = tl.load(G0 + ix, mask, 0.) * tl.load(D0 + ix, mask, 0.)
    prod += tl.load(G1 + ix, mask, 0.) * tl.load(D1 + ix, mask, 0.)
    tl.store(GR + row, tl.sum(prod, axis=1), row < N)


class _PackedTangent(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r, module):
        n = r.numel()
        z = r.new_empty((2 * n, 10))
        _bessel_tangent[(triton.cdiv(n, 32),)](
            r.contiguous(), module.freq, z, n, module.cutoff, module.power, module.prefactor, 32
        )
        # Within each network the rows are [radial values; distance slopes].
        z = z.expand(2, -1, -1)
        for i, weight in enumerate(module.weights):
            z = torch.bmm(z, weight)
            gain = module.last_gain if i == 2 else 1.
            _silu_and_tangent[(triton.cdiv(n * 64, 256), 2)](z, n * 64, gain, 256)
        ctx.save_for_backward(z[0, n:], z[1, n:])
        return z[0, :n], z[1, :n]

    @staticmethod
    @once_differentiable
    def backward(ctx, g0, g1):
        d0, d1 = ctx.saved_tensors
        n = d0.shape[0]
        gr = d0.new_empty((n, 1))
        _two_adjoints[(triton.cdiv(n, 8),)](g0.contiguous(), g1.contiguous(), d0, d1, gr, n, 8)
        return gr, None


class _Select(torch.nn.Module):
    def __init__(self, index):
        super().__init__()
        self.index = index

    def forward(self, features):
        return features[self.index]


class _Packed:
    def __init__(self, model):
        from mace.modules.radial import BesselBasis, PolynomialCutoff
        emb = model.radial_embedding
        if (type(emb.bessel_fn) is not BesselBasis or type(emb.cutoff_fn) is not PolynomialCutoff
                or hasattr(emb, "distance_transform") or not getattr(emb, "apply_cutoff", True)
                or len(model.interactions) != 2):
            raise ValueError("requires the MP-0 Small Bessel/polynomial input embedding")
        self.embedding = emb.forward
        self.freq = emb.bessel_fn.bessel_weights.detach().contiguous()
        self.cutoff = float(emb.cutoff_fn.r_max)
        self.power = int(emb.cutoff_fn.p)
        self.prefactor = float(emb.bessel_fn.prefactor)
        if self.freq.numel() != 10 or self.power < 1:
            raise ValueError("unexpected radial basis")
        networks = [list(layer.conv_tp_weights.children()) for layer in model.interactions]
        if any(len(network) != 4 for network in networks):
            raise ValueError("expected two radial networks of four bias-free layers")
        self.weights = []
        gain = [1., 1.]
        for i, expected in enumerate(((10, 64), (64, 64), (64, 64))):
            pair = []
            for j, net in enumerate(networks):
                layer = net[i]
                if tuple(layer.weight.shape) != expected or getattr(layer.act, "f", None) is not F.silu:
                    raise ValueError("expected the fixed MP-0 Small radial prefix")
                # e3nn separately scales the input weight and normalized SiLU.
                # Fold each preceding SiLU gain into this frozen weight.
                pair.append(layer.weight.detach() * (gain[j] / math.sqrt(layer.h_in * layer.var_in)))
                gain[j] = math.sqrt(layer.var_out) * (1. if layer.act._is_id else layer.act.cst)
            self.weights.append(torch.stack(pair))
        if gain[0] != gain[1]:
            raise ValueError("the final SiLU scales differ")
        self.last_gain = gain[0]

    def reverse_mode(self, lengths, attrs, index, numbers):
        base, cutoff = self.embedding(lengths, attrs, index, numbers)
        z = base.expand(2, -1, -1)
        for weight in self.weights:
            z = F.silu(torch.bmm(z, weight))
        z = z * self.last_gain
        return z.unbind(0), cutoff

    def tangent(self, lengths, attrs, index, numbers):
        return _PackedTangent.apply(lengths, self), None


@contextmanager
def install_exact_packed_radials(model, tangent=False):
    """Replace both radial prefixes while retaining each trained final projection."""
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError("freeze the inference checkpoint first")
    impl = _Packed(model)
    with ExitStack() as stack:
        fn = impl.tangent if tangent else impl.reverse_mode
        stack.enter_context(patch.object(model.radial_embedding, "forward", fn))
        for i, layer in enumerate(model.interactions):
            final = list(layer.conv_tp_weights.children())[-1]
            selector = torch.nn.Sequential(_Select(i), final)
            stack.enter_context(patch.object(layer, "conv_tp_weights", selector))
        yield
