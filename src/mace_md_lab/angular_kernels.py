"""Fused CUDA kernels for real Cartesian harmonics and their edge-vector adjoint."""

import torch
import triton
import triton.language as tl


@triton.jit
def _coordinates(V, i, valid, step0: tl.constexpr, step1: tl.constexpr):
    x = tl.load(V + i * step0, valid, other=0.0)
    y = tl.load(V + i * step0 + step1, valid, other=0.0)
    z = tl.load(V + i * step0 + 2 * step1, valid, other=0.0)
    radius = tl.sqrt(x * x + y * y + z * z)
    # Triton promotes ordinary Python float literals through FP32 on some
    # versions. Derive non-polynomial constants in the actual checkpoint dtype.
    epsilon = 1.0 / tl.full((), 1000000000000, V.dtype.element_ty)
    inv = 1.0 / tl.maximum(radius, epsilon)
    return x * inv, y * inv, z * inv, radius, inv


@triton.jit
def _forward(
    V,
    O,
    stride0: tl.constexpr,
    stride1: tl.constexpr,
    count,
    L: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < count
    x, y, z, _, _ = _coordinates(V, i, valid, stride0, stride1)
    out = O + i * WIDTH
    tl.store(out, 1.0, valid)
    if L >= 1:
        c1 = tl.sqrt(tl.full((), 3, V.dtype.element_ty))  # sqrt(3)
        tl.store(out + 1, c1 * x, valid)
        tl.store(out + 2, c1 * y, valid)
        tl.store(out + 3, c1 * z, valid)
    if L >= 2:
        s = x * x + z * z
        c2 = tl.sqrt(tl.full((), 15, V.dtype.element_ty))  # sqrt(15)
        tl.store(out + 4, c2 * x * z, valid)
        tl.store(out + 5, c2 * x * y, valid)
        tl.store(
            out + 6,
            tl.sqrt(tl.full((), 5, V.dtype.element_ty)) * (y * y - 0.5 * s),
            valid,
        )
        tl.store(out + 7, c2 * y * z, valid)
        tl.store(out + 8, 0.5 * c2 * (z * z - x * x), valid)
    if L >= 3:
        a = tl.sqrt(tl.full((), 35, V.dtype.element_ty) / 8.0)  # sqrt(35/8)
        b = tl.sqrt(tl.full((), 105, V.dtype.element_ty))  # sqrt(105)
        c = tl.sqrt(tl.full((), 21, V.dtype.element_ty) / 8.0)  # sqrt(21/8)
        d = tl.sqrt(tl.full((), 7, V.dtype.element_ty)) * 0.5  # sqrt(7)/2
        s = x * x + z * z
        tl.store(out + 9, a * x * (3 * z * z - x * x), valid)
        tl.store(out + 10, b * x * y * z, valid)
        tl.store(out + 11, c * x * (4 * y * y - s), valid)
        tl.store(out + 12, d * y * (2 * y * y - 3 * s), valid)
        tl.store(out + 13, c * z * (4 * y * y - s), valid)
        tl.store(out + 14, 0.5 * b * y * (z * z - x * x), valid)
        tl.store(out + 15, a * z * (z * z - 3 * x * x), valid)


@triton.jit
def _reverse(
    V,
    G,
    D,
    stride0: tl.constexpr,
    stride1: tl.constexpr,
    g0: tl.constexpr,
    g1: tl.constexpr,
    count,
    L: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < count
    x, y, z, r, invr = _coordinates(V, i, valid, stride0, stride1)
    base = G + i * g0
    gx = tl.full((BLOCK,), 0.0, V.dtype.element_ty)
    gy = tl.full((BLOCK,), 0.0, V.dtype.element_ty)
    gz = tl.full((BLOCK,), 0.0, V.dtype.element_ty)
    if L >= 1:
        c1 = tl.sqrt(tl.full((), 3, V.dtype.element_ty))
        gx += c1 * tl.load(base + g1, valid, other=0.0)
        gy += c1 * tl.load(base + 2 * g1, valid, other=0.0)
        gz += c1 * tl.load(base + 3 * g1, valid, other=0.0)
    if L >= 2:
        aa = tl.sqrt(tl.full((), 15, V.dtype.element_ty))
        bb = tl.sqrt(tl.full((), 5, V.dtype.element_ty))
        h4 = tl.load(base + 4 * g1, valid, other=0.0)
        h5 = tl.load(base + 5 * g1, valid, other=0.0)
        h6 = tl.load(base + 6 * g1, valid, other=0.0)
        h7 = tl.load(base + 7 * g1, valid, other=0.0)
        h8 = tl.load(base + 8 * g1, valid, other=0.0)
        gx += aa * (z * h4 + y * h5 - x * h8) - bb * x * h6
        gy += aa * (x * h5 + z * h7) + 2 * bb * y * h6
        gz += aa * (x * h4 + y * h7 + z * h8) - bb * z * h6
    if L >= 3:
        a = tl.sqrt(tl.full((), 35, V.dtype.element_ty) / 8.0)
        b = tl.sqrt(tl.full((), 105, V.dtype.element_ty))
        c = tl.sqrt(tl.full((), 21, V.dtype.element_ty) / 8.0)
        d = tl.sqrt(tl.full((), 7, V.dtype.element_ty)) * 0.5
        h9 = tl.load(base + 9 * g1, valid, other=0.0)
        h10 = tl.load(base + 10 * g1, valid, other=0.0)
        h11 = tl.load(base + 11 * g1, valid, other=0.0)
        h12 = tl.load(base + 12 * g1, valid, other=0.0)
        h13 = tl.load(base + 13 * g1, valid, other=0.0)
        h14 = tl.load(base + 14 * g1, valid, other=0.0)
        h15 = tl.load(base + 15 * g1, valid, other=0.0)
        xx, yy, zz = x * x, y * y, z * z
        gx += (
            3 * a * (zz - xx) * h9
            + b * y * z * h10
            + c * (4 * yy - 3 * xx - zz) * h11
            - 6 * d * y * x * h12
            - 2 * c * x * z * h13
            - b * x * y * h14
            - 6 * a * x * z * h15
        )
        gy += (
            b * x * z * h10
            + 8 * c * x * y * h11
            + d * (6 * yy - 3 * (xx + zz)) * h12
            + 8 * c * z * y * h13
            + 0.5 * b * (zz - xx) * h14
        )
        gz += (
            6 * a * x * z * h9
            + b * x * y * h10
            - 2 * c * x * z * h11
            - 6 * d * y * z * h12
            + c * (4 * yy - xx - 3 * zz) * h13
            + b * y * z * h14
            + 3 * a * (zz - xx) * h15
        )
    epsilon = 1.0 / tl.full((), 1000000000000, V.dtype.element_ty)
    dot = tl.where(r >= epsilon, x * gx + y * gy + z * gz, 0.0)
    tl.store(D + 3 * i, (gx - x * dot) * invr, valid)
    tl.store(D + 3 * i + 1, (gy - y * dot) * invr, valid)
    tl.store(D + 3 * i + 2, (gz - z * dot) * invr, valid)


def forward(vectors, lmax):
    n = vectors.shape[0]
    width = (lmax + 1) ** 2
    result = vectors.new_empty((n, width))
    if n:
        with torch.cuda.device(vectors.device):
            _forward[(triton.cdiv(n, 128),)](
                vectors, result, *vectors.stride(), n, lmax, width, 128
            )
    return result


def backward(vectors, cotangent, lmax):
    n = vectors.shape[0]
    result = vectors.new_empty((n, 3))
    if n:
        with torch.cuda.device(vectors.device):
            _reverse[(triton.cdiv(n, 128),)](
                vectors,
                cotangent,
                result,
                *vectors.stride(),
                *cotangent.stride(),
                n,
                lmax,
                128
            )
    return result
