"""Exact first derivatives of e3nn's real, component-normalized harmonics through l=3."""

from contextlib import contextmanager
from math import sqrt

import torch
from torch.autograd.function import once_differentiable


def _torch_forward(vectors, lmax):
    """Cartesian solid harmonics evaluated on normalized Cartesian edge directions."""
    u = vectors / vectors.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    x, y, z = u.unbind(-1)
    values = [torch.ones_like(x)]
    if lmax >= 1:
        values.extend([sqrt(3) * x, sqrt(3) * y, sqrt(3) * z])
    if lmax >= 2:
        s = x * x + z * z
        values.extend(
            [
                sqrt(15) * x * z,
                sqrt(15) * x * y,
                sqrt(5) * (y * y - 0.5 * s),
                sqrt(15) * y * z,
                0.5 * sqrt(15) * (z * z - x * x),
            ]
        )
    if lmax >= 3:
        a, b, c, d = sqrt(35 / 8), sqrt(105), sqrt(21 / 8), sqrt(7) / 2
        values.extend(
            [
                a * x * (3 * z * z - x * x),
                b * x * y * z,
                c * x * (4 * y * y - s),
                d * y * (2 * y * y - 3 * s),
                c * z * (4 * y * y - s),
                0.5 * b * y * (z * z - x * x),
                a * z * (z * z - 3 * x * x),
            ]
        )
    return torch.stack(values, dim=-1)


def _torch_vjp(vectors, cotangent, lmax):
    radius = vectors.norm(dim=-1, keepdim=True)
    divisor = radius.clamp_min(1e-12)
    u = vectors / divisor
    x, y, z = u.unbind(-1)
    g = cotangent.unbind(-1)
    gx = gy = gz = torch.zeros_like(x)
    if lmax >= 1:
        gx, gy, gz = sqrt(3) * g[1], sqrt(3) * g[2], sqrt(3) * g[3]
    if lmax >= 2:
        a, b = sqrt(15), sqrt(5)
        gx = gx + a * (z * g[4] + y * g[5] - x * g[8]) - b * x * g[6]
        gy = gy + a * (x * g[5] + z * g[7]) + 2 * b * y * g[6]
        gz = gz + a * (x * g[4] + y * g[7] + z * g[8]) - b * z * g[6]
    if lmax >= 3:
        a, b, c, d = sqrt(35 / 8), sqrt(105), sqrt(21 / 8), sqrt(7) / 2
        xx, yy, zz = x * x, y * y, z * z
        gx = gx + (
            3 * a * (zz - xx) * g[9]
            + b * y * z * g[10]
            + c * (4 * yy - 3 * xx - zz) * g[11]
            - 6 * d * y * x * g[12]
            - 2 * c * x * z * g[13]
            - b * x * y * g[14]
            - 6 * a * x * z * g[15]
        )
        gy = gy + (
            b * x * z * g[10]
            + 8 * c * x * y * g[11]
            + d * (6 * yy - 3 * (xx + zz)) * g[12]
            + 8 * c * z * y * g[13]
            + 0.5 * b * (zz - xx) * g[14]
        )
        gz = gz + (
            6 * a * x * z * g[9]
            + b * x * y * g[10]
            - 2 * c * x * z * g[11]
            - 6 * d * y * z * g[12]
            + c * (4 * yy - xx - 3 * zz) * g[13]
            + b * y * z * g[14]
            + 3 * a * (zz - xx) * g[15]
        )
    grad_u = torch.stack([gx, gy, gz], dim=-1)
    projection = u * (u * grad_u).sum(-1, keepdim=True)
    return (grad_u - torch.where(radius >= 1e-12, projection, 0.0)) / divisor


class _AngularFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vectors, lmax):
        ctx.save_for_backward(vectors)
        ctx.lmax = lmax
        if vectors.is_cuda:
            from .angular_kernels import forward

            return forward(vectors, lmax)
        return _torch_forward(vectors, lmax)

    @staticmethod
    @once_differentiable
    def backward(ctx, cotangent):
        (vectors,) = ctx.saved_tensors
        if vectors.is_cuda:
            from .angular_kernels import backward

            return backward(vectors, cotangent, ctx.lmax), None
        return _torch_vjp(vectors, cotangent, ctx.lmax), None


class AngularAdjoint(torch.nn.Module):
    """Inference-only module for MACE position derivatives; no trainable weights."""

    def __init__(self, lmax):
        super().__init__()
        if lmax not in range(4):
            raise ValueError("supported angular degrees are 0 through 3")
        self.lmax = lmax

    def forward(self, vectors):
        if vectors.ndim != 2 or vectors.shape[1] != 3:
            raise ValueError("expected (E,3) Cartesian edge vectors")
        if vectors.dtype not in (torch.float32, torch.float64):
            raise ValueError("the angular adjoint requires FP32 or FP64")
        return _AngularFunction.apply(vectors, self.lmax)


@contextmanager
def install_angular_adjoint(model):
    """Temporarily specialize the angular block of a frozen stock MACE model.

    Use inside the candidate path only. Native controls retain the original
    module. Backward computes physical forces; force fitting and Hessians need
    further derivatives and must use the original e3nn graph.
    """
    from e3nn.o3 import SphericalHarmonics

    original = model.spherical_harmonics
    if type(original) is not SphericalHarmonics:
        raise ValueError("expected an unmodified e3nn SphericalHarmonics module")
    lmax = int(original._lmax)
    if (
        not original.normalize
        or original.normalization != "component"
        or list(original._ls_list) != list(range(lmax + 1))
        or lmax > 3
    ):
        raise ValueError(
            "expected normalized component harmonics in degree order 0 through lmax<=3"
        )
    # Pin the exact installed e3nn basis, normalization and ordering against a
    # small deterministic probe, outside the callback and timing.
    ref = next(model.parameters())
    replacement = AngularAdjoint(lmax)
    probe = torch.tensor(
        [[1.0, 2.0, 3.0], [-2.0, 1.0, 1.0], [0.7, -1.3, 0.4], [-0.8, -0.6, 0.1]],
        dtype=ref.dtype,
        device=ref.device,
    )
    with torch.no_grad():
        torch.testing.assert_close(
            _torch_forward(probe, lmax),
            original(probe),
            atol=1e-12 if ref.dtype == torch.float64 else 2e-6,
            rtol=1e-12 if ref.dtype == torch.float64 else 2e-6,
        )
    model.spherical_harmonics = replacement
    try:
        yield replacement
    finally:
        model.spherical_harmonics = original
