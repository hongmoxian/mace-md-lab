"""Distances and normalized Cartesian edge vectors with exact first derivatives.

With no epsilon, unit(0)=0 and its selected derivative at zero is zero. With
epsilon>0, unit=d/max(r,epsilon), including the linear branch at r=0 (the
radial branch is used at the kink). Distance is always the true Euclidean norm
and its selected gradient is zero at coincidence. A false edge_mask entry gives
zero outputs and zero gradients, including for a padded -1 endpoint. Shifts are
Cartesian vectors, so their derivatives propagate to a differentiable cell.
The analytic backend differentiates once; use native for force training/Hessians.
"""

import math

import torch
from torch.autograd.function import once_differentiable

try:
    import triton
    import triton.language as tl
except ImportError:  # The CPU implementation requires only PyTorch.
    triton = None


if triton is not None:

    @triton.jit
    def _cartesian(
        P,
        I,
        S,
        M,
        edge,
        pos0: tl.constexpr,
        pos1: tl.constexpr,
        idx0: tl.constexpr,
        idx1: tl.constexpr,
        sh0: tl.constexpr,
        sh1: tl.constexpr,
        ms: tl.constexpr,
        n,
        count,
        FROM_POS: tl.constexpr,
        SHIFT: tl.constexpr,
        MASK: tl.constexpr,
    ):
        active = edge < count
        if MASK:
            active = active & tl.load(M + edge * ms, edge < count, other=0).to(tl.int1)
        if FROM_POS:
            src = tl.load(I + edge * idx1, active, other=0)
            dst = tl.load(I + idx0 + edge * idx1, active, other=0)
            valid = active & (src >= 0) & (src < n) & (dst >= 0) & (dst < n)
            s = P + src * pos0
            t = P + dst * pos0
            x = tl.load(t, valid, other=0.0) - tl.load(s, valid, other=0.0)
            y = tl.load(t + pos1, valid, other=0.0) - tl.load(
                s + pos1, valid, other=0.0
            )
            z = tl.load(t + 2 * pos1, valid, other=0.0) - tl.load(
                s + 2 * pos1, valid, other=0.0
            )
            if SHIFT:
                shift = S + edge * sh0
                x += tl.load(shift, valid, other=0.0)
                y += tl.load(shift + sh1, valid, other=0.0)
                z += tl.load(shift + 2 * sh1, valid, other=0.0)
        else:
            src, dst, valid = edge, edge, active
            p = P + edge * pos0
            x = tl.load(p, valid, other=0.0)
            y = tl.load(p + pos1, valid, other=0.0)
            z = tl.load(p + 2 * pos1, valid, other=0.0)
        return x, y, z, src, dst, valid, active

    @triton.jit
    def _forward(
        P,
        I,
        S,
        M,
        R,
        U,
        n,
        count,
        pos0: tl.constexpr,
        pos1: tl.constexpr,
        idx0: tl.constexpr,
        idx1: tl.constexpr,
        sh0: tl.constexpr,
        sh1: tl.constexpr,
        ms: tl.constexpr,
        EPS: tl.constexpr,
        FROM_POS: tl.constexpr,
        SHIFT: tl.constexpr,
        MASK: tl.constexpr,
        WRITE_R: tl.constexpr,
        WRITE_U: tl.constexpr,
        B: tl.constexpr,
    ):
        edge = tl.program_id(0) * B + tl.arange(0, B)
        x, y, z, src, dst, valid, active = _cartesian(
            P,
            I,
            S,
            M,
            edge,
            pos0,
            pos1,
            idx0,
            idx1,
            sh0,
            sh1,
            ms,
            n,
            count,
            FROM_POS,
            SHIFT,
            MASK,
        )
        r = tl.sqrt(x * x + y * y + z * z)
        # Malformed active CUDA indices produce a visible NaN without out-of-bounds reads.
        bad = active & ~valid
        if WRITE_R:
            tl.store(R + edge, tl.where(bad, float("nan"), r), edge < count)
        if WRITE_U:
            if EPS > 0:
                # A bare Python epsilon may be rounded to FP32 by Triton promotion.
                # Materialize it directly in coordinate precision (e.g. 0.3 in FP64).
                epsilon = tl.full((), EPS, P.dtype.element_ty)
                denom = tl.maximum(r, epsilon)
            else:
                denom = tl.where(r > 0, r, 1.0)
            tl.store(U + edge * 3, tl.where(bad, float("nan"), x / denom), edge < count)
            tl.store(
                U + edge * 3 + 1, tl.where(bad, float("nan"), y / denom), edge < count
            )
            tl.store(
                U + edge * 3 + 2, tl.where(bad, float("nan"), z / denom), edge < count
            )

    @triton.jit
    def _reverse(
        P,
        I,
        S,
        M,
        R,
        GR,
        GU,
        GP,
        GS,
        n,
        count,
        pos0: tl.constexpr,
        pos1: tl.constexpr,
        idx0: tl.constexpr,
        idx1: tl.constexpr,
        sh0: tl.constexpr,
        sh1: tl.constexpr,
        ms: tl.constexpr,
        gr0: tl.constexpr,
        gu0: tl.constexpr,
        gu1: tl.constexpr,
        EPS: tl.constexpr,
        FROM_POS: tl.constexpr,
        SHIFT: tl.constexpr,
        MASK: tl.constexpr,
        SAVED_R: tl.constexpr,
        HAS_GR: tl.constexpr,
        HAS_GU: tl.constexpr,
        WANT_GP: tl.constexpr,
        WANT_GS: tl.constexpr,
        B: tl.constexpr,
    ):
        edge = tl.program_id(0) * B + tl.arange(0, B)
        x, y, z, src, dst, valid, active = _cartesian(
            P,
            I,
            S,
            M,
            edge,
            pos0,
            pos1,
            idx0,
            idx1,
            sh0,
            sh1,
            ms,
            n,
            count,
            FROM_POS,
            SHIFT,
            MASK,
        )
        if SAVED_R:
            r = tl.load(R + edge, valid, other=0.0)
        else:
            r = tl.sqrt(x * x + y * y + z * z)
        inv = tl.where(r > 0, 1.0 / tl.where(r > 0, r, 1.0), 0.0)
        if HAS_GR:
            radial = tl.load(GR + edge * gr0, valid, other=0.0) * inv
            gx, gy, gz = radial * x, radial * y, radial * z
        else:
            gx, gy, gz = x * 0.0, y * 0.0, z * 0.0
        if HAS_GU:
            q = GU + edge * gu0
            a = tl.load(q, valid, other=0.0)
            b = tl.load(q + gu1, valid, other=0.0)
            c = tl.load(q + gu1 * 2, valid, other=0.0)
            if EPS > 0:
                epsilon = tl.full((), EPS, P.dtype.element_ty)
                inv_denom = 1.0 / tl.maximum(r, epsilon)
                project = r >= epsilon
            else:
                inv_denom = inv
                project = r > 0
            ux, uy, uz = x * inv_denom, y * inv_denom, z * inv_denom
            dot = tl.where(project, ux * a + uy * b + uz * c, 0.0)
            gx += (a - ux * dot) * inv_denom
            gy += (b - uy * dot) * inv_denom
            gz += (c - uz * dot) * inv_denom
        if WANT_GP:
            if FROM_POS:
                # A periodic self edge depends on its image shift alone.
                push = valid & (src != dst)
                tl.atomic_add(GP + 3 * src, -gx, push, sem="relaxed")
                tl.atomic_add(GP + 3 * src + 1, -gy, push, sem="relaxed")
                tl.atomic_add(GP + 3 * src + 2, -gz, push, sem="relaxed")
                tl.atomic_add(GP + 3 * dst, gx, push, sem="relaxed")
                tl.atomic_add(GP + 3 * dst + 1, gy, push, sem="relaxed")
                tl.atomic_add(GP + 3 * dst + 2, gz, push, sem="relaxed")
            else:
                tl.store(GP + 3 * edge, tl.where(valid, gx, 0.0), edge < count)
                tl.store(GP + 3 * edge + 1, tl.where(valid, gy, 0.0), edge < count)
                tl.store(GP + 3 * edge + 2, tl.where(valid, gz, 0.0), edge < count)
        if WANT_GS:
            tl.store(GS + 3 * edge, tl.where(valid, gx, 0.0), edge < count)
            tl.store(GS + 3 * edge + 1, tl.where(valid, gy, 0.0), edge < count)
            tl.store(GS + 3 * edge + 2, tl.where(valid, gz, 0.0), edge < count)


def _displacement(p, index, shifts, mask):
    if index is None:
        d = p
    else:
        # Replace padded indices before gathering; they may be -1.
        i = index if mask is None else torch.where(mask[None, :], index, 0)
        d = p.index_select(0, i[1]) - p.index_select(0, i[0])
        if shifts is not None:
            d = d + shifts
    return d if mask is None else torch.where(mask[:, None], d, 0.0)


def _normalization(d, r, epsilon):
    if epsilon:
        return d / r.clamp_min(epsilon)[:, None]
    positive = r > 0
    denom = torch.where(positive, r, 1.0)
    return torch.where(positive[:, None], d / denom[:, None], 0.0)


def _strides(p, index, shifts, mask):
    return (
        *p.stride(),
        *(index.stride() if index is not None else (0, 0)),
        *(shifts.stride() if shifts is not None else (0, 0)),
        mask.stride(0) if mask is not None else 0,
    )


class _EdgeGeometry(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p, index, shifts, mask, epsilon, want_r, want_u):
        ctx.set_materialize_grads(False)
        count = p.shape[0] if index is None else index.shape[1]
        if p.is_cuda:
            if triton is None:
                raise RuntimeError("analytic CUDA geometry requires Triton")
            r = p.new_empty((count,)) if want_r else None
            u = p.new_empty((count, 3)) if want_u else None
            if count:
                with torch.cuda.device(p.device):
                    _forward[(triton.cdiv(count, 128),)](
                        p,
                        index,
                        shifts,
                        mask,
                        r,
                        u,
                        len(p),
                        count,
                        *_strides(p, index, shifts, mask),
                        epsilon,
                        index is not None,
                        shifts is not None,
                        mask is not None,
                        want_r,
                        want_u,
                        128,
                        enable_fp_fusion=False
                    )
        else:
            d = _displacement(p, index, shifts, mask)
            raw = d.norm(dim=-1)
            u = _normalization(d, raw, epsilon) if want_u else None
            r = raw if want_r else None
        ctx.save_for_backward(p, index, shifts, mask, r)
        ctx.epsilon = epsilon
        return r, u

    @staticmethod
    @once_differentiable
    def backward(ctx, gr, gu):
        p, index, shifts, mask, r = ctx.saved_tensors
        eps = ctx.epsilon
        want_p, want_s = ctx.needs_input_grad[0], ctx.needs_input_grad[2]
        if gr is None and gu is None:
            return (None,) * 7
        count = p.shape[0] if index is None else index.shape[1]
        if p.is_cuda:
            gp = (
                (p.new_zeros(p.shape) if index is not None else p.new_empty(p.shape))
                if want_p
                else None
            )
            gs = p.new_empty((count, 3)) if want_s else None
            if count:
                with torch.cuda.device(p.device):
                    _reverse[(triton.cdiv(count, 128),)](
                        p,
                        index,
                        shifts,
                        mask,
                        r,
                        gr,
                        gu,
                        gp,
                        gs,
                        len(p),
                        count,
                        *_strides(p, index, shifts, mask),
                        gr.stride(0) if gr is not None else 0,
                        *(gu.stride() if gu is not None else (0, 0)),
                        eps,
                        index is not None,
                        shifts is not None,
                        mask is not None,
                        r is not None,
                        gr is not None,
                        gu is not None,
                        want_p,
                        want_s,
                        128,
                        enable_fp_fusion=False
                    )
        else:
            d = _displacement(p, index, shifts, mask)
            if r is None:
                r = d.norm(dim=1)
            positive = r > 0
            inv = torch.where(positive, 1.0 / torch.where(positive, r, 1.0), 0.0)
            gd = d * (gr * inv)[:, None] if gr is not None else torch.zeros_like(d)
            if gu is not None:
                invden = 1.0 / r.clamp_min(eps) if eps else inv
                u = d * invden[:, None]
                dot = (gu * u).sum(-1)
                dot = torch.where(r >= eps if eps else positive, dot, 0.0)
                gd = gd + (gu - u * dot[:, None]) * invden[:, None]
            if mask is not None:
                gd = torch.where(mask[:, None], gd, 0.0)
            gp = gd if want_p else None
            if want_p and index is not None:
                i = index if mask is None else torch.where(mask[None, :], index, 0)
                edge_grad = torch.where((i[0] != i[1])[:, None], gd, 0.0)
                gp = (
                    torch.zeros_like(p)
                    .index_add_(0, i[0], -edge_grad)
                    .index_add_(0, i[1], edge_grad)
                )
            gs = gd if want_s else None
        return gp, None, gs, None, None, None, None


def _geometry(p, index, shifts, mode, epsilon, quantities, mask):
    if p.ndim != 2 or p.shape[1] != 3 or p.dtype not in (torch.float32, torch.float64):
        raise ValueError("coordinates must be FP32 or FP64 with shape (*,3)")
    if index is not None and (
        index.ndim != 2
        or index.shape[0] != 2
        or index.dtype not in (torch.int32, torch.int64)
        or index.device != p.device
    ):
        raise ValueError(
            "edge_index must be a (2,E) integer tensor on the coordinate device"
        )
    count = p.shape[0] if index is None else index.shape[1]
    if count and not len(p):
        raise ValueError("a nonempty edge list needs at least one position")
    if shifts is not None and (
        shifts.shape != (count, 3)
        or shifts.dtype != p.dtype
        or shifts.device != p.device
    ):
        raise ValueError(
            "Cartesian shifts must be (E,3) on the coordinate dtype/device"
        )
    if mask is not None and (
        mask.shape != (count,) or mask.dtype != torch.bool or mask.device != p.device
    ):
        raise ValueError("edge_mask must be boolean (E,) on the coordinate device")
    eps = 0.0 if epsilon is None else float(epsilon)
    if not math.isfinite(eps) or eps < 0:
        raise ValueError("distance_epsilon must be nonnegative and finite")
    if quantities not in ("both", "distance", "unit"):
        raise ValueError("quantities must be 'both', 'distance' or 'unit'")
    want_r, want_u = quantities != "unit", quantities != "distance"
    if mode == "native":
        d = _displacement(p, index, shifts, mask)
        raw = d.norm(dim=1)
        r = raw if want_r else None
        u = _normalization(d, raw, eps) if want_u else None
    elif mode == "analytic":
        r, u = _EdgeGeometry.apply(p, index, shifts, mask, eps, want_r, want_u)
    else:
        raise ValueError("mode must be 'native' or 'analytic'")
    return (r, u) if want_r and want_u else (r if want_r else u)


def edge_geometry(
    displacements,
    mode="native",
    distance_epsilon=None,
    *,
    quantities="both",
    edge_mask=None
):
    """Return distance (E,) and/or unit (E,3) for a tensor of Cartesian edge vectors."""
    return _geometry(
        displacements, None, None, mode, distance_epsilon, quantities, edge_mask
    )


def edge_geometry_from_positions(
    positions,
    edge_index,
    shifts=None,
    mode="native",
    distance_epsilon=None,
    *,
    quantities="both",
    edge_mask=None
):
    """Evaluate d=positions[dst]-positions[src]+shifts and project its first derivative.

    Active endpoints must be valid indices. A false edge_mask allows arbitrary padded
    endpoints and completely removes that edge. Use quantities='distance' or 'unit'
    to return one tensor and skip the other output. Default 'both' returns (r, unit).
    """
    if edge_index is None:
        raise ValueError("edge_index is required")
    return _geometry(
        positions, edge_index, shifts, mode, distance_epsilon, quantities, edge_mask
    )
