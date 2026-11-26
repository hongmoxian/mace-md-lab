"""Fixed, scalar-distance filters for the benchmark SchNet and PaiNN models.

The exact schedule moves each layer's cutoff through its last affine filter
projection. The table schedule interpolates those same filters and differentiates
the interpolation polynomial in the reverse pass.
"""

import math
from time import perf_counter

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from .geometry import edge_geometry_from_positions


@triton.jit
def _gaussian_features(R, MU, COEFF, OMEGA, OUT, DY, edges, width: tl.constexpr,
                       weighted: tl.constexpr, cols: tl.constexpr):
    row = tl.program_id(0) * 8 + tl.arange(0, 8)
    col = tl.arange(0, cols)
    r = tl.load(R + row, row < edges, other=0.0)
    omega = tl.load(OMEGA)
    angle = r * omega
    gate = 0.5 * (1.0 + libdevice.cos(angle))
    dg = -0.5 * omega * libdevice.sin(angle)
    mu = tl.load(MU + col, col < width, other=0.0)
    coeff = tl.load(COEFF + col, col < width, other=0.0)
    u = r[:, None] - mu[None, :]
    value = libdevice.exp(coeff[None, :] * u * u)
    deriv = (2 * coeff[None, :] * u) * value
    if weighted:
        deriv = deriv * gate[:, None] + value * dg[:, None]
        value *= gate[:, None]
    ptr = row[:, None] * (width + 1) + col[None, :]
    active = (row[:, None] < edges) & (col[None, :] <= width)
    tl.store(OUT + ptr, tl.where(col[None, :] == width, gate[:, None], value), active)
    tl.store(DY + ptr, tl.where(col[None, :] == width, dg[:, None], deriv), active)


@triton.jit
def _feature_adjoint(G, DY, DR, edges, width: tl.constexpr, cols: tl.constexpr):
    row = tl.program_id(0) * 8 + tl.arange(0, 8)
    col = tl.arange(0, cols)
    ptr = row[:, None] * width + col[None, :]
    mask = (row[:, None] < edges) & (col[None, :] < width)
    g = tl.load(G + ptr, mask, other=0.0)
    dy = tl.load(DY + ptr, mask, other=0.0)
    tl.store(DR + row, tl.sum(g * dy, axis=1), row < edges)


class _GaussianGate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r, mu, coeff, omega, weighted):
        width = mu.numel()
        out = r.new_empty((r.numel(), width + 1))
        dy = torch.empty_like(out)
        if r.numel():
            _gaussian_features[(triton.cdiv(r.numel(), 8),)](
                r, mu, coeff, omega, out, dy, r.numel(), width, weighted,
                triton.next_power_of_2(width + 1),
            )
        ctx.save_for_backward(dy)
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad):
        (dy,) = ctx.saved_tensors
        dr = dy.new_empty((dy.shape[0],))
        if dr.numel():
            _feature_adjoint[(triton.cdiv(dr.numel(), 8),)](
                grad.contiguous(), dy, dr, dr.numel(), dy.shape[1],
                triton.next_power_of_2(dy.shape[1]),
            )
        return dr, None, None, None, None


@triton.jit
def _cubic_values(R, COEF, OUT, width: tl.constexpr, n: tl.constexpr,
                  inv_h: tl.constexpr, tile: tl.constexpr):
    edge = tl.program_id(0)
    c = tl.program_id(1) * tile + tl.arange(0, tile)
    t = tl.load(R + edge) * inv_h
    k = tl.minimum(t.to(tl.int32), n - 1)
    t -= k.to(tl.float64)
    base = (k * 4 * width) + c
    a = tl.load(COEF + base, c < width, other=0.0)
    b = tl.load(COEF + base + width, c < width, other=0.0)
    d = tl.load(COEF + base + 2 * width, c < width, other=0.0)
    f = tl.load(COEF + base + 3 * width, c < width, other=0.0)
    tl.store(OUT + edge * width + c, a + t * (b + t * (d + t * f)), c < width)


@triton.jit
def _cubic_adjoint(R, COEF, GRAD, DR, width: tl.constexpr, n: tl.constexpr,
                   inv_h: tl.constexpr, tile: tl.constexpr):
    edge = tl.program_id(0)
    c = tl.arange(0, tile)
    t = tl.load(R + edge) * inv_h
    k = tl.minimum(t.to(tl.int32), n - 1)
    t -= k.to(tl.float64)
    base = (k * 4 * width) + c
    b = tl.load(COEF + base + width, c < width, other=0.0)
    d = tl.load(COEF + base + 2 * width, c < width, other=0.0)
    f = tl.load(COEF + base + 3 * width, c < width, other=0.0)
    g = tl.load(GRAD + edge * width + c, c < width, other=0.0)
    tl.store(DR + edge, tl.sum(g * (b + t * (2.0 * d + t * 3.0 * f)), axis=0) * inv_h)


class _Hermite(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r, coef, inv_h):
        n, _, width = coef.shape
        out = r.new_empty((r.numel(), width))
        ctx.save_for_backward(r, coef)
        ctx.inv_h = inv_h
        if r.numel():
            _cubic_values[(r.numel(), triton.cdiv(width, 128))](
                r, coef, out, width, n, inv_h, 128
            )
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad):
        r, coef = ctx.saved_tensors
        n, _, width = coef.shape
        dr = torch.empty_like(r)
        if r.numel():
            _cubic_adjoint[(r.numel(),)](
                r, coef, grad.contiguous(), dr, width, n, ctx.inv_h,
                triton.next_power_of_2(width), num_warps=4
            )
        return dr, None, None


class RadialFilters:
    """Prepared distance-only filters for an isolated, fixed-weight molecule."""

    def __init__(self, adapter, family, schedule, knots):
        self.adapter, self.family, self.schedule = adapter, family, schedule
        self.cutoff = adapter.cutoff
        gpu_tensor = adapter.numbers if family == "schnet" else adapter.z
        if not gpu_tensor.is_cuda or knots < 3 or schedule not in ("direct", "exact", "gaussian", "hermite"):
            raise ValueError("requires a supported FP64 CUDA trajectory, schedule and grid")
        if family == "schnet":
            m = adapter.model
            if (adapter.metadata["version"] != "2.6.1" or adapter.metadata["target"] != "QM9 U0"
                    or m.dipole or m.scale != 1.0):
                raise ValueError("requires the pinned scalar PyG U0 model")
            blocks = list(m.interactions)
            self.filter_width = m.num_filters
            self.first_weight = torch.cat([b.mlp[0].weight for b in blocks])
            self.first_bias = torch.cat([b.mlp[0].bias for b in blocks])
            self.last_weight = torch.stack([
                torch.cat((b.mlp[2].weight.T, b.mlp[2].bias[None])) for b in blocks
            ])
        elif family == "painn":
            rep = adapter.rep
            if rep.share_filters or len(rep.electronic_embeddings) or not isinstance(
                rep.filter_net.activation, torch.nn.Identity
            ):
                raise ValueError("requires the unshared affine filters of ethanol PaiNN")
            self.filter_width = 3 * rep.n_atom_basis
            self.last_weight = torch.cat((rep.filter_net.weight.T, rep.filter_net.bias[None]))
        else:
            raise ValueError(f"unsupported architecture {family}")
        if self.last_weight.dtype != torch.float64:
            raise ValueError("the benchmark is defined for FP64 weights")
        self.report = {"schedule": schedule}
        if schedule == "gaussian":
            if family == "schnet":
                basis = m.distance_expansion
                self.offsets = basis.offset
                self.coeffs = torch.full_like(self.offsets, basis.coeff)
            else:
                from schnetpack.nn import CosineCutoff, GaussianRBF

                if type(rep.radial_basis) is not GaussianRBF or type(rep.cutoff_fn) is not CosineCutoff:
                    raise ValueError("the fused derivative requires Gaussian bases and a cosine cutoff")
                self.offsets = rep.radial_basis.offsets
                self.coeffs = -0.5 / rep.radial_basis.widths.square()
            self.omega = self.offsets.new_tensor(math.pi / self.cutoff)
            self.report["basis_width"] = self.offsets.numel()
        if schedule == "hermite":
            self.prepare_table(knots)

    def exact_filters(self, r):
        if self.family == "painn":
            rep = self.adapter.rep
            d = r[:, None]
            if self.schedule == "gaussian":
                features = _GaussianGate.apply(r, self.offsets, self.coeffs, self.omega, True)
            else:
                basis = rep.radial_basis(d)[:, 0, :]
                augmented = torch.cat((basis, torch.ones_like(d)), dim=-1)
                features = augmented * rep.cutoff_fn(d)
            return features @ self.last_weight
        m = self.adapter.model
        if self.schedule == "gaussian":
            features = _GaussianGate.apply(r, self.offsets, self.coeffs, self.omega, False)
            basis, gate = features[:, :-1], features[:, -1]
        else:
            basis = m.distance_expansion(r)
            gate = 0.5 * (torch.cos(r * math.pi / self.cutoff) + 1)
        hidden = torch.nn.functional.linear(basis, self.first_weight, self.first_bias)
        hidden = m.interactions[0].mlp[1](hidden).view(len(r), len(m.interactions), self.filter_width)
        expanded = torch.cat((hidden, torch.ones_like(hidden[:, :, :1])), -1)
        out = torch.bmm((expanded * gate[:, None, None]).transpose(0, 1), self.last_weight)
        return out.transpose(0, 1).reshape(len(r), len(m.interactions) * self.filter_width)

    def prepare_table(self, knots):
        device = self.last_weight.device
        torch.cuda.synchronize(device)
        start = perf_counter()
        grid = torch.linspace(0.0, self.cutoff, knots, device=device, dtype=torch.float64)
        h = self.cutoff / (knots - 1)
        def values_and_slopes(grid):
            values, slopes = [], []
            for piece in grid.split(256):
                y, dy = torch.func.jvp(self.exact_filters, (piece,), (torch.ones_like(piece),))
                values.append(y)
                slopes.append(dy)
            return torch.cat(values), torch.cat(slopes)
        y, dy = values_and_slopes(grid)
        delta = y[1:] - y[:-1]
        v0, v1 = dy[:-1] * h, dy[1:] * h
        self.coef = torch.stack((y[:-1], v0, 3 * delta - 2 * v0 - v1, v0 + v1 - 2 * delta),
                                dim=1).contiguous()
        self.inv_h = 1.0 / h
        sample = grid[:-1] + h * 0.371
        y_ref, dy_ref = values_and_slopes(sample)
        a, b, c, d = self.coef.unbind(1)
        t = 0.371
        y_err = (a + t * (b + t * (c + t * d)) - y_ref).abs().max().item()
        dy_err = ((b + t * (2 * c + t * 3 * d)) * self.inv_h - dy_ref).abs().max().item()
        torch.cuda.synchronize(device)
        self.report.update(knots=knots, width=y.shape[1], table_bytes=self.coef.numel()*8,
                           table_and_grid_validation_s=perf_counter()-start,
                           max_off_grid_value_error=y_err, max_off_grid_slope_error=dy_err,
                           validation_grid_interior_fraction=t)

    def energy(self, x, edges, mode):
        if self.family == "schnet":
            r = edge_geometry_from_positions(x, edges, mode=mode, quantities="distance")
        else:
            r, unit = edge_geometry_from_positions(x, edges, mode=mode)
        if self.schedule == "direct":
            if self.family == "schnet":
                m = self.adapter.model
                basis = m.distance_expansion(r)
                by_layer = [
                    block.mlp(basis) * (0.5 * (torch.cos(r * math.pi / block.conv.cutoff) + 1))[:, None]
                    for block in m.interactions
                ]
            else:
                rep = self.adapter.rep
                d = r[:, None]
                weights = rep.filter_net(rep.radial_basis(d)) * rep.cutoff_fn(d).unsqueeze(-1)
                by_layer = weights[:, 0, :].split(self.filter_width, dim=1)
        else:
            filters = self.exact_filters(r) if self.schedule != "hermite" else _Hermite.apply(
                r.contiguous(), self.coef, self.inv_h
            )
            by_layer = filters.split(self.filter_width, dim=1)
        src, dst = edges
        a = self.adapter
        if self.family == "schnet":
            m = a.model
            h = m.embedding(a.numbers)
            for weight, block in zip(by_layer, m.interactions, strict=True):
                terms = block.conv.lin1(h)[src] * weight
                summed = h.new_zeros((len(h), self.filter_width)).index_add_(0, dst, terms)
                h = h + block.lin(block.act(block.conv.lin2(summed)))
            h = m.lin2(m.act(m.lin1(h)))
            h = h * m.std + m.mean + m.atomref(a.numbers)
            return m.readout(h, a.batch, dim=0, dim_size=1).reshape(())
        from schnetpack.nn import scatter_add

        q = a.rep.embedding(a.z).unsqueeze(1)
        mu = q.new_zeros((len(q), 3, a.rep.n_atom_basis))
        for weights, block, mixing in zip(by_layer, a.rep.interactions, a.rep.mixing, strict=True):
            q, mu = block(q, mu, weights[:, None, :], -unit, dst, src, len(q))
            q, mu = mixing(q, mu)
        per_atom = a.head.outnet(q.squeeze(1))
        data = a._static_inputs()
        data["energy"] = scatter_add(per_atom, a.batch, dim_size=1).squeeze(-1)
        return a.model.postprocess(data)["energy"].reshape(()) * a.unit
