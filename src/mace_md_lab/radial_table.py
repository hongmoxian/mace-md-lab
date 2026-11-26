"""FP64 cubic Hermite lookup for the fixed, distance-only MP-0 Small radial MLPs.

The potential uses an interpolant between knots. The VJP evaluates the derivative
of that same interpolant. The grouped first interaction consumes 64 hidden
features; its learned final projection stays intact. The second interaction can
use either the same factorization or a table containing the 512 full outputs.
"""

from contextlib import contextmanager, ExitStack
from dataclasses import dataclass
from unittest.mock import patch

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable


@triton.jit
def _hermite_kernel(R, TABLE, OUT, SLOPE, edges, intervals: tl.constexpr,
                    cutoff: tl.constexpr, width: tl.constexpr, rows: tl.constexpr):
    edge = tl.program_id(0) * rows + tl.arange(0, rows)
    channel = tl.arange(0, width)
    radius = tl.load(R + edge, edge < edges, other=cutoff)
    scale = tl.full((), intervals, tl.float64) / tl.full((), cutoff, tl.float64)
    u = radius * scale
    index = tl.maximum(0, tl.minimum(u.to(tl.int32), intervals - 1))
    t = (u - index.to(tl.float64))[:, None]
    ptr = TABLE + index[:, None] * (4 * width) + channel[None, :]
    inside = ((edge < edges) & (radius >= 0) & (radius < cutoff))[:, None]
    value = tl.load(ptr, inside, other=0)
    linear = tl.load(ptr + width, inside, other=0)
    quadratic = tl.load(ptr + 2 * width, inside, other=0)
    cubic = tl.load(ptr + 3 * width, inside, other=0)
    # Clamp t for exterior candidate edges before polynomial evaluation.
    t = tl.where(inside, t, 0.0)
    value = ((cubic * t + quadratic) * t + linear) * t + value
    derivative = ((3.0 * cubic * t + 2.0 * quadratic) * t + linear) * scale
    out_idx = edge[:, None] * width + channel[None, :]
    tl.store(OUT + out_idx, value, edge[:, None] < edges)
    tl.store(SLOPE + out_idx, derivative, edge[:, None] < edges)


@triton.jit
def _radial_vjp(G, SLOPE, GR, edges, width: tl.constexpr, rows: tl.constexpr):
    edge = tl.program_id(0) * rows + tl.arange(0, rows)
    ch = tl.arange(0, width)
    addr = edge[:, None] * width + ch[None, :]
    active = edge[:, None] < edges
    incoming = tl.load(G + addr, active, other=0)
    slope = tl.load(SLOPE + addr, active, other=0)
    tl.store(GR + edge, tl.sum(incoming * slope, axis=1), edge < edges)


class _Hermite(torch.autograd.Function):
    @staticmethod
    def forward(ctx, radii, coeff, cutoff):
        if radii.ndim != 2 or radii.shape[-1] != 1:
            raise ValueError("expected one scalar length per edge")
        count = radii.shape[0]
        width = coeff.shape[-1]
        result = radii.new_empty((count, width))
        slope = torch.empty_like(result)
        if count:
            with torch.cuda.device(radii.device):
                _hermite_kernel[(triton.cdiv(count, 4),)](
                    radii.contiguous(), coeff, result, slope, count, coeff.shape[0],
                    cutoff, width, 4, num_warps=4,
                )
        ctx.save_for_backward(slope)
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, incoming):
        (slope,) = ctx.saved_tensors
        count, width = slope.shape
        result = slope.new_empty((count, 1))
        if count:
            with torch.cuda.device(slope.device):
                _radial_vjp[(triton.cdiv(count, 4),)](
                    incoming.contiguous(), slope, result, count, width, 4, num_warps=4,
                )
        return result, None, None


class _Lookup(torch.nn.Module):
    def __init__(self, coeff, cutoff):
        super().__init__()
        self.register_buffer("coeff", coeff)
        self.cutoff = cutoff

    def forward(self, radius):
        return _Hermite.apply(radius, self.coeff, self.cutoff)


def _exact_radial(embedding, layers, radius):
    from mace.modules.radial import BesselBasis

    basis = embedding.bessel_fn
    if type(basis) is BesselBasis:
        # The installed e3nn radial basis uses sin(omega*r)/r for r>0. Its
        # removable singularity at the zero knot has value omega and slope zero.
        safe = torch.where(radius == 0, torch.ones_like(radius), radius)
        basis_value = torch.where(radius == 0, basis.bessel_weights,
                                  torch.sin(basis.bessel_weights * radius) / safe)
        features = basis.prefactor * basis_value
    else:
        features = basis(radius)
    factors = embedding.cutoff_fn(radius)
    before_mlp = getattr(embedding, "apply_cutoff", True)
    hidden = features * factors if before_mlp else features
    for layer in layers:
        hidden = layer(hidden)
    return hidden if before_mlp else hidden * factors


@dataclass
class RadialTables:
    model: torch.nn.Module
    knots: int
    second_width: int
    cutoff: float
    lookup: tuple
    parameter_versions: tuple


def _versions(model):
    modules = [model.radial_embedding, *(layer.conv_tp_weights for layer in model.interactions)]
    return tuple((id(tensor), tensor._version) for module in modules
                 for tensor in (*module.parameters(), *module.buffers()))


def prepare_radial_tables(model, knots=8192, second_width=64):
    """Build frozen knot values and their directional derivatives outside timing."""
    from mace.modules.blocks import RadialEmbeddingBlock
    from mace.modules.models import ScaleShiftMACE
    from mace.modules.radial import BesselBasis, GaussianBasis, PolynomialCutoff

    embedding = model.radial_embedding
    if (type(model) is not ScaleShiftMACE or model.training
            or any(p.requires_grad for p in model.parameters())
            or len(model.interactions) != 2 or type(embedding) is not RadialEmbeddingBlock
            or type(embedding.bessel_fn) not in (BesselBasis, GaussianBasis)
            or type(embedding.cutoff_fn) is not PolynomialCutoff
            or hasattr(embedding, "distance_transform") or hasattr(model, "joint_embedding")):
        raise ValueError("expected fixed, distance-only MP-0 Small radial features")
    parameter = next(model.parameters())
    if parameter.device.type != "cuda" or parameter.dtype != torch.float64:
        raise ValueError("the tabulated implementation requires FP64 CUDA")
    if not isinstance(knots, int) or knots < 4 or second_width not in (64, 512):
        raise ValueError("use at least four knots and a second width of 64 or 512")
    cutoff = float(model.r_max)
    radii = torch.linspace(0, cutoff, knots, dtype=torch.float64, device=parameter.device)[:, None]
    step = cutoff / (knots - 1)
    lookups = []
    for index, interaction in enumerate(model.interactions):
        network = list(interaction.conv_tp_weights.children())
        widths = [layer.weight.shape for layer in network]
        if widths != [(10, 64), (64, 64), (64, 64), (64, 512)]:
            raise ValueError(f"unsupported radial widths {widths}")
        zero = torch.zeros((1, 64), device=parameter.device, dtype=torch.float64)
        if getattr(network[-1], "act", None) is not None or torch.count_nonzero(network[-1](zero)):
            raise ValueError("expected a bias-free linear last radial projection")
        width = 64 if index == 0 else second_width
        wanted = network[:-1] if width == 64 else network
        values, slopes = [], []
        for part in radii.split(512):
            # All rows are independent, so the tangent vector of ones gives
            # every component's own radial derivative in a single JVP.
            fn = lambda r: _exact_radial(embedding, wanted, r)
            y, dy = torch.autograd.functional.jvp(fn, part, torch.ones_like(part))
            values.append(y.detach())
            slopes.append(dy.detach())
        value, slope = torch.cat(values), torch.cat(slopes)
        if not torch.isfinite(value).all() or not torch.isfinite(slope).all():
            raise ValueError("nonfinite knot value or slope")
        if value[-1].abs().max() > 1e-12 or slope[-1].abs().max() > 1e-12:
            raise ValueError("the true radial value and slope must vanish at the cutoff")
        delta = value[1:] - value[:-1]
        s0, s1 = slope[:-1] * step, slope[1:] * step
        coeff = torch.stack((value[:-1], s0, 3 * delta - 2 * s0 - s1, s0 + s1 - 2 * delta), dim=1)
        lookups.append(_Lookup(coeff.contiguous(), cutoff))
    return RadialTables(model, knots, second_width, cutoff, tuple(lookups), _versions(model))


@contextmanager
def install_radial_table(model, knots=8192, second_width=64, *, tables=None):
    """Install before grouped-first, with optional prebuilt tables for this frozen model."""
    if tables is None:
        tables = prepare_radial_tables(model, knots, second_width)
    if (tables.model is not model or model.training or tables.parameter_versions != _versions(model)
            or any(p.requires_grad for p in model.parameters())):
        raise ValueError("the supplied coefficients require the original frozen radial parameters")
    with ExitStack() as stack:
        def use_radius(edge_lengths, node_attrs, edge_index, atomic_numbers):
            return edge_lengths, None

        stack.enter_context(patch.object(model.radial_embedding, "forward", use_radius))
        for index, (layer, lookup) in enumerate(zip(model.interactions, tables.lookup)):
            if index == 0 or tables.second_width == 64:
                end = list(layer.conv_tp_weights.children())[-1]
                replacement = torch.nn.Sequential(lookup, end)
            else:
                replacement = lookup
            stack.enter_context(patch.object(layer, "conv_tp_weights", replacement))
        yield {"knots": tables.knots, "cutoff_A": tables.cutoff, "first_width": 64,
               "second_width": tables.second_width, "interpolation": "cubic Hermite FP64"}
