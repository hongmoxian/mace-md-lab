"""Compile frozen scalar MACE contractions into sparse homogeneous polynomials.

The checkpoint defines every coefficient, including the Clebsch–Gordan basis.
Distinct permutations of a commutative monomial are combined in float64.
CUDA can compute feature derivatives in the forward pass or defer them to the VJP.
It supports first derivatives of energy for MD; use native MACE for training.
"""

from collections import Counter
from functools import lru_cache
from itertools import combinations_with_replacement, permutations

import numpy as np
import torch
from torch import nn
from torch.autograd.function import once_differentiable


def _combine(contraction):
    degree = int(contraction.correlation)
    dimension = int(contraction.U_tensors(1).shape[0])
    weights = [contraction.weights_max, *contraction.weights]
    terms, tables = [], []
    for order in range(1, degree + 1):
        coupling = contraction.U_tensors(order).detach().cpu().double().numpy()
        w = weights[degree - order].detach().cpu().double().numpy()
        if coupling.shape != (dimension,) * order + (w.shape[1],):
            raise ValueError("unexpected coefficient tensor for a scalar product")
        indices, rows = [], []
        for ix in combinations_with_replacement(range(dimension), order):
            # No magnitude threshold. Drop a term only when all path factors are zero.
            row = np.sum(
                [coupling[key] for key in sorted(set(permutations(ix)))], axis=0
            )
            if np.any(row):
                indices.append(ix)
                rows.append(row)
        if rows:
            # [element, term, feature channel]; adjacent CUDA threads read channels.
            tables.append(np.stack(rows) @ w)
            terms.extend(indices)
    if not terms:
        raise ValueError("scalar contraction has no nonzero coupling paths")
    return dimension, tuple(terms), np.concatenate(tables, axis=1)


@lru_cache(maxsize=32)
def _source(dimension, terms, scalar, stage):
    """One CUDA thread evaluates a channel. The JIT can keep x and its VJP in registers."""
    # Same term order, loads, coefficients and arithmetic in all three kernels.
    value_needed = stage in ("value", "value_and_derivative")
    derivative_needed = stage in ("vjp", "value_and_derivative")
    statements = []
    for t, ix in enumerate(terms):
        # Multiplying a factor multiple times calls for the multiplicity in its derivative.
        val = " * ".join(f"x{i}" for i in ix)
        lines = [
            f"const real a = coefficient({t}, ch, channels, table, attrs, ne, z, scale);"
        ]
        if value_needed:
            lines.append(f"value += a * ({val});")
        for i, count in (Counter(ix).items() if derivative_needed else ()):
            factors = list(ix)
            factors.remove(i)
            grad = " * ".join(f"x{k}" for k in factors) or "1"
            lines.append(f"g{i} += {count} * a * ({grad});")
        statements.append("{ " + " ".join(lines) + " }")
    feature_loads = "\n".join(
        f"const real x{i} = x[base + {i}];" for i in range(dimension)
    )
    grad_decl = (
        "\n".join(f"real g{i} = 0;" for i in range(dimension))
        if derivative_needed
        else ""
    )
    if stage == "vjp":
        # Here value_buffer is the incoming adjoint; we write the final dE/dx directly.
        output = "const real upstream = value_buffer[(long long)atom * channels + ch];"
        output += "\n" + "\n".join(
            f"dx[base + {i}] = upstream * g{i};" for i in range(dimension)
        )
    else:
        output = "value_buffer[(long long)atom * channels + ch] = value;"
        if derivative_needed:
            output += "\n" + "\n".join(
                f"dx[base + {i}] = g{i};" for i in range(dimension)
            )
    operations = "\n".join(statements)
    return f"""
typedef {scalar} real;
static __device__ __forceinline__ real coefficient(
    int term, int ch, int channels, const real *w, const real *attrs, int ne,
    int z, real scale) {{
    if (z >= 0) return scale * w[((long long)z * {len(terms)} + term) * channels + ch];
    if (z == -1) return 0;
    real a = 0;
    for (int e = 0; e < ne; ++e)
        a += attrs[e] * w[((long long)e * {len(terms)} + term) * channels + ch];
    return a;
}}
extern "C" __global__ void scalar_product(const real *x, const real *y, const real *table,
                                          real *value_buffer, real *dx, int channels, int ne) {{
    const int atom = blockIdx.x;
    const int ch = blockIdx.y * blockDim.x + threadIdx.x;
    const real *attrs = y + (long long)atom * ne;
    __shared__ int element;
    __shared__ real factor;
    if (threadIdx.x == 0) {{
        int which = -1;
        real weight = 0;
        for (int e = 0; e < ne; ++e) {{
            if (attrs[e] == 0) continue;
            if (which >= 0) {{ which = -2; break; }}
            which = e;
            weight = attrs[e];
        }}
        element = which; factor = weight;
    }}
    __syncthreads();
    if (ch >= channels) return;
    const int z = element;
    const real scale = factor;
    const long long base = ((long long)atom * channels + ch) * {dimension};
    {feature_loads}
    {grad_decl}
    real value = 0;
    {operations}
    {output}
}}
"""


@lru_cache(maxsize=32)
def _cuda_kernel(dimension, terms, dtype, stage):
    import cupy as cp

    return cp.RawKernel(
        _source(dimension, terms, dtype, stage),
        "scalar_product",
        options=("--std=c++11",),
        backend="nvrtc",
    )


def _launch(stage, x, attrs, table, value_buffer, dx, terms):
    import cupy as cp

    n, channels, dimension = x.shape
    if not n:
        return
    with cp.cuda.Device(x.device.index):
        stream = cp.cuda.ExternalStream(torch.cuda.current_stream(x.device).cuda_stream)
        kernel = _cuda_kernel(
            dimension, terms, "double" if x.dtype == torch.float64 else "float", stage
        )
        # RawKernel copies pointer values at launch; torch owns all the allocations.
        tensors = (x, attrs, table, value_buffer, dx)
        pointers = tuple(np.uint64(0 if t is None else t.data_ptr()) for t in tensors)
        kernel(
            (n, (channels + 63) // 64),
            (64,),
            (*pointers, np.int32(channels), np.int32(attrs.shape[1])),
            stream=stream,
        )


class _FrozenProductFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, attrs, table, terms, schedule):
        x, attrs = x.contiguous(), attrs.contiguous()
        out = x.new_empty(x.shape[:2])
        ctx.schedule = schedule
        if schedule == "precompute":
            dx = torch.empty_like(x)
            _launch("value_and_derivative", x, attrs, table, out, dx, terms)
            ctx.save_for_backward(dx)
        else:
            _launch("value", x, attrs, table, out, None, terms)
            ctx.terms = terms
            ctx.save_for_backward(x, attrs, table)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out):
        if ctx.schedule == "deferred":
            x, attrs, table = ctx.saved_tensors
            grad_x = torch.empty_like(x)
            _launch("vjp", x, attrs, table, grad_out.contiguous(), grad_x, ctx.terms)
        else:
            (dx,) = ctx.saved_tensors
            grad_x = grad_out.unsqueeze(-1) * dx
        return grad_x, None, None, None, None


class FrozenScalarProduct(nn.Module):
    """A frozen scalar output. Replaces a native, one-output SymmetricContraction."""

    def __init__(self, contraction, schedule="precompute"):
        super().__init__()
        if schedule not in ("precompute", "deferred"):
            raise ValueError("unknown frozen product schedule")
        self.schedule = schedule
        dim, terms, table = _combine(contraction)
        w = contraction.weights_max
        self.dimension, self.terms = dim, terms
        self.register_buffer(
            "table", torch.as_tensor(table, device=w.device, dtype=w.dtype).contiguous()
        )
        # Unity sentinel allows one formula for linear, quadratic and cubic terms.
        indices = [(*ix, *((dim,) * (3 - len(ix)))) for ix in terms]
        self.register_buffer(
            "indices",
            torch.tensor(indices, device=w.device, dtype=torch.long).T.contiguous(),
        )

    def forward(self, x, y):
        if x.shape[1:] != (self.table.shape[-1], self.dimension) or y.shape != (
            x.shape[0],
            self.table.shape[0],
        ):
            raise ValueError("unexpected scalar contraction input dimensions")
        if (
            x.dtype != self.table.dtype
            or y.dtype != x.dtype
            or x.device != self.table.device
            or y.device != x.device
        ):
            raise ValueError(
                "scalar contraction input dtype/device does not match weights"
            )
        if x.is_cuda:
            if y.requires_grad:
                raise ValueError("CUDA frozen products provide position gradients only")
            return _FrozenProductFn.apply(x, y, self.table, self.terms, self.schedule)
        vals = torch.cat((x, torch.ones_like(x[:, :, :1])), dim=-1)
        coeff = (y @ self.table.flatten(1)).view(
            len(y), self.table.size(1), self.table.size(2)
        )
        return (
            coeff.transpose(1, 2)
            * vals[:, :, self.indices[0]]
            * vals[:, :, self.indices[1]]
            * vals[:, :, self.indices[2]]
        ).sum(-1)


def install_frozen_products(model, atomic_numbers=None, schedule="precompute"):
    """Specialize scalar-only product blocks on this inference copy; return their count.

    All checkpoint species are covered. atomic_numbers optionally checks compatibility
    with a user's fixed-composition simulation. Vector-output blocks remain native.
    """
    from mace.modules.symmetric_contraction import SymmetricContraction

    if atomic_numbers is not None:
        available = set(model.atomic_numbers.detach().cpu().tolist())
        if not set(map(int, atomic_numbers)).issubset(available):
            raise ValueError("structure contains elements absent from the checkpoint")
    replacements = []
    for block in model.products:
        sym = block.symmetric_contractions
        if (
            type(sym) is not SymmetricContraction
            or len(sym.contractions) != 1
            or sym.irreps_out[0].ir.l != 0
            or sym.irreps_out[0].ir.p != 1
        ):
            continue
        c = sym.contractions[0]
        if not 1 <= c.correlation <= 3 or c.U_tensors(1).ndim != 2:
            continue
        if c.weights_max.dtype not in (torch.float32, torch.float64):
            raise ValueError("frozen scalar products support FP32 and FP64")
        if c.U_tensors(1).shape[0] > 16:
            continue
        if any(m._forward_hooks or m._forward_pre_hooks for m in sym.modules()):
            raise ValueError(
                "cannot specialize a scalar contraction with forward hooks"
            )
        replacements.append((block, FrozenScalarProduct(c, schedule=schedule)))
    for block, optimized in replacements:
        block.symmetric_contractions = optimized
    return len(replacements)
