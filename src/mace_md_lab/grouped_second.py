"""Experimental FP64 scalar-message convolution for the second MP-0 Small interaction.

Receiver-segmented kernels form and reduce directed messages in registers. The
backward recomputes their scalar factors and retains gradients with respect to
the sender features, the angular features and the radial network output.
"""

from contextlib import contextmanager
import math

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable


@triton.jit
def _reduce_by_receiver(
    X, Y, W, SEND, ORDER, START, OUT,
    CHANNELS: tl.constexpr, EDGE_TILE: tl.constexpr, CHANNEL_TILE: tl.constexpr,
):
    receiver = tl.program_id(0)
    chan = tl.program_id(1) * CHANNEL_TILE + tl.arange(0, CHANNEL_TILE)
    degree = tl.program_id(2)
    dim = 2 * degree + 1
    component = tl.arange(0, 8)
    begin = tl.load(START + receiver)
    end = tl.load(START + receiver + 1)
    value = tl.full((CHANNEL_TILE, 8), 0, tl.float64)
    for start in range(begin, end, EDGE_TILE):
        slot = start + tl.arange(0, EDGE_TILE)
        active = slot < end
        edge = tl.load(ORDER + slot, active, other=0).to(tl.int64)
        sender = tl.load(SEND + edge, active, other=0).to(tl.int64)
        features = tl.load(X + sender[:, None] * CHANNELS + chan[None, :], active[:, None], other=0)
        radial = tl.load(
            W + edge[:, None] * (4 * CHANNELS) + degree * CHANNELS + chan[None, :],
            active[:, None], other=0,
        )
        angular = tl.load(
            Y + edge[:, None] * 16 + degree * degree + component[None, :],
            active[:, None] & (component[None, :] < dim), other=0,
        )
        value += tl.sum((features * radial)[:, :, None] * angular[:, None, :], axis=0)
    offset = receiver * (16 * CHANNELS) + degree * degree * CHANNELS
    tl.store(OUT + offset + chan[:, None] * dim + component[None, :], value, component[None, :] < dim)


@triton.jit
def _message_vjp(X, Y, W, SEND, RECV, G, DX, DY, DW, CHANNELS: tl.constexpr):
    edge = tl.program_id(0)
    sender = tl.load(SEND + edge).to(tl.int64)
    receiver = tl.load(RECV + edge).to(tl.int64)
    chan = tl.arange(0, CHANNELS)
    component = tl.arange(0, 8)
    features = tl.load(X + sender * CHANNELS + chan)
    sender_gradient = tl.full((CHANNELS,), 0, tl.float64)
    for degree in tl.static_range(0, 4):
        dim = 2 * degree + 1
        coeff = tl.load(W + edge * (4 * CHANNELS) + degree * CHANNELS + chan)
        harmonic = tl.load(Y + edge * 16 + degree * degree + component, component < dim, other=0)
        block = receiver * (16 * CHANNELS) + degree * degree * CHANNELS
        incoming = tl.load(G + block + chan[:, None] * dim + component[None, :],
                           component[None, :] < dim, other=0)
        dot = tl.sum(incoming * harmonic[None, :], axis=1)
        sender_gradient += dot * coeff
        tl.store(DW + edge * (4 * CHANNELS) + degree * CHANNELS + chan, dot * features)
        harmonic_gradient = tl.sum(incoming * (coeff * features)[:, None], axis=0)
        tl.store(DY + edge * 16 + degree * degree + component, harmonic_gradient, component < dim)
    tl.atomic_add(DX + sender * CHANNELS + chan, sender_gradient, sem="relaxed")


class _ScalarConvolution(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, harmonics, weights, senders, receivers, layout):
        nodes, channels = features.shape
        output = features.new_empty((nodes, 16 * channels))
        if senders.numel() == 0:
            output.zero_()
        else:
            # Derive all metadata from this call's actual active edges. Two
            # graphs with equal numbers of edges can have different endpoints.
            if layout is None:
                sorted_receivers, order = receivers.sort()
                boundaries = torch.searchsorted(
                    sorted_receivers, torch.arange(nodes + 1, device=receivers.device, dtype=receivers.dtype)
                )
            else:
                order, boundaries = layout.order, layout.receiver_offsets
            with torch.cuda.device(features.device):
                _reduce_by_receiver[(nodes, channels // 32, 4)](
                    features, harmonics, weights, senders, order, boundaries, output,
                    channels, 8, 32, num_warps=4,
                )
        ctx.save_for_backward(features, harmonics, weights, senders, receivers)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        features, harmonics, weights, senders, receivers = ctx.saved_tensors
        grad_x = torch.zeros_like(features)
        grad_y = torch.empty_like(harmonics)
        grad_w = torch.empty_like(weights)
        if senders.numel():
            with torch.cuda.device(features.device):
                _message_vjp[(senders.numel(),)](
                    features, harmonics, weights, senders, receivers, gradient.contiguous(),
                    grad_x, grad_y, grad_w, features.shape[1], num_warps=4,
                )
        return grad_x, grad_y, grad_w, None, None, None


class _FusedTensorProduct(torch.nn.Module):
    def __init__(self, layout):
        super().__init__()
        self.layout = layout

    def forward(self, node_features, edge_attrs, weights, edge_index):
        edges = edge_index.shape[1]
        if (not node_features.is_cuda or node_features.dtype != torch.float64
                or node_features.shape[1] != 128 or edge_attrs.shape != (edges, 16)
                or weights.shape != (edges, 512) or edge_index.shape[0] != 2
                or edge_attrs.dtype != node_features.dtype or weights.dtype != node_features.dtype
                or edge_index.dtype not in (torch.int32, torch.int64)
                or any(t.device != node_features.device for t in (edge_attrs, weights, edge_index))):
            raise ValueError("expected GPU FP64 scalar UVU tensors for one MP-0 Small interaction")
        return _ScalarConvolution.apply(
            node_features.contiguous(), edge_attrs.contiguous(), weights.contiguous(),
            edge_index[0].contiguous(), edge_index[1].contiguous(), self.layout,
        )


@contextmanager
def install_second_interaction(model, atomic_numbers=None, *, layout=None):
    """Temporarily replace only the second interaction's tensor product and reduction.

    The node order and composition may change on each call. The optional atom
    numbers support the same call convention as the first-layer installer.
    """
    from mace.modules.blocks import RealAgnosticResidualInteractionBlock

    # The convolution reads every activation/weight tensor on each call. Its
    # first derivative also accepts grad-enabled parameters in upstream models.
    if len(model.interactions) < 2:
        raise ValueError("expected a MACE model with a second interaction")
    layer = model.interactions[1]
    original = layer.conv_tp
    if type(layer) is not RealAgnosticResidualInteractionBlock or hasattr(layer, "conv_fusion"):
        raise ValueError("expected the ordinary e3nn residual interaction")
    irreps = (str(original.irreps_in1), str(original.irreps_in2), str(original.irreps_out))
    if irreps != ("128x0e", "1x0e+1x1o+1x2e+1x3o", "128x0e+128x1o+128x2e+128x3o"):
        raise ValueError(f"unsupported scalar tensor-product layout {irreps}")
    if len(original.instructions) != 4 or any(
        (i.i_in1, i.i_in2, i.i_out, i.connection_mode, i.has_weight, i.path_shape)
        != (0, degree, degree, "uvu", True, (128, 1))
        or not math.isclose(i.path_weight, math.sqrt(2 * degree + 1))
        for degree, i in enumerate(original.instructions)
    ):
        raise ValueError("tensor-product paths differ from the normalized scalar UVU layout")
    layer.conv_tp = _FusedTensorProduct(layout)
    layer.conv_fusion = True
    try:
        yield {"second_interaction": "segmented_scalar_uvu", "channels": 128, "l_max": 3}
    finally:
        layer.conv_tp = original
        del layer.conv_fusion
