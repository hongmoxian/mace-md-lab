"""Receiver/species aggregation for the first scalar MACE-MP-0 Small interaction.

This fixed-checkpoint GPU path supports energy inference and one coordinate or
strain derivative on an unpadded single structure with fixed atom identities.
It sorts the current edge indices on the GPU. A topology change therefore takes
effect even when the new graph has the same number of edges.
"""

from contextlib import contextmanager
import math
from unittest.mock import patch

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable


@triton.jit
def _group_moments(H, Y, order, offsets, out, groups, edge_tile: tl.constexpr):
    group = tl.program_id(0)
    harmonic = tl.program_id(1)
    begin = tl.load(offsets + group).to(tl.int32)
    end = tl.load(offsets + group + 1).to(tl.int32)
    row = tl.arange(0, edge_tile)
    feature = tl.arange(0, 64)
    total = tl.full((64,), 0, tl.float64)
    for start in range(begin, end, edge_tile):
        valid = start + row < end
        edge = tl.load(order + start + row, valid, other=0).to(tl.int32)
        y = tl.load(Y + edge * 16 + harmonic, valid, other=0.0)
        h = tl.load(H + edge[:, None] * 64 + feature[None, :], valid[:, None], other=0.0)
        total += tl.sum(y[:, None] * h, axis=0)
    tl.store(out + (harmonic * groups + group) * 64 + feature, total)


@triton.jit
def _moment_adjoint(H, Y, group_index, dM, dH, dY, groups):
    edge = tl.program_id(0)
    group = tl.load(group_index + edge).to(tl.int32)
    harmonic = tl.arange(0, 16)
    feature = tl.arange(0, 64)
    h = tl.load(H + edge * 64 + feature)
    y = tl.load(Y + edge * 16 + harmonic)
    dm = tl.load(dM + (harmonic[:, None] * groups + group) * 64 + feature[None, :])
    tl.store(dH + edge * 64 + feature, tl.sum(dm * y[:, None], axis=0))
    tl.store(dY + edge * 16 + harmonic, tl.sum(dm * h[None, :], axis=1))


class _Moments(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, y, group_index, offsets, order):
        h, y = h.contiguous(), y.contiguous()
        groups = offsets.numel() - 1
        moments = h.new_empty((16, groups, 64))
        with torch.cuda.device(h.device):
            _group_moments[(groups, 16)](
                h, y, order, offsets, moments, groups, 16, num_warps=4
            )
        ctx.groups = groups
        ctx.save_for_backward(h, y, group_index)
        return moments

    @staticmethod
    @once_differentiable
    def backward(ctx, dm):
        h, y, group_index = ctx.saved_tensors
        dh, dy = torch.empty_like(h), torch.empty_like(y)
        if group_index.numel():
            with torch.cuda.device(h.device):
                _moment_adjoint[(group_index.numel(),)](
                    h, y, group_index, dm.contiguous(), dh, dy, ctx.groups, num_warps=4
                )
        return dh, dy, None, None, None


class _FirstInteraction:
    def __init__(self, model, atomic_numbers, fold_linear, layout):
        self.layout = layout
        from mace.modules.blocks import RealAgnosticResidualInteractionBlock
        from mace.modules.models import MACE, ScaleShiftMACE

        layer = model.interactions[0]
        if (
            type(model) not in (MACE, ScaleShiftMACE)
            or model.training
            or any(p.requires_grad for p in model.parameters())
            or hasattr(model, "joint_embedding")
            or type(layer) is not RealAgnosticResidualInteractionBlock
            or hasattr(layer, "conv_fusion")
            or str(layer.conv_tp.irreps_in1) != "128x0e"
            or str(layer.conv_tp.irreps_in2) != "1x0e+1x1o+1x2e+1x3o"
            or str(layer.conv_tp.irreps_out) != "128x0e+128x1o+128x2e+128x3o"
        ):
            raise ValueError("expected the native first scalar interaction of MACE-MP-0 Small")
        instructions = layer.conv_tp.instructions
        if len(instructions) != 4 or any(
            (i.i_in1, i.i_in2, i.i_out, i.connection_mode, i.has_weight, i.path_shape)
            != (0, degree, degree, "uvu", True, (128, 1))
            or not math.isclose(i.path_weight, math.sqrt(2 * degree + 1))
            for degree, i in enumerate(instructions)
        ):
            raise ValueError("expected the normalized scalar UVU tensor-product paths")
        param = next(layer.parameters())
        if param.device.type != "cuda" or param.dtype != torch.float64:
            raise ValueError("the grouped first-interaction benchmark requires CUDA FP64")
        self.layer = layer
        self.fold_linear = fold_linear
        radial = list(layer.conv_tp_weights.children())
        self.radial_prefix = radial[:-1]
        dtype, device = param.dtype, param.device
        self.n_atoms = len(atomic_numbers)
        if not self.n_atoms:
            raise ValueError("a nonempty fixed composition is required")
        present = sorted(set(map(int, atomic_numbers)))
        self.n_species = len(present)
        table = model.atomic_numbers.detach().cpu().tolist()
        if not set(present).issubset(table):
            raise ValueError("an input element is absent from the checkpoint")
        one_hot = torch.eye(len(table), device=device, dtype=dtype)[[table.index(z) for z in present]]
        self.species = torch.tensor([present.index(int(z)) for z in atomic_numbers], device=device)
        self.expected_attrs = one_hot.index_select(0, self.species)
        self.checked_attrs = None
        self.checked_version = None
        self.boundaries = torch.arange(self.n_atoms * self.n_species + 1, device=device)

        with torch.no_grad():
            weight = radial[-1](torch.eye(64, device=device, dtype=dtype))
            if (
                weight.shape != (64, 512)
                or getattr(radial[-1], "act", None) is not None
                or torch.count_nonzero(radial[-1](torch.zeros(1, 64, device=device, dtype=dtype)))
            ):
                raise ValueError("expected a bias-free, linear final 64-to-512 radial projection")
            source = layer.linear_up(model.node_embedding(one_hot))
            # Each angular order uses one learned projection for all its m components.
            # Fold the position-independent source features into each element's matrix.
            projected = weight.reshape(64, 4, 128).permute(1, 0, 2)[:, None] * source[None, :, None, :]
            if fold_linear:
                # The native receiver linear only mixes channels within one
                # angular order. Send the coefficient rows through it once,
                # with the same input in each magnetic component of that order.
                if getattr(layer.linear, "bias_numel", 0):
                    raise ValueError("cannot fold a receiver linear containing a bias")
                blocks = []
                for degree in range(4):
                    block = projected[degree].reshape(-1, 128, 1)
                    block = block.expand(-1, -1, 2 * degree + 1)
                    blocks.append(block.reshape(self.n_species * 64, -1))
                linear_weights = layer.linear(torch.cat(blocks, dim=1)) / layer.avg_num_neighbors
                reshaped = layer.reshape(linear_weights)
                if reshaped.shape != (self.n_species * 64, 128, 16):
                    raise ValueError("unsupported receiver linear output layout")
                self.weight = reshaped.permute(2, 0, 1).contiguous()
            else:
                degrees = [degree for degree in range(4) for _ in range(2 * degree + 1)]
                self.weight = projected[degrees].reshape(16, self.n_species * 64, 128).contiguous()

    def __call__(
        self, node_attrs, node_feats, edge_attrs, edge_feats, edge_index, cutoff=None,
        lammps_class=None, lammps_natoms=(0, 0), first_layer=False,
    ):
        if lammps_class is not None or len(node_attrs) != self.n_atoms:
            raise ValueError("the grouped first interaction requires an unpadded single structure")
        if self.checked_attrs is not node_attrs or self.checked_version != node_attrs._version:
            if not torch.equal(node_attrs, self.expected_attrs):
                raise ValueError("the element ordering differs from the installed fixed composition")
            self.checked_attrs = node_attrs
            self.checked_version = node_attrs._version
        skip = self.layer.skip_tp(node_feats, node_attrs)
        hidden = edge_feats
        for transform in self.radial_prefix:
            hidden = transform(hidden)
        if cutoff is not None:
            hidden = hidden * cutoff
        # Distinct periodic images and any repeated input edges contribute independently.
        if self.layout is None:
            group = edge_index[1] * self.n_species + self.species.index_select(0, edge_index[0])
            sorted_group, order = torch.sort(group)
            offsets = torch.searchsorted(sorted_group, self.boundaries)
        else:
            group, offsets, order = self.layout.groups, self.layout.group_offsets, self.layout.order
        moments = _Moments.apply(hidden, edge_attrs, group, offsets, order)
        message = torch.bmm(moments.view(16, self.n_atoms, self.n_species * 64), self.weight)
        if self.fold_linear:
            return message.permute(1, 2, 0).contiguous(), skip
        blocks = [
            piece.permute(1, 2, 0).reshape(self.n_atoms, -1)
            for piece in message.split((1, 3, 5, 7), dim=0)
        ]
        out = self.layer.linear(torch.cat(blocks, dim=1)) / self.layer.avg_num_neighbors
        return self.layer.reshape(out), skip


@contextmanager
def install_grouped_first(model, atomic_numbers, *, fold_linear=False, layout=None):
    """Install the first-interaction replacement temporarily on one model instance."""
    impl = _FirstInteraction(model, atomic_numbers, fold_linear, layout)
    with patch.object(impl.layer, "forward", impl):
        yield {
            "operator": "grouped_species_first_interaction",
            "kernel": "Triton segmented edge moments and adjoint, batched FP64 projection",
            "atoms": impl.n_atoms,
            "species": impl.n_species,
            "dense_group_slots": impl.n_atoms * impl.n_species,
            "fold_receiver_linear": fold_linear,
        }
