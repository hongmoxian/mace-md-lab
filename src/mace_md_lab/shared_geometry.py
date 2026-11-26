"""Reuse fixed geometric features across independent standard MACE checkpoints.

A frame has one positions leaf. Each member keeps its own learned interactions
and differentiates its own scalar energy through the shared geometry to that
leaf. The cache lives only for the duration of one evaluate() call.
"""

from contextvars import ContextVar
from copy import copy

import torch
from e3nn import o3
from mace.modules.blocks import RadialEmbeddingBlock
from mace.modules.models import MACE, ScaleShiftMACE
from mace.modules.radial import (
    AgnesiTransform,
    BesselBasis,
    ChebychevBasis,
    GaussianBasis,
    PolynomialCutoff,
    SoftTransform,
)


class _FrameFeature(torch.nn.Module):
    """On the first member use the real module; subsequent members read the result."""

    def __init__(self, original, key, frame_cache):
        super().__init__()
        self.original = original
        self.key = key
        self.frame_cache = frame_cache

    def forward(self, *args):
        cache = self.frame_cache.get()
        if self.key not in cache:
            cache[self.key] = self.original(*args)
        return cache[self.key]


def _geometry_signature(model):
    if type(model) not in (MACE, ScaleShiftMACE):
        raise ValueError(
            "shared features require eager standard MACE or ScaleShiftMACE"
        )
    sh = model.spherical_harmonics
    rad = model.radial_embedding
    if type(sh) is not o3.SphericalHarmonics or type(rad) is not RadialEmbeddingBlock:
        raise ValueError(
            "unrecognized spherical-harmonics or radial embedding implementation"
        )
    basis = rad.bessel_fn
    transform = getattr(rad, "distance_transform", None)
    if (
        type(basis) not in (BesselBasis, GaussianBasis, ChebychevBasis)
        or type(rad.cutoff_fn) is not PolynomialCutoff
        or type(transform) not in (type(None), AgnesiTransform, SoftTransform)
    ):
        raise ValueError("unrecognized radial basis, cutoff or distance transform")
    for block in (sh, rad):
        if any(True for _ in block.parameters()):
            raise ValueError("sharing a parameterized geometry module is unsupported")
        for submodule in block.modules():
            if submodule._forward_hooks or submodule._forward_pre_hooks:
                raise ValueError(
                    "geometry modules with forward hooks require independent evaluation"
                )
    # Most options are tensor buffers; non-buffer forward options are listed here.
    return (
        tuple(model.atomic_numbers.detach().cpu().tolist()),
        float(model.r_max),
        tuple(getattr(model, "heads", ("Default",))),
        sh.normalize,
        sh.normalization,
        tuple(sh._ls_list),
        sh._lmax,
        sh._is_range_lmax,
        str(sh.irreps_in),
        str(sh.irreps_out),
        type(basis),
        type(transform),
        getattr(rad, "apply_cutoff", True),
        getattr(basis, "coeff", None),
        getattr(basis, "num_basis", None),
    )


class SharedGeometry:
    """Inference-only view over a compatible, fixed collection of checkpoints.

    Inputs follow MACE's AtomicData batch dictionary with one unpadded graph.
    Output energies [M] and conservative forces [M,N,3] retain model units.
    The caller owns the checkpoint references and must keep them unchanged for
    the lifetime of the evaluator. No original module or checkpoint is patched.
    """

    def __init__(self, models):
        models = tuple(models)
        if len(models) < 2:
            raise ValueError("shared geometry requires at least two members")
        ref = models[0]
        signature = _geometry_signature(ref)
        reference_parameter = next(ref.parameters())
        self.device = reference_parameter.device
        self.dtype = reference_parameter.dtype
        self.frame_cache = ContextVar("mace_md_lab_frame_features")
        features = {
            name: _FrameFeature(getattr(ref, name), name, self.frame_cache)
            for name in ("spherical_harmonics", "radial_embedding")
        }
        ref_buffers = dict(ref.radial_embedding.named_buffers())
        self.views = []
        for model in models:
            if _geometry_signature(model) != signature:
                raise ValueError(
                    "committee species, cutoff, heads or geometry options differ"
                )
            first = next(model.parameters())
            if first.device != self.device or first.dtype != self.dtype:
                raise ValueError(
                    "committee members require matching precision and device"
                )
            member_buffers = dict(model.radial_embedding.named_buffers())
            if member_buffers.keys() != ref_buffers.keys() or any(
                b.device != ref_buffers[n].device
                or b.dtype != ref_buffers[n].dtype
                or not torch.equal(b, ref_buffers[n])
                for n, b in member_buffers.items()
            ):
                raise ValueError("radial embedding constants differ between members")

            # nn.Module looks up children in _modules. Only this private, shallow
            # view gets feature readers. All other children are this member's
            # original, read-only weights and tensor-product implementations.
            view = copy(model)
            object.__setattr__(view, "_modules", {**model._modules, **features})
            self.views.append(view)

    @torch.enable_grad()
    def evaluate(self, batch):
        """Return each member's E and F on one common set of physical edges."""
        if torch.is_inference_mode_enabled():
            raise RuntimeError(
                "MACE forces require autograd; disable torch.inference_mode"
            )
        if batch["ptr"].numel() != 2:
            raise ValueError("expected one unpadded structure")
        pos = batch["positions"].detach().requires_grad_(True)
        if pos.device != self.device or pos.dtype != self.dtype:
            raise ValueError("positions must match checkpoint precision and device")
        frame = {**batch, "positions": pos}
        energies, forces = [], []
        token = self.frame_cache.set({})
        try:
            for k, model in enumerate(self.views):
                energy = model(
                    frame, training=False, compute_force=False, compute_stress=False
                )["energy"].reshape(())
                # Later members still need the shared geometry's backward graph.
                # Member-specific graphs are released when their energy goes out
                # of scope; neither forces nor energies keep a live graph.
                grad = None
                if energy.requires_grad:
                    grad = torch.autograd.grad(
                        energy,
                        pos,
                        retain_graph=k + 1 < len(self.views),
                        create_graph=False,
                        allow_unused=True,
                    )[0]
                forces.append(
                    -grad.detach() if grad is not None else torch.zeros_like(pos)
                )
                energies.append(energy.detach())
                del energy  # release this member's private graph before the next forward
        finally:
            self.frame_cache.reset(token)
        return torch.stack(energies), torch.stack(forces)
