"""Single-structure energy, forces and stress from the MACE-MP-0 edge gradient."""

from contextlib import contextmanager, ExitStack
import math
from pathlib import Path

import torch


def verify_edge_checkpoint(path):
    """Qualify the exact edge-only energy checkpoint used by this experiment."""
    from .bench import file_hash

    small = "2ddb079cee0e131eaaf6912ba581b394551ead283e95c99cfe78c605d10b5736"
    if file_hash(Path(path)) != small:
        raise ValueError("the experimental edge stress route is qualified only for MACE-MP-0 Small")


@contextmanager
def install_edge_virial(model, volume, stress_result):
    """Use one edge gradient for both atomic forces and the global configurational stress.

    The tested MP-0 energy depends on geometry only through Cartesian directed
    edge vectors. Symmetric strain is affine on positions and periodic shifts.
    Supply the current volume in Å³. Use None when the cell can vary.
    """
    from mace.modules.models import ScaleShiftMACE

    if (type(model) is not ScaleShiftMACE or hasattr(model, "joint_embedding")
            or (volume is not None and (not math.isfinite(volume) or volume <= 0))):
        raise ValueError("expected the MP-0 scalar edge-energy model and a valid cell")
    call = {}

    def request(_module, inputs, kwargs):
        if (kwargs.get("training") or kwargs.get("compute_hessian")
                or kwargs.get("lammps_mliap") or kwargs.get("compute_atomic_stresses")):
            raise ValueError("the edge virial probe supports single-structure first derivatives")
        data = inputs[0] if inputs else kwargs["data"]
        if data["cell"].numel() != 9:
            raise ValueError("supply one periodic structure")
        call["data"] = data
        kwargs.update(compute_force=False, compute_stress=False, compute_virials=False,
                      compute_displacement=False, compute_edge_forces=True)
        return inputs, kwargs

    def capture_vectors(_module, inputs):
        # These are the exact edge displacements supplied to e3nn, including the
        # periodic shifts chosen by this frame's neighbor list.
        call["vectors"] = inputs[0].detach()

    @torch.no_grad()
    def combine(_module, inputs, output):
        data = call.pop("data")
        vectors = call.pop("vectors")
        # In MACE 0.3.16 edge_forces is +dE/d(vector), with vector from
        # sender to receiver. Each directed edge carries its own derivative.
        edge_grad = output["edge_forces"].detach()
        sender, receiver = data["edge_index"]
        force = torch.zeros_like(data["positions"])
        force.index_add_(0, sender, edge_grad)
        force.index_add_(0, receiver, edge_grad, alpha=-1.0)
        tensor = edge_grad.T @ vectors
        symmetric = (tensor + tensor.T) * 0.5
        if volume is None:
            cell = data["cell"].reshape(3, 3)
            current_volume = torch.dot(cell[0], torch.linalg.cross(cell[1], cell[2])).abs()
        else:
            current_volume = volume
        stress = (symmetric / current_volume).reshape(1, 3, 3)
        output["forces"] = force
        output["stress"] = stress
        output["virials"] = -symmetric.reshape(1, 3, 3)
        stress_result["tensor"] = stress
        return output

    with ExitStack() as stack:
        stack.callback(model.register_forward_pre_hook(request, with_kwargs=True).remove)
        stack.callback(model.spherical_harmonics.register_forward_pre_hook(capture_vectors).remove)
        stack.callback(model.register_forward_hook(combine).remove)
        yield
