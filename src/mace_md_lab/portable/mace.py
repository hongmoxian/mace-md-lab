"""Adapt the distance and direction inputs of ordinary molecular MACE."""

from copy import copy
from importlib.metadata import version
from pathlib import Path
from types import FunctionType, MethodType

import numpy as np
import torch
from ase import Atoms
from e3nn import o3
from mace.calculators import MACECalculator
from mace.modules.models import MACE, ScaleShiftMACE

from .geometry import edge_geometry_from_positions


def _with_global(fn, name, value):
    """Bind a dependency on a private Python function without changing upstream."""
    if name not in fn.__code__.co_names or name not in fn.__globals__:
        raise ValueError(f"unexpected MACE function {fn.__qualname__}: missing {name}")
    clone = FunctionType(
        fn.__code__,
        {**fn.__globals__, name: value},
        fn.__name__,
        fn.__defaults__,
        fn.__closure__,
    )
    clone.__kwdefaults__ = fn.__kwdefaults__
    return clone


def _energy_view(model, mode):
    # In the pinned upstream energy-only forward, ctx.vectors feeds SH and a dtype
    # cast. Virial/edge-force outputs would require the actual unnormalized vector.
    def directions_and_lengths(positions, edge_index, shifts):
        r, direction = edge_geometry_from_positions(
            positions, edge_index, shifts, mode=mode, distance_epsilon=1e-12
        )
        return direction, r.unsqueeze(-1)

    sh = model.spherical_harmonics
    if type(sh) is not o3.SphericalHarmonics or not sh.normalize:
        raise ValueError("expected unmodified normalized e3nn spherical harmonics")
    directional_sh = copy(sh)
    directional_sh.normalize = False
    view = copy(model)
    view._modules = {**model._modules, "spherical_harmonics": directional_sh}
    upstream = type(model).forward
    prepare = _with_global(
        upstream.__globals__["prepare_graph"],
        "get_edge_vectors_and_lengths",
        directions_and_lengths,
    )
    view.forward = MethodType(_with_global(upstream, "prepare_graph", prepare), view)
    return view


class MolecularMACE:
    """Fixed atom ordering, nonperiodic molecular energy, conservative first forces."""

    def __init__(self, weights, device, dtype, atomic_numbers):
        if version("mace-torch") != "0.3.16":
            raise ValueError("this adapter supports mace-torch==0.3.16")
        self.device = torch.device(device)
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        if self.dtype not in (torch.float32, torch.float64):
            raise ValueError("use float32 or float64")
        self.numbers = torch.as_tensor(atomic_numbers).detach().cpu().numpy().copy()
        if (
            self.numbers.ndim != 1
            or not len(self.numbers)
            or not np.issubdtype(self.numbers.dtype, np.integer)
        ):
            raise ValueError("atomic_numbers must be a nonempty integer sequence")
        self.calculator = MACECalculator(
            model_paths=[str(weights)],
            device=str(self.device),
            default_dtype=str(self.dtype).split(".")[-1],
        )
        self.model = self.calculator.models[0].eval()
        if type(self.model) not in (MACE, ScaleShiftMACE) or hasattr(
            self.model, "joint_embedding"
        ):
            raise ValueError(
                "only ordinary scalar MACE/ScaleShiftMACE without auxiliary embeddings"
            )
        if not set(self.numbers.tolist()).issubset(self.calculator.z_table.zs):
            raise ValueError("input contains species unsupported by checkpoint")
        self.cutoff = float(self.calculator.r_max)
        self.views = {
            mode: _energy_view(self.model, mode) for mode in ("native", "analytic")
        }
        # Build the upstream element/head encodings once. Replace every geometry
        # field before evaluation. This template can be constructed before capture.
        pos = np.zeros((len(self.numbers), 3))
        pos[:, 0] = np.arange(len(self.numbers)) * self.cutoff * 0.4
        template = self._official_batch(pos)
        self.static = {
            k: v
            for k, v in template.items()
            if k not in ("positions", "edge_index", "shifts", "unit_shifts")
        }
        self.metadata = {
            "family": "MACE",
            "checkpoint": Path(weights).name,
            "mace_version": version("mace-torch"),
            "architecture": type(self.model).__name__,
            "head": self.calculator.head,
            "energy_unit": "eV",
            "length_unit": "angstrom",
            "periodic": False,
            "geometry": "shared distance/unit vector, stock e3nn harmonics and interactions",
            "cutoff_A": self.cutoff,
        }

    def _official_batch(self, pos_numpy):
        atoms = Atoms(numbers=self.numbers, positions=pos_numpy, pbc=False)
        data = self.calculator._atoms_to_batch(atoms).to_dict()
        return {
            k: (
                v.to(dtype=self.dtype)
                if torch.is_tensor(v) and v.is_floating_point()
                else v
            )
            for k, v in data.items()
        }

    @staticmethod
    def _output(model, batch):
        return model(
            batch,
            compute_stress=False,
            compute_force=False,
            compute_edge_forces=False,
            compute_virials=False,
            compute_displacement=False,
            training=False,
        )["energy"].reshape(())

    def native_energy(self, positions):
        """Original unmodified model and upstream topology; preserve position gradient."""
        data = self._official_batch(positions.detach().cpu().numpy())
        data["positions"] = positions
        return self._output(self.model, data)

    def energy(self, positions, edge_index, mode):
        if mode not in self.views:
            raise ValueError(f"unsupported geometry mode {mode!r}")
        if (
            positions.shape != (len(self.numbers), 3)
            or positions.dtype != self.dtype
            or positions.device != next(self.model.parameters()).device
        ):
            raise ValueError("positions must match adapter molecule, dtype and device")
        if (
            edge_index.ndim != 2
            or edge_index.shape[0] != 2
            or edge_index.dtype != torch.long
            or edge_index.device != positions.device
        ):
            raise ValueError("supply a device-local long tensor of shape (2,E)")
        shifts = positions.new_zeros((edge_index.shape[1], 3))
        data = {
            **self.static,
            "positions": positions,
            "edge_index": edge_index,
            "shifts": shifts,
            "unit_shifts": shifts,
        }
        return self._output(self.views[mode], data)


def load_adapter(weights, device, dtype, atomic_numbers):
    return MolecularMACE(weights, device, dtype, atomic_numbers)
