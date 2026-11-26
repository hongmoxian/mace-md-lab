"""Single-structure, fixed-cell NVE for standard MACE energy checkpoints.

ASE positions, velocities and masses are the integration state in float64. Model
inputs use the calculator's checkpoint dtype. A Verlet skin extends the candidate
list only. Every forward pass sees directed periodic images strictly inside r_max.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from ase import units
from mace.data.neighborhood import get_neighborhood
from mace.modules.models import MACE, ScaleShiftMACE


def validate_calculator(calculator):
    """Validate the single-checkpoint, scalar-energy interface used by this driver."""
    if getattr(calculator, "model_type", None) != "MACE" or len(calculator.models) != 1:
        raise ValueError("expected a single scalar-energy MACECalculator")
    if getattr(calculator, "use_compile", False):
        raise ValueError("use an eager MACECalculator for the resident backend")
    if type(calculator.models[0]) not in (MACE, ScaleShiftMACE):
        raise ValueError("the resident backend supports MACE and ScaleShiftMACE")
    if getattr(calculator, "pad_num_atoms", 0) or getattr(
        calculator, "pad_num_edges", 0
    ):
        raise ValueError("use an unpadded MACECalculator for the resident backend")
    if getattr(calculator, "compute_atomic_stresses", False):
        raise ValueError("the resident NVE backend computes only energy and forces")
    if float(calculator.length_units_to_A) != 1.0:
        raise ValueError("the checkpoint must take distances in angstrom")
    scale = float(calculator.energy_units_to_eV)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("energy_units_to_eV must be positive and finite")
    if not callable(getattr(calculator, "_atoms_to_batch", None)):
        raise ValueError(
            "this MACECalculator has no supported AtomicData batch adapter"
        )


@torch.enable_grad()
def evaluate_prepared(calculator, batch, num_real_atoms=None):
    """Evaluate a compatible prepared batch and return detached eV and eV/Å.

    The caller validates each calculator once. Shared batches require identical
    element encoding, head ordering/selection, model input schema, cutoff, dtype
    and device. Each forward gets its own differentiation leaf and dictionary.
    """
    if torch.is_inference_mode_enabled():
        raise RuntimeError(
            "MACE force evaluation requires autograd; disable inference_mode"
        )
    inputs = batch.copy()
    inputs["positions"] = batch["positions"].detach().requires_grad_(True)
    output = calculator.models[0](
        inputs,
        training=False,
        compute_force=True,
        compute_stress=False,
        compute_virials=False,
    )
    if num_real_atoms is not None:
        # Identical graph-level and atom-level slicing to the official ASE path.
        output = calculator._slice_real_outputs(output, num_real_atoms)
    energy = output["energy"].reshape(())
    forces = output["forces"]
    scale = float(calculator.energy_units_to_eV)
    if scale != 1.0:
        energy, forces = energy * scale, forces * scale
    return energy.detach(), forces.detach()


class ResidentMACE:
    """Run stock MACE using a reusable graph template and resident MD state.

    Positions may remain unwrapped. On a GPU, checking the maximum displacement
    synchronizes one scalar per neighbor validity check. A rebuild transfers
    positions to the host for MACE's upstream periodic neighbor-list routine.
    With certify_edges=True a cutoff-gap displacement bound also permits reuse
    of the filtered physical graph between candidate rebuilds. A miss performs
    the usual strict cutoff filter; an atom pair exactly at cutoff has zero gap.
    """

    def __init__(
        self,
        atoms,
        calculator,
        skin=0.5,
        certify_edges=False,
        capture_forces=False,
        pad_capture=False,
    ):
        validate_calculator(calculator)
        if not len(atoms) or not np.all(atoms.pbc) or atoms.cell.rank != 3:
            raise ValueError(
                "expected a nonempty, three-dimensional periodic structure"
            )
        cell = np.asarray(atoms.cell.array, dtype=np.float64)
        if not np.isfinite(cell).all() or np.linalg.matrix_rank(cell) != 3:
            raise ValueError("a finite, nonsingular cell is required")
        if atoms.constraints:
            raise ValueError(
                "remove constraints or use an integrator supporting constraints"
            )
        skin = float(skin)
        if not math.isfinite(skin) or skin < 0:
            raise ValueError("skin must be nonnegative and finite")

        self.calculator = calculator
        self.cutoff = float(calculator.r_max)
        self.skin = skin
        self.certify_edges = bool(certify_edges)
        if not math.isfinite(self.cutoff) or self.cutoff <= 0:
            raise ValueError("MACE r_max must be positive and finite")
        parameter = next(calculator.models[0].parameters())
        self.device = parameter.device
        self.dtype = parameter.dtype
        self._replay = None
        if pad_capture and not capture_forces:
            raise ValueError("pad_capture requires capture_forces=True")
        if pad_capture and getattr(calculator.models[0], "embedding_specs", None):
            raise ValueError(
                "padded replay supports standard MACE inputs without extended embeddings"
            )
        if capture_forces:
            from .cuda_replay import ForceReplay

            self._replay = ForceReplay(self.device, padded=pad_capture)
        self._cell = cell.copy()
        self._pbc = tuple(bool(value) for value in atoms.pbc)
        self._positions = self._state_tensor(atoms.positions)
        velocities = atoms.get_velocities()
        if velocities is None:
            velocities = np.zeros_like(atoms.positions)
        self._velocities = self._state_tensor(velocities)
        masses = np.asarray(atoms.get_masses(), dtype=np.float64)
        if not np.isfinite(masses).all() or np.any(masses <= 0):
            raise ValueError("atomic masses must be positive and finite")
        if not np.isfinite(atoms.positions).all() or not np.isfinite(velocities).all():
            raise ValueError("positions and velocities must be finite")
        self._masses = self._state_tensor(masses[:, None])

        # This preserves the exact upstream element ordering, selected head,
        # per-atom attributes and any static auxiliary fields from the calculator.
        template = calculator._atoms_to_batch(atoms).to_dict()
        geometry_keys = {"positions", "edge_index", "shifts", "unit_shifts"}
        self._static = {
            key: (
                value.to(dtype=self.dtype)
                if torch.is_tensor(value) and value.is_floating_point()
                else value
            )
            for key, value in template.items()
            if key not in geometry_keys
        }
        self._shift_dtype = template["unit_shifts"].dtype
        self._reference = None
        self._edge_reference = None
        self._physical_edges = None
        self.certificate_hits = 0
        self.edge_refreshes = 0
        self._force = None
        self._force_version = None
        self.rebuilds = 0

    def _state_tensor(self, array):
        return torch.tensor(array, dtype=torch.float64, device=self.device)

    @property
    def positions(self):
        """Unwrapped Cartesian positions in Å, on the integration device."""
        return self._positions

    @property
    def velocities(self):
        """Cartesian velocities in ASE internal units, on the integration device."""
        return self._velocities

    @property
    def captures(self):
        return self._replay.captures if self._replay is not None else 0

    @property
    def replays(self):
        return self._replay.replays if self._replay is not None else 0

    @property
    def fallback_evaluations(self):
        return self._replay.fallback_evaluations if self._replay is not None else 0

    @property
    def capture_capacity(self):
        return self._replay.capacity if self._replay is not None else 0

    def kinetic_energy(self):
        return 0.5 * (self._masses * self.velocities.square()).sum()

    def _topology_action(self, positions):
        # 2: rebuild candidate list, 1: filter candidates, 0: reuse physical graph.
        if self._reference is None or self.skin == 0:
            return 2
        candidate_move2 = (positions - self._reference).square().sum(dim=1).max()
        candidate_valid = candidate_move2 < (0.5 * self.skin) ** 2
        if not self.certify_edges:
            return 1 if bool(candidate_valid) else 2

        # The skin protects every periodic image omitted by the candidate list.
        # For each candidate, |r_now-r_exact| <= 2 max_i |x_now,i-x_exact,i|.
        # Use separate reference coordinates when the physical graph has changed
        # since the last candidate-list rebuild. Both tests share one host sync.
        if self._edge_reference is self._reference:
            edge_move2 = candidate_move2
        else:
            edge_move2 = (positions - self._edge_reference).square().sum(dim=1).max()
        edge_valid = edge_move2 < self._certified_move2
        action = (~candidate_valid).to(torch.int32) * 2 + (~edge_valid).to(torch.int32)
        return min(2, int(action))

    def _rebuild(self, positions):
        host_positions = positions.detach().cpu().numpy()
        if not np.isfinite(host_positions).all():
            raise ValueError("positions must remain finite")
        edge_index, shifts, unit_shifts, _ = get_neighborhood(
            positions=host_positions,
            cutoff=self.cutoff + self.skin,
            pbc=self._pbc,
            cell=self._cell.copy(),
        )
        self._candidates = torch.as_tensor(
            edge_index, device=self.device, dtype=torch.long
        )
        self._candidate_shifts = self._state_tensor(shifts)
        self._candidate_units = torch.as_tensor(
            unit_shifts, device=self.device, dtype=self._shift_dtype
        )
        self._reference = positions.detach().clone()
        self.rebuilds += 1

    def _exact_edges(self, positions, rebuilt):
        edge_index = self._candidates
        shifts, units = self._candidate_shifts, self._candidate_units
        if self.skin > 0:
            src, dst = edge_index
            vectors = positions[dst] - positions[src] + shifts
            distance2 = vectors.square().sum(dim=1)
            # Gather indices once: each separate boolean-index operation would
            # require a dynamically sized CUDA allocation and synchronization.
            keep = torch.nonzero(distance2 < self.cutoff**2).flatten()
            if self.certify_edges:
                if distance2.numel():
                    gap = (distance2.sqrt() - self.cutoff).abs().amin()
                    # Round inward when a candidate is at or within roundoff of
                    # the cutoff. Large unwrapped coordinates scale the guard.
                    scale = (
                        1.0 + self.cutoff + positions.abs().amax() + shifts.abs().amax()
                    )
                    guard = (32 * torch.finfo(torch.float64).eps) * scale
                    self._certified_move2 = (0.5 * (gap - guard).clamp_min(0)).square()
                else:
                    # Only the separate skin criterion can admit a new image.
                    self._certified_move2 = math.inf
                self._edge_reference = (
                    self._reference if rebuilt else positions.detach().clone()
                )
            edge_index = edge_index.index_select(1, keep)
            shifts = shifts.index_select(0, keep)
            units = units.index_select(0, keep)
        self.edge_refreshes += 1
        return edge_index, shifts.to(dtype=self.dtype), units

    @torch.no_grad()
    def prepare(self, positions=None):
        """Build the exact cutoff graph for positions using cached candidate images."""
        if positions is None:
            positions = self.positions
        else:
            positions = torch.as_tensor(
                positions, device=self.device, dtype=torch.float64
            )
        if positions.shape != self.positions.shape:
            raise ValueError("positions must match the original (N,3) structure")
        action = self._topology_action(positions)
        if action:
            if action == 2:
                self._rebuild(positions)
            self._physical_edges = self._exact_edges(positions, rebuilt=(action == 2))
        else:
            self.certificate_hits += 1
        edge_index, shifts, unit_shifts = self._physical_edges
        batch = self._static.copy()
        batch.update(
            positions=positions.detach().to(dtype=self.dtype),
            edge_index=edge_index,
            shifts=shifts,
            unit_shifts=unit_shifts,
        )
        return batch

    def evaluate(self, positions=None):
        """Compute energy and conservative forces for current or supplied positions."""
        batch = self.prepare(positions)
        if self._replay is None:
            result = evaluate_prepared(self.calculator, batch)
        else:
            result = self._replay(
                self.calculator, batch, self.edge_refreshes, evaluate_prepared
            )
        if positions is None or positions is self.positions:
            self._force = result[1]
            self._force_version = self.positions._version
        return result

    def step(self, dt_fs):
        """Advance one velocity-Verlet step in femtoseconds; return final E and F."""
        dt_fs = float(dt_fs)
        if not math.isfinite(dt_fs) or dt_fs <= 0:
            raise ValueError("the timestep in fs must be positive and finite")
        dt = dt_fs * units.fs
        if self._force is None or self._force_version != self.positions._version:
            self.evaluate()
        with torch.no_grad():
            v_half = self.velocities + (0.5 * dt) * self._force / self._masses
            trial = self.positions + dt * v_half
        energy, force = self.evaluate(trial)
        with torch.no_grad():
            self._positions.copy_(trial)
            self._velocities.copy_(v_half + (0.5 * dt) * force / self._masses)
        self._force = force
        self._force_version = self.positions._version
        return energy, force
