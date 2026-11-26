"""Driver-model MD and same-frame force disagreement for scalar MACE committees."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib

import numpy as np
import torch

from .runtime import ResidentMACE, evaluate_prepared, validate_calculator


@dataclass
class Observation:
    """One observation. Energies are eV, forces and disagreement are eV/Å."""

    energies: torch.Tensor  # (models,)
    forces: torch.Tensor  # (models, atoms, 3)

    @property
    def atom_force_disagreement(self):
        # Population variance across members, summing all three Cartesian components.
        diff = self.forces - self.forces.mean(dim=0)
        return diff.square().sum(dim=-1).mean(dim=0).sqrt()

    @property
    def max_force_disagreement(self):
        return self.atom_force_disagreement.max()


def _weight_fingerprint(model):
    """Catch repeated trained weights even if checkpoints have different filenames."""
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if not isinstance(value, torch.Tensor):
            raise ValueError("committee requires tensor-valued state dictionaries")
        value = value.detach().cpu().contiguous()
        digest.update(repr((name, tuple(value.shape), str(value.dtype))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class MACECommittee:
    """Use a native multi-member calculator and retain its member ordering.

    The first member propagates MD. All members independently differentiate their
    scalar energy on the selected observation frames. No mean force drives MD.
    """

    def __init__(
        self, atoms, calculator, skin=0.5, geometry="auto", certify_edges=False
    ):
        if geometry not in {"auto", "off", "require"}:
            raise ValueError("geometry must be auto, off or require")
        models = calculator.models
        if len(models) < 2 or len({id(model) for model in models}) != len(models):
            raise ValueError("supply at least two distinct scalar MACE models")
        if float(calculator.energy_units_to_eV) != 1.0:
            raise ValueError("committee requires native model energy in eV")
        self.members = []
        self.calculator = calculator
        first_parameter = next(models[0].parameters())
        self.members_sha256 = []
        # The upstream multi-model calculator uses element/head ordering of member 0.
        # Incompatible orderings would silently feed the wrong one-hot encoding.
        for model in models:
            if list(map(int, model.atomic_numbers)) != list(calculator.z_table.zs):
                raise ValueError(
                    "all members need the same ordered atomic-number table"
                )
            if list(getattr(model, "heads", ["Default"])) != list(
                calculator.available_heads
            ):
                raise ValueError("all members need the same ordered prediction heads")
            if float(model.r_max) != float(calculator.r_max):
                raise ValueError("all members need the same radial cutoff")
            if any(
                p.device != first_parameter.device or p.dtype != first_parameter.dtype
                for p in model.parameters()
                if p.is_floating_point()
            ):
                raise ValueError(
                    "all member parameters must have the same dtype and device"
                )
            # Member views reuse the same already-loaded weights as the native reference.
            member = copy.copy(calculator)
            member.models = [model]
            member.num_models = 1
            validate_calculator(member)
            self.members.append(member)
            self.members_sha256.append(_weight_fingerprint(model))
        if len(set(self.members_sha256)) != len(models):
            raise ValueError(
                "identical member weights cannot form an independent committee"
            )
        self.skin = float(skin)
        self.certify_edges = bool(certify_edges)
        self.reset(atoms)
        self.geometry = None
        self.observer_geometry = None
        self.geometry_reason = None
        if geometry != "off":
            from .shared_geometry import SharedGeometry

            try:
                self.geometry = SharedGeometry(models)
                if len(models) > 2:
                    self.observer_geometry = SharedGeometry(models[1:])
            except (ValueError, NotImplementedError) as exc:
                if geometry == "require":
                    raise
                self.geometry = None
                self.observer_geometry = None
                self.geometry_reason = str(exc)
        else:
            self.geometry_reason = "disabled by caller"

    def reset(self, atoms):
        """Start a new independent replay or trajectory from this ASE structure."""
        self.driver = ResidentMACE(
            atoms, self.members[0], skin=self.skin, certify_edges=self.certify_edges
        )

    @staticmethod
    def _evaluate(members, shared, batch):
        if shared is not None:
            energy, force = shared.evaluate(batch)
            return Observation(energy.detach().reshape(-1), force.detach())
        values = [evaluate_prepared(calc, batch) for calc in members]
        return Observation(
            torch.stack([v[0] for v in values]), torch.stack([v[1] for v in values])
        )

    def observe(self, positions=None, *, share_geometry=True):
        """Evaluate all members at fixed positions; no integration or force averaging."""
        batch = self.driver.prepare(positions)
        shared = self.geometry if share_geometry else None
        return self._evaluate(self.members, shared, batch)

    def step(self, dt_fs, *, observe=False):
        """Always propagate member 0. Return (driver_E, driver_F, observation or None)."""
        energy, force = self.driver.step(dt_fs)
        if not observe:
            return energy, force, None
        batch = self.driver.prepare()
        rest = self._evaluate(self.members[1:], self.observer_geometry, batch)
        result = Observation(
            torch.cat((energy.reshape(1), rest.energies)),
            torch.cat((force.unsqueeze(0), rest.forces)),
        )
        return energy, force, result


def native_observation(calc, atoms):
    """Native ASE reference. calculate() ensures even identical warmup frames run."""
    calc.calculate(atoms)
    return (
        np.asarray(calc.results["energy_comm"]).reshape(-1),
        np.asarray(calc.results["forces_comm"]),
    )


def disagreement(forces):
    """Population vector-force disagreement from a host array (members, atoms, 3)."""
    centered = forces - forces.mean(axis=0)
    return np.sqrt(np.square(centered).sum(axis=-1).mean(axis=0))


def main():
    from .committee_cli import main as run

    return run()


if __name__ == "__main__":
    raise SystemExit(main())
