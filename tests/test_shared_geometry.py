"""CPU checks for exact feature sharing across distinct scalar MACE members."""

import os
from pathlib import Path
import runpy
import unittest

import torch
from ase import Atoms
from mace.calculators import MACECalculator

from mace_md_lab.shared_geometry import SharedGeometry


class SharedGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.addClassCleanup(torch.set_default_dtype, torch.get_default_dtype())
        cls.addClassCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_default_dtype(torch.float64)
        torch.set_num_threads(1)
        fixture_dir = os.environ.get("MACE_MD_LAB_FIXTURES")
        if fixture_dir:
            paths = sorted(Path(fixture_dir).glob("member-*.model"))
            if len(paths) < 2:
                raise ValueError(
                    "MACE_MD_LAB_FIXTURES must contain at least two example members"
                )
            cls.calc = MACECalculator(
                model_paths=[str(path) for path in paths],
                device="cpu",
                default_dtype="float64",
            )
        else:
            example = (
                Path(__file__).resolve().parents[1] / "examples" / "make_fixture.py"
            )
            make = runpy.run_path(str(example))["model"]
            with torch.random.fork_rng(devices=[]):
                models = [make(seed, channels=2) for seed in (17, 29)]
            cls.calc = MACECalculator(
                models=models, device="cpu", default_dtype="float64"
            )
        cls.models = cls.calc.models
        cls.atoms = Atoms(
            "Cu4",
            pbc=True,
            cell=[[9.7, 0, 0], [0.6, 9.0, 0], [0.2, 0.3, 9.2]],
            positions=[
                [0.7, 1.1, 2.3],
                [2.8, 2.0, 1.6],
                [4.1, 0.8, 1.9],
                [1.4, 3.5, 3.0],
            ],
        )

    def batch(self, atoms):
        return self.calc._atoms_to_batch(atoms).to_dict()

    def native_energies(self, atoms):
        batch = self.batch(atoms)
        with torch.no_grad():
            return torch.stack(
                [
                    model(dict(batch), training=False, compute_force=False)[
                        "energy"
                    ].reshape(())
                    for model in self.models
                ]
            )

    def test_member_forces_and_refresh(self):
        originals = [
            (model.spherical_harmonics, model.radial_embedding) for model in self.models
        ]
        shared = SharedGeometry(self.models)
        atoms = self.atoms.copy()
        batch = self.batch(atoms)
        e0, f0 = shared.evaluate(batch)
        # A nontrivial committee must have different forces on an asymmetric frame.
        self.assertGreater((f0[0] - f0[1]).abs().max().item(), 1e-8)
        atoms.positions[1] += [0.213, -0.04, 0.065]
        moved_batch = self.batch(atoms)
        e1, f1 = shared.evaluate(moved_batch)
        self.assertGreater((f1 - f0).abs().max().item(), 1e-8)
        for data, expected_e, expected_f in [(batch, e0, f0), (moved_batch, e1, f1)]:
            for k, model in enumerate(self.models):
                inputs = dict(data, positions=data["positions"].detach().clone())
                result = model(
                    inputs, training=False, compute_force=True, compute_stress=False
                )
                torch.testing.assert_close(
                    expected_e[k], result["energy"].reshape(()), rtol=1e-9, atol=1e-10
                )
                torch.testing.assert_close(
                    expected_f[k], result["forces"], rtol=1e-8, atol=1e-10
                )
        # An independent numerical derivative of each untouched member's scalar energy.
        h = 1e-5
        plus, minus = atoms.copy(), atoms.copy()
        plus.positions[0, 2] += h
        minus.positions[0, 2] -= h
        minus_gradient = -(self.native_energies(plus) - self.native_energies(minus)) / (
            2 * h
        )
        torch.testing.assert_close(f1[:, 0, 2], minus_gradient, atol=2e-8, rtol=2e-5)
        for (angular, radial), model in zip(originals, self.models):
            self.assertIs(angular, model.spherical_harmonics)
            self.assertIs(radial, model.radial_embedding)

    def test_rejects_different_fixed_radial_constants(self):
        weights = self.models[1].radial_embedding.bessel_fn.bessel_weights
        original = weights.detach().clone()
        with torch.no_grad():
            try:
                weights[0] += 1e-3
                with self.assertRaisesRegex(ValueError, "radial embedding constants"):
                    SharedGeometry(self.models)
            finally:
                weights.copy_(original)


if __name__ == "__main__":
    unittest.main()
