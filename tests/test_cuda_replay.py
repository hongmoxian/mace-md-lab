"""Real MACE force-graph replay with changing exact periodic connectivity."""

from pathlib import Path
import runpy
import unittest

import numpy as np
import torch
from ase import Atoms
from mace.calculators import MACECalculator

from mace_md_lab.runtime import ResidentMACE


class CudaReplay(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_capture_refreshes_edges_and_preserves_old_outputs(self):
        fixture = Path(__file__).resolve().parents[1] / "examples" / "make_fixture.py"
        build = runpy.run_path(str(fixture))["model"]
        with torch.random.fork_rng(devices=[]):
            model = build(seed=19, channels=2).double()
        calc = MACECalculator(models=model, device="cuda", default_dtype="float64")
        atoms = Atoms(
            "Cu3",
            positions=[[0, 0, 0], [3, 0, 0], [10, 0, 0]],
            cell=np.eye(3) * 30,
            pbc=True,
        )
        ref = ResidentMACE(atoms, calc, skin=0)
        cap = ResidentMACE(atoms, calc, skin=0.8, capture_forces=True)
        saved = None
        # Same count but different neighbors, then equivalent unwrapped image,
        # followed by a change in edge count that must trigger a fresh capture.
        for positions in (
            atoms.positions,
            [[0, 0, 0], [7, 0, 0], [10, 0, 0]],
            [[0, 0, 0], [37, 0, 0], [10, 0, 0]],
            [[0, 0, 0], [3, 0, 0], [4.1, 0.5, 0]],
        ):
            pair = cap.evaluate(positions)
            exact = ref.evaluate(positions)
            for x, y in zip(pair, exact):
                torch.testing.assert_close(x, y, rtol=1e-9, atol=1e-9)
            if saved is None:
                saved = [(x, x.clone()) for x in pair]
            for x, y in saved:
                torch.testing.assert_close(x, y, rtol=0, atol=0)
        self.assertEqual(cap.captures, 2)
        self.assertEqual(cap.replays, 4)
        self.assertEqual(cap.fallback_evaluations, 0)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_padded_isolation_and_capacity_growth(self):
        fixture = Path(__file__).resolve().parents[1] / "examples" / "make_fixture.py"
        build = runpy.run_path(str(fixture))["model"]
        with torch.random.fork_rng(devices=[]):
            model = build(seed=23, channels=2).double()
        # Give the padding atom a nonzero isolated atomic energy. Accidentally
        # taking the energy sum over both graphs then fails the equivalence check.
        with torch.no_grad():
            model.atomic_energies_fn.atomic_energies.fill_(2.0)
        calc = MACECalculator(models=model, device="cuda", default_dtype="float64")
        atoms = Atoms(
            "Cu4",
            positions=[[0, 0, 0], [3, 0, 0], [10, 0, 0], [18, 0, 0]],
            cell=np.eye(3) * 50,
            pbc=True,
        )
        exact = ResidentMACE(atoms, calc, skin=0)
        padded = ResidentMACE(
            atoms, calc, skin=0.8, capture_forces=True, pad_capture=True
        )
        cases = (
            (atoms.positions, 2),
            ([[0, 0, 0], [7, 0, 0], [10, 0, 0], [18, 0, 0]], 2),
            ([[0, 0, 0], [57, 0, 0], [10, 0, 0], [18, 0, 0]], 2),
            ([[0, 0, 0], [3, 0, 0], [10, 0, 0], [13, 0, 0]], 4),
            (atoms.positions, 2),
            ([[0, 0, 0], [1, 0, 0], [2, 0.2, 0], [3, 0, 0]], 12),
        )
        for i, (pos, nedges) in enumerate(cases):
            observed = padded.evaluate(pos)
            for x, y in zip(observed, exact.evaluate(pos)):
                torch.testing.assert_close(x, y, rtol=1e-9, atol=1e-9)
            self.assertEqual(padded.captures, 1 if i < 5 else 2)
            # Structural assertions expose stale cross-graph connectivity even if
            # a particular set of weights happens to suppress its contribution.
            batch = padded._replay._inputs
            edge = batch["edge_index"]
            self.assertTrue(bool((edge[:, :nedges] < len(atoms)).all()))
            self.assertTrue(bool((edge[:, nedges:] == len(atoms)).all()))
            self.assertEqual(batch["ptr"].tolist(), [0, len(atoms), len(atoms) + 1])
        self.assertEqual(padded.replays, len(cases))
        self.assertEqual(padded.fallback_evaluations, 0)


if __name__ == "__main__":
    unittest.main()
