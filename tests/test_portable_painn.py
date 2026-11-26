"""Check the energy-only fast path against SchNetPack's full Energy + Forces output."""

import os
from pathlib import Path
import unittest

import torch
from ase.io import read

from mace_md_lab.portable.painn import load_adapter
from mace_md_lab.portable.runner import MolecularGraph


@unittest.skipUnless(
    os.environ.get("SCHNETPACK_PAINN_MODEL"),
    "requires official rMD17 ethanol checkpoint",
)
class PaiNNReferenceTest(unittest.TestCase):
    def test_upstream_forces_and_mean_offset(self):
        xyz = Path(__file__).resolve().parents[1] / "examples/structures/ethanol.xyz"
        atoms = read(xyz)
        for dtype, atol in ((torch.float64, 1e-11), (torch.float32, 2e-5)):
            with self.subTest(dtype=dtype):
                adapter = load_adapter(
                    os.environ["SCHNETPACK_PAINN_MODEL"],
                    "cpu",
                    dtype,
                    torch.tensor(atoms.numbers),
                )
                x = torch.tensor(atoms.positions, dtype=dtype, requires_grad=True)
                official = adapter.model(adapter._official_inputs(x))
                ref_energy = official["energy"].detach().reshape(()) * adapter.unit
                ref_forces = official["forces"].detach() * adapter.unit
                edges = MolecularGraph(x, adapter.cutoff).update(x)
                for mode in ("native", "analytic"):
                    with self.subTest(mode=mode):
                        e = adapter.energy(x, edges, mode)
                        f = -torch.autograd.grad(e, x)[0]
                        torch.testing.assert_close(e, ref_energy, atol=atol, rtol=0)
                        torch.testing.assert_close(
                            f, ref_forces.to(f), atol=atol, rtol=0
                        )


if __name__ == "__main__":
    unittest.main()
