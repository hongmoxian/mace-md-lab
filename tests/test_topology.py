"""Exact graph membership regressions using MACE's own fresh per-frame graph."""

from pathlib import Path
import runpy
import unittest

import numpy as np
import torch
from ase import Atoms
from mace.calculators import MACECalculator
from mace.data.neighborhood import get_neighborhood

from mace_md_lab.runtime import ResidentMACE


def image_keys(edges, shifts):
    """A directed neighbor is identified by *both* atom indices and its lattice image."""
    return sorted(map(tuple, np.column_stack((edges.T, shifts)).tolist()))


class ExactTopology(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture = Path(__file__).resolve().parents[1] / "examples" / "make_fixture.py"
        build = runpy.run_path(str(fixture))["model"]
        with torch.random.fork_rng(devices=[]):
            model = build(seed=11, channels=2).double()
        cls.calculator = MACECalculator(
            models=model, device="cpu", default_dtype="float64"
        )

    def new_pair(self, radius):
        atoms = Atoms(
            "Cu2", positions=[[0, 0, 0], [radius, 0, 0]], cell=np.eye(3) * 20, pbc=True
        )
        return atoms, ResidentMACE(atoms, self.calculator, skin=0.8, certify_edges=True)

    def check_graph(self, runtime, atoms, positions):
        pos = np.asarray(positions, dtype=np.float64)
        expected, _, units, _ = get_neighborhood(
            pos,
            cutoff=runtime.cutoff,
            cell=atoms.cell.array.copy(),
            pbc=tuple(atoms.pbc),
        )
        graph = runtime.prepare(pos)
        edges = graph["edge_index"].cpu().numpy()
        shifts = graph["unit_shifts"].cpu().numpy()
        self.assertEqual(image_keys(edges, shifts), image_keys(expected, units))
        np.testing.assert_allclose(
            graph["shifts"].cpu().numpy(),
            shifts @ atoms.cell.array,
            rtol=1e-14,
            atol=1e-14,
        )
        return edges.shape[1]

    def test_cutoff_crossing_inside_skin_and_exact_equality(self):
        atoms, driver = self.new_pair(4.8)
        # The second frame reuses the exact graph. Subsequent motion crosses
        # r_max without invalidating the candidate Verlet list.
        sizes = []
        for r in (4.8, 4.82, 5.05, 5.0, 4.999):
            sizes.append(self.check_graph(driver, atoms, [[0, 0, 0], [r, 0, 0]]))
        self.assertEqual(sizes, [2, 2, 0, 0, 2])
        self.assertEqual(driver.rebuilds, 1)
        self.assertEqual(driver.certificate_hits, 1)
        self.assertEqual(driver.edge_refreshes, 4)

    def test_empty_candidates_and_omitted_edge(self):
        atoms, driver = self.new_pair(6.0)
        # Initially the pair is beyond r_max+skin. An infinite empty-list gap
        # cannot authorize old indices after the separate skin bound is exceeded.
        sizes = [
            self.check_graph(driver, atoms, [[0, 0, 0], [r, 0, 0]])
            for r in (6.0, 5.8, 5.2, 4.99)
        ]
        self.assertEqual(sizes, [0, 0, 0, 2])
        self.assertEqual(driver.rebuilds, 2)
        self.assertEqual(driver.certificate_hits, 1)

    def test_periodic_images_and_repeatability(self):
        cell = np.array([[3.0, 0.2, 0.0], [0.1, 2.8, 0.3], [0.1, 0.0, 3.3]])
        atoms = Atoms("Cu", positions=[[0, 0, 0]], cell=cell, pbc=True)
        counters = []
        for _ in range(2):
            driver = ResidentMACE(atoms, self.calculator, skin=0.8, certify_edges=True)
            for translation in (np.zeros(3), np.array([1e-5, 0, 0]), cell[0], -cell[2]):
                count = self.check_graph(driver, atoms, atoms.positions + translation)
                self.assertGreater(count, 1)  # nonzero self-images in a small skew cell
            counters.append(
                (driver.rebuilds, driver.certificate_hits, driver.edge_refreshes)
            )
        self.assertGreater(counters[0][1], 0)
        self.assertEqual(counters[0], counters[1])


if __name__ == "__main__":
    unittest.main()
