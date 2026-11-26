import unittest

import torch

from mace_md_lab.portable.runner import ForceCall, MolecularGraph


class HarmonicPairs:
    @staticmethod
    def energy(x, edges, mode):
        src, dst = edges
        return 0.25 * (x[dst] - x[src]).square().sum()


class PortableRunnerTests(unittest.TestCase):
    def test_exact_cutoff_crossing_and_reuse(self):
        x = torch.tensor([[0., 0., 0.], [1.1, 0., 0.]], dtype=torch.float64)
        graph = MolecularGraph(x, 1.)
        self.assertEqual(graph.update(x).size(1), 0)
        self.assertEqual(graph.update(x + .01).size(1), 0)
        self.assertEqual(graph.refreshes, 1)
        x[1, 0] = 1.
        self.assertEqual(graph.update(x).size(1), 0)
        x[1, 0] = .999
        self.assertEqual(graph.update(x).size(1), 2)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_recording_tracks_changed_edges(self):
        x = torch.arange(9, device="cuda", dtype=torch.float64).reshape(3, 3)
        eager = ForceCall(HarmonicPairs(), "native", False)
        replay = ForceCall(HarmonicPairs(), "native", True)
        for i, pair in enumerate(([[0, 1], [1, 0]], [[0, 2], [2, 0]], [[1], [2]])):
            edges = torch.tensor(pair, dtype=torch.long, device=x.device)
            expected = eager(x, edges)
            got = replay(x, edges)
            for a, b in zip(got, expected):
                torch.testing.assert_close(a, b)
            self.assertEqual(replay.captures, 1 if i < 2 else 2)


if __name__ == "__main__":
    unittest.main()
