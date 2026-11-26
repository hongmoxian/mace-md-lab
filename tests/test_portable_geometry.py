"""Value, force and differentiable periodic cell checks for common Cartesian geometry."""

import unittest
import torch
from mace_md_lab.portable.geometry import edge_geometry, edge_geometry_from_positions

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


class PortableGeometry(unittest.TestCase):
    def test_partial_and_missing_cotangents(self):
        for dev in DEVICES:
            for dtype in (torch.float64, torch.float32):
                tol = 2e-11 if dtype == torch.float64 else 4e-5
                p = torch.tensor(
                    [
                        [0.0, 0.0, 0.0],
                        [0.2, 0.1, 0.0],
                        [1.0, -0.4, 2.0],
                        [0.5, 0.0, 0.0],
                    ],
                    dtype=dtype,
                    device=dev,
                    requires_grad=True,
                )
                wr = torch.tensor([0.7, 0.2, -1.0, 1.5], dtype=dtype, device=dev)
                wu = torch.arange(12, dtype=dtype, device=dev).reshape(3, 4).T / 11
                for eps in (None, 0.5):
                    for used in ("distance", "unit", "both"):

                        def calculate(mode, quantities):
                            output = edge_geometry(p, mode, eps, quantities=quantities)
                            if quantities == "both":
                                r, u = output
                            else:
                                r, u = (
                                    (output, None)
                                    if quantities == "distance"
                                    else (None, output)
                                )
                            terms = []
                            if used != "unit":
                                terms.append((r * wr).sum())
                            if used != "distance":
                                terms.append((u * wu).sum())
                            return output, torch.autograd.grad(sum(terms), p)[0]

                        baseline, dn = calculate("native", "both")
                        output, da = calculate("analytic", "both")
                        for x, y in zip(baseline, output):
                            torch.testing.assert_close(y, x, atol=tol, rtol=tol)
                        torch.testing.assert_close(da, dn, atol=tol, rtol=tol)
                        if used != "both":
                            _, ds = calculate("analytic", used)
                            torch.testing.assert_close(ds, dn, atol=tol, rtol=tol)

    def test_periodic_self_edges_and_mask(self):
        for dev in DEVICES:
            for dtype in (torch.float32, torch.float64):
                tol = 4e-5 if dtype == torch.float32 else 2e-11
                p = torch.tensor(
                    [[0.0, 0.3, -0.8], [0.1, 0.3, -0.6], [1.0, -0.3, 0.4]],
                    device=dev,
                    dtype=dtype,
                    requires_grad=True,
                )
                ij = torch.tensor(
                    [[2, 1], [0, 2], [2, 2], [1, 1], [1, 2], [-1, 100], [2, 1]],
                    device=dev,
                ).T
                shifts = torch.tensor(
                    [
                        [0.0, 0.0, 0.0],
                        [0.2, -0.3, 1.0],
                        [1.0, 0.1, 0.15],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [20.0, -4.0, 0.0],
                        [-0.3, 0.7, -0.1],
                    ],
                    device=dev,
                    dtype=dtype,
                    requires_grad=True,
                )
                mask = torch.tensor([1, 1, 1, 1, 1, 0, 1], device=dev, dtype=torch.bool)
                wr = torch.linspace(-1, 2, len(mask), device=dev, dtype=dtype)
                wu = (
                    torch.arange(len(mask) * 3, device=dev, dtype=dtype)
                    .reshape(3, -1)
                    .T
                    / 9
                )
                for eps in (None, 0.3):
                    src, dst = torch.where(mask[None, :], ij, 0)
                    d = torch.where(mask[:, None], p[dst] - p[src] + shifts, 0.0)
                    r0 = torch.linalg.vector_norm(d, dim=1)
                    u0 = (
                        d / r0.clamp_min(eps)[:, None]
                        if eps
                        else torch.where(
                            (r0 > 0)[:, None],
                            d / torch.where(r0 > 0, r0, 1.0)[:, None],
                            0.0,
                        )
                    )
                    dn, sn = torch.autograd.grad(
                        (r0 * wr).sum() + (u0 * wu).sum(), (p, shifts)
                    )
                    r, u = edge_geometry_from_positions(
                        p, ij, shifts, "analytic", eps, edge_mask=mask
                    )
                    da, sa = torch.autograd.grad(
                        (r * wr).sum() + (u * wu).sum(), (p, shifts)
                    )
                    for x, y in [(r0, r), (u0, u), (dn, da), (sn, sa)]:
                        torch.testing.assert_close(y, x, atol=tol, rtol=tol)
                    torch.testing.assert_close(
                        da.sum(0), torch.zeros_like(da[0]), rtol=0, atol=tol
                    )
                    self.assertEqual(sa[5].abs().sum().item(), 0.0)
                    self.assertGreater(sa[2].abs().sum().item(), 0.0)

    def test_empty_edges_and_missing_shift(self):
        for dev in DEVICES:
            p = torch.randn(5, 3, device=dev, dtype=torch.float64, requires_grad=True)
            for e in [
                torch.empty((2, 0), dtype=torch.long, device=dev),
                torch.tensor([[0, 2, 0], [1, 0, 4]], device=dev),
            ]:
                grads = []
                for mode in ("native", "analytic"):
                    r = edge_geometry_from_positions(
                        p, e, mode=mode, quantities="distance"
                    )
                    grads.append(torch.autograd.grad(r.sum(), p)[0])
                torch.testing.assert_close(*grads, rtol=1e-11, atol=1e-11)

    def test_finite_difference(self):
        for dev in DEVICES:
            g = torch.Generator().manual_seed(2026)
            p = (
                torch.randn(5, 3, generator=g, dtype=torch.float64)
                .to(dev)
                .requires_grad_(True)
            )
            v = torch.randn(5, 3, generator=g, dtype=torch.float64).to(dev)
            v /= v.norm()
            ij = torch.tensor(
                [[0, 2, 1, 1, 3, 3, 4], [1, 4, 2, 3, 1, 2, 0]], device=dev
            )
            s = torch.randn(7, 3, generator=g, dtype=torch.float64).to(dev)
            w = torch.randn(3, generator=g, dtype=torch.float64).to(dev)

            def energy(coords):
                r, u = edge_geometry_from_positions(coords, ij, s, mode="analytic")
                return (r.sin() * (u @ w).tanh()).sum()

            pr = (torch.autograd.grad(energy(p), p)[0] * v).sum()
            with torch.no_grad():
                fd = (energy(p + 1e-5 * v) - energy(p - 1e-5 * v)) / (2e-5)
            torch.testing.assert_close(pr, fd, atol=1e-8, rtol=1e-7)

    def test_cell_derivative(self):
        for dev in DEVICES:
            p = torch.tensor(
                [[0.3, 0.1, 0.4], [0.7, -0.2, 0.5]], device=dev, dtype=torch.float64
            )
            ij = torch.tensor([[0, 1], [1, 0]], device=dev)
            c = torch.eye(3, device=dev, dtype=torch.float64, requires_grad=True)
            images = torch.tensor(
                [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]], device=dev, dtype=torch.float64
            )
            grads = []
            for mode in ("native", "analytic"):
                r, u = edge_geometry_from_positions(p, ij, images @ c, mode)
                grads.append(
                    torch.autograd.grad(r.sum() + u[:, 0].square().sum(), c)[0]
                )
            torch.testing.assert_close(*grads, rtol=1e-11, atol=1e-11)


if __name__ == "__main__":
    unittest.main()
