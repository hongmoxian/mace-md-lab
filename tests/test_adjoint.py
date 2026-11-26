"""Compare every angular output and its force pullback with the original e3nn basis."""

import unittest

import torch
from mace_md_lab._e3nn import o3

from mace_md_lab.adjoint import AngularAdjoint, install_angular_adjoint


class AngularDerivatives(unittest.TestCase):
    def check_device(self, device):
        for dtype in (torch.float32, torch.float64):
            generator = torch.Generator(device=device).manual_seed(209)
            # Noncontiguous input and a count spanning two GPU thread blocks.
            raw = torch.randn((3, 131), device=device, dtype=dtype, generator=generator)
            raw[:, 0] = 0
            raw[:, 1] = torch.tensor([2e-14, 3e-14, 5e-14], device=device, dtype=dtype)
            v = raw.T.detach().requires_grad_(True)
            for lmax in (1, 2, 3):
                oracle = o3.SphericalHarmonics(
                    list(range(lmax + 1)), normalize=True, normalization="component"
                ).to(device)
                actual = AngularAdjoint(lmax)(v)
                expected = oracle(v)
                weights = torch.randn(
                    (actual.shape[1], len(v)),
                    device=device,
                    dtype=dtype,
                    generator=generator,
                ).T
                ref_grad = torch.autograd.grad(expected, v, weights)[0]
                grad = torch.autograd.grad(actual, v, weights)[0]
                tol = 4e-6 if dtype == torch.float32 else 3e-13
                torch.testing.assert_close(actual, expected, rtol=tol, atol=tol)
                # At |v| < 1e-12 the upstream clamped normalization can produce
                # derivatives of size 1e12. Use a relative comparison there.
                torch.testing.assert_close(
                    grad[:2], ref_grad[:2], rtol=tol * 4, atol=tol
                )
                torch.testing.assert_close(
                    grad[2:], ref_grad[2:], rtol=tol * 4, atol=tol * 4
                )

    def test_cpu_against_e3nn(self):
        self.check_device("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_against_e3nn(self):
        self.check_device("cuda")

    def test_restores_on_exception(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(
                    torch.zeros((), dtype=torch.float64), requires_grad=False
                )
                self.spherical_harmonics = o3.SphericalHarmonics(
                    [0, 1, 2, 3], True, "component"
                )

        m = Model()
        original = m.spherical_harmonics
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            with install_angular_adjoint(m):
                self.assertIsInstance(m.spherical_harmonics, AngularAdjoint)
                raise RuntimeError("intentional")
        self.assertIs(m.spherical_harmonics, original)


if __name__ == "__main__":
    unittest.main()
