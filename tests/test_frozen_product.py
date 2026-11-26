import unittest

import torch
from torch import nn

from mace_md_lab.frozen_product import FrozenScalarProduct


class _Contraction(nn.Module):
    """Arbitrary non-symmetric path tensors stress the permutation coefficients."""

    def __init__(self, dtype=torch.float64):
        super().__init__()
        generator = torch.Generator().manual_seed(25)

        def draw(shape):
            return torch.randn(shape, dtype=dtype, generator=generator)

        self.correlation = 3
        dim, species, channels = 4, 3, 7
        for order in (1, 2, 3):
            self.register_buffer(f"U_{order}", draw((dim,) * order + (order + 1,)))
        self.weights_max = nn.Parameter(draw((species, 4, channels)))
        self.weights = nn.ParameterList(
            [
                nn.Parameter(draw((species, 3, channels))),
                nn.Parameter(draw((species, 2, channels))),
            ]
        )

    def U_tensors(self, degree):
        return getattr(self, f"U_{degree}")

    def forward(self, x, y):
        e1 = torch.einsum("ik,skc,ns,nci->nc", self.U_1, self.weights[1], y, x)
        e2 = torch.einsum("ijk,skc,ns,nci,ncj->nc", self.U_2, self.weights[0], y, x, x)
        e3 = torch.einsum(
            "ijlk,skc,ns,nci,ncj,ncl->nc", self.U_3, self.weights_max, y, x, x, x
        )
        return e1 + e2 + e3


class ProductTests(unittest.TestCase):
    def compare(self, device, dtype):
        native = _Contraction(dtype=dtype).to(device)
        x = torch.randn(6, 7, 4, device=device, dtype=dtype).requires_grad_(True)
        # Include all-zero graph padding, one-hot species, and fractional attributes.
        y = x.new_tensor(
            [[0, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 0.3], [0.2, 0, 0.8], [0, 0, 1]]
        )
        reference = native(x, y)
        tol = 1e-10 if dtype == torch.float64 else 3e-4
        grad = torch.randn(x.shape[1], len(y), device=device, dtype=dtype).T
        reference_grad = torch.autograd.grad(reference, x, grad)[0]
        values, vjps = [], []
        for schedule in ("precompute", "deferred"):
            with self.subTest(schedule=schedule):
                optimized = FrozenScalarProduct(native, schedule)
                result = optimized(x, y)
                torch.testing.assert_close(result, reference, rtol=tol, atol=tol)
                actual_grad = torch.autograd.grad(result, x, grad)[0]
                torch.testing.assert_close(
                    actual_grad, reference_grad, rtol=tol, atol=tol
                )
                self.assertEqual(actual_grad[0].abs().max().item(), 0.0)
                values.append(result.detach())
                vjps.append(actual_grad)
        # Both kernels accumulate the same sorted terms, although instruction
        # scheduling and final fused scaling can differ between compiled kernels.
        torch.testing.assert_close(values[0], values[1], rtol=tol, atol=tol)
        torch.testing.assert_close(vjps[0], vjps[1], rtol=tol, atol=tol)

    def test_cpu_scalar_products(self):
        self.compare("cpu", torch.float64)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_scalar_products(self):
        try:
            import cupy
        except ImportError:
            self.skipTest("CuPy unavailable for NVRTC")
        for dtype in (torch.float64, torch.float32):
            with self.subTest(dtype=dtype):
                self.compare("cuda", dtype)
