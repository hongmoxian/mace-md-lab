"""Regression checks for basis values and the fused feature-to-distance gradient."""
import math
import unittest

import torch


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class RadialDerivative(unittest.TestCase):
    def test_basis_and_vector_jacobian_product(self):
        from mace_md_lab.portable.periodic_radial import _Radial

        for mode in range(5):
            with self.subTest(mode=mode):
                rc = 5.
                p = 6
                distances = [0.02, 0.92, 1.7, 3.9, rc - 1e-5, rc]
                if mode not in (0,4):
                    distances = [0.] + distances + [rc + 1e-5]
                r = torch.tensor(distances, dtype=torch.float64, device="cuda", requires_grad=True)
                freq = r.new_tensor([1.] if mode == 3 else [.6, 1.3, 1.8])
                scale = (-.32 if mode == 2 else 2./rc if mode in (0,4) else 1.)
                series = tuple(float(math.comb(k+3,3)) for k in range(p))
                coeffs = tuple(range(p+1)) if mode == 0 else series
                if mode == 4:
                    coeffs = (3.,)
                pred = _Radial.apply(r,freq,rc,coeffs,mode,scale)
                x = r/rc
                if mode == 0:
                    env = 1. - ((p+1)*(p+2)/2)*x**p + p*(p+2)*x**(p+1) - p*(p+1)/2*x**(p+2)
                elif mode == 4:
                    sq = r*r
                    env = torch.where(r<3.,1.,(rc**2-sq)**2*(rc**2+2.*sq-27.)/(rc**2-9.)**3)
                else:
                    u = ((rc-r)/rc).clamp(0.,1.)
                    env = u**4 * sum(c*(1-u)**k for k,c in enumerate(series))
                if mode in (0,1,4):
                    val = freq*torch.sinc(r[:,None]*freq/math.pi)*scale
                elif mode == 2:
                    val = ((r[:,None]-freq)**2*scale).exp()
                else:
                    val = torch.ones_like(r[:,None])
                ref = val*env[:,None]
                upstream = torch.cos(torch.arange(ref.numel(),device=r.device,dtype=r.dtype)).reshape_as(ref)
                dx, = torch.autograd.grad(pred,r,upstream)
                honest, = torch.autograd.grad(ref,r,upstream)
                torch.testing.assert_close(pred,ref,atol=2e-11,rtol=2e-10)
                torch.testing.assert_close(dx,honest,atol=2e-10,rtol=2e-9)
