"""Frozen radial expansion and first Cartesian-chain derivative for 7net-0 and DPA4.

One CUDA program evaluates each edge tile. Its reverse contracts feature cotangents
with analytic radial slopes without constructing a feature Jacobian. The DPA4
envelope uses the upstream positive-coefficient polynomial near its cutoff.
"""
from contextlib import contextmanager
from types import MethodType

import torch
from torch.autograd.function import once_differentiable
import triton
import triton.language as tl


@triton.jit
def _basis(R, W, G, OUT, N, C: tl.constexpr, WIDTH: tl.constexpr,
           RC: tl.constexpr, COEFFS: tl.constexpr, MODE: tl.constexpr,
           SCALE: tl.constexpr, REVERSE: tl.constexpr, BR: tl.constexpr):
    e = tl.program_id(0)*BR + tl.arange(0, BR)
    c = tl.arange(0, WIDTH)
    r = tl.load(R+e, e < N, 1.)
    w = tl.load(W+c, c < C, 0.)
    if MODE == 0:  # SevenNet polynomial (edges already satisfy the physical cutoff)
        x = r / RC
        xpower = tl.full((BR,), 1., r.dtype)
        for k in tl.static_range(len(COEFFS)-1):
            xpower *= x
        p: tl.constexpr = len(COEFFS)-1
        a: tl.constexpr = (p+1)*(p+2)/2
        b: tl.constexpr = p*(p+2)
        d: tl.constexpr = p*(p+1)/2
        env = 1. + xpower * (-a+x*(b-d*x))
        envgrad = -a*p*xpower / tl.where(r!=0.,r,1.) * (1.-x)*(1.-x)
    elif MODE == 4:  # SevenNet XPLOR envelope between onset and cutoff
        on: tl.constexpr = COEFFS[0]
        a0: tl.constexpr = on*on
        b0: tl.constexpr = RC*RC
        denom: tl.constexpr = (b0-a0)*(b0-a0)*(b0-a0)
        sq = r*r
        gap = b0-sq
        env = tl.where(r<on,1.,gap*gap*(b0+2.*sq-3.*a0)/denom)
        envgrad = tl.where(r<on,0.,-12.*r*gap*(sq-a0)/denom)
    else:
        u = tl.minimum(1., tl.maximum(0., (RC-r)/RC))
        x = 1.-u
        series = tl.full((BR,), COEFFS[len(COEFFS)-1], r.dtype)
        ds = tl.full((BR,), 0., r.dtype)
        for j in tl.static_range(len(COEFFS)-1):
            ds = ds*x + series
            series = COEFFS[len(COEFFS)-2-j] + x*series
        env = (u*u)*(u*u)*series
        envgrad = tl.where((r>=0.) & (r<=RC), (u*u*u)*(u*ds-4.*series)/RC, 0.)
    if MODE == 2:  # Gaussian DPA4
        delta = r[:,None]-w[None,:]
        val = tl.exp(SCALE*delta*delta)
        dv = 2.*SCALE*delta*val
    elif MODE == 3:  # standalone DPA4 message envelope
        val = tl.full((BR, WIDTH), 1., r.dtype)
        dv = tl.full((BR, WIDTH), 0., r.dtype)
    else:
        z = r[:,None]*w[None,:]
        sn = tl.sin(z)
        div = 1./tl.where(r!=0., r, 1.)
        val = SCALE*sn*div[:,None]
        dv = (SCALE*w[None,:]*tl.cos(z)-val)*div[:,None]
        if MODE == 1:  # stable limit for DPA4 sinc
            small = tl.abs(z)<0.001
            val = tl.where(small,SCALE*w[None,:]*(1.-z*z/6.+z*z*z*z/120.),val)
            dv = tl.where(small,SCALE*w[None,:]*w[None,:]*(-z/3.+z*z*z/30.),dv)
    ix = e[:,None]*C+c[None,:]
    ok = (e[:,None]<N) & (c[None,:]<C)
    if REVERSE:
        g = tl.load(G+ix,ok,0.)
        out = tl.sum(tl.where(ok,g*(dv*env[:,None]+val*envgrad[:,None]),0.),axis=1)
        tl.store(OUT+e,out,e<N)
    else:
        tl.store(OUT+ix,val*env[:,None],ok)


class _Radial(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r, freq, rc, coeffs, mode, scale):
        r, freq = r.contiguous(), freq.detach().contiguous().reshape(-1)
        n, c = r.numel(), freq.numel()
        y = torch.empty((n,c), device=r.device, dtype=r.dtype)
        args = (c, triton.next_power_of_2(c), rc, coeffs, mode, scale)
        if n:
            _basis[(triton.cdiv(n,32),)](r,freq,r,y,n,*args,False,32,enable_fp_fusion=False)
        ctx.save_for_backward(r,freq)
        ctx.args = args
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        r, freq = ctx.saved_tensors
        dx = torch.empty_like(r)
        if r.numel():
            _basis[(triton.cdiv(r.numel(),32),)](r,freq,grad.contiguous(),dx,r.numel(),
                                                *ctx.args,True,32,enable_fp_fusion=False)
        return dx,None,None,None,None,None


@contextmanager
def fused_radial(model, family, angular=False):
    """Replace qualified operators during inference only and restore existing forwards."""
    edits = []
    calls = {"radial": 0, "envelope": 0}

    def bind(module, fn):
        edits.append((module, module.__dict__.get("forward")))
        module.forward = MethodType(fn, module)

    try:
        if family == "sevennet":
            from sevenn import _keys as key
            from sevenn.nn.edge_embedding import BesselBasis, PolynomialCutoff, XPLORCutoff, EdgeEmbedding
            blocks = [m for m in model.modules() if isinstance(m, EdgeEmbedding)]
            if len(blocks) != 1:
                raise ValueError("expected one SevenNet edge embedding")
            block = blocks[0]
            radial, cutoff = block.basis_function, block.cutoff_function
            if not isinstance(radial, BesselBasis):
                raise ValueError("requires a frozen SevenNet Bessel basis")
            if isinstance(cutoff, XPLORCutoff):
                coeffs, rc, mode = (float(cutoff.r_on),), float(cutoff.r_cut), 4
            elif isinstance(cutoff, PolynomialCutoff):
                coeffs, rc, mode = tuple(range(int(cutoff.p)+1)), float(cutoff.cutoff_length), 0
            else:
                raise ValueError("unqualified SevenNet cutoff")
            if angular:
                from mace_md_lab.adjoint import AngularAdjoint
                sh = block.spherical.sph
                lmax = int(sh._lmax)
                if (not sh.normalize or sh.normalization != "component" or lmax > 3
                        or list(sh._ls_list) != list(range(lmax + 1))):
                    raise ValueError("unsupported SevenNet spherical-harmonics contract")
                op = AngularAdjoint(lmax)
                def harmonics(m, vec):
                    return op(vec)
                bind(block.spherical,harmonics)
            def seven(m, data):
                vec = data[key.EDGE_VEC]
                r = torch.linalg.norm(vec,dim=-1)
                data[key.EDGE_LENGTH] = r
                data[key.EDGE_EMBEDDING] = _Radial.apply(r,radial.coeffs,rc,coeffs,mode,float(radial.prefactor))
                data[key.EDGE_ATTR] = m.spherical(vec)
                calls["radial"] += 1
                return data
            bind(block, seven)
        elif family == "dpa4":
            from deepmd.pt.model.descriptor.sezm_nn.radial import C3CutoffEnvelope, RadialBasis
            radial = [m for m in model.modules() if isinstance(m,RadialBasis)]
            if len(radial) != 1:
                raise ValueError("expected one DPA4 radial expansion")
            rad = radial[0]
            mode = {"bessel": 1, "gaussian": 2}[rad.basis_type]
            scale = float(rad.gaussian_coeff.item()) if mode==2 else 1.
            coeffs = tuple(rad.envelope._series_coefficients)
            def dpa(m, r):
                calls["radial"] += 1
                return _Radial.apply(r,m.adam_freqs,m.rcut,coeffs,mode,scale)
            bind(rad,dpa)
            for env in model.modules():
                if isinstance(env,C3CutoffEnvelope) and env is not rad.envelope:
                    def envelope(m, r):
                        calls["envelope"] += 1
                        return _Radial.apply(r,m.rcut_tensor,m.rcut,tuple(m._series_coefficients),
                                             3,1.).reshape_as(r)
                    bind(env,envelope)
        else:
            raise ValueError(f"unsupported framework {family}")
        yield calls
    finally:
        for module,old in reversed(edits):
            if old is None:
                del module.forward
            else:
                module.forward = old
