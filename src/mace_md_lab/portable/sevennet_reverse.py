"""Inference-only edge-owned adjoint of SevenNet's e3nn uvu tensor product and sum."""
from contextlib import contextmanager
from functools import lru_cache
import hashlib
import importlib.util
from pathlib import Path
import tempfile

import torch
from torch.autograd.function import once_differentiable
import triton

_cache = tempfile.TemporaryDirectory(prefix="sevennet-adjoint-")


@lru_cache(None)
def _path_kernel(lx, ly, lo, scale):
    """Specialize sparse Clebsch-Gordan paths, retaining e3nn's path normalization."""
    from e3nn.o3 import wigner_3j

    cg = wigner_3j(lx, ly, lo, dtype=torch.float64) * scale
    sx, sy, so = cg.shape
    terms = [(i, j, k, float(cg[i,j,k])) for i in range(sx)
             for j in range(sy) for k in range(so) if abs(cg[i,j,k]) > 1e-12]
    lines = ["""import triton
import triton.language as tl

@triton.jit
def path(X,Y,W,S,D,G,DX,DY,DW,N,XS:tl.constexpr,YS:tl.constexpr,WS:tl.constexpr,
         GS:tl.constexpr,X0:tl.constexpr,Y0:tl.constexpr,W0:tl.constexpr,G0:tl.constexpr,
         C:tl.constexpr,B:tl.constexpr,NE:tl.constexpr):
    e = tl.program_id(0)*NE + tl.arange(0,NE)
    c = tl.arange(0,B)
    valid = (e[:,None]<N) & (c[None,:]<C)
    src = tl.load(S+e,e<N,0).to(tl.int32)
    dst = tl.load(D+e,e<N,0).to(tl.int32)
    w = tl.load(W+e[:,None]*WS+W0+c[None,:], valid, 0.)
""".rstrip()]
    def add(s):
        lines.append("    "+s)
    for i in range(sx):
        add(f"x{i} = tl.load(X+src[:,None]*XS+X0+c[None,:]*{sx}+{i},valid,0.)")
    for j in range(sy):
        add(f"y{j} = tl.load(Y+e*YS+Y0+{j},e<N,0.)[:,None]")
    for k in range(so):
        add(f"g{k} = tl.load(G+dst[:,None]*GS+G0+c[None,:]*{so}+{k},valid,0.)")
    # Contract receiver adjoints and sender features before differentiating angular inputs.
    for j in range(sy):
        inner = " + ".join(f"({v!r}*x{i}*g{k})" for i,jj,k,v in terms if jj==j)
        add(f"p{j} = {inner}")
        add(f"tl.atomic_add(DY+e*YS+Y0+{j},tl.sum(p{j}*w,axis=1),e<N,sem='relaxed')")
    add("dw = "+" + ".join(f"p{j}*y{j}" for j in range(sy)))
    add("tl.store(DW+e[:,None]*WS+W0+c[None,:],dw,valid)")
    for i in range(sx):
        inner = " + ".join(f"({v!r}*y{j}*g{k})" for ii,j,k,v in terms if ii==i)
        add(f"dx{i} = w*({inner})")
        add(f"tl.atomic_add(DX+src[:,None]*XS+X0+c[None,:]*{sx}+{i},dx{i},valid,sem='relaxed')")
    source="\n".join(lines)+"\n"
    name="seven_path_"+hashlib.sha256(source.encode()).hexdigest()[:20]
    file=Path(_cache.name)/(name+".py")
    file.write_text(source)
    spec=importlib.util.spec_from_file_location(name,file)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.path


class EdgeAdjoint:
    def __init__(self, conv):
        self.tp = conv.convolution
        tp=self.tp
        if tp.shared_weights or tp.internal_weights or conv.is_parallel:
            raise ValueError("requires local e3nn convolution with per-edge radial weights")
        self.paths=[]
        seen=set()
        weight_offset=0
        for ins in tp.instructions:
            a,b,o=ins.i_in1,ins.i_in2,ins.i_out
            c,ix=tp.irreps_in1[a]
            ny,iy=tp.irreps_in2[b]
            no,io=tp.irreps_out[o]
            if ins.connection_mode!="uvu" or not ins.has_weight or o in seen or ny!=1 or no!=c:
                raise ValueError("requires independent channel-preserving uvu paths")
            seen.add(o)
            kernel=_path_kernel(ix.l,iy.l,io.l,float(ins.path_weight))
            self.paths.append((kernel,tp.irreps_in1.slices()[a].start,tp.irreps_in2.slices()[b].start,
                               weight_offset,tp.irreps_out.slices()[o].start,c))
            weight_offset+=c
        if weight_offset!=tp.weight_numel or len(seen)!=len(tp.irreps_out):
            raise ValueError("uncovered e3nn path")
        self.calls=0

    def reverse(self,x,y,w,src,dst,g):
        dx=torch.zeros_like(x)
        dy=torch.zeros_like(y)
        dw=torch.empty_like(w)
        g=g.contiguous()
        n=src.numel()
        if n:
            for kernel,x0,y0,w0,g0,c in self.paths:
                kernel[(triton.cdiv(n,4),)](x,y,w,src,dst,g,dx,dy,dw,n,x.shape[1],y.shape[1],
                    w.shape[1],g.shape[1],x0,y0,w0,g0,c,triton.next_power_of_2(c),4,num_warps=4)
        self.calls+=1
        return dx,dy,dw


class _Message(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,y,w,src,dst,plan):
        from sevenn.nn.convolution import message_gather
        x,y,w,src,dst=(t.contiguous() for t in (x,y,w,src,dst))
        if x.dtype!=torch.float32 or not x.is_cuda:
            raise ValueError("qualified for CUDA float32")
        ctx.save_for_backward(x,y,w,src,dst)
        ctx.plan=plan
        return message_gather(x,dst,plan.tp(x[src],y,w))

    @staticmethod
    @once_differentiable
    def backward(ctx,g):
        dx,dy,dw=ctx.plan.reverse(*ctx.saved_tensors,g)
        return dx,dy,dw,None,None,None


@contextmanager
def use_adjoint(blocks):
    """Use the original radial network and forward contraction in every block."""
    restored=[]
    def wrapper(module,plan):
        def call(data):
            edge=data[module.key_edge_idx]
            w=module.weight_nn(data[module.key_weight_input])
            value=_Message.apply(data[module.key_x],data[module.key_filter],w,edge[1],edge[0],plan)
            data[module.key_x]=value/module.denominator
            return data
        return call
    try:
        for module,plan in blocks:
            restored.append((module,module.__dict__.get("forward")))
            module.forward=wrapper(module,plan)
        yield
    finally:
        for module,old in reversed(restored):
            if old is None:
                del module.forward
            else:
                module.forward=old
