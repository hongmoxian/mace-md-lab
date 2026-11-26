"""Compare DPA4 SO(2) convolution backward spans with synchronized fixed-cell MD."""
import argparse
from contextlib import contextmanager, nullcontext
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
from statistics import median
from time import perf_counter

from ase.io import read
from deepmd.calculator import DP
from deepmd.pt.model.descriptor.sezm_nn.so2 import SO2Convolution
import torch
from torch.utils._pytree import tree_flatten, tree_unflatten

from benchmark_periodic_transfer import deep_eval
from mace_md_lab.xl_fixed_step import ArrayVerlet


def event():
    marker = torch.cuda.Event(enable_timing=True)
    marker.record()
    return marker


class InputBoundary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, record, *values):
        ctx.record = record
        ctx.set_materialize_grads(False)
        return tuple(values)

    @staticmethod
    def backward(ctx, *grads):
        ctx.record["end"] = event()
        return (None, *grads)


def timed(runner, steps):
    torch.cuda.synchronize()
    start = perf_counter()
    runner.advance(steps)
    torch.cuda.synchronize()
    return 1000 * (perf_counter() - start) / steps


def describe(blocks):
    fields = ("channels","lmax","mmax","n_focus","focus_dim","head_dim","n_atten_head",
              "needs_local_frame","use_flash_atten","edge_cartesian","mixing_layers",
              "so2_attn_res_mode","use_triton_infer","radial_so2_mode","hidden_channels")
    return {n: {**{k:getattr(m,k,None) for k in fields}, "children":[k for k,_ in m.named_children()]}
            for n,m in blocks}


@contextmanager
def instrument(blocks, records):
    edits = []
    def wrap(name, fn):
        def call(*args, **kwargs):
            leaves, tree = tree_flatten((args, kwargs))
            indices = [i for i,t in enumerate(leaves) if isinstance(t,torch.Tensor) and t.requires_grad]
            rec = {"block": name}
            records.append(rec)
            marked = InputBoundary.apply(rec, *(leaves[i] for i in indices))
            for i,value in zip(indices,marked):
                leaves[i] = value
            positional, keywords = tree_unflatten(leaves,tree)
            rec["f0"] = event()
            result = fn(*positional, **keywords)
            rec["f1"] = event()
            def begin(g):
                rec["start"] = event()
                return g
            result.register_hook(begin)
            return result
        return call
    try:
        for name, module in blocks:
            edits.append((module,module.__dict__.get("forward")))
            module.forward = wrap(name,module.forward)
        yield
    finally:
        for module, old in reversed(edits):
            if old is None:
                del module.forward
            else:
                module.forward = old


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("model","structure","output"):
        p.add_argument("--"+k,type=Path,required=True)
    p.add_argument("--steps",type=int,default=4)
    p.add_argument("--repeats",type=int,default=4)
    p.add_argument("--variant",choices=("stock","combined","full","edge","linear","packed","transport","local","reverse"),default="stock")
    a=p.parse_args()
    if min(a.steps,a.repeats)<1:
        p.error("steps and repeats must be positive")
    atoms=read(a.structure)
    calc=DP(model=str(a.model))
    model=calc.dp.deep_eval.dp
    blocks=[(n,m) for n,m in model.named_modules() if isinstance(m,SO2Convolution)]
    if not blocks or not next(model.parameters()).is_cuda:
        raise ValueError("requires eager DPA4 and CUDA")
    configs=describe(blocks)
    print("BLOCKS",list(configs.values())[:1],len(blocks),flush=True)
    def runner():
        return ArrayVerlet(atoms,deep_eval(atoms,calc),.5)
    ctx=nullcontext()
    if a.variant!="stock":
        from mace_md_lab.portable.dpa4_reverse import use_adjoint
        ctx=use_adjoint(blocks,a.variant)
    trials=[]
    with ctx:
        runner().advance(8)
        for i in range(a.repeats):
            row={}
            for tag in (("bare","traced") if i%2==0 else ("traced","bare")):
                path=runner()
                path.advance(3)
                if tag=="bare":
                    row["bare_ms"]=timed(path,a.steps)
                    continue
                rec=[]
                with instrument(blocks,rec):
                    row["traced_ms"]=timed(path,a.steps)
                if len(rec)!=len(blocks)*a.steps:
                    raise RuntimeError("missing SO2Convolution call")
                origin=rec[0]["f0"]
                spans=sorted((origin.elapsed_time(r["start"]),origin.elapsed_time(r["end"])) for r in rec)
                end=-1.
                union=0.
                for b,e in spans:
                    union+=max(0.,e-max(end,b))
                    end=max(e,end)
                row["backward_ms"]=union/a.steps
                row["block_ms"]={n:sum(r["start"].elapsed_time(r["end"]) for r in rec if r["block"]==n)/a.steps for n,_ in blocks}
                row["forward_ms"]=sum(r["f0"].elapsed_time(r["f1"]) for r in rec)/a.steps
            trials.append(row)
    share=median(r["backward_ms"]/r["traced_ms"] for r in trials)
    result=dict(route=a.variant, atoms=len(atoms),model_sha256=hashlib.sha256(a.model.read_bytes()).hexdigest(),
                fixture_sha256=hashlib.sha256(a.structure.read_bytes()).hexdigest(),fixture=a.structure.name,torch=str(torch.__version__),
                deepmd=version("deepmd-kit"),dtype=str(next(model.parameters()).dtype),gpu=torch.cuda.get_device_name(),
                blocks=configs,steps=a.steps,warmup=3,dt_fs=.5,trials=trials,share=share)
    a.output.write_text(json.dumps(result,indent=2)+"\n")
    print("SHARE",len(atoms),share,flush=True)

if __name__=="__main__":
    main()
