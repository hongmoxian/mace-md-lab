"""Paired SevenNet-0 MD timing with stock forward and alternative message adjoints."""
import argparse
from contextlib import nullcontext
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
from statistics import median

import numpy as np
import torch
from ase.io import read
from sevenn.calculator import SevenNetCalculator
from sevenn.nn.convolution import IrrepsConvolution

from benchmark_periodic_transfer import SevenGraph
from profile_sevennet_reverse import measure
from mace_md_lab.portable.sevennet_reverse import EdgeAdjoint, use_adjoint
from mace_md_lab.xl_fixed_step import ArrayVerlet


def check_adjoints(blocks):
    from sevenn.nn.convolution import message_gather

    generator = torch.Generator(device="cuda").manual_seed(183)
    def draw(rows, cols):
        return torch.randn(rows, cols, device="cuda", generator=generator, requires_grad=True)
    reports = []
    for module, plan in blocks:
        tp = module.convolution
        src = torch.randint(9, (29,), device="cuda", generator=generator)
        dst = torch.randint(9, (29,), device="cuda", generator=generator)
        x, y, w = draw(9,tp.irreps_in1.dim), draw(29,tp.irreps_in2.dim), draw(29,tp.weight_numel)
        reference = message_gather(x,dst,tp(x[src],y,w))
        upstream = draw(9,tp.irreps_out.dim).detach()
        expected = torch.autograd.grad(reference,(x,y,w),upstream)
        with torch.no_grad():
            actual = plan.reverse(x,y,w,src,dst,upstream)
        for a,b in zip(actual,expected):
            torch.testing.assert_close(a,b,rtol=4e-5,atol=4e-5)
        reports.append({k:float((a-b).abs().max()) for k,a,b in zip(("node","angular","radial"),actual,expected)})
    return reports


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("model","structure","output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--steps",type=int,default=8)
    p.add_argument("--repeats",type=int,default=8)
    p.add_argument("--adjoint-first",action="store_true",help="reverse the first timing pair")
    a=p.parse_args()
    atoms=read(a.structure)
    calc=SevenNetCalculator(model=str(a.model),device="cuda")
    names=[(n,m) for n,m in calc.model.named_modules() if isinstance(m,IrrepsConvolution)]
    blocks=[(m,EdgeAdjoint(m)) for n,m in names]
    assert blocks
    local_grad_errors=check_adjoints(blocks)
    def path():
        return ArrayVerlet(atoms,SevenGraph(atoms,calc,1.),.5)
    official=atoms.copy()
    official.calc=calc
    warm=path()
    frames=[]
    for _ in range(5):
        warm.advance(1)
        official.positions=warm.positions.copy()
        frames.append((warm.positions.copy(),official.get_potential_energy(),official.get_forces(),official.get_stress()))
    errs={}
    for tag in ("stock","adjoint"):
        with use_adjoint(blocks) if tag=="adjoint" else nullcontext():
            evaluate=SevenGraph(atoms,calc,1.)
            delta=[]
            for pos,e,f,s in frames:
                ee,ff,ss=evaluate(pos)
                delta.append([abs(e-ee)/len(atoms),np.abs(f-ff).max(),np.abs(s-ss).max()])
            errs[tag]=np.max(delta,axis=0).tolist()
            path().advance(6)
    passed=all(np.all(np.asarray(x)<=[2e-5,2e-4,2e-5]) for x in errs.values())
    print("GATE",errs,passed,flush=True)
    if not passed:
        raise RuntimeError("checkpoint comparison failed")
    rows=[]
    for i in range(a.repeats):
        row={}
        for tag in (("stock","adjoint") if (i+int(a.adjoint_first))%2==0 else ("adjoint","stock")):
            with use_adjoint(blocks) if tag=="adjoint" else nullcontext():
                runner=path()
                runner.advance(3)
                row[tag]=measure(runner,a.steps)
        rows.append(row)
    m={key:median(r[key] for r in rows) for key in rows[0]}
    data=dict(model_sha256=hashlib.sha256(a.model.read_bytes()).hexdigest(),fixture=a.structure.name,
              fixture_sha256=hashlib.sha256(a.structure.read_bytes()).hexdigest(),gpu=torch.cuda.get_device_name(),
              sevennet=version("sevenn"),torch=str(torch.__version__),atoms=len(atoms),steps=a.steps,warmup=3,
              dt_fs=.5,first_route=list(rows[0])[0],paired_trials=rows,medians=m,saved_fraction=1-m["adjoint"]/m["stock"],
              faster_pairs=sum(r["adjoint"]<r["stock"] for r in rows),
              efs_max_abs=errs,passed=passed,local_adjoint_max_abs=local_grad_errors,
              backward_calls={n:plan.calls for (n,_),(_,plan) in zip(names,blocks)})
    a.output.write_text(json.dumps(data,indent=2)+"\n")
    print("TIMING",m,data["saved_fraction"],flush=True)
if __name__=="__main__":
    main()
