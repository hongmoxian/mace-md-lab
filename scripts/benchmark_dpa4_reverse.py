"""Full-step DPA4 transfer ablations and shared-geometry acceptance tests."""
import argparse
from contextlib import nullcontext
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
from statistics import median

from ase.io import read
from deepmd.calculator import DP
from deepmd.pt.model.descriptor.sezm_nn.so2 import SO2Convolution
import numpy as np
import torch

from benchmark_periodic_transfer import deep_eval
from profile_dpa4_reverse import timed
from mace_md_lab.portable.dpa4_reverse import use_adjoint
from mace_md_lab.xl_fixed_step import ArrayVerlet


def check_block_vjps(blocks,evaluate,positions,route):
    """Compare independent coordinate-path input adjoints on a subset of actual edges."""
    captured={}
    handles=[]
    for name,module in blocks:
        def grab(mod,args,key=name):
            x,cache,rad=args[:3]
            count=min(len(cache.src),48)
            values={}
            for k,v in cache._asdict().items():
                if k in ("D_to_m_cache","Dt_from_m_cache"):
                    values[k]={} if v is not None else None
                elif isinstance(v,torch.Tensor):
                    per_node=k in ("deg","inv_sqrt_deg")
                    values[k]=(v if per_node else v[:count]).detach().clone()
                else:
                    values[k]=v
            captured[key]=(x.detach().clone(),cache._replace(**values),rad[:count].detach().clone())
        handles.append(module.register_forward_pre_hook(grab))
    try:
        evaluate(positions)
    finally:
        for h in handles:
            h.remove()
    report={}
    for name,module in blocks:
        xv,base,rv=captured[name]
        upstream=torch.cos(torch.arange(xv.numel(),device=xv.device,dtype=xv.dtype)).reshape_as(xv)
        def one():
            x=xv.detach().requires_grad_()
            r=rv.detach().requires_grad_()
            env=base.edge_env.detach().requires_grad_()
            d=base.D_full.detach().requires_grad_()
            dt=base.Dt_full.detach().requires_grad_()
            cache=base._replace(edge_env=env,D_full=d,Dt_full=dt,
                                D_to_m_cache={},Dt_from_m_cache={})
            result=module(x,cache,r)
            grad=torch.autograd.grad(result,(x,r,env,d,dt),upstream)
            return (result.detach(),*grad)
        ref=one()
        with use_adjoint([(name,module)],route):
            out=one()
        vals={}
        for key,g,h in zip(("output","node","radial","envelope","rotation","inverse_rotation"),ref,out):
            torch.testing.assert_close(h,g,rtol=2e-4,atol=8e-5)
            vals[key]=float((g-h).abs().max())
        report[name]=vals
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("model","structure","output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--routes",nargs="+",choices=["stock","edge","linear","combined","full","packed","transport","local","reverse"],default=["stock","local"])
    p.add_argument("--steps",type=int,default=8)
    p.add_argument("--repeats",type=int,default=8)
    p.add_argument("--check-vjp",choices=["edge","combined","full","transport","local","reverse"])
    a=p.parse_args()
    if a.steps < 1 or a.repeats < 1:
        p.error("steps and repeats must be positive")
    if len(set(a.routes)) != len(a.routes) or len(a.routes) < 2:
        p.error("select at least two distinct routes, including stock")
    atoms=read(a.structure)
    calc=DP(model=str(a.model))
    blocks=[(n,m) for n,m in calc.dp.deep_eval.dp.named_modules() if isinstance(m,SO2Convolution)]
    if not blocks or "stock" not in a.routes:
        raise ValueError("requires stock and eager SO2 convolutions")
    def path():
        return ArrayVerlet(atoms,deep_eval(atoms,calc),.5)
    def setup(route):
        return nullcontext() if route=="stock" else use_adjoint(blocks,route)
    official=atoms.copy()
    official.calc=calc
    initial=path()
    frames=[]
    for _ in range(5):
        initial.advance(1)
        official.positions=initial.positions.copy()
        frames.append((initial.positions.copy(),official.get_potential_energy(),official.get_forces(),official.get_stress()))
    errors={}
    checks={}
    counts={}
    limits=np.array([2e-5,2e-4,2e-5])
    for route in a.routes:
        with setup(route) as hits:
            eval_fn=deep_eval(atoms,calc)
            delta=[]
            for pos,e,f,s in frames:
                ee,ff,ss=eval_fn(pos)
                delta.append((abs(ee-e)/len(atoms),np.max(np.abs(ff-f)),np.max(np.abs(ss-s))))
            errors[route]=np.max(delta,axis=0).tolist()
            checks[route]=bool(np.all(np.asarray(errors[route])<=limits))
            print("GATE",route,errors[route],checks[route],flush=True)
            if not checks[route]:
                raise RuntimeError("failed common-frame E/F/stress")
            path().advance(6)
            if hits is not None:
                counts[route]=dict(hits)
                required={"edge":"edge","linear":"linear","combined":"edge","full":"gate",
                          "transport":"transport","local":"local","reverse":"transport"}
                if route in required and not hits[required[route]]:
                    raise RuntimeError(f"no dispatch for requested route {route}")
    block_vjp=(check_block_vjps(blocks,deep_eval(atoms,calc),atoms.positions,a.check_vjp) if a.check_vjp else None)
    orders=[]
    trials=[]
    for i in range(a.repeats):
        shift=i%len(a.routes)
        order=a.routes[shift:]+a.routes[:shift]
        if len(a.routes)>2 and i//len(a.routes)%2:
            order=order[::-1]
        row={}
        for route in order:
            with setup(route):
                runner=path()
                runner.advance(3)
                row[route]=timed(runner,a.steps)
        orders.append(order)
        trials.append(row)
    medians={r:median(t[r] for t in trials) for r in a.routes}
    result=dict(atoms=len(atoms),fixture=a.structure.name,model_sha256=hashlib.sha256(a.model.read_bytes()).hexdigest(),
                fixture_sha256=hashlib.sha256(a.structure.read_bytes()).hexdigest(),deepmd=version("deepmd-kit"),torch=str(torch.__version__),
                gpu=torch.cuda.get_device_name(),dt_fs=.5,warmup=3,steps=a.steps,trials=trials,
                orders=orders,medians=medians,saved={r:1-med/medians["stock"] for r,med in medians.items()},
                faster_pairs={r:sum(t[r]<t["stock"] for t in trials) for r in a.routes if r!="stock"},
                efs_limits=limits.tolist(),efs_max_abs=errors,passed=all(checks.values()),dispatch=counts,block_vjp=block_vjp)
    a.output.write_text(json.dumps(result,indent=2)+"\n")
    print("RESULT",medians,result["saved"],flush=True)

if __name__=="__main__":
    main()
