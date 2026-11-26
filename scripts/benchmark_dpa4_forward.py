"""Paired whole-step forward replay ablation with a moving-cutoff correctness gate."""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
from statistics import median
from importlib.metadata import version
from time import perf_counter

import numpy as np
from ase.io import read
from deepmd.calculator import DP
from deepmd.pt.model.descriptor.sezm_nn.so2 import SO2Convolution
import torch

from benchmark_periodic_transfer import deep_eval
from profile_dpa4_reverse import timed
from mace_md_lab.portable.dpa4_graph import replay_energy_core
from mace_md_lab.portable.dpa4_reverse import use_adjoint
from mace_md_lab.xl_fixed_step import ArrayVerlet


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "structure", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--routes", nargs="+", choices=("stock","local","front","joint","stockgraph"), default=["stock","local","front","joint","stockgraph"])
    p.add_argument("--bucket-policy", choices=("headroom", "aligned"), default="headroom")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--steps", type=int, default=6)
    a = p.parse_args()
    if min(a.steps, a.repeats) < 1 or len(set(a.routes)) != len(a.routes):
        p.error("positive lengths and distinct routes required")
    atoms = read(a.structure)
    calc = DP(model=str(a.model))
    evaluate = deep_eval(atoms,calc)
    def runner():
        return ArrayVerlet(atoms,evaluate,.5)
    def setup(route, stack, blocks, model):
        if route in ("local","front","joint"):
            stack.enter_context(use_adjoint(blocks,"local"))
        if route in ("front","joint","stockgraph"):
            return stack.enter_context(replay_energy_core(model,forward_only=route=="front", headroom=a.bucket_policy=="headroom"))
        return {}

    ref = atoms.copy()
    ref.calc = calc
    initial = runner()
    positions = []
    for _ in range(5):
        initial.advance(1)
        positions.append(initial.positions.copy())
    rng = np.random.default_rng(809)
    for amplitude in (.035,.12):
        positions.append(atoms.positions + rng.uniform(-amplitude, amplitude, atoms.positions.shape))
    moved = positions[-1].copy()
    moved[0] += atoms.cell.array[0]
    positions.extend((moved,positions[0]))
    refs = []
    for pos in positions:
        ref.positions = pos
        refs.append((ref.get_potential_energy(),ref.get_forces(),ref.get_stress()))
    trials = [{} for _ in range(a.repeats)]
    timed_captures = [{} for _ in range(a.repeats)]
    orders = []
    for i in range(a.repeats):
        shift = i % len(a.routes)
        order = a.routes[shift:] + a.routes[:shift]
        orders.append(order if i//len(a.routes)%2 == 0 else order[::-1])
    errors, gate_counts, counts, cold = {}, {}, {}, {}
    # Switching the local kernel binding is cheap. Graph instances must survive
    # interleaved trials, so each route receives its own evaluator/model.
    routes = {}
    with ExitStack() as lifetime:
        for route in a.routes:
            calc = DP(model=str(a.model))
            modules = list(calc.dp.deep_eval.dp.named_modules())
            blocks = [(n,m) for n,m in modules if isinstance(m,SO2Convolution)]
            model, = [m for _,m in modules if hasattr(m,"core_compute") and hasattr(m,"forward_common_lower")]
            metrics = setup(route,lifetime,blocks,model)
            fn = deep_eval(atoms,calc)
            torch.cuda.synchronize()
            t0 = perf_counter()
            first = ArrayVerlet(atoms,fn,.5)
            first.advance(12)
            torch.cuda.synchronize()
            cold[route] = dict(total_ms=1000*(perf_counter()-t0), steps=12,
                               force_calls=13, metrics=dict(metrics))
            delta = []
            for pos,(e,f,s) in zip(positions,refs,strict=True):
                ee,ff,ss = fn(pos)
                delta.append([abs(ee-e)/len(atoms),np.max(np.abs(ff-f)),np.max(np.abs(ss-s))])
            if route in ("front","joint","stockgraph") and not metrics["captures"]:
                raise RuntimeError("expected CUDA graph dispatch")
            errors[route] = np.max(delta,axis=0).tolist()
            gate_counts[route] = dict(metrics)
            print("GATE",route,errors[route],gate_counts[route],flush=True)
            if not np.all(np.asarray(errors[route]) <= [2e-5,2e-4,2e-5]):
                raise ValueError("failed shared-geometry E/F/stress gate")
            routes[route] = (fn,metrics)
        for i,order in enumerate(orders):
            for route in order:
                fn,metrics = routes[route]
                runner_i = ArrayVerlet(atoms,fn,.5)
                runner_i.advance(3)
                before = metrics.get("captures",0)
                trials[i][route] = timed(runner_i,a.steps)
                timed_captures[i][route] = metrics.get("captures",0) - before
        counts = {name: dict(pair[1]) for name,pair in routes.items()}
    med = {r:median(row[r] for row in trials) for r in a.routes}
    result = dict(fixture=a.structure.name, atoms=len(atoms), gpu=torch.cuda.get_device_name(), torch=str(torch.__version__),
                  deepmd=version("deepmd-kit"),model_sha256=hashlib.sha256(a.model.read_bytes()).hexdigest(),
                  fixture_sha256=hashlib.sha256(a.structure.read_bytes()).hexdigest(), dt_fs=.5, bucket_policy=a.bucket_policy, steps=a.steps,warmup=3,
                  orders=orders,trials=trials, medians=med, cold=cold, timed_captures=timed_captures,efs_max_abs=errors,gate_metrics=gate_counts,final_metrics=counts)
    print("MEDIANS",med,flush=True)
    a.output.write_text(json.dumps(result,indent=2)+"\n")


if __name__ == "__main__":
    main()
