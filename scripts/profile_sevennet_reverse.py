"""Time the coordinate-gradient pass through each SevenNet convolution during full MD."""
import argparse
from contextlib import contextmanager, nullcontext
import hashlib
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import torch
from ase.io import read
from sevenn.calculator import SevenNetCalculator
from sevenn.nn.convolution import IrrepsConvolution

from benchmark_periodic_transfer import SevenGraph
from mace_md_lab.xl_fixed_step import ArrayVerlet


def event():
    e = torch.cuda.Event(enable_timing=True)
    e.record()
    return e


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


@contextmanager
def instrument(blocks, records):
    edits = []
    def wrap(name, module, forward):
        def call(data):
            record = {"layer": name}
            records.append(record)
            keys = [k for k in (module.key_x, module.key_filter, module.key_weight_input)
                    if data[k].requires_grad]
            local = {k: data[k] for k in (module.key_x, module.key_filter, module.key_weight_input, module.key_edge_idx)}
            values = InputBoundary.apply(record, *(data[k] for k in keys))
            local.update(zip(keys, values))
            record["forward_start"] = event()
            result = forward(local)
            record["forward_end"] = event()
            def start(grad):
                record["start"] = event()
                return grad
            result[module.key_x].register_hook(start)
            data[module.key_x] = result[module.key_x]
            return data
        return call
    try:
        for name, module in blocks:
            old = module.__dict__.get("forward")
            edits.append((module, old))
            module.forward = wrap(name, module, module.forward)
        yield
    finally:
        for module, old in reversed(edits):
            if old is None:
                del module.forward
            else:
                module.forward = old


def measure(path, steps):
    torch.cuda.synchronize()
    start = perf_counter()
    path.advance(steps)
    torch.cuda.synchronize()
    return 1000 * (perf_counter() - start) / steps


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for option in ("model", "structure", "output"):
        p.add_argument("--" + option, type=Path, required=True)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--repeats", type=int, default=4)
    p.add_argument("--adjoint", action="store_true", help="profile the ported reverse on the same driver")
    a = p.parse_args()
    atoms = read(a.structure)
    calc = SevenNetCalculator(model=str(a.model), device="cuda")
    blocks = [(n, m) for n, m in calc.model.named_modules() if isinstance(m, IrrepsConvolution)]
    if not blocks:
        raise ValueError("no e3nn IrrepsConvolution blocks")
    info = {}
    for n,m in blocks:
        tp=m.convolution
        info[n] = dict(x=str(tp.irreps_in1), y=str(tp.irreps_in2), out=str(tp.irreps_out),
                       paths=len(tp.instructions), weights=tp.weight_numel)
    ctx = nullcontext()
    if a.adjoint:
        from mace_md_lab.portable.sevennet_reverse import EdgeAdjoint, use_adjoint
        ctx = use_adjoint([(m,EdgeAdjoint(m)) for _,m in blocks])
    with ctx:
        ArrayVerlet(atoms, SevenGraph(atoms,calc,1.), .5).advance(12)
        rows=[]
        for i in range(a.repeats):
            row={}
            for mode in (("stock","traced") if i%2==0 else ("traced","stock")):
                path=ArrayVerlet(atoms, SevenGraph(atoms,calc,1.), .5)
                path.advance(3)
                if mode=="stock":
                    row["stock_ms"]=measure(path,a.steps)
                    continue
                records=[]
                with instrument(blocks,records):
                    row["traced_ms"]=measure(path,a.steps)
                ends=[(r["start"],r["end"]) for r in records]
                if len(ends)!=a.steps*len(blocks):
                    raise RuntimeError("backward boundary count mismatch")
                ref=records[0]["forward_start"]
                times=sorted((ref.elapsed_time(b),ref.elapsed_time(e)) for b,e in ends)
                union=0.; last=-1.
                for start,end in times:
                    union+=max(0.,end-max(start,last))
                    last=max(last,end)
                row["reverse_union_ms"]=union/a.steps
                row["layers"]={n:sum(r["start"].elapsed_time(r["end"]) for r in records if r["layer"]==n)/a.steps for n,_ in blocks}
                row["forward_layers_ms"]=sum(r["forward_start"].elapsed_time(r["forward_end"]) for r in records)/a.steps
            rows.append(row)
    data=dict(model_sha256=hashlib.sha256(a.model.read_bytes()).hexdigest(),fixture=a.structure.name,
              fixture_sha256=hashlib.sha256(a.structure.read_bytes()).hexdigest(),gpu=torch.cuda.get_device_name(),
              route="adjoint" if a.adjoint else "stock",atoms=len(atoms),steps=a.steps,warmup=3,dt_fs=.5,blocks=info,trials=rows,
              share=median(r["reverse_union_ms"]/r["traced_ms"] for r in rows))
    a.output.write_text(json.dumps(data,indent=2)+"\n")
    print("PROFILE",len(atoms),data["share"],[round(r["stock_ms"],2) for r in rows],flush=True)

if __name__ == "__main__":
    main()
