"""Matched short NVE tests of prepared drivers and a frozen radial derivative."""
import argparse
from contextlib import nullcontext
import hashlib
from importlib.metadata import version
import json
import sys
from pathlib import Path
from statistics import median
from time import perf_counter

from ase import units
from ase.io import read
from ase.md.verlet import VelocityVerlet
from ase.stress import full_3x3_to_voigt_6_stress
import numpy as np
import torch

from mace_md_lab.portable.periodic_radial import fused_radial
from mace_md_lab.xl_fixed_step import ArrayVerlet


class SevenGraph:
    """Use a checked candidate graph and retain the physical cutoff at every step."""
    def __init__(self, atoms, calc, skin):
        self.atoms, self.calc, self.skin = atoms.copy(), calc, skin
        self.ref = None

    def __call__(self, pos):
        from sevenn import _keys as key
        from sevenn.atom_graph_data import AtomGraphData
        from sevenn.train.dataload import unlabeled_atoms_to_graph
        if self.ref is None or 2*np.linalg.norm(pos-self.ref,axis=1).max() >= self.skin-1.e-12:
            self.atoms.positions[:] = pos
            d = unlabeled_atoms_to_graph(self.atoms,self.calc.cutoff+self.skin,with_shift=True)
            self.ij = d.pop(key.EDGE_IDX)
            self.shift = d.pop(key.CELL_SHIFT) @ self.atoms.cell.array
            d.pop(key.CELL)
            self.static = d
            self.ref = pos.copy()
        vec = pos[self.ij[1]]-pos[self.ij[0]]+self.shift
        active = np.einsum("ij,ij->i",vec,vec) < self.calc.cutoff**2
        d = dict(self.static)
        d.update({key.POS:pos,key.EDGE_IDX:self.ij[:,active],key.EDGE_VEC:vec[active]})
        graph = AtomGraphData.from_numpy_dict(d).to(self.calc.device)
        result = self.calc.output_to_results(self.calc.model(graph))
        return result["energy"], result["forces"], result["stress"]


def deep_eval(atoms, calc):
    if any(k in atoms.info for k in ("fparam","aparam","charge_spin")):
        raise ValueError("requires a field-free fixture")
    types = [calc.type_dict[k] for k in atoms.get_chemical_symbols()]
    cell = atoms.cell.array.reshape(1,9)
    volume = atoms.get_volume()
    def evaluate(pos):
        e, f, v = calc.dp.eval(pos.reshape(1,-1),cell,types)[:3]
        return float(e[0,0]), f[0], full_3x3_to_voigt_6_stress(-v[0].reshape(3,3)/volume)
    return evaluate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--framework", choices=["sevennet","dpa4"], required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--structure", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--memory", action="store_true", help="run fresh processes for process-peak comparisons")
    p.add_argument("--memory-route", choices=["official","driver","driver_kernel","driver_radial_angular"])
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--angular", action="store_true", help="include existing MACE angular operator on SevenNet")
    p.add_argument("--repeats", type=int, default=6)
    a = p.parse_args()
    if a.memory and not a.memory_route:
        from benchmark_peak_memory import monitor
        rows = {}
        routes = ["official","driver","driver_kernel"]
        if a.angular:
            routes.append("driver_radial_angular")
        for name in routes:
            rows[name] = monitor([__file__,*sys.argv[1:],"--memory-route",name])
            a.output.write_text(json.dumps({"framework":a.framework,"routes":rows},indent=2)+"\n")
        return
    atoms = read(a.structure,index=0)
    if not np.all(atoms.pbc) or atoms.constraints or not atoms.has("momenta"):
        raise ValueError("supply a periodic unconstrained fixture with initial momenta")
    if min(a.steps,a.repeats,a.warmup)<1:
        p.error("positive counts required")
    if a.framework == "dpa4":
        from deepmd.calculator import DP
        calc = DP(model=str(a.model))
        model = calc.dp.deep_eval.dp
        new_eval = lambda: deep_eval(atoms,calc)
        pkg = "deepmd-kit"
    else:
        from sevenn.calculator import SevenNetCalculator
        calc = SevenNetCalculator(model=str(a.model),device="cuda")
        model = calc.model
        new_eval = lambda: SevenGraph(atoms,calc,1.)
        pkg = "sevenn"
    if not next(model.parameters()).is_cuda:
        raise ValueError("expected CUDA model")
    def fresh_ase():
        box = atoms.copy()
        calc.reset()
        box.calc = calc
        dynamics = VelocityVerlet(box,.5*units.fs,logfile=None)
        box.get_forces()
        return box,dynamics
    if a.memory_route:
        name = a.memory_route
        op = fused_radial(model,a.framework,angular=name=="driver_radial_angular") if name.startswith("driver_") else nullcontext()
        with op as hits:
            if name == "official":
                box,dynamics = fresh_ase()
                advance = dynamics.run
            else:
                path = ArrayVerlet(atoms,new_eval(),.5)
                advance = path.advance
            advance(a.warmup)
            torch.cuda.synchronize()
            start = perf_counter()
            advance(a.steps)
            torch.cuda.synchronize()
            elapsed = 1000*(perf_counter()-start)/a.steps
            if hits is not None and hits["radial"] < a.steps:
                raise RuntimeError("radial replacement bypassed during memory test")
            print("MEMRESULT "+json.dumps(dict(atoms=len(atoms),gpu=torch.cuda.get_device_name(),
                steps=a.steps,warmup=a.warmup,ms_step=elapsed,
                model_sha256=hashlib.sha256(a.model.read_bytes()).hexdigest(),
                torch_allocated_peak_MiB=torch.cuda.max_memory_allocated()/2**20,
                torch_reserved_peak_MiB=torch.cuda.max_memory_reserved()/2**20)),flush=True)
        return
    box,dyn = fresh_ase()
    ref = []
    for _ in range(5):
        ref.append((box.positions.copy(),box.get_potential_energy(),box.get_forces(),box.get_stress()))
        dyn.run(1)
    names = ["official","driver","driver_kernel"]
    if a.angular:
        if a.framework != "sevennet":
            p.error("--angular requires SevenNet")
        names.append("driver_radial_angular")
    def operators(name):
        if name in ("driver_kernel","driver_radial_angular"):
            return fused_radial(model,a.framework,angular=name=="driver_radial_angular")
        return nullcontext()
    times = {name: [] for name in names}
    errors = {}
    calls = {}
    for name in names[1:]:
        cm = operators(name)
        with cm as hits:
            evaluate = new_eval()
            gaps = []
            for pos,e,f,s in ref:
                pred = evaluate(pos)
                gaps.append([abs(pred[0]-e)/len(atoms),np.max(np.abs(pred[1]-f)),np.max(np.abs(pred[2]-s))])
            errors[name] = np.max(gaps,axis=0).tolist()
            if hits is not None and hits["radial"] == 0:
                raise RuntimeError("radial replacement was bypassed")
    orders = []
    for i in range(a.repeats):
        shift = i % len(names)
        seq = names[shift:]+names[:shift]
        if i//len(names)%2:
            seq = seq[::-1]
        orders.append(seq)
        for name in seq:
            cm = operators(name)
            with cm as hits:
                if name=="official":
                    at, dynamics = fresh_ase()
                    advance = dynamics.run
                else:
                    path = ArrayVerlet(atoms,new_eval(),.5)
                    advance = path.advance
                advance(a.warmup)
                torch.cuda.synchronize()
                t = perf_counter()
                advance(a.steps)
                torch.cuda.synchronize()
                times[name].append(1000*(perf_counter()-t)/a.steps)
                if name == "official" and "stress" not in at.calc.results:
                    raise RuntimeError("missing stress")
                if hits is not None:
                    calls.setdefault(name, {})[i] = dict(hits)
                    if hits["radial"] < a.steps:
                        raise RuntimeError("insufficient native replacement calls")
    limits = np.array([2e-5,2e-4,2e-5])
    valid = all(np.all(np.array(gap)<=limits) for gap in errors.values())
    result = dict(model=a.model.name,sha256=hashlib.sha256(a.model.read_bytes()).hexdigest(),
            fixture=a.structure.name,fixture_sha256=hashlib.sha256(a.structure.read_bytes()).hexdigest(),
            framework=a.framework,version=version(pkg),torch=str(torch.__version__),gpu=torch.cuda.get_device_name(),
            dtype=str(next(model.parameters()).dtype),atoms=len(atoms),dt_fs=.5,steps=a.steps,
            warmup=a.warmup,orders=orders,ms_per_step=times,medians={k:median(v) for k,v in times.items()},
            max_abs_E_per_atom_F_stress=errors,thresholds=limits.tolist(),geometry_checks=len(ref),
            replacement_calls=calls,passed=valid)
    a.output.write_text(json.dumps(result,indent=2)+"\n")
    if not valid:
        raise RuntimeError("same-frame numerical tolerance failed")


if __name__ == "__main__":
    main()
