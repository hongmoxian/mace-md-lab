"""Compare prepared XL evaluation, ASE, and fixed-cell Verlet in moving MD."""

import argparse
from contextlib import contextmanager, ExitStack
import json
import os
from pathlib import Path
from statistics import median
from time import perf_counter
from unittest.mock import patch

import numpy as np
import torch

from benchmark_xl_step import fixed_pair_bound
from mace_md_lab.bench import ASEPath, file_hash, numerical_gates, package_version, starting_structure, write_report
from mace_md_lab.xl_fixed_step import PreparedCalculator, PreparedXL, ArrayVerlet


R0_VARIANTS = {
    "ase_upstream_r0": "upstream",
    "ase_factored_r0": "factored",
    "arrays_upstream_r0": "upstream",
    "arrays_factored_r0": "factored",
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True, help="original trusted MACE checkpoint")
    p.add_argument("--symmetrix-json", type=Path, required=True)
    p.add_argument("--structure", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=48)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeats", type=int, default=6)
    p.add_argument("--skin", type=float, default=1.0)
    p.add_argument("--dt-fs", type=float, default=0.5)
    p.add_argument("--kernel-launch-policy", choices=("automatic", "static"), default="automatic")
    p.add_argument("--product-dir", type=Path)
    p.add_argument("--routes", nargs="+", choices=["ase", "certificate", "ase_prepared", "arrays_stock",
                   "arrays_prepared", "ase_upstream", "arrays_upstream", "ase_product", "arrays_product",
                   *R0_VARIANTS],
                   default=["ase", "certificate", "ase_prepared", "arrays_stock", "arrays_prepared"])
    p.set_defaults(frame=0, seed=17, temperature=300., dtype="float64")
    a = p.parse_args()
    if (len(set(a.routes)) != len(a.routes)
            or min(a.steps, a.repeats, a.dt_fs, a.skin) <= 0 or a.warmup < 0
            or a.output.resolve() in {q.resolve() for q in (a.symmetrix_json, a.model, a.structure)}):
        p.error("require unique routes, positive timed dimensions and a separate report path")
    from symmetrix import load_backend, selected_backend
    load_backend("cuda12-sm70")
    from symmetrix import Symmetrix, calculator
    from symmetrix.neighbor_graph import NeighborReuseDecision
    from mace.calculators import MACECalculator
    if package_version("symmetrix-xl") != "0.2.0" or selected_backend().get("selector") != "cuda12-sm70":
        raise RuntimeError("benchmark pinned to cuda12-sm70 in Symmetrix-XL 0.2.0")
    os.environ["SYMMETRIX_NEIGHBOR_BACKEND"] = "kokkos"
    atoms, velocities = starting_structure(a)
    kw = dict(dtype="float64", execution_profile="speed", streamed_edges="direct", neighbor_skin=a.skin,
              kernel_launch_policy=a.kernel_launch_policy,
              dispersion=False)
    engine = Symmetrix(str(a.symmetrix_json), **kw)
    def product_variant(name):
        if name.endswith("upstream") or name in R0_VARIANTS:
            return "upstream"
        if name.endswith("product"):
            return "horner"
        return None

    product_engine, r0_engine, product_info = None, None, {}
    compact_sha256 = file_hash(a.symmetrix_json)
    if any(product_variant(x) for x in a.routes):
        if a.product_dir is None:
            p.error("supply --product-dir when timing the native M1 plugin")
        from mace_md_lab.xl_product import install_xl_product, install_xl_r0
        if any(product_variant(name) and name not in R0_VARIANTS for name in a.routes):
            product_engine = Symmetrix(str(a.symmetrix_json), **kw)
        if any(name in R0_VARIANTS for name in a.routes):
            # The pristine stock reference trajectory is always recorded first.
            # When no stock route is timed, reuse its evaluator exclusively for R0.
            needs_stock = any(product_variant(name) is None for name in a.routes)
            r0_engine = Symmetrix(str(a.symmetrix_json), **kw) if needs_stock else engine
    original = calculator.neighbor_topology_reuse
    certificate = fixed_pair_bound(original, NeighborReuseDecision)

    @contextmanager
    def path(name, structure):
        with ExitStack() as stack:
            variant = product_variant(name)
            if name in R0_VARIANTS:
                calc = r0_engine
            else:
                calc = product_engine if variant else engine
            if variant:
                product_info[variant] = install_xl_product(calc, a.product_dir, variant=variant)
                if compact_sha256 != product_info[variant]["compact_sha256"]:
                    raise ValueError("compiled product belongs to a different compact model")
            if name in R0_VARIANTS:
                r0_variant = R0_VARIANTS[name]
                info = install_xl_r0(calc, a.product_dir, variant=r0_variant)
                if info["compact_sha256"] != compact_sha256:
                    raise ValueError("compiled R0 belongs to a different compact model")
                product_info["r0_" + r0_variant] = info
            calc.reset()
            calc._neighbor_cache = None
            if name == "certificate":
                stack.enter_context(patch.object(calculator, "neighbor_topology_reuse", certificate))
            if name in ("ase", "certificate", "ase_product", "ase_upstream", "ase_upstream_r0", "ase_factored_r0"):
                route = ASEPath(structure, calc, a.dt_fs)
                builds = lambda: calc.neighbor_cache_build_count
            elif name == "arrays_stock":
                current = structure.copy()
                def evaluate(position):
                    current.positions[:] = position
                    calc.calculate(current, properties=["forces", "stress"])
                    r = calc.results
                    return r["energy"], r["forces"], r["stress"]
                route = ArrayVerlet(structure, evaluate, a.dt_fs)
                builds = lambda: calc.neighbor_cache_build_count
            else:
                prepared = PreparedXL(structure, calc, a.skin)
                if name == "ase_prepared":
                    route = ASEPath(structure, PreparedCalculator(structure, prepared), a.dt_fs)
                else:
                    route = ArrayVerlet(structure, prepared.evaluate, a.dt_fs)
                builds = lambda: prepared.rebuilds
            yield route, builds

    def snap(route):
        if isinstance(route, ASEPath):
            if "stress" not in route.atoms.calc.results:
                raise ValueError("missing every-step stress")
            return (*route.snapshot(), route.atoms.get_stress())
        return route.snapshot()

    def trajectory(name):
        with path(name, atoms) as (route, _):
            frames = [snap(route)]
            for _ in range(a.steps + a.warmup):
                route.advance(1)
                frames.append(snap(route))
        if not all(np.all(np.isfinite(x)) for frame in frames for x in frame):
            raise ValueError("nonfinite output on " + name)
        return frames

    gates = {**numerical_gates(a), "stress_eV_A3": 1e-7}
    reference = trajectory("ase")
    if np.array_equal(reference[0][0], reference[-1][0]):
        raise ValueError("stationary trajectory")
    selected = [0, len(reference)//2, len(reference)-1]
    mace = MACECalculator(model_paths=[str(a.model)], device="cuda", default_dtype="float64")
    stock_values = []
    for i in selected:
        fixed = atoms.copy()
        fixed.positions[:] = reference[i][0]
        fixed.calc = mace
        stock_values.append((fixed.get_potential_energy(), fixed.get_forces(), fixed.get_stress()))
    results = {}
    for name in a.routes:
        frames = trajectory(name)
        comparisons = []
        for i, stock in zip(selected, stock_values):
            fixed = atoms.copy()
            fixed.positions[:] = reference[i][0]
            with path(name, fixed) as (route, _):
                sample = snap(route)
            comparisons.append((sample, reference[i], stock))
        errors = {
            "energy_eV_atom": max(abs(x[2]-m[0])/len(atoms) for x, _, m in comparisons),
            "force_eV_A": max(abs(x[3]-m[1]).max() for x, _, m in comparisons),
            "stress_eV_A3": max(abs(x[5]-m[2]).max() for x, _, m in comparisons),
            "position_A": max(abs(x[0]-s[0]).max() for x, s in zip(frames,reference)),
            "velocity_A_fs": max(abs(x[1]-s[1]).max() for x, s in zip(frames,reference)),
            "nve_difference_eV_atom": max(abs((x[2]+x[4]-frames[0][2]-frames[0][4])
                                                -(s[2]+s[4]-reference[0][2]-reference[0][4]))/len(atoms)
                                               for x,s in zip(frames,reference)),
        }
        xl_errors = {"force_eV_A": max(abs(x[3]-s[3]).max() for x,s,_ in comparisons),
                     "energy_eV_atom": max(abs(x[2]-s[2])/len(atoms) for x,s,_ in comparisons),
                     "stress_eV_A3": max(abs(x[5]-s[5]).max() for x,s,_ in comparisons)}
        passed = all(np.isfinite(v) and v <= gates[k] for k,v in errors.items())
        passed = passed and all(np.isfinite(v) and v <= gates[k] for k,v in xl_errors.items())
        results[name] = dict(errors=errors, common_frames_vs_xl=xl_errors, passed=bool(passed),
                             trials_ms=[], timed_rebuilds=[])
        if name in R0_VARIANTS:
            from benchmark_xl_product import profile_native_steps
            with path(name, atoms) as (route, _):
                route.advance(a.warmup)
                torch.cuda.synchronize()
                trace = profile_native_steps(route)
            kernels = ("symmetrix_r0_forward", "symmetrix_r0_coordinate_reverse",
                       "symmetrix_m1_forward", "symmetrix_m1_reverse")
            active = all(trace["calls_in_two_steps"][kernel] >= 2 for kernel in kernels)
            results[name].update(trace=trace, modules_executed=active,
                                 passed=bool(passed and active))
        print("checked",name,bool(passed),flush=True)
    orders = []
    if all(r["passed"] for r in results.values()):
        for trial in range(a.repeats):
            shift = (trial//2) % len(a.routes)
            order = a.routes[shift:] + a.routes[:shift]
            if trial % 2:
                order.reverse()
            orders.append(order)
            for name in order:
                with path(name, atoms) as (route, builds):
                    route.advance(a.warmup)
                    torch.cuda.synchronize()
                    start_count = builds()
                    t = perf_counter()
                    route.advance(a.steps)
                    torch.cuda.synchronize()
                    elapsed = (perf_counter()-t)*1000/a.steps
                    results[name]["trials_ms"].append(elapsed)
                    results[name]["timed_rebuilds"].append(builds()-start_count)
        for r in results.values():
            r["median_ms_step"] = median(r["trials_ms"])
    passed = all(r["passed"] for r in results.values())
    report = dict(passed=passed, gates=gates, results=results, trial_orders=orders,
                  model_sha256=file_hash(a.model), compact_sha256=file_hash(a.symmetrix_json),
                  structure_sha256=file_hash(a.structure), fixture=a.structure.name,
                  atoms=len(atoms), velocities=velocities, period="fixed-cell fully periodic",
                  dt_fs=a.dt_fs, warmup=a.warmup, steps=a.steps, repeats=a.repeats, skin_A=a.skin,
                  selected_plan=engine.execution_plan["selected_id"], gpu=torch.cuda.get_device_name(),
                  symmetrix=package_version("symmetrix-xl"), selector="cuda12-sm70",
                  native_neighbor_builder="kokkos", dtype="float64", observables="E/F/stress on host every step",
                  kernel_launch_policy=a.kernel_launch_policy, launch_tuning_status=engine.kernel_launch_tuning_status,
                  numerical_scope="Common frames against original MACE and XL; all moving frames against stock XL",
                  product_info=product_info)
    write_report(a.output, report)
    print("MEDIANS",{k:v.get("median_ms_step") for k,v in results.items()}, flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
