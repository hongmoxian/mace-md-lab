"""Compare native force algorithms on the same prepared, FP64 Symmetrix-XL driver.

The two built-in reference routes are upstream and factored. A trusted JSON
manifest can add a route with base "r0" ("upstream" or "factored"), a list
"install" containing {"module": "mace_md_lab.xl_...", "config": {...}}, and
optional additional "kernels" expected in a trace. An installer has signature
install_algorithm(calc, compact_path, config) and returns artifact metadata.
All paths in the manifest are interpreted from the working directory. For a
locked comparison, set --require-speedup to the candidate name and use eight
interleaved repeats. Use --m0-dir for the tuned Cu reference. Add --trace for
a separate, untimed witness of native dispatch. On CUDA clusters use a
node-local SYMMETRIX_JIT_CACHE, e.g. export SYMMETRIX_JIT_CACHE="$(mktemp -d)".
"""
import argparse
from collections import Counter
from importlib import import_module
import json
import math
import os
from pathlib import Path
import re
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch

from mace_md_lab.bench import file_hash, numerical_gates, package_version, starting_structure, write_report
from mace_md_lab.force_checks import force_directions
from mace_md_lab.xl_fixed_step import ArrayVerlet, PreparedXL
from mace_md_lab.xl_product import install_xl_product, install_xl_r0


def read_routes(path):
    routes = {"upstream": {"r0": "upstream"}, "factored": {"r0": "factored"}}
    extra = json.loads(path.read_text()) if path else {}
    for name, route in extra.items():
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in routes:
            raise ValueError(f"invalid or reserved route name {name}")
        if set(route) - {"r0", "install", "kernels"}:
            raise ValueError(f"unknown route field in {name}")
        if route.get("r0", "upstream") not in ("upstream", "factored"):
            raise ValueError(f"unknown base R0 in {name}")
        if not route.get("install"):
            raise ValueError(f"no algorithm selected for {name}")
        routes[name] = route
    return routes


def setup_engine(args, spec, digest, species):
    from symmetrix import Symmetrix

    calc = Symmetrix(str(args.symmetrix_json), dtype="float64", execution_profile="speed",
                     streamed_edges="direct", neighbor_skin=args.skin, kernel_launch_policy="static",
                     dispersion=False)
    modules = {"m1": install_xl_product(calc, args.product_dir, variant="upstream"),
               "r0": install_xl_r0(calc, args.product_dir, variant=spec.get("r0", "upstream"))}
    if args.m0_dir:
        modules["m0"] = install_xl_product(calc, args.m0_dir, variant="upstream", stage="m0")
    if any(info["compact_sha256"] != digest for info in modules.values()):
        raise ValueError("generated reference module and compact checkpoint hashes differ")
    for specifier in spec.get("install", []):
        name = specifier["module"]
        if not re.fullmatch(r"mace_md_lab\.xl_[A-Za-z_0-9]+", name) or name in modules:
            raise ValueError(f"invalid or duplicate native installer {name}")
        config = specifier["config"]
        if "atomic_numbers" in config and sorted(config["atomic_numbers"]) != species:
            raise ValueError(f"{name} was qualified for a different set of atomic numbers")
        meta = import_module(name).install_algorithm(calc, args.symmetrix_json, config)
        artifacts = meta if isinstance(meta, list) else [meta]
        if not artifacts or any(not isinstance(info, dict) or not info.get("cubin_sha256")
                                or info.get("compact_sha256") != digest for info in artifacts):
            raise ValueError(f"{name} did not attest the compiled module for this checkpoint")
        modules[name] = meta
    if args.harmonic_storage is not None:
        calc.evaluator._set_harmonic_storage_policy(args.harmonic_storage, True)
        if calc.evaluator.harmonic_storage_policy != args.harmonic_storage:
            raise ValueError(f"native evaluator rejected {args.harmonic_storage}")
    if calc.execution_plan["selected_id"] != "mh0-direct-speed":
        raise ValueError(f"unexpected native plan {calc.execution_plan}")
    return calc, modules


def new_path(atoms, engine, args):
    engine.reset()
    engine._neighbor_cache = None
    prepared = PreparedXL(atoms, engine, args.skin)
    return ArrayVerlet(atoms, prepared.evaluate, args.dt_fs), prepared


def gather(atoms, engine, args):
    path, _ = new_path(atoms, engine, args)
    frames = [path.snapshot()]
    for _ in range(args.warmup + args.steps):
        path.advance(1)
        frames.append(path.snapshot())
    if any(not np.isfinite(value).all() for frame in frames for value in frame):
        raise FloatingPointError("nonfinite MD observable")
    return frames


def efs_errors(values, reference, n):
    return {"energy_eV_atom": float(abs(values[0]-reference[0])/n),
            "force_eV_A": float(np.max(np.abs(values[1]-reference[1]))),
            "stress_eV_A3": float(np.max(np.abs(values[2]-reference[2])))}


def frame_errors(atoms, engine, args, base_frames, mace_values):
    frames = gather(atoms, engine, args)
    n = len(atoms)
    ref0 = base_frames[0][2] + base_frames[0][4]
    e0 = frames[0][2] + frames[0][4]
    moving = {
        "position_A": max(float(np.abs(a[0]-b[0]).max()) for a,b in zip(frames,base_frames)),
        "velocity_A_fs": max(float(np.abs(a[1]-b[1]).max()) for a,b in zip(frames,base_frames)),
        "nve_difference_eV_atom": max(abs((a[2]+a[4]-e0)-(b[2]+b[4]-ref0))/n
                                      for a,b in zip(frames,base_frames)),
    }
    # Evaluate forces again at the *same* positions even when trajectories diverge.
    common = PreparedXL(atoms, engine, args.skin)
    xl_errors, mace_errors = [], []
    for step, frame in enumerate(base_frames):
        values = common.evaluate(frame[0])
        xl_errors.append(efs_errors(values, (frame[2], frame[3], frame[5]), n))
        if step in mace_values:
            mace_errors.append(efs_errors(values, mace_values[step], n))
    return moving, {k:max(x[k] for x in xl_errors) for k in xl_errors[0]}, {
        k:max(x[k] for x in mace_errors) for k in mace_errors[0]}


def finite_differences(atoms, engine, args):
    def efs(at):
        return PreparedXL(at, engine, args.skin).evaluate(at.positions)

    _, forces, stress = efs(atoms)
    force_checks, strain_checks = {}, {}
    for label, direction in force_directions(atoms, 17):
        projection = float((forces*direction).sum())
        energies = []
        h = 1e-4
        for sign in (-1, 1):
            at = atoms.copy()
            at.positions[:] += sign*h*direction
            energies.append(efs(at)[0])
        force_checks[label] = abs((energies[0]-energies[1])/(2*h)-projection)
    for label, index, i, j in (("xx", 0, 0, 0), ("xy", 5, 0, 1)):
        basis = np.zeros((3,3))
        basis[i,j] += 0.5
        basis[j,i] += 0.5
        energies = []
        h = 1e-5
        for sign in (-1, 1):
            at = atoms.copy()
            at.set_cell(atoms.cell.array @ (np.eye(3) + sign*h*basis), scale_atoms=True)
            energies.append(efs(at)[0])
        strain_checks[label] = abs((energies[1]-energies[0])/(2*h*atoms.get_volume())-stress[index])
    return {"displacement_A": 1e-4, "strain": 1e-5, "force_error_eV_A": force_checks,
            "stress_error_eV_A3": strain_checks, "passed": bool(
                max(force_checks.values()) <= 2e-5 and max(strain_checks.values()) <= 1e-7)}


def witness(path, expected):
    from torch.profiler import ProfilerActivity, profile

    with TemporaryDirectory() as temporary:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as trace:
            path.advance(2)
            torch.cuda.synchronize()
        target = Path(temporary) / "trace.json"
        trace.export_chrome_trace(str(target))
        events = json.loads(target.read_text())["traceEvents"]
    calls, times = Counter(), Counter()
    for event in events:
        if event.get("cat") == "kernel" and event.get("ph") == "X":
            for symbol in expected:
                if symbol in event["name"]:
                    calls[symbol] += 1
                    times[symbol] += event["dur"]/2000.
    return {"steps": 2, "counts": dict(calls), "ms_per_step": dict(times),
            "passed": all(calls[k]>=2 for k in expected)}


def score(candidate, reference):
    ratios = [x/y for x,y in zip(candidate, reference)]
    ratio = median(candidate)/median(reference)
    wins = sum(x<1. for x in ratios)
    needed = math.ceil(7*len(ratios)/8)
    return {"ratio_of_medians": ratio, "time_reduction_pct": 100*(1.-ratio),
            "paired_ratios": ratios, "wins": wins, "wins_required": needed,
            "trials": len(ratios), "meets_10pct_and_win_rule": bool(ratio<=0.90 and wins>=needed)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("model", "symmetrix-json", "structure", "product-dir", "output"):
        p.add_argument("--"+key, type=Path, required=True)
    p.add_argument("--harmonic-storage", choices=("y-only-direct-v1",),
                   help="set the same explicit native harmonic storage policy in every engine; "
                        "omit to keep the upstream native default")
    p.add_argument("--m0-dir", type=Path, help="also install the official generated M0 in every engine")
    p.add_argument("--routes-json", type=Path, help="trusted additional native algorithm route manifest")
    p.add_argument("--steps", type=int, default=48)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeats", type=int, default=8)
    p.add_argument("--require-speedup", metavar="ROUTE",
                   help="exit with failure if the named route fails the declared relative speed target")
    p.add_argument("--trace", action="store_true", help="verify native kernel execution in an untimed two-step trace")
    p.set_defaults(frame=0, seed=17, temperature=300., dtype="float64", dt_fs=0.5, skin=1.)
    a = p.parse_args()
    if min(a.steps, a.repeats, a.warmup)<1 or a.output.resolve() in {
            q.resolve() for q in (a.model, a.symmetrix_json, a.structure, a.routes_json) if q is not None}:
        p.error("use positive steps/warmup/repeats and a distinct report path")
    specs = read_routes(a.routes_json)
    if a.require_speedup and a.repeats != 8:
        p.error("locked speed acceptance requires eight paired repeats; omit --require-speedup for development")
    if a.require_speedup and (a.require_speedup not in specs or a.require_speedup == "upstream"):
        p.error("--require-speedup must name an included non-reference route")
    from symmetrix import load_backend, selected_backend
    from mace.calculators import MACECalculator

    os.environ["SYMMETRIX_NEIGHBOR_BACKEND"] = "kokkos"
    load_backend("cuda12-sm70")
    if selected_backend().get("selector")!="cuda12-sm70" or package_version("symmetrix-xl")!="0.2.0":
        raise RuntimeError("requires Symmetrix-XL 0.2.0 cuda12-sm70")
    atoms, velocities = starting_structure(a)
    digest = file_hash(a.symmetrix_json)
    species = sorted(set(map(int, atoms.numbers)))
    engines, modules = {}, {}
    engines["upstream"], modules["upstream"] = setup_engine(a, specs["upstream"], digest, species)
    base = gather(atoms, engines["upstream"], a)
    if np.array_equal(base[0][0],base[-1][0]):
        raise RuntimeError("stationary reference trajectory")
    mace = MACECalculator(model_paths=[str(a.model)], device="cuda", default_dtype="float64")
    mace_values = {}
    for step in sorted({0,len(base)//2,len(base)-1}):
        at = atoms.copy()
        at.positions[:] = base[step][0]
        at.calc = mace
        mace_values[step] = (at.get_potential_energy(),at.get_forces(),at.get_stress())
    del at, mace
    torch.cuda.empty_cache()
    for name, spec in specs.items():
        if name != "upstream":
            engines[name], modules[name] = setup_engine(a, spec, digest, species)
    mace_gates = {**numerical_gates(a),"stress_eV_A3":1e-7}
    xl_gates = {"energy_eV_atom":1e-10, "force_eV_A":1e-9, "stress_eV_A3":1e-10}
    results, order = {}, []
    for name, engine in engines.items():
        moving, xl, original = frame_errors(atoms, engine, a, base, mace_values)
        differences = finite_differences(atoms, engine, a)
        ok = all(v<=mace_gates[k] for k,v in {**original,**moving}.items())
        ok = ok and all(v<=xl_gates[k] for k,v in xl.items()) and differences["passed"]
        entry = {"trajectory_vs_XL":moving, "EFS_vs_XL_all_frames":xl,
                 "EFS_vs_original_MACE_three_frames":original, "finite_differences":differences,
                 "trials_ms":[], "timed_rebuilds":[]}
        if a.trace:
            path, _ = new_path(atoms, engine, a)
            path.advance(a.warmup)
            torch.cuda.synchronize()
            symbols = ["symmetrix_r0_forward","symmetrix_r0_coordinate_reverse",
                       "symmetrix_m1_forward","symmetrix_m1_reverse",*specs[name].get("kernels",[])]
            if a.m0_dir:
                symbols.extend(["symmetrix_m0_forward","symmetrix_m0_reverse"])
            entry["trace"] = witness(path, sorted(set(symbols)))
            ok = ok and entry["trace"]["passed"]
        results[name] = {**entry,"passed":ok}
        print("validated",name,ok,flush=True)

    passed = all(r["passed"] for r in results.values())
    if passed:
        names = list(engines)
        for trial in range(a.repeats):
            shift = trial//2 % len(names)
            seq = names[shift:]+names[:shift]
            if trial%2:
                seq.reverse()
            order.append(seq)
            for name in seq:
                path, prepared = new_path(atoms, engines[name], a)
                path.advance(a.warmup)
                torch.cuda.synchronize()
                before = prepared.rebuilds
                t0 = perf_counter()
                path.advance(a.steps)
                torch.cuda.synchronize()
                elapsed = (perf_counter()-t0)*1000/a.steps
                results[name]["trials_ms"].append(elapsed)
                results[name]["timed_rebuilds"].append(prepared.rebuilds-before)
        for entry in results.values():
            equal_builds = entry["timed_rebuilds"] == results["upstream"]["timed_rebuilds"]
            entry["same_timed_rebuilds"] = equal_builds
            entry["passed"] = entry["passed"] and equal_builds
            entry["median_ms_step"] = median(entry["trials_ms"])
            entry["vs_upstream"] = score(entry["trials_ms"],results["upstream"]["trials_ms"])
            entry["vs_factored"] = score(entry["trials_ms"],results["factored"]["trials_ms"])
        passed = all(r["passed"] for r in results.values())
    policies = {name: engine.evaluator.harmonic_storage_policy for name, engine in engines.items()}
    policy_matched = (len(set(policies.values())) == 1
                      and (a.harmonic_storage is None or policies["upstream"] == a.harmonic_storage))
    passed = passed and policy_matched
    report = {"passed":passed,"harmonic_storage_requested":a.harmonic_storage,
              "native_harmonic_storage_by_route":policies,"same_harmonic_storage_policy":policy_matched, "performance_rule":{"reference":"upstream","max_ratio":0.90,
              "paired_win_fraction":0.875,"estimator":"ratio of trial medians"},
              "results":results, "orders":order, "routes":specs, "modules":modules,
              "route_manifest_sha256":file_hash(a.routes_json) if a.routes_json else None,
              "compact_sha256":digest,"model_sha256":file_hash(a.model),"structure_sha256":file_hash(a.structure),
              "driver":"PreparedXL + ArrayVerlet","dtype":"float64","dt_fs":a.dt_fs,"skin_A":a.skin,
              "symmetrix":package_version("symmetrix-xl"),"mace":package_version("mace-torch"),
              "gpu":torch.cuda.get_device_name(),"launch":"static","plans":{n:e.execution_plan["selected_id"] for n,e in engines.items()},
              "neighbor_backend":"kokkos","observables":"E/F/stress returned to host every step",
              "atoms":len(atoms),"steps":a.steps,"warmup":a.warmup,"repeats":a.repeats,
              "velocities":velocities,"gates_vs_MACE":mace_gates,"gates_vs_XL":xl_gates,
              "generated_m0":bool(a.m0_dir),"trace_outside_timing":a.trace}
    if a.require_speedup:
        report["required_route"] = a.require_speedup
        report["accepted"] = passed and results[a.require_speedup]["vs_upstream"]["meets_10pct_and_win_rule"]
    write_report(a.output,report)
    print({n:r.get("median_ms_step") for n,r in results.items()},flush=True)
    if not passed:
        raise SystemExit("accuracy, native dispatch, rebuild-count or harmonic-storage gate failed")
    if a.require_speedup and not report["accepted"]:
        raise SystemExit("the required route missed the predefined speed target")


if __name__ == "__main__":
    main()
