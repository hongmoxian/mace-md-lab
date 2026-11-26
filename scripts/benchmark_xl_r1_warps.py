"""Development ablation of the native scalar-source R1 in a prepared XL driver."""
import argparse
import json
import os
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch
from ase.io import read

from mace_md_lab.xl_fixed_step import ArrayVerlet, PreparedXL
from mace_md_lab.xl_product import install_xl_product, install_xl_r0
from mace_md_lab.xl_r1_algorithm import WARPS, digest, install_algorithm


def trace_four_steps(path):
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities) as profiler:
        path.advance(4)
        torch.cuda.synchronize()
    with TemporaryDirectory() as tmp:
        target = Path(tmp) / "trace.json"
        profiler.export_chrome_trace(str(target))
        events = json.loads(target.read_text())["traceEvents"]

    stats = {}
    for stage in ("forward_v2", "reverse_fused_v2"):
        name = "symmetrix_factorized_" + stage
        kernels = [event for event in events if event.get("cat") == "kernel" and name in event["name"]]
        if len(kernels) != 4:
            raise RuntimeError(f"Expected four actual R1 kernel dispatches for {name}")
        stats[stage + "_ms"] = sum(event["dur"] for event in kernels) * 1e-3 / 4
        stats[stage + "_calls"] = len(kernels)
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact", type=Path, required=True)
    parser.add_argument("--structure", type=Path, required=True)
    parser.add_argument("--r0-directory", type=Path, required=True)
    parser.add_argument("--r1-directory", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", choices=["stock", "upstream", *WARPS], required=True)
    parser.add_argument("--y-only", action="store_true", help="use the upstream on-demand harmonic gradient policy for every route")
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    variants = args.variants
    if variants[0] != "stock" or "upstream" not in variants or len(variants) != len(set(variants)):
        parser.error("include stock first and a compiler-matched upstream NVCC control")
    if min(args.steps, args.repeats) < 1 or args.output.resolve() in {args.compact.resolve(), args.structure.resolve()}:
        parser.error("use positive test sizes and an output distinct from all inputs")

    from symmetrix import load_backend

    load_backend("cuda12-sm70")
    from symmetrix import Symmetrix

    os.environ["SYMMETRIX_NEIGHBOR_BACKEND"] = "kokkos"
    atoms = read(args.structure)
    engines = {}
    results = {v: dict(artifact={}, trial_ms=[], rebuilds=[]) for v in variants}
    for key in ("stock", "modified"):
        engine = Symmetrix(
            str(args.compact), dtype="float64", execution_profile="speed", streamed_edges="direct",
            neighbor_skin=1.0, kernel_launch_policy="static", dispersion=False,
        )
        install_xl_product(engine, args.r0_directory, variant="upstream")
        install_xl_r0(engine, args.r0_directory, variant="upstream")
        engines[key] = engine
    active = None

    def new_path(name):
        nonlocal active
        engine = engines["stock" if name == "stock" else "modified"]
        if name != "stock" and name != active:
            config = dict(directory=str(args.r1_directory), variant=name)
            results[name]["artifact"] = install_algorithm(engine, args.compact, config)
            active = name
        if args.y_only:
            engine.evaluator._set_harmonic_storage_policy("y-only-direct-v1", True)
        geometry = PreparedXL(atoms, engine, 1.0)
        return ArrayVerlet(atoms, geometry.evaluate, 0.5), geometry

    warmup = 5
    ref, _ = new_path("stock")
    reference = [ref.snapshot()]
    for _ in range(args.steps + warmup):
        ref.advance(1)
        reference.append(ref.snapshot())

    gates = {"position_A": 1e-7, "velocity_A_fs": 1e-8, "energy_eV": 1e-8,
             "force_eV_A": 1e-8, "kinetic_eV": 1e-7, "stress_eV_A3": 1e-8}
    for name in variants:
        path, _ = new_path(name)
        maxima = np.zeros(len(gates))
        for step, frame in enumerate(reference):
            if step:
                path.advance(1)
            deltas = [np.max(np.abs(np.asarray(x) - y)) for x, y in zip(path.snapshot(), frame)]
            maxima = np.maximum(maxima, deltas)
        row = results[name]
        row["moving_maxima"] = dict(zip(gates, maxima.tolist()))
        row["passed"] = bool(np.all(maxima < list(gates.values())))
        row.update(trace_four_steps(path))
        if args.y_only:
            native = engines["stock" if name == "stock" else "modified"].evaluator
            row["direct_harmonic_launches"] = int(native.execution_direct_harmonic_launch_count)
            if row["direct_harmonic_launches"] < 2:
                raise RuntimeError("Missing dispatch for the on-demand harmonic-gradient policy")
        print(name, row["passed"], row["forward_v2_ms"], row["reverse_fused_v2_ms"], flush=True)

    orders = []
    passed = all(row["passed"] for row in results.values())
    if passed:
        for repeat in range(args.repeats):
            offset = (repeat // 2) % len(variants)
            order = variants[offset:] + variants[:offset]
            if repeat % 2:
                order.reverse()
            orders.append(order)
            for name in order:
                path, geometry = new_path(name)
                path.advance(warmup)
                torch.cuda.synchronize()
                initial_rebuilds = geometry.rebuilds
                start = perf_counter()
                path.advance(args.steps)
                torch.cuda.synchronize()
                results[name]["trial_ms"].append(1e3 * (perf_counter() - start) / args.steps)
                results[name]["rebuilds"].append(geometry.rebuilds - initial_rebuilds)
        for row in results.values():
            row["median_ms"] = median(row["trial_ms"])

    report = dict(
        results=results, orders=orders, structure_sha256=digest(args.structure),
        compact_sha256=digest(args.compact), steps=args.steps, warmup=warmup, repeats=args.repeats,
        gpu=torch.cuda.get_device_name(), dtype="float64", prepared_driver=True,
        upstream_generated_r0_m1=True, y_only=args.y_only,
    )
    args.output.parent.mkdir(exist_ok=True, parents=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print({k: v.get("median_ms") for k, v in results.items()}, flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
