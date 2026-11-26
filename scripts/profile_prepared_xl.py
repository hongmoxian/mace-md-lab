"""Measure CUPTI activity within complete prepared-XL MD steps, including native R1."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch
from torch.profiler import profile, ProfilerActivity, record_function

from mace_md_lab.bench import file_hash, package_version, starting_structure, write_report
from mace_md_lab.xl_fixed_step import ArrayVerlet, PreparedXL
from mace_md_lab.xl_product import install_xl_product, install_xl_r0
from profile_xl_cuda import summarize_trace


def kernel_family(event):
    name = event["name"]
    if event["cat"] != "kernel":
        return event["cat"]
    if name.startswith("symmetrix_"):
        return name
    match = re.search(r"MACEKokkos<[^>]+>::(\w+)", name)
    if match:
        return match.group(1)
    if "build_periodic_neighbor_graph_kokkos" in name:
        return "build_periodic_neighbor_graph_kokkos"
    if "gemm" in name.lower() or "gemv" in name.lower():
        return "BLAS"
    if "spherical_harmonics" in name:
        return "spherical_harmonics"
    return "other"


def detailed_activities(trace, steps):
    result = summarize_trace(trace, steps, classify=kernel_family)
    result.pop("top_kernel_duration_sum_ms_step")
    records = [e for e in trace["traceEvents"] if e.get("ph") == "X" and e.get("dur", 0) > 0]
    bounds = sorted((e["ts"], e["ts"] + e["dur"]) for e in records
                    if e.get("cat") == "user_annotation" and e["name"].startswith("xl-md-step-"))
    totals, calls = Counter(), Counter()
    for event in records:
        category = event.get("cat")
        if category not in {"kernel", "gpu_memcpy", "gpu_memset", "cuda_runtime", "cuda_driver"}:
            continue
        begin, end = event["ts"], event["ts"] + event["dur"]
        for left, right in bounds:
            overlap = min(end, right) - max(begin, left)
            if overlap > 0:
                key = (category, event["name"])
                totals[key] += overlap / (1000 * steps)
                calls[key] += 1
    result["all_named_activity"] = [
        {"category": key[0], "name": key[1], "calls": calls[key], "ms_per_step": value}
        for key, value in totals.most_common()
    ]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("structure", "symmetrix-json", "module-dir", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--trials", type=int, default=3)
    parser.set_defaults(frame=0, seed=17, temperature=300.0)
    args = parser.parse_args()
    if min(args.steps, args.warmup, args.trials) < 1 or args.output.resolve() in {
            args.structure.resolve(), args.symmetrix_json.resolve()}:
        parser.error("supply positive run lengths and a separate output")
    from symmetrix import load_backend, selected_backend
    load_backend("cuda12-sm70")
    from symmetrix import Symmetrix
    if selected_backend().get("selector") != "cuda12-sm70" or package_version("symmetrix-xl") != "0.2.0":
        raise RuntimeError("profile is qualified for the SM70 build of Symmetrix-XL 0.2.0")
    os.environ["SYMMETRIX_NEIGHBOR_BACKEND"] = "kokkos"
    atoms, velocity_source = starting_structure(args)
    engine = Symmetrix(str(args.symmetrix_json), dtype="float64", execution_profile="speed",
                       streamed_edges="direct", neighbor_skin=1.0, dispersion=False,
                       kernel_launch_policy="static")
    compact_hash = file_hash(args.symmetrix_json)
    m1 = install_xl_product(engine, args.module_dir, variant="upstream")
    routes = {}

    def warm():
        engine.reset()
        engine._neighbor_cache = None
        prepared = PreparedXL(atoms, engine, 1.0)
        path = ArrayVerlet(atoms, prepared.evaluate, 0.5)
        path.advance(args.warmup)
        torch.cuda.synchronize()
        return path, prepared

    for variant in ("upstream", "factored"):
        r0 = install_xl_r0(engine, args.module_dir, variant=variant)
        if any(artifact["compact_sha256"] != compact_hash for artifact in (m1, r0)):
            raise ValueError("operator module belongs to different compact model")
        untraced = []
        for _ in range(args.trials):
            path, _ = warm()
            start = perf_counter()
            path.advance(args.steps)
            torch.cuda.synchronize()
            untraced.append((perf_counter() - start) * 1000 / args.steps)
        path, prepared = warm()
        pos0 = path.positions.copy()
        builds = []
        with TemporaryDirectory() as directory:
            filename = Path(directory) / "trace.json"
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as recorder:
                for i in range(args.steps):
                    before = prepared.rebuilds
                    with record_function(f"xl-md-step-{i}"):
                        path.advance(1)
                        torch.cuda.synchronize()
                    builds.append(prepared.rebuilds - before)
            recorder.export_chrome_trace(str(filename))
            stats = detailed_activities(json.loads(filename.read_text()), args.steps)
        wanted = ("symmetrix_r0_forward", "symmetrix_r0_coordinate_reverse",
                  "symmetrix_m1_forward", "symmetrix_m1_reverse",
                  "symmetrix_factorized_forward_v2", "symmetrix_factorized_reverse_fused_v2")
        observed = stats["all_named_activity"]
        native_calls = {k: sum(row["calls"] for row in observed if row["category"] == "kernel"
                               and k in row["name"]) for k in wanted}
        print("native dispatch", native_calls, "families", stats["mean_group_ms"], flush=True)
        if any(native_calls[k] < args.steps for k in wanted):
            raise ValueError("trace lacks R0, R1 or M1 dispatches")
        if (not all(np.isfinite(x).all() for x in path.snapshot())
                or np.array_equal(pos0, path.positions)):
            raise ValueError("trajectory did not move or contains nonfinite observables")
        routes[variant] = dict(r0=r0, m1=m1, native_dispatch=native_calls, activity=stats, per_step_graph_builds=builds,
                               uninstrumented_ms_step=untraced, median_uninstrumented_ms_step=median(untraced),
                               max_traced_displacement_A=float(np.max(abs(path.positions-pos0))))
        print(variant, sorted(stats["mean_group_ms"].items(), key=lambda item: -item[1])[:16], flush=True)
    report = dict(routes=routes, steps=args.steps, warmup=args.warmup, untraced_trials=args.trials,
                  atoms=len(atoms), velocity_source=velocity_source, skin_A=1.0, dt_fs=0.5, dtype="float64",
                  fixture_sha256=file_hash(args.structure), compact_sha256=compact_hash,
                  gpu=torch.cuda.get_device_name(), torch=str(torch.__version__), cuda=torch.version.cuda,
                  xl_version=package_version("symmetrix-xl"), execution_plan=engine.execution_plan,
                  kernel_launch_policy="static", observables="host E, forces and full stress per step",
                  scope="Fully periodic fixed-cell ArrayVerlet and PreparedXL; elapsed GPU union and event-name "
                  "sums are reported separately. CUDA API durations overlap device work. Per-step traced "
                  "wall time includes CUPTI overhead. Compare actual performance using paired untraced benchmarks.")
    write_report(args.output, report)


if __name__ == "__main__":
    main()
