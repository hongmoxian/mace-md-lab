"""Attribute CUDA activity of warm, moving stock Symmetrix-XL steps on one GPU."""

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile, record_function

from mace_md_lab.bench import ASEPath, file_hash, package_version, starting_structure, write_report


def activity_group(event):
    name = event["name"].lower()
    if event["cat"] != "kernel":
        return "device_copy_or_set"
    for operator in ("r0", "m0", "m1"):
        if "symmetrix_" + operator + "_" in name:
            direction = "reverse" if "reverse" in name else "forward"
            return f"{operator}_{direction}"
    if "symmetrix_factorized_" in name or any(
        word in name for word in ("r1_forward_kernel", "r1_source_kernel", "r1_edge_kernel", "r1_reverse_fused_kernel")
    ):
        direction = "reverse" if any(word in name for word in ("source", "edge", "reverse")) else "forward"
        return f"r1_{direction}"
    return "other_kernels"


def summarize_trace(trace, step_count, *, prefix="xl-md-step-", classify=activity_group):
    events = trace["traceEvents"]
    markers = {e["name"]: e for e in events if e.get("cat") == "user_annotation"
               and e["name"].startswith(prefix) and e.get("ph") == "X"}
    if len(markers) != step_count:
        raise RuntimeError("CUPTI trace lacks the requested CPU step boundaries")
    device_events = [e for e in events if e.get("cat") in {"kernel", "gpu_memcpy", "gpu_memset"}
                     and e.get("ph") == "X" and e.get("dur", 0) > 0]
    per_step, kernel_names = [], Counter()
    stream_ids = set()
    for step in range(step_count):
        marker = markers[f"{prefix}{step}"]
        start, end = marker["ts"], marker["ts"] + marker["dur"]
        changes = defaultdict(Counter)
        count = Counter()
        for event in device_events:
            left, right = max(start, event["ts"]), min(end, event["ts"] + event["dur"])
            if left >= right:
                continue
            group = classify(event)
            changes[left][group] += 1
            changes[right][group] -= 1
            stream_ids.add(str(event["args"].get("stream")))
            count[group] += 1
            if event["cat"] == "kernel":
                kernel_names[event["name"]] += (right - left) / 1000
        if not count:
            raise RuntimeError("no CUPTI device activity inside a timed step")
        counts, times, previous = Counter(), Counter(), start
        for when, change in sorted(changes.items()):
            running = [k for k, v in counts.items() if v > 0]
            if running:
                group = running[0] if len(running) == 1 else "overlapping_groups"
                times[group] += (when - previous) / 1000
            counts.update(change)
            previous = when
        wall = marker["dur"] / 1000
        busy = sum(times.values())
        per_step.append(dict(wall_ms=wall, gpu_busy_union_ms=busy, no_device_event_ms=wall-busy,
                             group_ms=dict(times), calls=dict(count)))
    labels = sorted({k for s in per_step for k in s["group_ms"]})
    group_means = {k: sum(s["group_ms"].get(k, 0) for s in per_step) / step_count for k in labels}
    return dict(per_step=per_step, mean_group_ms=group_means, observed_streams=sorted(stream_ids),
                mean_gpu_busy_union_ms=sum(s["gpu_busy_union_ms"] for s in per_step) / step_count,
                mean_traced_wall_ms=sum(s["wall_ms"] for s in per_step) / step_count,
                top_kernel_duration_sum_ms_step=[
                    {"kernel": name, "ms": value / step_count} for name, value in kernel_names.most_common(20)
                ])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symmetrix-json", required=True, type=Path)
    p.add_argument("--structure", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--trials", type=int, default=3)
    p.set_defaults(frame=0, seed=17, temperature=300.)
    args = p.parse_args()
    if min(args.steps, args.warmup, args.trials) <= 0:
        p.error("steps, warmup, and trials must be positive")
    if args.output.resolve() in {args.structure.resolve(), args.symmetrix_json.resolve()}:
        p.error("output overlaps an input")

    from symmetrix import load_backend, selected_backend

    load_backend("cuda12-sm70")
    from symmetrix import Symmetrix

    if selected_backend().get("selector") != "cuda12-sm70" or package_version("symmetrix-xl") != "0.2.0":
        raise RuntimeError("this tracer is qualified for Symmetrix-XL 0.2.0 on SM70")
    os.environ["SYMMETRIX_NEIGHBOR_BACKEND"] = "kokkos"
    atoms, velocities = starting_structure(args)
    calc = Symmetrix(str(args.symmetrix_json), dtype="float64", execution_profile="speed",
                     streamed_edges="direct", neighbor_skin=1.0, dispersion=False)

    def warm_path():
        calc.reset()
        calc._neighbor_cache = None
        path = ASEPath(atoms, calc, 0.5)
        path.advance(args.warmup)
        torch.cuda.synchronize()
        return path

    timings = []
    for _ in range(args.trials):
        path = warm_path()
        start = perf_counter()
        path.advance(args.steps)
        torch.cuda.synchronize()
        timings.append(1000*(perf_counter()-start)/args.steps)
    path = warm_path()
    pos0 = path.atoms.positions.copy()
    with TemporaryDirectory() as temporary:
        filename = str(Path(temporary) / "activity.json")
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as trace:
            for step in range(args.steps):
                with record_function(f"xl-md-step-{step}"):
                    path.advance(1)
                    torch.cuda.synchronize()
        trace.export_chrome_trace(filename)
        summary = summarize_trace(json.loads(Path(filename).read_text()), args.steps)
    displacement = float(np.max(np.abs(path.atoms.positions - pos0)))
    if displacement <= 1e-8 or calc.neighbor_graph_backend != "kokkos":
        raise RuntimeError("expected moving atoms and Kokkos neighbors")
    plan = calc.execution_plan
    report = dict(symmetrix=package_version("symmetrix-xl"), torch=str(torch.__version__),
                  cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(), engine=str(plan),
                  r0_diagnostics=calc.r0_diagnostics, steps=args.steps, warmup=args.warmup,
                  dt_fs=0.5, skin_A=1.0, builder=calc.neighbor_graph_backend, atoms=len(atoms),
                  dtype="float64", observables=["energy","forces","stress"], velocity_source=velocities,
                  structure_sha256=file_hash(args.structure), converted_sha256=file_hash(args.symmetrix_json),
                  max_displacement_A=displacement, uninstrumented_ms_step=timings,
                  median_uninstrumented_ms_step=median(timings), activity=summary,
                  scope="CUDA kernel and copy activity partitioned by union within per-step host markers. "
                        "Unrecognized kernels remain other_kernels. Trace times include profiling overhead. "
                        "Individual top-name totals can overlap across streams.")
    write_report(args.output, report)


if __name__ == "__main__":
    main()
