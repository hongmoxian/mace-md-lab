"""Record individual GPU kernels and host API calls of the captured exact Au/water EFS path."""

import argparse
from collections import Counter
from contextlib import ExitStack
import json
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import torch
from mace.calculators import MACECalculator
from torch.profiler import ProfilerActivity, profile, record_function

from mace_md_lab.bench import file_hash, package_version, starting_structure, write_report
from mace_md_lab.candidate_replay import CandidatePath, explicit_cutoff
from mace_md_lab.edge_virial import install_edge_virial, verify_edge_checkpoint
from mace_md_lab.grouped_first import install_grouped_first
from mace_md_lab.grouped_second import install_second_interaction
from mace_md_lab.transforms import optimized_model
from profile_xl_cuda import summarize_trace


PREFIX = "candidate-md-step-"


def kernel_group(event):
    if event["cat"] != "kernel":
        return "device_copy_or_set"
    name = event["name"].lower()
    if any(x in name for x in ("gemm", "gemv")):
        return "dense_linear_algebra"
    if "scalar_product" in name:
        return "frozen_scalar_product"
    if "_group_moments" in name or "_moment_adjoint" in name:
        return "grouped_first"
    if "_reduce_by_receiver" in name or "_message_vjp" in name:
        return "grouped_second"
    if name in ("_forward", "_reverse"):
        return "angular"
    if any(x in name for x in ("scatter", "gather", "index", "radix", "searchsorted", "nonzero")):
        return "indexing_sorting_scatter"
    if "reduce" in name:
        return "other_reductions"
    if "elementwise" in name:
        return "other_elementwise"
    return "other_kernels"


def api_activity(trace, steps):
    events = trace["traceEvents"]
    spans = [(e["ts"], e["ts"] + e["dur"]) for e in events if
             e.get("cat") == "user_annotation" and e.get("ph") == "X" and e["name"].startswith(PREFIX)]
    times, counts = Counter(), Counter()
    kernels = Counter()
    for e in events:
        if e.get("cat") not in ("cuda_runtime", "cuda_driver", "kernel") or e.get("ph") != "X":
            continue
        for start, end in spans:
            length = min(end, e["ts"] + e["dur"]) - max(start, e["ts"])
            if length <= 0:
                continue
            name = e["name"]
            if e["cat"] == "kernel":
                kernels[name] += 1
            else:
                key = (e["cat"], name)
                times[key] += length / 1000
                counts[key] += 1
    return ([dict(api=name, category=cat, calls_per_step=counts[(cat, name)]/steps,
                  ms_step=ms/steps) for (cat, name), ms in times.most_common()],
            {name: count/steps for name, count in kernels.items()})


def runner_counts(path):
    r = path.runner
    return {name: int(getattr(r, name)) for name in ("captures", "replays", "rebuilds", "edge_refreshes")}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "structure", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--skin", type=float, default=0.25)
    p.add_argument("--macro-extension", type=float, default=1.)
    p.set_defaults(frame=0, seed=17, temperature=300.)
    a = p.parse_args()
    if not 8 <= a.steps <= 12 or min(a.warmup, a.trials, a.skin, a.macro_extension) <= 0:
        p.error("use 8–12 steps and positive warmup, trials and Verlet buffers")
    if a.output.resolve() in (a.structure.resolve(), a.model.resolve()):
        p.error("use a separate output path")
    verify_edge_checkpoint(a.model)
    atoms, velocities = starting_structure(a)
    calc = MACECalculator(model_paths=[str(a.model)], device="cuda", default_dtype="float64")
    model = calc.models[0].eval().requires_grad_(False)
    layout, stress = SimpleNamespace(), {}
    with ExitStack() as stack:
        stack.enter_context(optimized_model(model, "both", atoms.numbers))
        stack.enter_context(explicit_cutoff(model, calc.r_max))
        stack.enter_context(install_grouped_first(model, atoms.numbers, fold_linear=True, layout=layout))
        stack.enter_context(install_second_interaction(model, layout=layout))
        stack.enter_context(install_edge_virial(model, atoms.get_volume(), stress))

        def warm():
            path = CandidatePath(atoms, calc, 0.5, a.skin, layout, True, a.macro_extension)
            path.advance(a.warmup)
            torch.cuda.synchronize()
            return path

        def step_host(path):
            path.advance(1)
            return torch.cat((path.energy.reshape(-1), path.forces.reshape(-1),
                              stress["tensor"].reshape(-1))).cpu()

        timings = []
        for _ in range(a.trials):
            path = warm()
            start = perf_counter()
            for _ in range(a.steps):
                step_host(path)
            torch.cuda.synchronize()
            timings.append(1000 * (perf_counter() - start) / a.steps)
        path = warm()
        before = runner_counts(path)
        pos0 = path.runner.positions.cpu().numpy().copy()
        with TemporaryDirectory() as temporary:
            target = Path(temporary) / "trace.json"
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                for i in range(a.steps):
                    with record_function(f"{PREFIX}{i}"):
                        host_observables = step_host(path)
            prof.export_chrome_trace(str(target))
            trace = json.loads(target.read_text())
        after = runner_counts(path)
        delta = {k: after[k] - v for k, v in before.items()}
        edge_audit = path.runner.verify_physical_edges()
        displacement = float(np.max(abs(path.runner.positions.cpu().numpy() - pos0)))
        if delta["replays"] != a.steps or delta["captures"] or displacement <= 1e-8:
            raise RuntimeError(f"expected one warm graph replay per moving step: {delta}")
        if not torch.isfinite(host_observables).all():
            raise RuntimeError("nonfinite energy, force or stress")
    activity = summarize_trace(trace, a.steps, prefix=PREFIX, classify=kernel_group)
    api, kernels = api_activity(trace, a.steps)
    if not any("_group_moments" in name for name in kernels):
        raise RuntimeError("trace did not expose captured force-graph node kernels")
    for row in activity["top_kernel_duration_sum_ms_step"]:
        row["calls_per_step"] = kernels[row["kernel"]]
    report = dict(passed=True, variant="replay", host_efs_each_step=True, steps=a.steps,
                  warmup=a.warmup, trials=a.trials, dt_fs=0.5, dtype="float64", atoms=len(atoms),
                  skin_A=a.skin, macro_extension_A=a.macro_extension, velocity_source=velocities,
                  gpu=torch.cuda.get_device_name(), torch=str(torch.__version__), cuda=torch.version.cuda,
                  mace=package_version("mace-torch"), model_sha256=file_hash(a.model),
                  structure_sha256=file_hash(a.structure), max_displacement_A=displacement,
                  traced_graph_updates=delta, edge_membership_final=edge_audit,
                  uninstrumented_ms_step=timings, median_uninstrumented_ms_step=median(timings),
                  activity=activity, host_cuda_api=api, gpu_kernel_calls_per_step=sum(kernels.values()),
                  scope="Whole moving steps include CPU delivery of E/F/stress. Device time is the union "
                        "inside CPU markers. API duration overlaps device work and is not additive. "
                        "The part of marker wall without device activity includes profiler perturbation.")
    write_report(a.output, report)
    print("Uninstrumented median ms/step", median(timings))


if __name__ == "__main__":
    main()
