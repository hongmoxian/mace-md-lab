"""Time XL's native M1 or R0 kernel variants with a moving Au/water trajectory."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

from mace_md_lab.bench import ASEPath, file_hash, starting_structure, write_report
from mace_md_lab.xl_product import install_xl_product, install_xl_r0


def profile_native_steps(path):
    """Collect native kernel counts and per-step GPU times for two moving steps."""
    with TemporaryDirectory() as tmp:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as trace:
            path.advance(2)
            torch.cuda.synchronize()
        target = str(Path(tmp) / "trace.json")
        trace.export_chrome_trace(target)
        events = json.loads(Path(target).read_text())["traceEvents"]
        names_seen = Counter()
        durations = Counter()
        for e in events:
            if e.get("cat") != "kernel":
                continue
            label = e["name"]
            for needle in ("symmetrix_m0_forward", "symmetrix_m0_reverse", "::reverse_M0", "::compute_M0",
                           "symmetrix_m1_forward", "symmetrix_m1_reverse", "::reverse_M1", "::compute_M1",
                           "symmetrix_r0_forward", "symmetrix_r0_coordinate_reverse",
                           "standard_r0::launch_forward", "standard_r0::launch_coordinate_reverse"):
                if needle in label:
                    names_seen[needle] += 1
                    durations[needle] += e["dur"] / 2000.
                    break
        return {"calls_in_two_steps": names_seen, "ms_per_step": durations}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symmetrix-json", type=Path, required=True)
    p.add_argument("--artifact-dir", type=Path, required=True)
    p.add_argument("--structure", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=48)
    p.add_argument("--repeats", type=int, default=6)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--stage", choices=("m0", "m1", "r0"), default="m1")
    p.add_argument("--kernel-launch-policy", choices=("automatic", "static"), default="automatic")
    p.set_defaults(frame=0, seed=17, temperature=300.)
    a = p.parse_args()
    if min(a.steps, a.repeats, a.warmup) < 1 or a.output.resolve() in {a.symmetrix_json.resolve(), a.structure.resolve()}:
        p.error("supply positive step counts and a separate output file")

    from symmetrix import load_backend
    load_backend("cuda12-sm70")
    from symmetrix import Symmetrix

    os.environ["SYMMETRIX_NEIGHBOR_BACKEND"] = "kokkos"
    atoms, velocity_source = starting_structure(a)
    def engine():
        return Symmetrix(str(a.symmetrix_json), dtype="float64", execution_profile="speed",
                         streamed_edges="direct", neighbor_skin=1., dispersion=False,
                         kernel_launch_policy=a.kernel_launch_policy)

    stock, candidate = engine(), engine()
    test_r0 = a.stage == "r0"
    test_m0 = a.stage == "m0"
    common_modules = {}
    if test_r0 or test_m0:
        for calc in (stock, candidate):
            common_modules["m1"] = install_xl_product(calc, a.artifact_dir, variant="upstream")
            if test_m0:
                common_modules["r0"] = install_xl_r0(calc, a.artifact_dir, variant="upstream")
    details = {}
    def setup(name):
        calc = stock if name == "stock" else candidate
        if name != "stock":
            if test_r0:
                details[name] = install_xl_r0(calc, a.artifact_dir, variant=name)
            else:
                details[name] = install_xl_product(calc, a.artifact_dir, variant=name, stage=a.stage)
        calc.reset()
        calc._neighbor_cache = None
        return calc, ASEPath(atoms, calc, 0.5)

    def frames(name):
        calc, path = setup(name)
        trajectory = []
        for step in range(a.warmup + a.steps + 1):
            if step:
                path.advance(1)
            trajectory.append((*path.snapshot(), path.atoms.get_stress().copy()))
        return trajectory

    reference = frames("stock")
    if np.array_equal(reference[0][0], reference[-1][0]):
        raise ValueError("stationary trajectory")
    names = ("stock", "upstream") if test_m0 else ("stock", "upstream", "factored" if test_r0 else "horner")
    gates = {"position_A": 1e-6, "velocity_A_fs": 1e-7, "force_eV_A": 1e-6,
             "energy_eV_atom": 1e-8, "stress_eV_A3": 1e-7, "nve_eV_atom": 1e-8}
    traces, checks = {}, {}
    for name in names:
        traj = frames(name)
        calc, path = setup(name)
        err = dict.fromkeys(gates, 0.0)
        fixed = atoms.copy()
        fixed.calc = calc
        for s, ref in zip(traj, reference):
            fixed.positions[:] = ref[0]
            values = {
                "position_A": np.max(abs(s[0] - ref[0])),
                "velocity_A_fs": np.max(abs(s[1] - ref[1])),
                "energy_eV_atom": abs(fixed.get_potential_energy() - ref[2]) / len(atoms),
                "force_eV_A": np.max(abs(fixed.get_forces() - ref[3])),
                "stress_eV_A3": np.max(abs(fixed.get_stress() - ref[5])),
                "nve_eV_atom": abs((s[2]+s[4]-traj[0][2]-traj[0][4])
                                   -(ref[2]+ref[4]-reference[0][2]-reference[0][4])) / len(atoms),
            }
            if not np.isfinite(list(values.values())).all():
                raise ValueError("nonfinite trajectory")
            for key, value in values.items():
                err[key] = max(err[key], float(value))
        checks[name] = err
        if any(err[key] > gates[key] for key in gates):
            raise ValueError(f"{name} differs from native XL: {err}")
        calc, path = setup(name)
        path.advance(a.warmup)
        torch.cuda.synchronize()
        traces[name] = profile_native_steps(path)
        print(name, traces[name], flush=True)
    times = {n: [] for n in names}
    order = []
    for rep in range(a.repeats):
        rotate = (rep // 2) % len(names)
        sequence = names[rotate:] + names[:rotate]
        if rep % 2:
            sequence = sequence[::-1]
        order.append(sequence)
        for name in sequence:
            _, path = setup(name)
            path.advance(a.warmup)
            torch.cuda.synchronize()
            start = perf_counter()
            path.advance(a.steps)
            torch.cuda.synchronize()
            times[name].append(1000 * (perf_counter() - start) / a.steps)
    ms = {k: median(v) for k, v in times.items()}
    kernels = (("symmetrix_r0_forward", "symmetrix_r0_coordinate_reverse") if test_r0 else
               (f"symmetrix_{a.stage}_forward", f"symmetrix_{a.stage}_reverse"))
    admitted = all(traces[name]["calls_in_two_steps"][kernel] >= 2
                   for name in names[1:] for kernel in kernels)
    report = dict(passed=admitted, module_executed=admitted, checks=checks, gates=gates, traces=traces,
                  stage=a.stage, all_paths_use_upstream_generated_m1=test_r0 or test_m0,
                  common_modules=common_modules, kernel_launch_policy=a.kernel_launch_policy,
                  artifact=details, steps=a.steps, warmup=a.warmup, repeats=a.repeats, atoms=len(atoms),
                  velocity_source=velocity_source, dt_fs=0.5, selector="cuda12-sm70", dtype="float64",
                  profile="speed", skin_A=1., neighbor_builder="kokkos", gpu=torch.cuda.get_device_name(),
                  precision="float64", observables=["energy","forces","stress"], driver="ASE VelocityVerlet",
                  structure_sha256=file_hash(a.structure), model_sha256=file_hash(a.symmetrix_json),
                  raw_ms_step=times, median_ms_step=ms, orders=order)
    write_report(a.output, report)
    print(ms, "native module executed:", admitted, flush=True)
    if not admitted:
        raise RuntimeError("CUPTI did not observe each candidate forward and reverse kernel in both steps")


if __name__ == "__main__":
    main()
