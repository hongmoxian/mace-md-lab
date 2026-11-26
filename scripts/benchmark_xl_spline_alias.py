"""Qualify native FP64 radial table row aliasing and compare moving MD with a fixed driver."""
import argparse
import json
import os
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import torch
from ase.io import read

from mace_md_lab.bench import file_hash, write_report
from mace_md_lab.xl_fixed_step import PreparedXL, ArrayVerlet
from mace_md_lab.xl_product import install_xl_product
from mace_md_lab.xl_spline_alias import QUALIFIED_COMPACT, load_module, qualify


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("compact", "artifacts", "m1", "model", "structure", "output", "audit"):
        p.add_argument("--"+name, required=True, type=Path)
    p.add_argument("--m0", type=Path)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--steps", type=int, default=48)
    p.add_argument("--repeats", type=int, default=6)
    p.add_argument("--trace-only", action="store_true",
                   help="collect untimed R0/R1 CUPTI activity for the qualified row aliases")
    a = p.parse_args()
    if min(a.steps, a.repeats) < 1 or a.warmup < 0 or file_hash(a.compact) != QUALIFIED_COMPACT:
        p.error("positive timed dimensions and the qualified FP64 compact export are required")
    os.environ["SYMMETRIX_NEIGHBOR_BACKEND"] = "kokkos"
    from symmetrix import load_backend
    load_backend("cuda12-sm70")
    from symmetrix import Symmetrix
    atoms = read(a.structure)
    engine = Symmetrix(str(a.compact), dtype="float64", execution_profile="speed",
                       streamed_edges="direct", neighbor_skin=1., kernel_launch_policy="static",
                       dispersion=False)
    m1 = install_xl_product(engine, a.m1, variant="upstream")
    if a.m0:
        install_xl_product(engine, a.m0, variant="upstream", stage="m0")
    audit = qualify(engine, atoms, a.artifacts)
    write_report(a.audit, audit)
    valid = [s for s in ("r0", "r1") if audit["stages"][s]["passed"]]
    cases = {"control": ()}
    cases.update({s: (s,) for s in valid})
    if len(valid) == 2:
        cases["both"] = tuple(valid)
    print("native coefficient checks", json.dumps(audit), flush=True)
    if not valid:
        raise ValueError("neither radial table qualifies")

    def route(name, start=atoms):
        for stage in ("r0", "r1"):
            load_module(engine, a.artifacts, stage, "alias" if stage in cases[name] else "control")
        engine.reset()
        prepared = PreparedXL(start, engine, 1.)
        return ArrayVerlet(start, prepared.evaluate, 0.5), prepared

    if a.trace_only:
        from benchmark_xl_algorithms import witness
        kernels = ["symmetrix_r0_forward_v1", "symmetrix_r0_coordinate_reverse_v1",
                   "symmetrix_factorized_forward_v2", "symmetrix_factorized_reverse_fused_v2"]
        observations = {name: [] for name in cases}
        for trial in range(a.repeats):
            for name in list(cases)[::1 if trial % 2 == 0 else -1]:
                r, _ = route(name)
                r.advance(a.warmup)
                torch.cuda.synchronize()
                observations[name].append(witness(r, kernels))
        passed = all(t["passed"] for rows in observations.values() for t in rows)
        report = dict(passed=passed, traces=observations, audit=audit, gpu=torch.cuda.get_device_name(),
                      compact_sha256=QUALIFIED_COMPACT, structure_sha256=file_hash(a.structure),
                      warmup=a.warmup, repeats=a.repeats, traced_steps=2)
        write_report(a.output, report)
        if not passed:
            raise ValueError("missing expected native R0/R1 kernel dispatch")
        return

    ref, _ = route("control")
    frames = [ref.snapshot()]
    for _ in range(a.warmup + a.steps):
        ref.advance(1)
        frames.append(ref.snapshot())

    from mace.calculators import MACECalculator
    original = MACECalculator(model_paths=[str(a.model)], device="cuda", default_dtype="float64")
    common = []
    for index in (0, len(frames) // 2, len(frames) - 1):
        fixed = atoms.copy()
        fixed.positions[:] = frames[index][0]
        fixed.calc = original
        common.append((fixed, fixed.get_potential_energy(), fixed.get_forces(), fixed.get_stress()))
    results = {}
    for name in cases:
        errors = dict(force_eV_A=0., energy_eV_atom=0., stress_eV_A3=0.)
        for fixed, energy, forces, stress in common:
            path, _ = route(name, fixed)
            errors["force_eV_A"] = max(errors["force_eV_A"], np.max(abs(forces - path.forces)))
            errors["stress_eV_A3"] = max(errors["stress_eV_A3"], np.max(abs(stress - path.stress)))
            errors["energy_eV_atom"] = max(errors["energy_eV_atom"], abs(energy - path.energy) / len(fixed))
        r, _ = route(name)
        compare = np.zeros(6)
        for i, expected in enumerate(frames):
            if i:
                r.advance(1)
            actual = r.snapshot()
            compare = np.maximum(compare, [np.max(np.abs(x - y)) for x, y in zip(actual, expected)])
        passed = bool(errors["energy_eV_atom"] <= 1e-8 and errors["force_eV_A"] <= 1e-6
                      and errors["stress_eV_A3"] <= 1e-7 and np.all(compare <= 1e-11))
        results[name] = dict(mace_errors=errors, xl_trajectory_errors=dict(zip(
            ("positions_A", "velocities_A_fs", "energy_eV", "forces_eV_A", "kinetic_eV", "stress_eV_A3"), compare)),
            passed=passed, raw_ms_step=[], timed_neighbor_rebuilds=[])
    passed = all(row["passed"] for row in results.values())
    orders = []
    if passed:
        for trial in range(a.repeats):
            names = list(cases)
            shift = (trial // 2) % len(names)
            names = names[shift:] + names[:shift]
            if trial % 2:
                names.reverse()
            orders.append(names)
            for name in names:
                r, prep = route(name)
                r.advance(a.warmup)
                before = prep.rebuilds
                torch.cuda.synchronize()
                start = perf_counter()
                r.advance(a.steps)
                torch.cuda.synchronize()
                results[name]["raw_ms_step"].append(1000 * (perf_counter() - start) / a.steps)
                results[name]["timed_neighbor_rebuilds"].append(prep.rebuilds - before)
    medians = {name: median(row["raw_ms_step"]) for name, row in results.items() if row["raw_ms_step"]}
    print("Native alias medians", medians, flush=True)
    write_report(a.output, dict(passed=passed, steps=a.steps, repeats=a.repeats, warmup=a.warmup, dt_fs=.5,
                 rows=results, orders=orders, medians_ms_step=medians, audit=audit, gpu=torch.cuda.get_device_name(),
                 structure_sha256=file_hash(a.structure), compact_sha256=QUALIFIED_COMPACT, model_sha256=file_hash(a.model),
                 native_m1=m1["cubin_sha256"], generated_m0=bool(a.m0), native_precision="FP64",
                 driver="PreparedXL and ArrayVerlet; E/F/stress returned on each step", skin_A=1.,
                 compiler="R0 and R1 use nvcc for all four routes; original factored R0 and stock generated R1 are controls"))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
