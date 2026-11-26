"""Compare the species-constant residual fold with the same prepared XL driver."""
import argparse
import os
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import torch

from mace_md_lab.bench import file_hash, starting_structure, write_report
from mace_md_lab.xl_fixed_step import ArrayVerlet, PreparedXL
from mace_md_lab.xl_product import install_xl_product, install_xl_r0
from mace_md_lab.xl_species_shift import compile_shifted_product, make_shifted_calculator


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symmetrix-json", type=Path, required=True)
    p.add_argument("--product-dir", type=Path, required=True)
    p.add_argument("--shift-dir", type=Path, required=True)
    p.add_argument("--structure", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--study", choices=("species", "harmonics"), default="species",
                   help="compare the species-shift kernel or the official recomputed-gradient storage control")
    p.add_argument("--stock-m0-dir", type=Path, help="install the stronger stock M0 when applicable")
    p.add_argument("--steps", type=int, default=24)
    p.add_argument("--repeats", type=int, default=4)
    p.add_argument("--dt-fs", type=float, default=0.5)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--skin", type=float, default=1.0)
    p.set_defaults(frame=0, seed=17, temperature=300.)
    a = p.parse_args()
    if min(a.steps, a.repeats) < 1 or a.warmup < 0:
        p.error("positive timed dimensions required")
    from symmetrix import load_backend
    load_backend("cuda12-sm70")
    from symmetrix import Symmetrix
    os.environ["SYMMETRIX_NEIGHBOR_BACKEND"] = "kokkos"
    if a.compile:
        compile_shifted_product(a.symmetrix_json, a.shift_dir)
    kw = dict(dtype="float64", execution_profile="speed", streamed_edges="direct", kernel_launch_policy="static",
            neighbor_skin=a.skin, dispersion=False)
    stock = Symmetrix(str(a.symmetrix_json), **kw)
    if a.stock_m0_dir:
        install_xl_product(stock, a.stock_m0_dir, stage="m0", variant="upstream")
    if a.study == "species":
        other, meta = make_shifted_calculator(a.symmetrix_json, a.shift_dir, **kw)
        alternative = "shifted"
    else:
        other = Symmetrix(str(a.symmetrix_json), **kw)
        if a.stock_m0_dir:
            install_xl_product(other, a.stock_m0_dir, stage="m0", variant="upstream")
        meta = {}
        alternative = "y_only"
    engines = {"upstream": stock, alternative: other}
    for e in engines.values():
        install_xl_product(e, a.product_dir, variant="upstream")
        install_xl_r0(e, a.product_dir, variant="upstream")
    if a.study == "harmonics":
        other.evaluator._set_harmonic_storage_policy("y-only-direct-v1", True)
        meta["harmonic_storage_policy"] = other.evaluator.harmonic_storage_policy
    atoms, vel_source = starting_structure(a)

    def runner(key, at):
        backend = PreparedXL(at, engines[key], a.skin)
        return ArrayVerlet(at, backend.evaluate, a.dt_fs), backend

    def trajectory(key):
        path, _ = runner(key, atoms)
        frames = [path.snapshot()]
        for _ in range(a.steps + a.warmup):
            path.advance(1)
            frames.append(path.snapshot())
        return frames

    original = trajectory("upstream")
    candidate = trajectory(alternative)
    if np.array_equal(original[0][0], original[-1][0]):
        raise ValueError("expected moving atoms")
    common = []
    for i in (0, len(original) // 2, len(original) - 1):
        fixed = atoms.copy()
        fixed.positions[:] = original[i][0]
        path, _ = runner(alternative, fixed)
        common.append((original[i], path.snapshot()))

    def delta_nve(s, start):
        return (s[2] + s[4] - start[2] - start[4]) / len(atoms)

    errors = dict(energy_eV_atom=max(abs(t[2] - s[2]) / len(atoms) for t, s in common),
                force_eV_A=max(abs(t[3] - s[3]).max() for t, s in common),
                stress_eV_A3=max(abs(t[5] - s[5]).max() for t, s in common),
                pos_A=max(abs(t[0] - s[0]).max() for t, s in zip(original, candidate)),
                velocity_A_fs=max(abs(t[1] - s[1]).max() for t, s in zip(original, candidate)),
                nve_eV_atom=max(abs(delta_nve(t, original[0]) - delta_nve(s, candidate[0])) for t, s in zip(original, candidate)))
    bounds = dict(energy_eV_atom=1e-9, force_eV_A=1e-8, stress_eV_A3=1e-9, pos_A=1e-8, velocity_A_fs=1e-9, nve_eV_atom=1e-9)
    passed = all(np.isfinite(v) and v <= bounds[k] for k, v in errors.items())
    print("validation", errors, passed, flush=True)
    timings = {k: [] for k in engines}
    rebuilds = {k: [] for k in engines}
    traces = {}
    if passed:
        from benchmark_xl_product import profile_native_steps
        for key in engines:
            path, _ = runner(key, atoms)
            path.advance(a.warmup)
            traces[key] = profile_native_steps(path)
        if a.study == "harmonics" and other.evaluator.execution_direct_harmonic_launch_count < 2:
            raise ValueError("missing direct harmonic evaluation")
        if a.study == "species" and traces[alternative]["calls_in_two_steps"].get("symmetrix_m0_forward", 0) < 2:
            raise ValueError("missing native shifted M0")
        for trial in range(a.repeats):
            order = list(engines) if trial % 2 == 0 else list(reversed(engines))
            for key in order:
                path, graph = runner(key, atoms)
                path.advance(a.warmup)
                torch.cuda.synchronize()
                before = graph.rebuilds
                t = perf_counter()
                path.advance(a.steps)
                torch.cuda.synchronize()
                timings[key].append((perf_counter() - t) * 1000 / a.steps)
                rebuilds[key].append(graph.rebuilds - before)
    report = dict(passed=bool(passed), errors_vs_prepared_xl={k: float(v) for k, v in errors.items()}, gates=bounds,
                compact_sha256=file_hash(a.symmetrix_json), fixture_sha256=file_hash(a.structure),
                study=a.study, manifest=meta, steps=a.steps, repeats=a.repeats, warmup=a.warmup, dt_fs=a.dt_fs, neighbor_skin=a.skin,
                velocities=vel_source, device=torch.cuda.get_device_name(), dtype="float64", kernel_launch_policy="static",
                driver="PreparedXL/ArrayVerlet with host E/F/stress", both_generated_r0_m1=True,
                stock_generated_m0=bool(a.stock_m0_dir), traces=traces, ms_step=timings,
                median_ms_step={k: median(v) for k, v in timings.items() if v}, timed_rebuilds=rebuilds)
    write_report(a.output, report)
    print("medians", report["median_ms_step"], flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
