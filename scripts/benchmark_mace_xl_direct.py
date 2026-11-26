"""Compare stock MACE, tuned upstream XL and the locked XL algorithms on moving atoms."""
import argparse
import os
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import torch

from benchmark_xl_algorithms import efs_errors, new_path, read_routes, setup_engine
from mace_md_lab.bench import ASEPath, file_hash, package_version, starting_structure, write_report
from mace_md_lab.xl_fixed_step import PreparedXL


def measure(call, steps):
    torch.cuda.synchronize()
    start = perf_counter()
    call()
    torch.cuda.synchronize()
    return (perf_counter() - start) * 1000 / steps


def ase_evaluator(atoms, calculator):
    sample = atoms.copy()
    sample.calc = calculator
    calculator.reset()

    def evaluate(positions):
        sample.set_positions(positions)
        forces = sample.get_forces()
        return sample.get_potential_energy(), forces, sample.get_stress()

    return evaluate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("model", "symmetrix-json", "structure", "product-dir", "output"):
        p.add_argument("--" + key, required=True, type=Path)
    p.add_argument("--m0-dir", type=Path)
    p.add_argument("--routes-json", type=Path, default=Path("examples/native-xl-locked.json"))
    p.add_argument("--steps", type=int, default=16)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeats", type=int, default=8)
    p.set_defaults(frame=0, seed=17, temperature=300., dt_fs=0.5, skin=1.,
                   harmonic_storage="y-only-direct-v1")
    a = p.parse_args()
    if min(a.steps, a.warmup, a.repeats) < 1:
        p.error("steps, warmup and repeats must be positive")
    if a.output.resolve() in {q.resolve() for q in (a.model, a.structure, a.symmetrix_json, a.routes_json)}:
        p.error("output must be distinct from all inputs")
    os.environ["SYMMETRIX_NEIGHBOR_BACKEND"] = "kokkos"
    from mace.calculators import MACECalculator
    from symmetrix import load_backend, selected_backend

    load_backend("cuda12-sm70")
    if selected_backend().get("selector") != "cuda12-sm70" or package_version("symmetrix-xl") != "0.2.0":
        raise RuntimeError("requires Symmetrix-XL 0.2.0 cuda12-sm70")
    atoms, velocities = starting_structure(a)
    species = sorted(set(map(int, atoms.numbers)))
    routes = read_routes(a.routes_json)
    compact_hash = file_hash(a.symmetrix_json)
    mace = MACECalculator(model_paths=[str(a.model)], device="cuda", default_dtype="float64")
    engines = {key: setup_engine(a, routes[key], compact_hash, species)[0]
               for key in ("upstream", "locked")}
    names = ["mace_ase", "xl_ase", "xl_prepared", "xl_locked"]

    def make_path(name):
        if name in ("xl_prepared", "xl_locked"):
            key = "locked" if name == "xl_locked" else "upstream"
            return new_path(atoms, engines[key], a)[0]
        calc = mace if name == "mace_ase" else engines["upstream"]
        calc.reset()
        if name == "xl_ase":
            calc._neighbor_cache = None
        return ASEPath(atoms, calc, a.dt_fs)

    # Precompute distinct positions once, outside either performance measurement.
    reference = make_path("mace_ase")
    frames = []
    for i in range(a.warmup + a.steps + 1):
        if i:
            reference.advance(1)
        state = reference.snapshot()
        frames.append((state[0], (state[2], state[3], reference.atoms.get_stress())))
    if any(np.array_equal(x[0], y[0]) for x, y in zip(frames, frames[1:])):
        raise RuntimeError("benchmark requires different consecutive positions")

    def make_eval(name):
        if name == "mace_ase":
            return ase_evaluator(atoms, mace)
        engine = engines["locked" if name == "xl_locked" else "upstream"]
        engine.reset()
        engine._neighbor_cache = None
        if name == "xl_ase":
            return ase_evaluator(atoms, engine)
        return PreparedXL(atoms, engine, a.skin).evaluate

    errors = {}
    for name in names:
        evaluate = make_eval(name)
        values = [efs_errors(evaluate(pos), truth, len(atoms)) for pos, truth in frames]
        errors[name] = {key: float(np.max([v[key] for v in values])) for key in values[0]}

    md = {name: [] for name in names}
    inference = {name: [] for name in names}
    orders = []
    for trial in range(a.repeats):
        shift = trial % len(names)
        order = names[shift:] + names[:shift]
        if (trial // len(names)) % 2:
            order.reverse()
        orders.append(order)
        for name in order:
            path = make_path(name)
            path.advance(a.warmup)
            md[name].append(measure(lambda: path.advance(a.steps), a.steps))
            # Both ASE calculators normally compute E/F/stress on every force call.
            if name in ("mace_ase", "xl_ase") and "stress" not in path.atoms.calc.results:
                raise RuntimeError(f"{name} did not produce stress in its timed MD force call")
            evaluate = make_eval(name)
            for pos, _ in frames[:a.warmup + 1]:
                evaluate(pos)
            def replay_frames():
                for pos, _ in frames[a.warmup + 1:]:
                    evaluate(pos)
            inference[name].append(measure(replay_frames, a.steps))

    thresholds = {"energy_eV_atom": 1e-8, "force_eV_A": 1e-6, "stress_eV_A3": 1e-7}
    passed = all(row[k] <= threshold for row in errors.values() for k, threshold in thresholds.items())
    result = {"model_sha256": file_hash(a.model), "compact_sha256": compact_hash,
              "structure_sha256": file_hash(a.structure), "structure": a.structure.name,
              "formula": atoms.get_chemical_formula(), "atoms": len(atoms),
              "device": torch.cuda.get_device_name(), "torch": str(torch.__version__),
              "mace": package_version("mace-torch"), "ase": package_version("ase"),
              "xl": package_version("symmetrix-xl"), "dtype": "float64", "dt_fs": a.dt_fs,
              "temperature_K": a.temperature, "velocity_seed": a.seed, "velocities": velocities,
              "steps": a.steps, "warmup": a.warmup, "orders": orders,
              "generated_upstream_m0": bool(a.m0_dir), "skin_A": a.skin,
              "upstream_harmonic_storage": a.harmonic_storage,
              "md_ms_per_step": md, "md_medians": {n: median(v) for n, v in md.items()},
              "common_frames_efs_ms": inference,
              "common_frames_efs_medians": {n: median(v) for n, v in inference.items()},
              "common_frames_checked": len(frames), "max_errors_vs_mace": errors,
              "thresholds": thresholds, "passed": passed}
    write_report(a.output, result)
    if not passed:
        raise SystemExit("same-frame MACE accuracy gate failed")


if __name__ == "__main__":
    main()
