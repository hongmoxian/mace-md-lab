"""Benchmark the speed-oriented Symmetrix-XL ASE path with the original MACE weights.

Optional setup on a V100 with a compatible CUDA 12 / MACE environment::

    python -m pip install symmetrix-xl==0.2.0 symmetrix-xl-cuda12-sm70==0.2.0
    symmetrix_extract_mace --model original.model --chemical-symbols Cu Au O H --num-spline-points 8192 --output compact.json
    export SYMMETRIX_JIT_CACHE="$(mktemp -d)"
    python scripts/benchmark_backend.py --model original.model --symmetrix-json compact.json \
        --structure initial.extxyz --output comparison.json

Use the official wheel and selector corresponding to your GPU architecture. Select
the elements in your structure on export, or omit the subset for a full-domain
export. Use a node-local JIT cache on filesystems lacking atomic no-replace rename.
"""

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import torch
from mace.calculators import MACECalculator
from mace_md_lab.transforms import optimized_model
from mace_md_lab.bench import (
    ASEPath,
    file_hash,
    make_path,
    numerical_gates,
    package_version,
    starting_structure,
    sync,
)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--symmetrix-json", required=True, type=Path)
    parser.add_argument("--structure", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--dt-fs", type=float, default=0.5)
    parser.add_argument("--skin", type=float, default=0.12)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--symmetrix-selector", default="cuda12-sm70")
    args = parser.parse_args()
    if (
        min(args.steps, args.repeats) < 1
        or args.warmup < 0
        or not (args.dt_fs > 0 and args.skin >= 0)
    ):
        parser.error("invalid number of steps, repeats, time step or skin")
    if args.output.resolve() in {
        p.resolve() for p in (args.model, args.symmetrix_json, args.structure)
    }:
        parser.error("output must not overwrite an input")
    args.certify_edges = True
    return args


def main():
    args = arguments()
    from symmetrix import load_backend, selected_backend

    # Force the requested GPU backend and reject any automatic CPU fallback.
    load_backend(args.symmetrix_selector)
    from symmetrix import Symmetrix

    selected = selected_backend()
    if selected.get(
        "selector"
    ) != args.symmetrix_selector or not args.symmetrix_selector.startswith("cuda"):
        raise RuntimeError(f"unexpected loaded Symmetrix backend {selected}")
    dev = torch.device("cuda")
    calc = MACECalculator(
        model_paths=[str(args.model)], device="cuda", default_dtype=args.dtype
    )
    atoms, velocities = starting_structure(args)
    sym = Symmetrix(
        str(args.symmetrix_json),
        dtype=args.dtype,
        execution_profile="speed",
        streamed_edges="direct",
        neighbor_skin=args.skin,
        dispersion=False,
    )
    paths = [
        "ase_native",
        "resident_capture",
        "resident_padded",
        "both_exact",
        "both_padded",
        "symmetrix_direct_ase",
    ]
    transform_info = {}

    @contextmanager
    def new_path(name, structure):
        if name == "symmetrix_direct_ase":
            sym.reset()
            try:
                yield ASEPath(structure, sym, args.dt_fs)
            finally:
                sym.reset()
        elif name in ("both_exact", "both_padded"):
            graph_mode = (
                "resident_capture" if name == "both_exact" else "resident_padded"
            )
            with optimized_model(calc.models[0], "both", structure.numbers) as meta:
                transform_info.update(meta)
                with make_path(graph_mode, structure, calc, args) as path:
                    yield path
        else:
            with make_path(name, structure, calc, args) as path:
                yield path

    def trajectory(name):
        samples, energy_changes = [], []
        with new_path(name, atoms) as path:
            for step in range(args.steps + 1):
                if step:
                    path.advance(1)
                snap = path.snapshot()
                if any(not np.isfinite(np.asarray(v)).all() for v in snap):
                    raise ValueError(f"{name} produced nonfinite output")
                samples.append(snap)
                energy_changes.append((snap[2] + snap[4]) / len(atoms))
            stress = (
                "stress" in path.atoms.calc.results
                if isinstance(path, ASEPath)
                else False
            )
        return samples, np.asarray(energy_changes) - energy_changes[0], stress

    ref, ref_drift, _ = trajectory("ase_native")
    if np.abs(ref[0][0] - ref[-1][0]).max() < 1e-12:
        raise ValueError("trajectory does not move")
    tolerances = numerical_gates(args)
    checks = {}
    for name in paths:
        tr, drift, stress = trajectory(name)
        # Validate on the identical starting geometry and final reference geometry.
        fixed = atoms.copy()
        fixed.positions = ref[-1][0]
        with new_path(name, fixed) as frozen:
            on_last = frozen.snapshot()
        pairs = [(tr[0], ref[0]), (on_last, ref[-1])]
        errors = {
            "energy_eV_atom": max(
                abs(float(a[2]) - float(b[2])) / len(atoms) for a, b in pairs
            ),
            "force_eV_A": max(float(np.max(np.abs(a[3] - b[3]))) for a, b in pairs),
            "position_A": max(
                float(np.max(np.abs(a[0] - b[0]))) for a, b in zip(tr, ref)
            ),
            "velocity_A_fs": max(
                float(np.max(np.abs(a[1] - b[1]))) for a, b in zip(tr, ref)
            ),
            "nve_difference_eV_atom": float(np.max(np.abs(drift - ref_drift))),
        }
        checks[name] = {
            "errors": errors,
            "passed": all(v <= tolerances[k] for k, v in errors.items()),
            "stress_returned": stress,
            "max_NVE_excursion_meV_atom": 1000 * float(np.max(np.abs(drift))),
        }
    passed = all(c["passed"] for c in checks.values())
    trials = {p: [] for p in paths}
    replay_stats = {p: [] for p in paths if p.startswith(("both_", "resident_"))}
    if passed:
        for rep in range(args.repeats):
            shift = rep % len(paths)
            for name in paths[shift:] + paths[:shift]:
                with new_path(name, atoms) as p:
                    p.advance(args.warmup)
                    sync(dev)
                    if name in replay_stats:
                        counts0 = (
                            p.runner.replays,
                            p.runner.fallback_evaluations,
                            p.runner.captures,
                        )
                    start = perf_counter()
                    p.advance(args.steps)
                    sync(dev)
                    ms = (perf_counter() - start) * 1000 / args.steps
                    if name in replay_stats:
                        replay, fallback, capture = [
                            a - b
                            for a, b in zip(
                                (
                                    p.runner.replays,
                                    p.runner.fallback_evaluations,
                                    p.runner.captures,
                                ),
                                counts0,
                            )
                        ]
                        if fallback or replay != args.steps:
                            raise RuntimeError(
                                f"{name} did not replay every timed step"
                            )
                        replay_stats[name].append(
                            {
                                "replays": replay,
                                "fallbacks": fallback,
                                "recaptures": capture,
                            }
                        )
                trials[name].append(ms)
    plan = sym.execution_plan
    with args.symmetrix_json.open() as f:
        compact = json.load(f)
    spline_grid = compact.get("compact_radial", {}).get("num_spline_points")
    output = {
        "model_sha256": file_hash(args.model),
        "converted_sha256": file_hash(args.symmetrix_json),
        "input_sha256": file_hash(args.structure),
        "atoms": len(atoms),
        "formula": atoms.get_chemical_formula(),
        "input_velocities": velocities,
        "frame": args.frame,
        "device": torch.cuda.get_device_name(dev),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "mace": package_version("mace-torch"),
        "symmetrix": package_version("symmetrix-xl"),
        "symmetrix_selected_backend": selected.get("selector"),
        "symmetrix_plan": {
            k: v
            for k, v in plan.items()
            if k
            in (
                "selected_id",
                "selected_algorithm",
                "requested_profile",
                "state",
                "selection_source",
            )
        },
        "compact_radial_spline_points": spline_grid,
        "compact_atomic_numbers": compact.get("atomic_numbers"),
        "symmetrix_dtype": args.dtype,
        "mace_dtype": args.dtype,
        "warmup_steps": args.warmup,
        "steps": args.steps,
        "repeats": args.repeats,
        "skin_A": args.skin,
        "dt_fs": args.dt_fs,
        "max_errors_allowed": tolerances,
        "checks": checks,
        "passed": passed,
        "transform": transform_info,
        "cuda_graph_diagnostics": replay_stats,
        "raw_ms_per_step": trials,
        "median_ms_per_step": {k: median(v) for k, v in trials.items() if v},
        "timing_scope": "Single trajectory on one GPU. Warmed full velocity-Verlet steps with no logging and synchronized endpoints. Symmetrix uses native compiled direct/speed via ASE. Original MACE uses ASE; resident exact-size and padded paths use device-resident state. Compilation, initial forces and validation excluded.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "passed": passed,
                "medians_ms": output["median_ms_per_step"],
                "symmetrix": checks["symmetrix_direct_ase"],
            }
        )
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
