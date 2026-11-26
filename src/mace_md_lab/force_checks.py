"""Check same-weight conservative forces and short NVE on a fixed-cell structure."""

import argparse
from contextlib import nullcontext
import math
from pathlib import Path

import numpy as np
import torch
from mace.calculators import MACECalculator

from .bench import (
    file_hash,
    host,
    make_path,
    numerical_gates,
    package_version,
    starting_structure,
    topology_counts,
    write_report,
)
from .runtime import ResidentMACE


def args_for_checks(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model", required=True, type=Path, help="trusted local MACE checkpoint"
    )
    p.add_argument("--structure", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    p.add_argument(
        "--adjoint", choices=["none", "angular", "product", "both"], default="none"
    )
    p.add_argument(
        "--backend", choices=["auto", "eager", "raw", "padded"], default="auto"
    )
    p.add_argument("--head")
    p.add_argument("--steps", type=int, default=48)
    p.add_argument("--dt-fs", type=float, default=0.5)
    p.add_argument("--skin", type=float, default=0.5)
    p.add_argument("--certify-edges", action="store_true")
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument(
        "--fd-h",
        type=float,
        nargs="+",
        help="central-difference displacement lengths in A; default depends on dtype",
    )
    p.add_argument(
        "--nve-budget-mev",
        type=float,
        default=1.0,
        help="maximum short-run NVE excursion in meV/atom",
    )
    a = p.parse_args(argv)
    if a.fd_h is None:
        a.fd_h = (
            [0.003, 0.001, 0.0003, 0.0001]
            if a.dtype == "float64"
            else [0.06, 0.04, 0.025, 0.015]
        )
    floats = [a.dt_fs, a.skin, a.temperature, a.nve_budget_mev, *a.fd_h]
    if (
        a.steps < 1
        or a.seed < 0
        or not np.isfinite(floats).all()
        or min(a.dt_fs, a.skin, a.nve_budget_mev) <= 0
        or a.temperature < 0
        or len(set(a.fd_h)) < 2
        or len(set(a.fd_h)) != len(a.fd_h)
        or min(a.fd_h) <= 0
    ):
        p.error(
            "require finite positive dt, skin, budget and at least two distinct positive FD lengths"
        )
    if (
        not a.model.is_file()
        or not a.structure.is_file()
        or a.output.resolve() in {a.model.resolve(), a.structure.resolve()}
    ):
        p.error("supply two existing input files and a separate report path")
    return a


def force_directions(atoms, seed):
    """Use a whole-cell random unit direction and a localized symmetric bond stretch."""
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=(len(atoms), 3))
    if len(atoms) > 1:
        direction -= direction.mean(axis=0)
    result = [("random_normalized", direction / np.linalg.norm(direction))]
    oxy = np.flatnonzero(atoms.numbers == 8)
    hyd = np.flatnonzero(atoms.numbers == 1)
    if len(oxy) and len(hyd):
        i = int(oxy[0])
        distances = atoms.get_distances(i, hyd, mic=True)
        j = int(hyd[np.argmin(distances)])
        bond = atoms.get_distance(i, j, mic=True, vector=True)
        length = np.linalg.norm(bond)
        if 0.7 < length < 1.25:
            stretch = np.zeros_like(direction)
            stretch[j] = bond / (length * np.sqrt(2))
            stretch[i] = -stretch[j]
            result.append((f"OH_stretch_{i}_{j}", stretch))
    return result


def fixed_checks(initial, native, candidate, args, name):
    at = initial.copy()
    at.calc = native
    ref_e, ref_f = at.get_potential_energy(), at.get_forces(apply_constraint=False)
    runner = ResidentMACE(
        initial,
        candidate,
        skin=args.skin,
        certify_edges=args.certify_edges,
        capture_forces=name in ("resident_capture", "resident_padded"),
        pad_capture=name == "resident_padded",
    )

    def ef(x):
        e, f = runner.evaluate(x)
        return e.item(), host(f)

    ref_x = initial.positions.copy()
    e, f = ef(ref_x)
    actual_energy = abs(e - ref_e) / len(at)
    actual_force = float(abs(f - ref_f).max())
    total = float(np.linalg.norm(np.sum(f, axis=0)))
    shifted_e, shifted_f = ef(ref_x + np.array([0.53, -0.31, 0.17]))
    n = len(at)
    limits = numerical_gates(args)
    rows = []
    issues = []

    def check(metric, value, tolerance):
        if not math.isfinite(value) or value > tolerance:
            issues.append(f"{metric} {value:.4g} exceeded {tolerance:.4g}")

    check("fixed energy (eV/atom)", actual_energy, limits["energy_eV_atom"])
    check("fixed force (eV/A)", actual_force, limits["force_eV_A"])
    check(
        "translation energy (eV/atom)", abs(shifted_e - e) / n, limits["energy_eV_atom"]
    )
    check(
        "translation force (eV/A)",
        float(abs(shifted_f - f).max()),
        limits["force_eV_A"],
    )
    force_sum_tol = 1e-7 if args.dtype == "float64" else 5e-3
    check("net force norm (eV/A)", total, force_sum_tol)

    for label, u in force_directions(at, args.seed):
        projection = float(np.sum(f * u))
        reference_projection = float(np.sum(ref_f * u))
        tol = (1e-4 if args.dtype == "float64" else 0.02) * max(
            1.0, abs(reference_projection)
        )
        samples = []
        for h in args.fd_h:
            # Reference energies on both displaced geometries give an independent
            # conservative-force check for candidate gradients and cutoff handling.
            displaced = []
            for sign in (-1, 1):
                at.positions = ref_x + sign * h * u
                displaced.append(at.get_potential_energy())
            fd = (displaced[0] - displaced[1]) / (2 * h)
            samples.append(
                {
                    "h_A": h,
                    "negative_dE_dh_eV_A": float(fd),
                    "absolute_error_eV_A": abs(fd - projection),
                }
            )
        errors = sorted(row["absolute_error_eV_A"] for row in samples)
        check(f"second-best directional difference {label} (eV/A)", errors[1], tol)
        rows.append(
            {
                "direction": label,
                "candidate_force_projection_eV_A": projection,
                "reference_force_projection_eV_A": reference_projection,
                "absolute_tolerance_eV_A": tol,
                "required_passing_distinct_h": 2,
                "samples": samples,
            }
        )
    result = {
        "fixed_energy_error_eV_atom": actual_energy,
        "fixed_force_max_error_eV_A": actual_force,
        "candidate_net_force_norm_eV_A": total,
        "official_net_force_norm_eV_A": float(np.linalg.norm(ref_f.sum(0))),
        "net_force_norm_tolerance_eV_A": force_sum_tol,
        "translation_energy_change_eV_atom": abs(shifted_e - e) / n,
        "translation_max_force_change_eV_A": float(abs(shifted_f - f).max()),
        "finite_differences": rows,
    }
    return result, issues


def trajectory(initial, calc, args, name, dt, steps):
    # Keep the source atoms, momenta and time budget identical across comparisons.
    settings = argparse.Namespace(**vars(args))
    settings.dt_fs = dt
    snapshots = []
    with make_path(name, initial, calc, settings) as path:
        for step in range(steps + 1):
            if step:
                path.advance(1)
            snap = path.snapshot()
            if not all(np.isfinite(x).all() for x in snap):
                raise FloatingPointError(f"nonfinite {name} trajectory at step {step}")
            snapshots.append(snap)
        counts = topology_counts(path)
    return snapshots, counts


def energy_change(frames, n):
    h = np.array([s[2] + s[4] for s in frames])
    return 1000 * (h - h[0]) / n


def compare_trajectories(initial, official, candidate, args, backend):
    native, _ = trajectory(
        initial, official, args, "ase_native", args.dt_fs, args.steps
    )
    fast, counts = trajectory(initial, candidate, args, backend, args.dt_fs, args.steps)
    # Two integrations cover the same physical time. Half-step is a timestep
    # sensitivity check on the same checkpoint and backend, independent of ASE parity.
    refined, half_counts = trajectory(
        initial, candidate, args, backend, args.dt_fs / 2, 2 * args.steps
    )
    n = len(initial)
    drift_ref, drift_fast, drift_half = (
        energy_change(x, n) for x in (native, fast, refined)
    )
    maxabs = lambda x: float(np.max(np.abs(x)))
    diff = lambda k: max(maxabs(r[k] - s[k]) for r, s in zip(native, fast))
    errors = {
        "position_A": diff(0),
        "velocity_A_fs": diff(1),
        "energy_eV_atom": diff(2) / n,
        "force_eV_A": diff(3),
        "nve_difference_eV_atom": maxabs(drift_ref - drift_fast) / 1000,
    }
    limits = numerical_gates(args)
    issues = [
        f"NVE {key} {v:.4g} exceeded {limits[key]:.4g}"
        for key, v in errors.items()
        if v > limits[key]
    ]
    for label, drift in [
        ("native", drift_ref),
        ("resident", drift_fast),
        ("half_step", drift_half),
    ]:
        if maxabs(drift) > args.nve_budget_mev:
            issues.append(
                f"{label} NVE excursion exceeded {args.nve_budget_mev} meV/atom"
            )
    coarse, fine = maxabs(drift_fast), maxabs(drift_half)
    return {
        "physical_time_fs": args.steps * args.dt_fs,
        "dt_fs": args.dt_fs,
        "steps": args.steps,
        "max_reference_error_all_steps": errors,
        "reference_match_tolerances": limits,
        "max_excursion_meV_atom": {
            "official": maxabs(drift_ref),
            "candidate": coarse,
            "candidate_half_dt": fine,
        },
        "half_dt_fs": args.dt_fs / 2,
        "half_dt_steps": args.steps * 2,
        "half_dt_to_full_dt_excursion_ratio": fine / coarse if coarse else None,
        "excursion_budget_meV_atom": args.nve_budget_mev,
        "end_change_meV_atom": float(drift_fast[-1]),
        "temperature_K_initial": float(initial.get_temperature()),
        "max_candidate_net_force_eV_A": max(
            float(np.linalg.norm(s[3].sum(0))) for s in fast
        ),
        "candidate_counts_including_initial": counts,
        "half_step_candidate_counts_including_initial": half_counts,
        "half_step_final_position_RMS_relative_to_full_A": float(
            np.sqrt(np.mean(np.sum((refined[-1][0] - fast[-1][0]) ** 2, axis=1)))
        ),
    }, issues


def main(argv=None):
    args = args_for_checks(argv)
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda") or (
        device.type == "cuda" and not torch.cuda.is_available()
    ):
        raise ValueError(f"unavailable device {device}")
    if device.type == "cuda":
        torch.cuda.set_device(
            device.index if device.index is not None else torch.cuda.current_device()
        )
    backend = args.backend
    if backend == "auto":
        backend = "padded" if device.type == "cuda" else "eager"
    paths = {
        "eager": "resident_cached",
        "raw": "resident_capture",
        "padded": "resident_padded",
    }
    name = paths[backend]
    if device.type == "cpu" and backend != "eager":
        raise ValueError("graph capture needs CUDA")
    initial, velocity_source = starting_structure(args)
    report = {
        "checkpoint": args.model.name,
        "checkpoint_sha256": file_hash(args.model),
        "structure": args.structure.name,
        "structure_sha256": file_hash(args.structure),
        "n_atoms": len(initial),
        "formula": initial.get_chemical_formula(),
        "frame": args.frame,
        "device": str(device),
        "dtype": args.dtype,
        "adjoint": args.adjoint,
        "backend": backend,
        "skin_A": args.skin,
        "certify_edges": args.certify_edges,
        "velocity_source": velocity_source,
        "torch": str(torch.__version__),
        "mace": package_version("mace-torch"),
        "scope": "Fixed-cell short NVE and independent finite differences on the same energy model.",
    }
    try:
        options = {
            "model_paths": [args.model],
            "device": str(device),
            "default_dtype": args.dtype,
        }
        if args.head:
            options["head"] = args.head
        native = MACECalculator(**options)
        candidate = MACECalculator(**options)
        if args.head and (native.head != args.head or candidate.head != args.head):
            raise ValueError("requested model head was not selected")
        report["head"] = native.head
        if args.adjoint == "none":
            selection = nullcontext()
        else:
            from .transforms import optimized_model

            selection = optimized_model(
                candidate.models[0], args.adjoint, atomic_numbers=initial.numbers
            )
        with selection as metadata:
            report["optimization"] = metadata or {
                "mode": "none",
                "specialized_product_layers": 0,
            }
            fixed, issues = fixed_checks(initial, native, candidate, args, name)
            report["static"] = fixed
            md, md_issues = compare_trajectories(initial, native, candidate, args, name)
            report["nve"] = md
            issues.extend(md_issues)
        report["passed"] = not issues
        report["issues"] = issues
    except Exception as exc:
        report["passed"] = False
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        write_report(args.output, report)
        raise
    write_report(args.output, report)
    if not report["passed"]:
        raise SystemExit("Force/NVE checks failed; see " + str(args.output))
    print(f"Passed force and NVE checks for {len(initial)} atoms: {args.output}")
    return 0


if __name__ == "__main__":
    main()
