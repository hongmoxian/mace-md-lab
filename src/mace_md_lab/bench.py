"""Short, accuracy-gated comparison of fixed-cell NVE with one local MACE checkpoint."""

import argparse
from contextlib import contextmanager
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import platform
from statistics import median
from tempfile import NamedTemporaryFile
from time import perf_counter

import numpy as np
import torch
from ase import units
from ase.io import read
from ase.md.verlet import VelocityVerlet
from mace.calculators import MACECalculator

from .runtime import ResidentMACE

CAPTURE_PATHS = ("resident_capture", "resident_padded")


def file_hash(path):
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def host(tensor):
    return tensor.detach().cpu().numpy().copy()


def starting_structure(args):
    atoms = read(args.structure, index=args.frame)
    if len(atoms) == 0 or not np.all(atoms.pbc) or atoms.cell.rank != 3:
        raise ValueError("supply a nonempty 3D periodic fixed cell")
    if atoms.constraints:
        raise ValueError("this benchmark compares unconstrained NVE trajectories")
    masses = atoms.get_masses()
    if (
        not np.isfinite(atoms.positions).all()
        or not np.isfinite(atoms.cell.array).all()
        or not np.isfinite(masses).all()
        or np.any(masses <= 0)
    ):
        raise ValueError("invalid atomic coordinates, cell, or masses")
    if atoms.has("momenta"):
        source = "input momenta"
    else:
        source = "seeded Maxwell velocities (center-of-mass velocity removed for N > 1)"
        rng = np.random.default_rng(args.seed)
        v = rng.normal(size=(len(atoms), 3)) * np.sqrt(
            units.kB * args.temperature / masses[:, None]
        )
        if len(atoms) > 1:
            v -= np.average(v, axis=0, weights=masses)
        atoms.set_velocities(v)
    if not np.isfinite(atoms.get_velocities()).all():
        raise ValueError("nonfinite initial velocities")
    atoms.calc = None
    return atoms, source


class ASEPath:
    def __init__(self, initial, calculator, dt):
        self.atoms = initial.copy()
        self.atoms.calc = calculator
        self.dynamics = VelocityVerlet(self.atoms, timestep=dt * units.fs, logfile=None)
        # Pay the cost of the t=0 force before warming or timing either path.
        self.atoms.get_forces()

    def advance(self, steps):
        self.dynamics.run(steps)

    def snapshot(self):
        a = self.atoms
        return (
            a.positions.copy(),
            a.get_velocities().copy() * units.fs,
            a.get_potential_energy(),
            a.get_forces(),
            a.get_kinetic_energy(),
        )


class ResidentPath:
    def __init__(
        self,
        initial,
        calculator,
        dt,
        skin,
        certified=False,
        captured=False,
        padded=False,
    ):
        options = {"certify_edges": True} if certified else {}
        if captured:
            options["capture_forces"] = True
        if padded:
            options["pad_capture"] = True
        self.runner = ResidentMACE(initial.copy(), calculator, skin=skin, **options)
        self.dt = dt
        self.energy, self.forces = self.runner.evaluate()

    def advance(self, steps):
        for _ in range(steps):
            self.energy, self.forces = self.runner.step(self.dt)

    def snapshot(self):
        r = self.runner
        return (
            host(r.positions),
            host(r.velocities) * units.fs,
            self.energy.item(),
            host(self.forces),
            r.kinetic_energy().item(),
        )


@contextmanager
def make_path(name, initial, calc, args):
    # The native calculator can reuse results among calls. Different trials and
    # different stress settings must each start with a fresh t=0 evaluation.
    calc.reset()
    hook = None
    if name == "ase_ef":

        def forces_only(_model, positional, kwargs):
            kwargs["compute_stress"] = False
            return positional, kwargs

        hook = calc.models[0].register_forward_pre_hook(forces_only, with_kwargs=True)
    try:
        if name in ("ase_native", "ase_ef"):
            yield ASEPath(initial, calc, args.dt_fs)
        else:
            skin = 0.0 if name == "resident_rebuild" else args.skin
            captured = name in CAPTURE_PATHS
            certified = name == "resident_certified" or (
                captured and args.certify_edges
            )
            yield ResidentPath(
                initial,
                calc,
                args.dt_fs,
                skin,
                certified=certified,
                captured=captured,
                padded=name == "resident_padded",
            )
    finally:
        if hook is not None:
            hook.remove()
        calc.reset()


def topology_counts(path):
    if isinstance(path, ASEPath):
        return {}
    runner = path.runner
    counts = {"neighbor_rebuilds": runner.rebuilds}
    for key in (
        "certificate_hits",
        "edge_refreshes",
        "captures",
        "replays",
        "fallback_evaluations",
    ):
        if hasattr(runner, key):
            counts[key] = int(getattr(runner, key))
    return counts


def sample_trajectory(name, initial, calc, args):
    frames = {}
    sampled = {0, 1, args.steps // 2, args.steps}
    totals = []
    with make_path(name, initial, calc, args) as path:
        for step in range(args.steps + 1):
            if step:
                path.advance(1)
            snapshot = path.snapshot()
            if not all(np.isfinite(value).all() for value in snapshot):
                raise FloatingPointError(
                    f"{name} produced nonfinite state at step {step}"
                )
            totals.append((snapshot[2] + snapshot[4]) / len(initial))
            if step in sampled:
                frames[step] = snapshot
        stress = "stress" in calc.results if isinstance(path, ASEPath) else False
        counts = topology_counts(path)
    changes = np.asarray(totals) - totals[0]
    return frames, changes, stress, counts


def numerical_gates(args):
    if args.dtype == "float64":
        return dict(
            energy_eV_atom=1e-8,
            force_eV_A=1e-6,
            position_A=1e-6,
            velocity_A_fs=1e-7,
            nve_difference_eV_atom=1e-8,
        )
    return dict(
        energy_eV_atom=2e-4,
        force_eV_A=3e-3,
        position_A=5e-4,
        velocity_A_fs=1e-4,
        nve_difference_eV_atom=2e-4,
    )


def compare_trial(name, initial, calc, args, reference, drift_ref, thresholds):
    reasons = []
    frames, drift, stress, counts = sample_trajectory(name, initial, calc, args)
    if name == "ase_ef" and stress:
        reasons.append("stress remained enabled in the native energy-and-force control")
    # Only the initial geometry is exactly shared by independently integrated
    # paths. Also compare E/F on the final reference geometry without integration.
    fixed = initial.copy()
    fixed.set_positions(reference[args.steps][0])
    with make_path(name, fixed, calc, args) as frozen:
        final_geom = frozen.snapshot()
        if not all(np.isfinite(value).all() for value in final_geom):
            raise FloatingPointError(
                f"{name} produced nonfinite state on the final reference geometry"
            )
    errors_e = []
    errors_f = []
    for ref, trial in [(reference[0], frames[0]), (reference[args.steps], final_geom)]:
        errors_e.append(abs(ref[2] - trial[2]) / len(initial))
        errors_f.append(np.max(np.abs(ref[3] - trial[3])).item())
    maximum = {
        "energy_eV_atom": max(errors_e),
        "force_eV_A": max(errors_f),
        "position_A": max(
            float(np.abs(reference[t][0] - trial[0]).max())
            for t, trial in frames.items()
        ),
        "velocity_A_fs": max(
            float(np.abs(reference[t][1] - trial[1]).max())
            for t, trial in frames.items()
        ),
        "nve_difference_eV_atom": float(abs(drift - drift_ref).max()),
    }
    for key, val in maximum.items():
        if not math.isfinite(val) or val > thresholds[key]:
            reasons.append(f"{name}: {key} {val:.6g} exceeded {thresholds[key]:.6g}")
    metrics = {
        "max_error": maximum,
        "stress_returned": stress,
        "topology_counts_in_validation_including_initial": counts,
        "max_nve_excursion_meV_atom": 1000 * float(abs(drift).max()),
        "end_nve_change_meV_atom": 1000 * float(drift[-1]),
    }
    return metrics, reasons


def validate(initial, calc, paths, args):
    thresholds = numerical_gates(args)
    metrics = {}
    reasons = []
    reference, drift_ref, stress_ref, _ = sample_trajectory(
        "ase_native", initial, calc, args
    )
    excursion = np.max(np.abs(reference[args.steps][0] - reference[0][0])).item()
    if excursion < 1e-12:
        reasons.append("trajectory did not move; native ASE can reuse cached forces")
    metrics["ase_native"] = {
        "stress_returned": stress_ref,
        "max_nve_excursion_meV_atom": 1000 * float(abs(drift_ref).max()),
        "end_nve_change_meV_atom": 1000 * float(drift_ref[-1]),
        "maximum_initial_to_final_displacement_component_A": excursion,
    }
    controls_passed = not reasons
    for name in (p for p in paths if p != "ase_native"):
        try:
            metrics[name], issues = compare_trial(
                name, initial, calc, args, reference, drift_ref, thresholds
            )
        except Exception as exc:
            if name not in CAPTURE_PATHS:
                raise
            issues = [f"{type(exc).__name__}: {exc}"]
            metrics[name] = {"error": issues[0]}
        metrics[name]["passed"] = not issues
        reasons.extend(f"{name}: {reason}" for reason in issues)
        if name in CAPTURE_PATHS and issues:
            continue
        if issues:
            controls_passed = False
    return {
        "passed": not reasons,
        "controls_passed": controls_passed,
        "reasons": reasons,
        "absolute_tolerances": thresholds,
        "sampled_trajectory_steps": sorted(reference),
        "paths": metrics,
    }


def time_trial(name, initial, calc, args, device, rep):
    sync(device)
    setup_start = perf_counter()
    with make_path(name, initial, calc, args) as path:
        sync(device)
        setup_ms = 1000 * (perf_counter() - setup_start)
        path.advance(args.warmup)
        before = topology_counts(path)
        sync(device)
        start = perf_counter()
        path.advance(args.steps)
        sync(device)
        elapsed = perf_counter() - start
        raw = {
            "repeat": rep + 1,
            "seconds": elapsed,
            "setup_and_first_force_ms": setup_ms,
            "counts_before_timed_steps": before,
            "ms_step": 1000 * elapsed / args.steps,
        }
        after = topology_counts(path)
        raw.update({key: after[key] - before[key] for key in before})
        if name in CAPTURE_PATHS:
            raw["final_capacity_edges"] = int(path.runner.capture_capacity)
        return raw


def timings(initial, calc, paths, args, device):
    errors = {}
    records = {name: [] for name in paths}
    for rep in range(args.repeats):
        # Rotate path order to reduce systematic warm-cache / clock-order effects.
        for name in paths[rep % len(paths) :] + paths[: rep % len(paths)]:
            if name in errors:
                continue
            try:
                records[name].append(time_trial(name, initial, calc, args, device, rep))
            except Exception as exc:
                if name not in CAPTURE_PATHS:
                    raise
                errors[name] = f"{type(exc).__name__}: {exc}"
    result = {}
    for name, rows in records.items():
        result[name] = {"trials": rows}
        if name in errors:
            result[name]["error"] = errors[name]
        else:
            result[name]["median_ms_step"] = median(r["ms_step"] for r in rows)
    return result


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model", type=Path, required=True, help="trusted local .model checkpoint"
    )
    p.add_argument(
        "--structure", type=Path, required=True, help="ASE-readable periodic structure"
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--head", help="head name for a multihead checkpoint")
    p.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:N")
    p.add_argument("--dtype", default="float64", choices=("float32", "float64"))
    p.add_argument("--enable-oeq", action="store_true")
    p.add_argument("--enable-cueq", action="store_true")
    p.add_argument(
        "--pad-capture",
        action="store_true",
        help="also test CUDA replay with isolated padding; combine with --capture-forces for an ablation",
    )
    p.add_argument(
        "--capture-forces",
        action="store_true",
        help="also test CUDA graph force replay; match the requested edge policy",
    )
    p.add_argument(
        "--certify-edges",
        action="store_true",
        help="also test exact cutoff-gap certification of previous active edges",
    )
    p.add_argument(
        "--rebuild-ablation",
        action="store_true",
        help="also benchmark a resident runner that rebuilds the list every step",
    )
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, choices=(1, 2, 3), default=3)
    p.add_argument("--dt-fs", type=float, default=0.5)
    p.add_argument("--skin", type=float, default=0.5)
    p.add_argument(
        "--temperature",
        type=float,
        default=300.0,
        help="K; only used without input momenta",
    )
    p.add_argument("--seed", type=int, default=17)
    a = p.parse_args(argv)
    if (
        a.steps < 1
        or a.warmup < 1
        or not all(math.isfinite(v) for v in (a.dt_fs, a.skin, a.temperature))
    ):
        p.error("invalid integration parameters")
    if a.dt_fs <= 0 or a.skin <= 0 or a.temperature < 0 or a.seed < 0:
        p.error("dt and skin must be positive, temperature and seed nonnegative")
    if not a.model.is_file() or not a.structure.is_file():
        p.error("model and structure must be local files")
    if a.output.resolve() in (a.model.resolve(), a.structure.resolve()):
        p.error("output path must differ from the model and structure")
    return a


def write_report(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", dir=path.parent, suffix=".tmp", delete=False) as tmp:
        temp_path = Path(tmp.name)
        try:
            json.dump(result, tmp, indent=2, allow_nan=False)
            tmp.write("\n")
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise
    temp_path.replace(path)


def main(argv=None):
    args = arguments(argv)
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda") or (
        device.type == "cuda" and not torch.cuda.is_available()
    ):
        raise ValueError(f"unavailable device: {device}")
    if device.type == "cuda":
        torch.cuda.set_device(
            device.index if device.index is not None else torch.cuda.current_device()
        )
    elif args.enable_oeq or args.enable_cueq or args.capture_forces or args.pad_capture:
        raise ValueError("GPU acceleration flags require a CUDA device")
    atoms, velocity_source = starting_structure(args)
    paths = ["ase_native", "ase_ef", "resident_cached"]
    if args.rebuild_ablation:
        paths.append("resident_rebuild")
    if args.certify_edges:
        paths.append("resident_certified")
    if args.capture_forces:
        paths.append("resident_capture")
    if args.pad_capture:
        paths.append("resident_padded")
    report = {
        "schema_version": 1,
        "inputs": {
            "checkpoint": args.model.name,
            "checkpoint_sha256": file_hash(args.model),
            "structure": args.structure.name,
            "structure_sha256": file_hash(args.structure),
            "frame": args.frame,
            "atoms": len(atoms),
            "formula": atoms.get_chemical_formula(),
            "initial_velocities": velocity_source,
            "initial_temperature_K": float(atoms.get_temperature()),
        },
        "settings": {
            "device": str(device),
            "dtype": args.dtype,
            "steps": args.steps,
            "dt_fs": args.dt_fs,
            "warmup_steps": args.warmup,
            "repeats": args.repeats,
            "skin_A": args.skin,
            "seed": args.seed,
            "temperature_if_generated_K": args.temperature,
            "head": args.head,
            "enable_oeq": args.enable_oeq,
            "enable_cueq": args.enable_cueq,
            "capture_forces": args.capture_forces,
            "pad_capture": args.pad_capture,
            "certify_edges": args.certify_edges,
            "paths": paths,
        },
        "environment": {
            "machine": platform.machine(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "ase": package_version("ase"),
            "mace_torch": package_version("mace-torch"),
            "matscipy": package_version("matscipy"),
            "openequivariance": package_version("openequivariance"),
            "cuequivariance_torch": package_version("cuequivariance-torch"),
            "e3nn": package_version("e3nn"),
            "cuda_runtime": torch.version.cuda,
            "gpu": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "torch_threads": torch.get_num_threads(),
            "omp_threads": os.getenv("OMP_NUM_THREADS"),
        },
        "scope": "Single-rank fixed-cell NVE. Validation and host diagnostics excluded from warm timings. "
        "Includes all list rebuilds and endpoint device synchronization. Native ASE copies its outputs "
        "each step; the resident paths leave observations on device until read. ase_ef uses native "
        "MACECalculator with a scoped model hook that disables stress; weights and backend are shared.",
    }
    try:
        options = {
            "model_paths": [args.model],
            "device": str(device),
            "default_dtype": args.dtype,
        }
        if args.enable_oeq:
            options["enable_oeq"] = True
        if args.enable_cueq:
            options["enable_cueq"] = True
        if args.head:
            options["head"] = args.head
        calc = MACECalculator(**options)
        if args.head and calc.head != args.head:
            raise ValueError("requested MACE head was not selected")
        if args.enable_oeq and not getattr(calc, "_enable_oeq", False):
            raise ValueError("OEq requested but calculator did not activate it")
        if args.enable_cueq and not getattr(calc, "_enable_cueq", False):
            raise ValueError("CuEq requested but calculator did not activate it")
        model = calc.models[0]
        report["model"] = {
            "class": type(model).__name__,
            "selected_head": calc.head,
            "cutoff_A": float(calc.r_max),
            "parameter_count": sum(x.numel() for x in model.parameters()),
        }
        check = validate(atoms, calc, paths, args)
        report["validation"] = check
        if check["controls_passed"]:
            capture_paths = [name for name in paths if name in CAPTURE_PATHS]
            timed_paths = [
                name
                for name in paths
                if name not in CAPTURE_PATHS or check["paths"][name]["passed"]
            ]
            result = timings(atoms, calc, timed_paths, args, device)
            report["timing"] = result
            candidates = ["resident_cached"] + (
                ["resident_certified"] if args.certify_edges else []
            )
            candidates.extend(
                name
                for name in capture_paths
                if name in result and "error" not in result[name]
            )
            controls = [
                name
                for name in paths
                if name not in candidates and name not in CAPTURE_PATHS
            ]
            report["timing_notes"] = []
            matched_eager = (
                "resident_certified" if args.certify_edges else "resident_cached"
            )
            for name in capture_paths:
                if name not in candidates:
                    report["timing_notes"].append(
                        f"{name} failed validation or timing; its speedup is withheld"
                    )
            if capture_paths:
                report["timing_notes"].append(
                    f"{', '.join(capture_paths)} use the edge policy of {matched_eager}. Setup includes the "
                    "first force; timed steps include any recapture. Counter deltas cover only timed steps."
                )
            if args.rebuild_ablation and any(
                trial["neighbor_rebuilds"] != args.steps
                for trial in result["resident_rebuild"]["trials"]
            ):
                raise RuntimeError(
                    "per-step rebuild control did not rebuild at every timed step"
                )
            if all(
                trial["neighbor_rebuilds"] == args.steps
                for trial in result["resident_cached"]["trials"]
            ):
                report["timing_notes"].append(
                    "the cached driver rebuilt every step on this trajectory"
                )
            for candidate in candidates:
                if candidate == "resident_certified" and not any(
                    trial.get("certificate_hits", 0)
                    for trial in result[candidate]["trials"]
                ):
                    report["timing_notes"].append(
                        "no certificate hits during timing; its speedup is withheld"
                    )
                    continue
                baseline_names = (
                    controls
                    if candidate == "resident_cached"
                    else controls + ["resident_cached"]
                )
                if candidate in CAPTURE_PATHS:
                    rows = result[candidate]["trials"]
                    if any(
                        row.get("replays", 0) == 0 or row.get("fallback_evaluations", 0)
                        for row in rows
                    ):
                        report["timing_notes"].append(
                            f"a {candidate} trial lacked replay or fell back to eager; its speedup is withheld"
                        )
                        continue
                    if args.certify_edges:
                        baseline_names.append(matched_eager)
                    if (
                        candidate == "resident_padded"
                        and "speedup_resident_capture_vs" in report
                    ):
                        baseline_names.append("resident_capture")
                cost = result[candidate]["median_ms_step"]
                report[f"speedup_{candidate}_vs"] = {
                    name: result[name]["median_ms_step"] / cost
                    for name in baseline_names
                }
    except Exception as ex:
        report["failure"] = {"type": type(ex).__name__, "message": str(ex)}
        write_report(args.output, report)
        raise
    write_report(args.output, report)
    if not check["passed"] or any(
        "error" in row for row in report.get("timing", {}).values()
    ):
        raise SystemExit(
            "A requested path failed; see diagnostics and valid controls in "
            + str(args.output)
        )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "median_ms_step": {k: v["median_ms_step"] for k, v in result.items()},
                **{k: v for k, v in report.items() if k.startswith("speedup_")},
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    main()
