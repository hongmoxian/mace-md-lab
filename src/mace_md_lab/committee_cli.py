"""Compare committee inference on exactly the same frames, weights and observables."""

from __future__ import annotations

import argparse
from importlib.metadata import version
from itertools import islice
import json
import math
import os
from pathlib import Path
import platform
import tempfile
from time import perf_counter

import numpy as np
import torch
from ase import units
from ase.md.verlet import VelocityVerlet
from ase.io import iread
from mace.calculators import MACECalculator

from .committee import MACECommittee, disagreement, native_observation


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", type=Path, required=True)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--head")
    parser.add_argument("--skin", type=float, default=0.5)
    parser.add_argument(
        "--certify-edges",
        action="store_true",
        help="Reuse exact filtered edges while certified by the cutoff gap",
    )
    parser.add_argument(
        "--check-driver-steps",
        type=int,
        default=0,
        help="Additional short driver-only NVE and sparse observation validation",
    )
    parser.add_argument(
        "--observe-every",
        type=int,
        default=2,
        help="Observation cadence in the driver check",
    )
    parser.add_argument("--dt-fs", type=float, default=0.5)
    parser.add_argument(
        "--stride", type=int, default=1, help="Observe every k-th input frame"
    )
    parser.add_argument("--limit", "--max-frames", dest="limit", type=int, default=8)
    parser.add_argument("--repeats", "--repeat", dest="repeats", type=int, default=3)
    parser.add_argument(
        "--warmup", type=int, default=1, help="Untimed passes on the first frame"
    )
    parser.add_argument(
        "--geometry", choices=["auto", "require", "off"], default="auto"
    )
    parser.add_argument(
        "--force-atol", type=float, help="Max component or disagreement error, eV/Å"
    )
    parser.add_argument(
        "--energy-atol", type=float, help="Max total energy error per atom, eV"
    )
    args = parser.parse_args()
    if min(args.limit, args.stride, args.repeats) < 1 or args.warmup < 0:
        parser.error(
            "stride, frame limit and repeats must be positive; warmup must be nonnegative"
        )
    if (
        args.check_driver_steps < 0
        or args.observe_every < 1
        or not math.isfinite(args.dt_fs)
        or args.dt_fs <= 0
    ):
        parser.error("invalid driver-check steps, observation cadence or timestep")
    if len(args.models) < 2 or len({p.resolve() for p in args.models}) != len(
        args.models
    ):
        parser.error("at least two distinct checkpoint paths are required")
    if not math.isfinite(args.skin) or args.skin < 0:
        parser.error("skin must be nonnegative and finite")
    for tol in (args.force_atol, args.energy_atol):
        if tol is not None and (not math.isfinite(tol) or tol < 0):
            parser.error("tolerances must be finite and nonnegative")
    for path in [*args.models, args.trajectory]:
        if not path.is_file():
            parser.error(f"local file does not exist: {path}")
    if args.output.resolve() in {p.resolve() for p in [*args.models, args.trajectory]}:
        parser.error("output path must differ from the input paths")
    return args


def _read_frames(args):
    selected = islice(
        enumerate(iread(args.trajectory, index=":")), 0, None, args.stride
    )
    indexed = list(islice(selected, args.limit))
    if not indexed:
        raise ValueError("trajectory is empty")
    return indexed


def _validate_frames(indexed, calculator):
    first = indexed[0][1]
    for index, frame in indexed:
        if not (
            np.array_equal(frame.numbers, first.numbers)
            and np.array_equal(frame.cell.array, first.cell.array)
            and np.array_equal(frame.pbc, first.pbc)
        ):
            raise ValueError(
                f"frame {index} changes the fixed cell, periodicity or atom ordering"
            )
        if not np.isfinite(frame.positions).all():
            raise ValueError(f"invalid positions in frame {index}")
        # Static input attributes belong to the resident template.
        for old, new, keys in (
            (
                first.info,
                frame.info,
                set(calculator.info_keys) | set(calculator.info_keys.values()),
            ),
            (
                first.arrays,
                frame.arrays,
                set(calculator.arrays_keys)
                | set(calculator.arrays_keys.values())
                | {calculator.charges_key},
            ),
        ):
            for key in keys:
                if (key in old) != (key in new) or not np.array_equal(
                    old.get(key), new.get(key)
                ):
                    raise ValueError(
                        f"frame {index} changes static model input {key!r}"
                    )


def _write_json(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, encoding="utf-8", delete=False
    ) as f:
        partial = f.name
        try:
            json.dump(report, f, indent=2, allow_nan=False)
            f.write("\n")
        except BaseException:
            os.unlink(partial)
            raise
    os.replace(partial, path)


def _check_driver(committee, calc, frame, args, etol, ftol):
    """Compare a sparse observation run against a continuously independent driver."""
    committee.reset(frame)
    ref = frame.copy()
    leader = committee.members[0]
    leader.reset()
    ref.calc = leader
    dyn = VelocityVerlet(ref, timestep=args.dt_fs * units.fs, logfile=None)
    dx = dv = dforce_driver = de_driver = de_members = df_members = dsigma = 0.0
    observed = []
    for step in range(1, args.check_driver_steps + 1):
        observe = step % args.observe_every == 0 or step == args.check_driver_steps
        e, f, obs = committee.step(args.dt_fs, observe=observe)
        dyn.run(1)
        pos = committee.driver.positions.detach().cpu().numpy()
        vel = committee.driver.velocities.detach().cpu().numpy()
        dx = max(dx, float(np.abs(pos - ref.positions).max()))
        dv = max(dv, float(np.abs(vel - ref.get_velocities()).max()))
        de_driver = max(
            de_driver, abs(e.item() - ref.get_potential_energy()) / len(ref)
        )
        dforce_driver = max(
            dforce_driver, float(np.abs(f.cpu().numpy() - ref.get_forces()).max())
        )
        if observe:
            probe = frame.copy()
            probe.positions[:] = pos
            en, fn = native_observation(calc, probe)
            de_members = max(
                de_members,
                float(np.abs(obs.energies.cpu().numpy() - en).max() / len(ref)),
            )
            df_members = max(
                df_members, float(np.abs(obs.forces.cpu().numpy() - fn).max())
            )
            dsigma = max(
                dsigma,
                float(
                    np.abs(
                        obs.atom_force_disagreement.cpu().numpy() - disagreement(fn)
                    ).max()
                ),
            )
            observed.append(step)
        elif obs is not None:
            raise AssertionError(
                "unexpected committee evaluation between observation steps"
            )
    # FP32 trajectories can separate slightly through repeated force rounding.
    xtol = 1e-5 if args.dtype == "float32" else 1e-9
    return {
        "steps": args.check_driver_steps,
        "observation_steps": observed,
        "dt_fs": args.dt_fs,
        "max_position_error_A": dx,
        "max_velocity_error_A_per_ASE_time_unit": dv,
        "max_driver_energy_error_eV_atom": de_driver,
        "max_driver_force_error_eV_A": dforce_driver,
        "max_member_energy_error_eV_atom": de_members,
        "max_member_force_error_eV_A": df_members,
        "max_atom_disagreement_error_eV_A": dsigma,
        "position_and_velocity_atol_in_reported_units": xtol,
        "passed": bool(
            dx <= xtol
            and dv <= xtol
            and max(de_driver, de_members) <= etol
            and max(dforce_driver, df_members, dsigma) <= ftol
        ),
    }


def run(args):
    indexed = _read_frames(args)
    frames = [frame for _, frame in indexed]
    options = {} if args.head is None else {"head": args.head}
    calc = MACECalculator(
        model_paths=[str(p) for p in args.models],
        device=args.device,
        default_dtype=args.dtype,
        **options,
    )
    if args.head is not None and calc.head != args.head:
        raise ValueError(f"checkpoint does not support requested head {args.head!r}")
    _validate_frames(indexed, calc)
    committee = MACECommittee(
        frames[0],
        calc,
        skin=args.skin,
        geometry=args.geometry,
        certify_edges=args.certify_edges,
    )
    device = committee.driver.device
    methods = ["official_committee", "cached_topology_ef"]
    if committee.geometry is not None:
        methods.append("shared_geometry_ef")

    def evaluate(method, frame):
        if method == "official_committee":
            return native_observation(calc, frame)
        obs = committee.observe(
            frame.positions, share_geometry=method == "shared_geometry_ef"
        )
        return obs.energies.cpu().numpy(), obs.forces.cpu().numpy()

    force_tol = (
        args.force_atol
        if args.force_atol is not None
        else (2e-4 if args.dtype == "float32" else 1e-8)
    )
    energy_tol = (
        args.energy_atol
        if args.energy_atol is not None
        else (2e-5 if args.dtype == "float32" else 1e-8)
    )
    checks = {}
    for mode in methods[1:]:
        checks[mode] = {
            "max_energy_error_eV_per_atom_by_member": np.zeros(len(args.models)),
            "max_force_component_error_eV_A_by_member": np.zeros(len(args.models)),
            "max_atom_disagreement_error_eV_A": 0.0,
        }
    records = []
    for index, frame in indexed:
        ref_e, ref_f = evaluate(methods[0], frame)
        ref_e, ref_f = ref_e.astype(float), ref_f.astype(float)
        if ref_f.shape != (len(args.models), len(frame), 3) or ref_e.shape != (
            len(args.models),
        ):
            raise RuntimeError(
                f"upstream committee returned unexpected member dimensions for frame {index}"
            )
        if not (np.isfinite(ref_e).all() and np.isfinite(ref_f).all()):
            raise ValueError(f"nonfinite native MACE output in frame {index}")
        ref_dev = disagreement(ref_f)
        obs_record = {
            "source_frame": index,
            "native_energies_eV": ref_e.tolist(),
            "native_max_force_disagreement_eV_A": float(ref_dev.max()),
            "native_max_atom_index": int(ref_dev.argmax()),
        }
        for mode in methods[1:]:
            e, f = evaluate(mode, frame)
            if e.shape != ref_e.shape or f.shape != ref_f.shape:
                raise RuntimeError(f"{mode} output and reference dimensions differ")
            if not (np.isfinite(e).all() and np.isfinite(f).all()):
                raise ValueError(f"nonfinite {mode} output in frame {index}")
            f = f.astype(float)
            dev = disagreement(f)
            obs_record[mode + "_max_force_disagreement_eV_A"] = float(dev.max())
            metrics = checks[mode]
            for key, err in (
                (
                    "max_energy_error_eV_per_atom_by_member",
                    np.abs(e - ref_e) / len(frame),
                ),
                (
                    "max_force_component_error_eV_A_by_member",
                    np.abs(f - ref_f).max(axis=(1, 2)),
                ),
            ):
                metrics[key] = np.maximum(metrics[key], err)
            metrics["max_atom_disagreement_error_eV_A"] = max(
                float(np.abs(dev - ref_dev).max()),
                metrics["max_atom_disagreement_error_eV_A"],
            )
        records.append(obs_record)
    for metric in checks.values():
        de = metric["max_energy_error_eV_per_atom_by_member"]
        df = metric["max_force_component_error_eV_A_by_member"]
        metric["passed"] = bool(
            de.max() <= energy_tol
            and df.max() <= force_tol
            and metric["max_atom_disagreement_error_eV_A"] <= force_tol
        )
        metric["max_energy_error_eV_per_atom_by_member"] = de.tolist()
        metric["max_force_component_error_eV_A_by_member"] = df.tolist()
    passed = all(row["passed"] for row in checks.values())
    driver_check = None
    if args.check_driver_steps and passed:
        driver_check = _check_driver(
            committee, calc, frames[0], args, energy_tol, force_tol
        )
        passed = passed and driver_check["passed"]
    report = {
        "schema_version": 1,
        "benchmark": "same_frame_committee",
        "models": [p.name for p in args.models],
        "model_state_sha256": committee.members_sha256,
        "trajectory": args.trajectory.name,
        "frames": len(frames),
        "atoms": len(frames[0]),
        "input_stride": args.stride,
        "head": calc.head,
        "skin_A": args.skin,
        "cutoff_A": float(calc.r_max),
        "dtype": args.dtype,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch": str(torch.__version__),
        "mace": version("mace-torch"),
        "ase": version("ase"),
        "platform": platform.platform(),
        "torch_num_threads": torch.get_num_threads(),
        "energy_atol_eV_per_atom": energy_tol,
        "force_atol_eV_A": force_tol,
        "checks": checks,
        "passed": passed,
        "observations": records,
        "shared_geometry_inactive_reason": committee.geometry_reason,
        "driver_check": driver_check,
        "certify_edges": args.certify_edges,
        "timings": {},
        "warmup_first_frame": args.warmup,
        "repeats": args.repeats,
        "timing_scope": (
            "Host positions to member E/F host arrays plus the same NumPy vector-force "
            "disagreement. Native MACECalculator also computes stress; the two EF-only "
            "methods isolate intra-committee geometry sharing. Checkpoint loading, "
            "template creation, trajectory I/O and warmup are excluded."
        ),
    }
    if passed:
        timings = {
            mode: {"raw_ms_per_frame": [], "rebuilds_per_pass": []} for mode in methods
        }
        for rep in range(args.repeats):
            # Alternate method order; start each pass with independent resident topology state.
            offset = rep % len(methods)
            for mode in methods[offset:] + methods[:offset]:
                if mode != methods[0]:
                    committee.reset(frames[0])
                for _ in range(args.warmup):
                    _, f = evaluate(mode, frames[0])
                    disagreement(f)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                n0 = committee.driver.rebuilds
                start = perf_counter()
                for frame in frames:
                    _, f = evaluate(mode, frame)
                    disagreement(f)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed = (perf_counter() - start) * 1000 / len(frames)
                timings[mode]["raw_ms_per_frame"].append(elapsed)
                timings[mode]["rebuilds_per_pass"].append(
                    len(frames)
                    if mode == methods[0]
                    else committee.driver.rebuilds - n0
                )
        for mode in methods:
            timings[mode]["median_ms_per_frame"] = float(
                np.median(timings[mode]["raw_ms_per_frame"])
            )
        for mode in methods[1:]:
            timings[mode]["speedup_vs_official"] = (
                timings[methods[0]]["median_ms_per_frame"]
                / timings[mode]["median_ms_per_frame"]
            )
        if "shared_geometry_ef" in timings:
            timings["shared_geometry_ef"]["speedup_vs_cached_ef"] = (
                timings["cached_topology_ef"]["median_ms_per_frame"]
                / timings["shared_geometry_ef"]["median_ms_per_frame"]
            )
        report["timings"] = timings
    _write_json(args.output, report)
    return report


def main():
    args = _arguments()
    report = run(args)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "passed": report["passed"],
                "timings": report["timings"],
            },
            indent=2,
        )
    )
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
