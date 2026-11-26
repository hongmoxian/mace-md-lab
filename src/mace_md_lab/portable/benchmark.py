"""Moving-molecule ablation of analytic edge geometry and CUDA Graph replay."""

import argparse
import hashlib
from importlib import import_module
import json
import math
from pathlib import Path
from statistics import median
import tempfile
from time import perf_counter

import numpy as np
import torch
from ase import units
from ase.io import read

from .runner import MolecularNVE


def native_ef(adapter, x):
    leaf = x.detach().requires_grad_(True)
    energy = adapter.native_energy(leaf).reshape(())
    force = -torch.autograd.grad(energy, leaf)[0]
    return energy.detach(), force.detach()


def reference(adapter, atoms, device, dtype, dt_fs, steps):
    x = torch.as_tensor(atoms.positions, device=device, dtype=torch.float64)
    v = torch.as_tensor(atoms.get_velocities(), device=device, dtype=torch.float64)
    m = torch.as_tensor(atoms.get_masses(), device=device, dtype=torch.float64)[:, None]
    dt = dt_fs * units.fs
    e, f = native_ef(adapter, x.to(dtype))
    frames = []
    for i in range(steps + 1):
        if i:
            vh = v + 0.5 * dt * f / m
            x = x + dt * vh
            e, f = native_ef(adapter, x.to(dtype))
            v = vh + 0.5 * dt * f / m
        frames.append(
            (
                x.clone(),
                v.clone(),
                e.clone(),
                f.clone(),
                e + 0.5 * (m * v.square()).sum(),
            )
        )
    return frames


def validate(md, ref, adapter, n, dtype, direction):
    fe, ee, pe, ve, ne = 0.0, 0.0, 0.0, 0.0, 0.0
    t0 = md.e + md.kinetic()
    for i, (x, v, e, f, total) in enumerate(ref):
        if i:
            md.step()
        if not all(
            torch.isfinite(value).all().item()
            for value in (md.e, md.f, md.x, md.v, e, f, x, v, total)
        ):
            raise FloatingPointError(f"nonfinite molecular trajectory at step {i}")
        ee = max(ee, float(abs(md.e - e)) / n)
        fe = max(fe, float(abs(md.f - f).max()))
        pe = max(pe, float(abs(md.x - x).max()))
        ve = max(ve, float(abs(md.v - v).max()) * units.fs)
        ne = max(ne, float(abs((md.e + md.kinetic() - t0) - (total - ref[0][4]))) / n)
    # This frame is precisely shared, even when two integrated coordinate paths diverge.
    x = ref[0][0].to(dtype)
    e0, f0 = md.call(x, md.graph.update(x))
    fproj = float((f0 * direction).sum())
    hs = (0.001, 0.0003) if dtype == torch.float64 else (0.02, 0.05)
    errors = []
    for h in hs:
        ep = adapter.native_energy(x + h * direction)
        em = adapter.native_energy(x - h * direction)
        errors.append(abs(float(((em - ep) / (2 * h)).detach()) - fproj))
    gates = (
        [1e-8, 1e-6, 1e-6, 1e-6, 1e-8]
        if dtype == torch.float64
        else [2e-4, 3e-3, 5e-4, 1e-4, 2e-4]
    )
    values = [ee, fe, pe, ve, ne]
    fixed = float((f0 - ref[0][3]).abs().max())
    fdlimit = (5e-4 if dtype == torch.float64 else 0.15) * max(1.0, abs(fproj))
    passed = (
        all(math.isfinite(val) and val <= tol for val, tol in zip(values, gates))
        and fixed <= gates[1]
        and min(errors) <= fdlimit
    )
    return dict(
        passed=passed,
        max_energy_error_eV_atom=ee,
        max_force_component_error_eV_A=fe,
        same_initial_force_error_eV_A=fixed,
        max_position_error_A=pe,
        max_velocity_error_A_fs=ve,
        max_NVE_change_difference_eV_atom=ne,
        fd_displacements_A=hs,
        fd_errors_eV_A=errors,
        fd_tolerance_eV_A=fdlimit,
        force_projection_eV_A=fproj,
    )


def dump(path, data):
    path.parent.mkdir(exist_ok=True, parents=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write("\n")
        name = f.name
    Path(name).replace(path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--framework", choices=("mace", "schnet", "painn"), required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--structure", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--dt-fs", type=float, default=0.25)
    p.add_argument("--repeats", type=int, default=5)
    args = p.parse_args(argv)
    if (
        min(args.steps, args.warmup, args.repeats) < 1
        or not math.isfinite(args.dt_fs)
        or args.dt_fs <= 0
        or args.output.resolve() in (args.model.resolve(), args.structure.resolve())
    ):
        p.error("provide valid integration settings and a dedicated output file")
    atoms = read(args.structure, index=0)
    if len(atoms) < 2 or atoms.pbc.any() or atoms.constraints:
        raise ValueError("benchmark requires a nonperiodic, unconstrained molecule")
    rng = np.random.default_rng(17)
    v = rng.normal(size=(len(atoms), 3)) * np.sqrt(
        units.kB * 300 / atoms.get_masses()[:, None]
    )
    v -= np.average(v, axis=0, weights=atoms.get_masses())
    atoms.set_velocities(v)
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    if device.type == "cuda":
        torch.cuda.set_device(
            device.index if device.index is not None else torch.cuda.current_device()
        )

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    params = dict(
        framework=args.framework,
        checkpoint=args.model.name,
        checkpoint_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
        structure=args.structure.name,
        structure_sha256=hashlib.sha256(args.structure.read_bytes()).hexdigest(),
        formula=atoms.get_chemical_formula(),
        atoms=len(atoms),
        dtype=args.dtype,
        gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        steps=args.steps,
        repeats=args.repeats,
        warmup=args.warmup,
        dt_fs=args.dt_fs,
        velocity_seed=17,
        torch=str(torch.__version__),
        scope="Full synchronized velocity-Verlet on one finite molecule. "
        "Each timed window excludes setup and warmup and includes graph selection, force evaluation and integration. "
        "All four paths share weights, code for topology and units. Native upstream energy is a separate reference.",
    )
    report = dict(**params, validation={}, trials={})
    try:
        module = import_module("mace_md_lab.portable." + args.framework)
        adapter = module.load_adapter(
            args.model,
            device,
            dtype,
            torch.tensor(atoms.numbers, dtype=torch.long, device=device),
        )
        report["model"] = adapter.metadata
        report["cutoff_A"] = float(adapter.cutoff)
        ref = reference(adapter, atoms, device, dtype, args.dt_fs, args.steps)
        direction = rng.normal(size=(len(atoms), 3))
        direction -= direction.mean(axis=0)
        direction = torch.as_tensor(
            direction / np.linalg.norm(direction), dtype=dtype, device=device
        )
        paths = [
            (m, c)
            for c in ([False, True] if device.type == "cuda" else [False])
            for m in ("native", "analytic")
        ]

        def build(m, c):
            return MolecularNVE(atoms, adapter, dtype, device, m, c, args.dt_fs)

        def key(m, c):
            return m + ("_capture" if c else "_eager")

        for m, c in paths:
            k = key(m, c)
            dyn = build(m, c)
            check = validate(dyn, ref, adapter, len(atoms), dtype, direction)
            report["validation"][k] = check
            if not check["passed"]:
                raise ValueError(k + " numerical validation failed")
            report["trials"][k] = []
            del dyn
        for j in range(args.repeats):
            for m, c in paths[j % len(paths) :] + paths[: j % len(paths)]:
                sync()
                t = perf_counter()
                dyn = build(m, c)
                sync()
                setup = (perf_counter() - t) * 1000
                for _ in range(args.warmup):
                    dyn.step()
                n0 = dyn.graph.refreshes
                c0 = dyn.call.captures
                r0 = dyn.call.replays
                sync()
                t = perf_counter()
                for _ in range(args.steps):
                    dyn.step()
                sync()
                dt = (perf_counter() - t) * 1000 / args.steps
                report["trials"][key(m, c)].append(
                    dict(
                        ms_step=dt,
                        setup_and_first_force_ms=setup,
                        timed_graph_refreshes=dyn.graph.refreshes - n0,
                        timed_captures=dyn.call.captures - c0,
                        timed_replays=dyn.call.replays - r0,
                        final_directed_edges=dyn.graph.edges.shape[1],
                    )
                )
                del dyn
        report["median_ms_step"] = {
            k: median(r["ms_step"] for r in v) for k, v in report["trials"].items()
        }
        report["passed"] = True
    except Exception as e:
        report["passed"] = False
        report["failure"] = f"{type(e).__name__}: {e}"
        raise
    finally:
        dump(args.output, report)
    print(report["median_ms_step"])


if __name__ == "__main__":
    main()
