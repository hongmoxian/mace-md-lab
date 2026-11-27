"""Fixed-density liquid Cu, using the qualified MACE weights and the combined XL kernels."""

import argparse
from contextlib import ExitStack
from hashlib import sha256
import json
import os
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
from ase import units
from ase.build import bulk
from ase.io import write

from benchmark_periodic_official import OfficialMD, ase_evaluate, xl_routes
from benchmark_xl_algorithms import read_routes, setup_engine
from mace_md_lab.dynamics import ArrayVerlet
from mace_md_lab.portable.periodic import official_calculator, periodic_evaluator
from mace_md_lab.portable.qualified_models import CHECKPOINTS
from mace_md_lab.xl_fixed_step import PreparedXL


def density(temperature):
    # Brillo & Egry (2003), Table IV (rho in g/cm^3, liquidus 1084 deg C).
    return 7.90 - 0.000765 * (temperature - 1357.15)


def starting_atoms(n, temperature, rng):
    copies = round((n / 4) ** (1 / 3))
    assert 4 * copies**3 == n
    atoms = bulk("Cu", "fcc", a=3.62, cubic=True).repeat((copies,) * 3)
    side = (atoms.get_masses().sum() * 1.66053906660 / density(temperature)) ** (1 / 3)
    atoms.set_cell(np.eye(3) * side, scale_atoms=True)
    momenta = rng.normal(size=(n, 3)) * np.sqrt(
        atoms.get_masses()[:, None] * units.kB * 2800
    )
    momenta -= momenta.mean(axis=0)
    atoms.set_momenta(momenta)
    return atoms


def configure(a):
    from symmetrix import load_backend

    os.environ["SYMMETRIX_NEIGHBOR_BACKEND"] = "kokkos"
    load_backend("cuda12-sm70")
    compact = sha256(a.symmetrix_json.read_bytes()).hexdigest()
    kernel = read_routes(a.routes_json)["locked"]
    engine, modules = setup_engine(a, kernel, compact, [29])
    return engine, compact


def kinetic(path):
    return 0.5 * np.sum(path.momenta**2 * path.inverse_mass)


def rescale(path, temperature, rng, elapsed_fs, tau_fs):
    # Exact stochastic velocity-rescaling step (Bussi et al., 2007).
    dof = 3 * (len(path.positions) - 1)
    mean_k = dof * units.kB * temperature / 2
    c = np.exp(-elapsed_fs / tau_fs)
    amplitude = np.sqrt((1 - c) * mean_k / (dof * kinetic(path)))
    longitudinal = np.sqrt(c) + amplitude * rng.normal()
    alpha = np.sign(longitudinal) * np.sqrt(
        longitudinal**2 + amplitude**2 * rng.chisquare(dof - 1)
    )
    path.momenta *= alpha


def thermalize(path, temperature, time_ps, dt, rng):
    chunk = 10
    for _ in range(round(time_ps * 1000 / dt / chunk)):
        path.advance(chunk)
        rescale(path, temperature, rng, chunk * dt, 100.0)


def fcc_order(positions, length, copies):
    # Average squared magnitude at the three {200} reciprocal vectors.
    phase = 4 * np.pi * copies * positions / length
    return float(np.mean(abs(np.exp(1j * phase).mean(axis=0)) ** 2))


def json_file(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def atom_state(atoms, path):
    result = atoms.copy()
    result.positions = path.positions
    result.set_momenta(path.momenta)
    return result


def observables_error(result, target, n):
    e, f, s = result
    er, fr, sr = target
    return [abs(e - er) / n, np.max(abs(f - fr)), np.max(abs(s - sr))]


def pilot(atoms, a, evaluate, rng):
    import torch

    calc = official_calculator("mace", a.model)
    ref = ase_evaluate(atoms, calc)
    with ExitStack() as stack:
        independent = stack.enter_context(periodic_evaluator("mace", a.model, atoms))
        entries = {
            "official": (lambda: OfficialMD(atoms, calc, a.dt_fs), ref),
            "independent": (
                lambda: ArrayVerlet(atoms, independent, a.dt_fs),
                independent,
            ),
        }
        entries.update(xl_routes(a, atoms))
        state = ArrayVerlet(atoms, evaluate, a.dt_fs)
        frames = [state.positions.copy()]
        for _ in range(4):
            state.advance(12)
            frames.append(state.positions.copy())
        image = frames[-1].copy()
        image[0] += atoms.cell.array[0]
        frames += [image, frames[0]]
        targets = [ref(pos) for pos in frames]
        errors = {
            tag: np.max(
                [
                    observables_error(ev(pos), target, len(atoms))
                    for pos, target in zip(frames, targets)
                ],
                axis=0,
            ).tolist()
            for tag, (_, ev) in entries.items()
        }
        for key, err in errors.items():
            if not np.all(np.asarray(err) <= [1e-8, 1e-6, 1e-7]):
                raise RuntimeError(
                    f"{key} exceeded the common-geometry E/F/S gate: {err}"
                )
        rows = []
        names = list(entries)
        for i in range(4):
            row = {}
            for name in names[i:] + names[:i]:
                run = entries[name][0]()
                run.advance(5)
                torch.cuda.synchronize()
                start = perf_counter()
                run.advance(50)
                torch.cuda.synchronize()
                row[name] = (perf_counter() - start) * 1000 / 50
            rows.append(row)
        times = {k: median([x[k] for x in rows]) for k in names}
    drift = {}
    for dt in (1.0, a.dt_fs):
        state = ArrayVerlet(atoms, evaluate, dt)
        energies = [(state.energy + kinetic(state)) / len(atoms)]
        for _ in range(100):
            state.advance(round(20 / dt))
            energies.append((state.energy + kinetic(state)) / len(atoms))
        slope = np.polyfit(np.arange(101) * 0.020, energies, 1)[0]
        drift[str(dt)] = {
            "slope_meV_atom_ps": float(1000 * slope),
            "span_meV_atom": float(1000 * np.ptp(energies)),
        }
    if (
        drift[str(a.dt_fs)]["span_meV_atom"] > 2
        or abs(drift[str(a.dt_fs)]["slope_meV_atom_ps"]) > 0.3
    ):
        raise RuntimeError(f"pilot timestep drift: {drift}")
    print("PILOT", times, "NVE", drift, flush=True)
    return {
        "times_ms_step": times,
        "raw_ms_step": rows,
        "common_geometry_errors_EFS": errors,
        "nve_2ps": drift,
        "warmup": 5,
        "steps": 50,
        "repeats": 4,
        "gpu": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "symmetrix-json", "product-dir", "m0-dir", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument(
        "--routes-json", type=Path, default=Path("examples/native-xl-locked.json")
    )
    p.add_argument("--temperature", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--atoms", type=int, choices=[108, 256], default=108)
    p.add_argument("--ps", type=float, default=100)
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--dt-fs", type=float, default=2.0)
    p.set_defaults(harmonic_storage="y-only-direct-v1", skin=1.0)
    a = p.parse_args()
    if (
        not np.isfinite(a.dt_fs)
        or a.dt_fs <= 0
        or not all(np.isclose(t / a.dt_fs, round(t / a.dt_fs)) for t in (20, 50))
    ):
        p.error("the time step must be positive and divide both 20 and 50 fs")
    if (
        a.temperature <= 0
        or not np.isfinite(a.ps)
        or a.ps <= 0
        or not np.isclose(a.ps * 20, round(a.ps * 20))
    ):
        p.error(
            "supply a positive temperature and a production duration in multiples of 0.05 ps"
        )
    a.out.mkdir(exist_ok=True, parents=True)
    stamp = a.out / "result.json"
    if stamp.exists():
        raise FileExistsError(stamp)
    t0 = perf_counter()
    if sha256(a.model.read_bytes()).hexdigest() != CHECKPOINTS["mace"]:
        raise RuntimeError("checkpoint mismatch")
    rng = np.random.default_rng(a.seed)
    atoms = starting_atoms(a.atoms, a.temperature, rng)
    engine, compact = configure(a)
    # A fresh prepared graph is needed if an independent route used this engine.
    prepared = PreparedXL(atoms, engine, a.skin)
    state = ArrayVerlet(atoms, prepared.evaluate, a.dt_fs)
    init_s = perf_counter() - t0
    ts = perf_counter()
    thermalize(state, 2800, 5.0, a.dt_fs, rng)
    thermalize(state, a.temperature, 8.0, a.dt_fs, rng)
    eq_s = perf_counter() - ts
    start_atoms = atom_state(atoms, state)
    copies = round((len(atoms) / 4) ** (1 / 3))
    order = fcc_order(state.positions, atoms.cell[0, 0], copies)
    if order > 0.20:
        raise RuntimeError(f"preparation retained FCC order: {order}")
    write(a.out / "start.extxyz", start_atoms)
    benchmark = {}
    if a.pilot:
        benchmark = pilot(start_atoms, a, prepared.evaluate, rng)
        json_file(a.out / "pilot.json", benchmark)
        if (
            min(benchmark["times_ms_step"], key=benchmark["times_ms_step"].get)
            != "xl_plus"
        ):
            raise RuntimeError(
                "combined route was not the fastest; review production selection"
            )
        prepared = PreparedXL(atoms, engine, a.skin)
        state = ArrayVerlet(start_atoms, prepared.evaluate, a.dt_fs)
    stride = round(50 / a.dt_fs)
    nsamples = round(a.ps * 1000 / (stride * a.dt_fs))
    xyz = [state.positions.copy()]
    energies, temps, stress, orders = [], [], [], []
    selected = {i for i in np.linspace(0, nsamples, 6, dtype=int)}
    saved = {}

    def sample(index):
        energies.append((state.energy + kinetic(state)) / len(atoms))
        temps.append(2 * kinetic(state) / ((3 * len(atoms) - 3) * units.kB))
        stress.append(np.asarray(state.stress).copy())
        orders.append(fcc_order(state.positions, atoms.cell[0, 0], copies))
        if index in selected:
            saved[index] = (state.energy, state.forces.copy(), state.stress.copy())

    sample(0)
    start = perf_counter()
    for i in range(1, nsamples + 1):
        state.advance(stride)
        rescale(state, a.temperature, rng, stride * a.dt_fs, 1000.0)
        xyz.append(state.positions.copy())
        sample(i)
        if i % 400 == 0:
            print(
                "STEP",
                i * stride,
                "ps",
                i * stride * a.dt_fs / 1000,
                "s",
                round(perf_counter() - start, 2),
                flush=True,
            )
    md_s = perf_counter() - start
    np.savez_compressed(
        a.out / "samples.npz",
        xyz=np.array(xyz, dtype=np.float32),
        t_ps=np.arange(nsamples + 1) * stride * a.dt_fs / 1000,
        temp_K=temps,
        e_total_atom=energies,
        stress=stress,
        fcc_order=orders,
        box=atoms.cell.array,
    )
    ref = ase_evaluate(atoms, official_calculator("mace", a.model))
    errors = np.max(
        [observables_error(saved[i], ref(xyz[i]), len(atoms)) for i in sorted(saved)],
        axis=0,
    )
    if not np.all(errors <= [1e-8, 1e-6, 1e-7]):
        raise RuntimeError(f"production common-geometry gate failed: {errors}")
    write(a.out / "final.extxyz", atom_state(atoms, state))
    output = {
        "T": a.temperature,
        "N": len(atoms),
        "seed": a.seed,
        "dt_fs": a.dt_fs,
        "ps": a.ps,
        "density_g_cm3": density(a.temperature),
        "box_A": float(atoms.cell[0, 0]),
        "compact_sha256": compact,
        "source_sha256": CHECKPOINTS["mace"],
        "target_equil_ps": 8,
        "melt_ps": 5,
        "tau_production_fs": 1000,
        "init_s": init_s,
        "equil_s": eq_s,
        "production_s": md_s,
        "steps": nsamples * stride,
        "ms_step": md_s * 1000 / (nsamples * stride),
        "mean_T": float(np.mean(temps)),
        "mean_fcc_order": float(np.mean(orders)),
        "max_err_EFS": errors.tolist(),
        "tested_frames": len(saved),
        "rebuilds": prepared.rebuilds,
        "benchmark": benchmark,
        "process_body_s": perf_counter() - t0,
    }
    json_file(stamp, output)
    print(
        "COMPLETE",
        a.temperature,
        a.seed,
        output["ms_step"],
        "force error",
        errors[1],
        flush=True,
    )


if __name__ == "__main__":
    main()
