"""Measure complete, host-returned E/F/stress MD steps against each official ASE calculator."""

import argparse
from contextlib import ExitStack
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import torch
from ase import units
from ase.io import read
from ase.md.verlet import VelocityVerlet

from mace_md_lab.dynamics import ArrayVerlet
from mace_md_lab.portable.periodic import official_calculator, periodic_evaluator
from mace_md_lab.portable.qualified_models import CHECKPOINTS


class OfficialMD:
    def __init__(self, atoms, calculator, dt):
        self.atoms = atoms.copy()
        self.atoms.calc = calculator
        calculator.reset()
        self.md = VelocityVerlet(self.atoms, dt * units.fs, logfile=None)
        self.atoms.get_forces()

    def advance(self, count):
        self.md.run(count)
        if "stress" not in self.atoms.calc.results:
            raise RuntimeError("official force call did not provide stress")


def ase_evaluate(atoms, calculator):
    frame = atoms.copy()
    calculator.reset()
    frame.calc = calculator

    def evaluate(pos):
        frame.positions = pos
        return frame.get_potential_energy(), frame.get_forces(), frame.get_stress()

    return evaluate


def xl_routes(args, atoms):
    import os

    os.environ["SYMMETRIX_NEIGHBOR_BACKEND"] = "kokkos"
    from symmetrix import load_backend
    from benchmark_xl_algorithms import read_routes, setup_engine
    from mace_md_lab.xl_fixed_step import PreparedXL

    load_backend("cuda12-sm70")
    routes = read_routes(args.routes_json)
    compact = sha256(args.symmetrix_json.read_bytes()).hexdigest()
    engines = [
        setup_engine(args, routes[n], compact, sorted(set(map(int, atoms.numbers))))[0]
        for n in ("upstream", "locked")
    ]
    base, joint = engines

    def base_path():
        base._neighbor_cache = None
        return OfficialMD(atoms, base, args.dt_fs)

    return {
        "xl": (base_path, ase_evaluate(atoms, base)),
        "xl_plus": (
            lambda: ArrayVerlet(
                atoms, PreparedXL(atoms, joint, args.skin).evaluate, args.dt_fs
            ),
            PreparedXL(atoms, joint, args.skin).evaluate,
        ),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--framework", required=True, choices=tuple(CHECKPOINTS))
    for key in ("model", "structure", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--repeats", type=int, default=4)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--reverse-order", action="store_true")
    p.add_argument("--symmetrix-json", type=Path)
    p.add_argument("--product-dir", type=Path)
    p.add_argument("--m0-dir", type=Path)
    p.add_argument(
        "--routes-json", type=Path, default=Path("examples/native-xl-locked.json")
    )
    p.set_defaults(harmonic_storage="y-only-direct-v1", skin=1.0, dt_fs=0.5)
    a = p.parse_args()
    if min(a.steps, a.repeats, a.warmup) < 1 or (
        a.symmetrix_json and (a.framework != "mace" or not a.product_dir)
    ):
        p.error(
            "positive window sizes required; optional XL requires MACE and its compiled artifacts"
        )
    atoms = read(a.structure)
    if not atoms.has("momenta"):
        p.error("supply a moving fixture with momenta")
    calc = official_calculator(a.framework, a.model)
    ref = ase_evaluate(atoms, calc)
    reference = OfficialMD(atoms, calc, a.dt_fs)
    frames = []
    for _ in range(5):
        reference.advance(1)
        frames.append(reference.atoms.positions.copy())
    rng = np.random.default_rng(809)
    displaced = atoms.positions + rng.uniform(-0.13, 0.13, atoms.positions.shape)
    wrapped = displaced.copy()
    wrapped[0] += atoms.cell.array[0]
    frames.extend((displaced, wrapped, frames[0]))
    targets = [ref(pos) for pos in frames]
    limits = np.array(
        [1e-8, 1e-6, 1e-7] if a.framework == "mace" else [2e-5, 2e-4, 2e-5]
    )
    with ExitStack() as ctx:
        t0 = perf_counter()
        evaluate = ctx.enter_context(periodic_evaluator(a.framework, a.model, atoms))
        first = evaluate(atoms.positions)
        torch.cuda.synchronize()
        setup_s = perf_counter() - t0
        print("INITIAL", first[0], setup_s, flush=True)
        entries = {
            "official": (lambda: OfficialMD(atoms, calc, a.dt_fs), ref),
            "independent": (lambda: ArrayVerlet(atoms, evaluate, a.dt_fs), evaluate),
        }
        if a.symmetrix_json:
            entries.update(xl_routes(a, atoms))
        errors = {}
        for tag, (_, ev) in entries.items():
            vals = []
            for pos, (e, f, s) in zip(frames, targets):
                en, force, stress = ev(pos)
                vals.append(
                    [
                        abs(en - e) / len(atoms),
                        np.abs(force - f).max(),
                        np.abs(stress - s).max(),
                    ]
                )
            errors[tag] = np.max(vals, axis=0).tolist()
            print("GATE", tag, errors[tag], flush=True)
            if not np.all(np.array(errors[tag]) <= limits):
                raise RuntimeError(f"official comparison failed for {tag}")
        names = list(entries)
        if a.reverse_order:
            names.reverse()
        rows, orders = [], []
        for i in range(a.repeats):
            order = names[i % len(names) :] + names[: i % len(names)]
            orders.append(order)
            row = {}
            for tag in order:
                path = entries[tag][0]()
                path.advance(a.warmup)
                torch.cuda.synchronize()
                t0 = perf_counter()
                path.advance(a.steps)
                torch.cuda.synchronize()
                row[tag] = 1000 * (perf_counter() - t0) / a.steps
            rows.append(row)
        times = {tag: median(t[tag] for t in rows) for tag in entries}
        speeds = {tag: times["official"] / t for tag, t in times.items()}
        package = {"mace": "mace-torch", "sevennet": "sevenn", "dpa4": "deepmd-kit"}[
            a.framework
        ]
        result = dict(
            framework=a.framework,
            checkpoint=a.model.name,
            model_sha256=CHECKPOINTS[a.framework],
            package_version=version(package),
            ase=version("ase"),
            gpu=torch.cuda.get_device_name(),
            torch=str(torch.__version__),
            fixture=a.structure.name,
            atoms=len(atoms),
            fixture_sha256=sha256(a.structure.read_bytes()).hexdigest(),
            dt_fs=a.dt_fs,
            steps=a.steps,
            warmup=a.warmup,
            repeats=a.repeats,
            orders=orders,
            trials=rows,
            ms_step=times,
            speedup_vs_official=speeds,
            setup_and_first_independent_force_s=setup_s,
            common_frames=len(frames),
            max_abs_efs=errors,
            limits=limits.tolist(),
            passed=True,
        )
        if a.symmetrix_json:
            result["xl_version"] = version("symmetrix-xl")
            result["xl_compact_sha256"] = sha256(
                a.symmetrix_json.read_bytes()
            ).hexdigest()
        a.output.write_text(json.dumps(result, indent=2) + "\n")
        print("RESULT", times, speeds, flush=True)


if __name__ == "__main__":
    main()
