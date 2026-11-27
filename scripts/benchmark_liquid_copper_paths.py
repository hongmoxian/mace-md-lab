"""Time equal-length, thermostatted liquid-Cu production windows starting from the same melt."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import torch
from ase.io import read

from benchmark_periodic_official import OfficialMD, xl_routes
from liquid_copper_md import rescale
from mace_md_lab.dynamics import ArrayVerlet
from mace_md_lab.portable.periodic import official_calculator, periodic_evaluator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in (
        "model",
        "symmetrix-json",
        "product-dir",
        "m0-dir",
        "structure",
        "output",
    ):
        parser.add_argument("--" + key, required=True, type=Path)
    parser.add_argument(
        "--routes-json", type=Path, default=Path("examples/native-xl-locked.json")
    )
    parser.add_argument("--temperature", type=float, default=1520.0)
    parser.add_argument("--seed", type=int, default=7759)
    parser.add_argument("--ps", type=float, default=10.0)
    parser.set_defaults(harmonic_storage="y-only-direct-v1", skin=1.0, dt_fs=2.0)
    a = parser.parse_args()
    if (
        a.temperature <= 0
        or not np.isfinite(a.ps)
        or a.ps < 0.10
        or not np.isclose(a.ps * 20, round(a.ps * 20))
    ):
        parser.error(
            "supply a positive temperature and at least 0.10 ps in increments of 0.05 ps"
        )
    atoms = read(a.structure)
    calc = official_calculator("mace", a.model)
    inv_mass = 1.0 / atoms.get_masses()[:, None]
    chunks = round(a.ps * 1000 / 50)
    results = {}
    paths = {}
    with ExitStack() as stack:
        evaluate = stack.enter_context(periodic_evaluator("mace", a.model, atoms))
        create = {
            "official": lambda: OfficialMD(atoms, calc, a.dt_fs),
            "independent": lambda: ArrayVerlet(atoms, evaluate, a.dt_fs),
        }
        create.update({k: v[0] for k, v in xl_routes(a, atoms).items()})
        for tag in ("independent", "xl", "official", "xl_plus"):
            warm = create[tag]()
            warm.advance(10)
            run = create[tag]()
            rng = np.random.default_rng(a.seed)
            series = []
            first = []
            torch.cuda.synchronize()
            start = perf_counter()
            for i in range(chunks):
                run.advance(25)
                if isinstance(run, OfficialMD):
                    state = SimpleNamespace(
                        positions=run.atoms.positions,
                        momenta=run.atoms.get_momenta(),
                        inverse_mass=inv_mass,
                    )
                    rescale(state, a.temperature, rng, 50.0, 1000.0)
                    run.atoms.set_momenta(state.momenta)
                    pos = run.atoms.positions
                    if i < 2:
                        first.append(run.atoms.get_forces().copy())
                else:
                    rescale(run, a.temperature, rng, 50.0, 1000.0)
                    pos = run.positions
                    if i < 2:
                        first.append(run.forces.copy())
                series.append(pos.copy())
            torch.cuda.synchronize()
            elapsed = perf_counter() - start
            results[tag] = {
                "seconds": elapsed,
                "ms_step": elapsed * 1000 / (chunks * 25),
            }
            paths[tag] = np.asarray(series[:2]), np.asarray(first)
            print("WINDOW", tag, results[tag], flush=True)
    for tag, (pos, forces) in paths.items():
        results[tag]["max_position_error_A_50fs_100fs"] = np.max(
            abs(pos - paths["official"][0]), axis=(1, 2)
        ).tolist()
        results[tag]["max_force_error_eVA_50fs_100fs"] = np.max(
            abs(forces - paths["official"][1]), axis=(1, 2)
        ).tolist()
    a.output.write_text(
        json.dumps(
            {
                "atoms": len(atoms),
                "temperature_K": a.temperature,
                "dt_fs": a.dt_fs,
                "thermostat_seed": a.seed,
                "ps": chunks * 0.05,
                "steps": chunks * 25,
                "gpu": torch.cuda.get_device_name(),
                "routes": results,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
