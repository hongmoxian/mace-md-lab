"""Measure native XL radial grid size at fixed force kernels and host-visible MD output."""
import argparse
import json
import os
from pathlib import Path
from statistics import median
import tempfile
from time import perf_counter

import numpy as np
import torch
from ase.io import read

from mace_md_lab.bench import file_hash, write_report
from mace_md_lab.xl_fixed_step import ArrayVerlet, PreparedXL
from mace_md_lab.xl_product import install_xl_product, install_xl_r0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("compact", "artifacts", "structure", "model", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--grids", type=int, nargs="+", default=[8192, 4096, 2048])
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()
    grids = args.grids
    if len(set(grids)) != len(grids) or min(grids) < 4 or min(args.steps, args.repeats) < 1 or args.warmup < 0:
        parser.error("grid sizes must be unique and at least four, with a positive timing window")
    os.environ["SYMMETRIX_NEIGHBOR_BACKEND"] = "kokkos"
    from symmetrix import load_backend
    load_backend("cuda12-sm70")
    from symmetrix import Symmetrix
    from mace.calculators import MACECalculator

    atoms = read(args.structure)
    model = json.loads(args.compact.read_text())
    if model["compact_radial"]["num_spline_points"] != grids[0]:
        parser.error("first grid must match the original compact file")
    if len(set(atoms.numbers)) < 2:
        parser.error("this Au/water control requires a multi-element case")
    original = MACECalculator(model_paths=[str(args.model)], device="cuda", default_dtype="float64")

    def upstream(position):
        fixed = atoms.copy()
        fixed.positions[:] = position
        fixed.calc = original
        return fixed.get_potential_energy(), fixed.get_forces(), fixed.get_stress()

    source_hash = file_hash(args.compact)
    with tempfile.TemporaryDirectory() as directory:
        engines, hashes = {}, {}
        for grid in grids:
            path = args.compact
            if grid != grids[0]:
                path = Path(directory) / f"radial-{grid}.json"
                model["compact_radial"]["num_spline_points"] = grid
                path.write_text(json.dumps(model))
            engine = Symmetrix(str(path), dtype="float64", execution_profile="speed",
                               streamed_edges="direct", neighbor_skin=1., kernel_launch_policy="static",
                               dispersion=False)
            # Only the native coefficient input changes. The compact weights and both
            # execution contracts remain identical by construction.
            m1 = install_xl_product(engine, args.artifacts, variant="upstream")
            r0 = install_xl_r0(engine, args.artifacts, variant="factored")
            if any(metadata["compact_sha256"] != source_hash for metadata in (m1, r0)):
                raise ValueError("source compact hash does not match the compiled contracts")
            engines[grid] = engine
            hashes[grid] = file_hash(path)

        def route(grid):
            engine = engines[grid]
            engine.reset()
            prepared = PreparedXL(atoms, engine, 1.)
            return ArrayVerlet(atoms, prepared.evaluate, 0.5), prepared

        def trajectory(grid):
            path, _ = route(grid)
            snapshots = [path.snapshot()]
            for _ in range(args.steps + args.warmup):
                path.advance(1)
                snapshots.append(path.snapshot())
            return snapshots

        frames = trajectory(grids[0])
        reference = [upstream(frame[0]) for frame in frames]
        results = {}
        for grid in grids:
            states = trajectory(grid)
            prepared = PreparedXL(atoms, engines[grid], 1.)
            outputs = [prepared.evaluate(frame[0]) for frame in frames]
            energy = max(abs(ref[0] - pred[0]) / len(atoms) for ref, pred in zip(reference, outputs))
            force = max(np.max(abs(ref[1] - pred[1])) for ref, pred in zip(reference, outputs))
            stress = max(np.max(abs(ref[2] - pred[2])) for ref, pred in zip(reference, outputs))
            position = max(np.max(abs(x[0] - y[0])) for x, y in zip(states, frames))
            velocity = max(np.max(abs(x[1] - y[1])) for x, y in zip(states, frames))
            drift = max(abs((x[2] + x[4] - states[0][2] - states[0][4])
                            - (y[2] + y[4] - frames[0][2] - frames[0][4])) / len(atoms)
                        for x, y in zip(states, frames))
            qualified = (energy <= 1e-8 and force <= 1e-6 and stress <= 1e-7
                         and position <= 1e-6 and velocity <= 1e-7 and drift <= 1e-8)
            results[grid] = dict(energy_eV_atom=energy, force_eV_A=force, stress_eV_A3=stress,
                                 position_A=position, velocity_A_fs=velocity,
                                 nve_difference_eV_atom=drift, times=[], rebuilds=[], qualified=bool(qualified))
        print("grid accuracy", [(g, results[g]["force_eV_A"], results[g]["qualified"]) for g in grids], flush=True)
        orders = []
        for rep in range(args.repeats):
            first = (rep // 2) % len(grids)
            order = grids[first:] + grids[:first]
            if rep % 2:
                order.reverse()
            orders.append(order)
            for grid in order:
                path, prepared = route(grid)
                path.advance(args.warmup)
                before = prepared.rebuilds
                torch.cuda.synchronize()
                start = perf_counter()
                path.advance(args.steps)
                torch.cuda.synchronize()
                results[grid]["times"].append((perf_counter() - start) * 1000 / args.steps)
                results[grid]["rebuilds"].append(prepared.rebuilds - before)
    for row in results.values():
        row["median_ms_step"] = median(row["times"])
    write_report(args.output, dict(
        results=results, orders=orders, steps=args.steps, warmup=args.warmup, repeats=args.repeats,
        model_sha256=file_hash(args.model), compact_sha256=hashes, structure_sha256=file_hash(args.structure),
        m1=m1["cubin_sha256"], r0=r0["cubin_sha256"], gpu=torch.cuda.get_device_name(), dtype="float64",
        scope="Alternative interpolation of the same weights; prepared ArrayVerlet with host E/F/stress; "
              "stock M1 and factored R0 on every route",
    ))
    print("grid times", [(g, results[g]["median_ms_step"]) for g in grids], flush=True)


if __name__ == "__main__":
    main()
