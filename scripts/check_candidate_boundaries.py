"""Compare captured MP-0 Small candidate forces to fresh MACE on cutoff crossings and reimaging."""
import argparse
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from ase.io import read
from ase.stress import full_3x3_to_voigt_6_stress
from mace.calculators import MACECalculator
from mace.data.neighborhood import get_neighborhood

from mace_md_lab.bench import file_hash, write_report
from mace_md_lab.candidate_replay import CandidateRunner, explicit_cutoff
from mace_md_lab.edge_virial import install_edge_virial, verify_edge_checkpoint
from mace_md_lab.grouped_first import install_grouped_first
from mace_md_lab.grouped_second import install_second_interaction
from mace_md_lab.transforms import optimized_model


def select_crossing(atoms, cutoff):
    edge, shifts, _, _ = get_neighborhood(positions=atoms.positions, cutoff=cutoff + 0.08,
                                         pbc=tuple(atoms.pbc), cell=atoms.cell.array.copy())
    vec = atoms.positions[edge[1]] - atoms.positions[edge[0]] + shifts
    radius = np.linalg.norm(vec, axis=1)
    eligible = np.flatnonzero(edge[0] != edge[1])
    pick = eligible[np.argmin(np.abs(radius[eligible] - cutoff))]
    target = int(edge[1, pick])
    unit = vec[pick] / radius[pick]
    base = atoms.positions.copy()
    base[target] += (cutoff - radius[pick]) * unit
    return base, target, unit, int(edge[0, pick]), shifts[pick].tolist()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "structure", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--route", choices=("replay", "schedtangent"), default="replay")
    p.add_argument("--skin", type=float, default=0.25)
    args = p.parse_args()
    if args.output.resolve() in (args.model.resolve(), args.structure.resolve()):
        p.error("choose a separate output")
    verify_edge_checkpoint(args.model)
    atoms = read(args.structure, index=0)
    kwargs = dict(model_paths=[str(args.model)], default_dtype="float64", device="cuda")
    stock, fast = MACECalculator(**kwargs), MACECalculator(**kwargs)
    model = fast.models[0].eval().requires_grad_(False)
    cutoff = float(fast.r_max)
    base, target, unit, sender, image = select_crossing(atoms, cutoff)
    rows = []
    skin = args.skin
    with ExitStack() as stack:
        layout = SimpleNamespace()
        stack.enter_context(optimized_model(model, "both", atoms.numbers))
        runner_type = CandidateRunner
        if args.route == "schedtangent":
            from mace_md_lab.candidate_schedule import ScheduledRunner
            from mace_md_lab.exact_packed_radial import install_exact_packed_radials

            stack.enter_context(install_exact_packed_radials(model, tangent=True))
            runner_type = ScheduledRunner
        stack.enter_context(explicit_cutoff(model, cutoff))
        stack.enter_context(install_grouped_first(model, atoms.numbers, fold_linear=True, layout=layout))
        stack.enter_context(install_second_interaction(model, layout=layout))
        stress = {}
        stack.enter_context(install_edge_virial(model, atoms.get_volume(), stress))
        runner = runner_type(atoms, fast, skin, layout, capture=True, macro_extension=1.0)
        for wrap, distance in [(False, -0.0003), (False, 0.0003), (False, -0.0003),
                               (False, 0.16), (True, -0.0003), (False, -0.0003)]:
            coords = base.copy()
            coords[target] += distance * unit
            if wrap:
                coords[target] += atoms.cell.array[0]
            runner.positions.copy_(runner._state_tensor(coords))
            e, f = runner.evaluate()
            s = full_3x3_to_voigt_6_stress(stress["tensor"].cpu().numpy().reshape((3, 3)))
            frame = atoms.copy()
            frame.positions = coords
            frame.calc = stock
            differences = dict(energy_eV_atom=abs(e.item()-frame.get_potential_energy())/len(atoms),
                               force_eV_A=float(np.max(abs(f.cpu().numpy()-frame.get_forces()))),
                               stress_eV_A3=float(np.max(abs(s-frame.get_stress()))))
            rows.append(dict(offset_from_cutoff_A=distance, lattice_shift_applied=wrap,
                             macro_builds=runner.rebuilds, candidate_refreshes=runner.edge_refreshes,
                             captures=runner.captures, replays=runner.replays,
                             **runner.verify_physical_edges(), **differences))
    gates = dict(energy_eV_atom=1e-8, force_eV_A=1e-6, stress_eV_A3=1e-7)
    passed = all(r[k] <= bound for r in rows for k, bound in gates.items())
    passed = passed and rows[0]["active"] != rows[1]["active"]
    passed = passed and rows[0]["active"] == rows[2]["active"]
    passed = passed and rows[0]["captures"] == rows[2]["captures"] == 1
    passed = passed and rows[0]["candidate_refreshes"] == rows[2]["candidate_refreshes"]
    passed = passed and rows[3]["candidate_refreshes"] > rows[2]["candidate_refreshes"]
    passed = passed and rows[4]["macro_builds"] > rows[3]["macro_builds"]
    data = dict(passed=bool(passed), route=args.route, rows=rows, gates=gates, skin_A=skin, cutoff_A=cutoff,
                structure_sha256=file_hash(args.structure), model_sha256=file_hash(args.model),
                selected_sender=sender, selected_receiver=target, selected_cartesian_shift_A=image)
    write_report(args.output, data)
    print("Passed", passed, flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
