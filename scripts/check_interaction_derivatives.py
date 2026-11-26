"""Check forces and stress against derivatives of the modified model's own energy."""

import argparse
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch
from ase.io import read
from mace.calculators import MACECalculator

from mace_md_lab.bench import file_hash, write_report
from mace_md_lab.force_checks import force_directions
from mace_md_lab.transforms import optimized_model


def evaluate(atoms, calculator):
    atoms.calc = calculator
    return (
        float(atoms.get_potential_energy()),
        atoms.get_forces().copy(),
        atoms.get_stress().copy(),
    )


def check_model(initial, calculator, reference):
    energy, forces, stress = evaluate(initial.copy(), calculator)
    errors = {
        "energy_eV_atom": abs(energy - reference[0]) / len(initial),
        "force_eV_A": float(abs(forces - reference[1]).max()),
        "stress_eV_A3": float(abs(stress - reference[2]).max()),
    }
    directions = force_directions(initial, 17)
    checks = []
    for label, direction in directions:
        projection = float((forces * direction).sum())
        errors_fd = {}
        for step in (2e-4, 1e-4):
            energies = []
            for displacement in (-step, step):
                at = initial.copy()
                at.positions += displacement * direction
                at.calc = calculator
                energies.append(at.get_potential_energy())
            numerical_force = (energies[0] - energies[1]) / (2 * step)
            errors_fd[str(step)] = abs(float(numerical_force) - projection)
        checks.append({"direction": label, "projection_eV_A": projection,
                       "absolute_error_eV_A": errors_fd})

    # ASE reports tensile-positive stress. A symmetric shear uses two
    # half increments so dE/d(engineering strain)/V gives sigma_xy.
    strain_errors = {}
    for label, voigt, a, b in [("xx", 0, 0, 0), ("zz", 2, 2, 2), ("xy", 5, 0, 1)]:
        basis = np.zeros((3, 3))
        basis[a, b] += 0.5
        basis[b, a] += 0.5
        values = {}
        for step in (2e-5, 1e-5):
            energies = []
            for strain in (-step, step):
                at = initial.copy()
                at.set_cell(initial.cell.array @ (np.eye(3) + strain * basis), scale_atoms=True)
                at.calc = calculator
                energies.append(at.get_potential_energy())
            derivative = (energies[1] - energies[0]) / (2 * step * initial.get_volume())
            values[str(step)] = abs(float(derivative) - stress[voigt])
        strain_errors[label] = values
    passed = (
        errors["energy_eV_atom"] <= 1e-8 and errors["force_eV_A"] <= 1e-6
        and errors["stress_eV_A3"] <= 1e-7
        and all(max(row["absolute_error_eV_A"].values()) <= 2e-5 for row in checks)
        and all(max(values.values()) <= 1e-7 for values in strain_errors.values())
    )
    return {"errors_vs_original": errors, "force_differences": checks,
            "strain_difference_errors_eV_A3": strain_errors, "passed": bool(passed)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--structure", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    table_widths = {"table64edge": 64, "table512edge": 512}
    parser.add_argument("--modes", nargs="+", choices=["first", "folded", "second", "both", "bothfolded", "bothfoldededge", "tangentedge", *table_widths],
                        default=["first", "second", "bothfolded"])
    args = parser.parse_args()
    if args.output.resolve() in (args.model.resolve(), args.structure.resolve()):
        parser.error("output overlaps an input")
    if set(args.modes).intersection(("bothfoldededge", "tangentedge", *table_widths)):
        from mace_md_lab.edge_virial import verify_edge_checkpoint

        verify_edge_checkpoint(args.model)
    atoms = read(args.structure, index=0)
    calc = MACECalculator(model_paths=[str(args.model)], default_dtype="float64", device="cuda")
    model = calc.models[0].eval().requires_grad_(False)
    reference = evaluate(atoms.copy(), calc)
    rows = {}
    for mode in args.modes:
        with ExitStack() as stack:
            stack.enter_context(optimized_model(model, "both", atoms.numbers))
            route = mode
            if mode == "tangentedge":
                from mace_md_lab.exact_packed_radial import install_exact_packed_radials

                stack.enter_context(install_exact_packed_radials(model, tangent=True))
                route = "bothfoldededge"
            if mode in table_widths:
                from mace_md_lab.radial_table import install_radial_table

                stack.enter_context(install_radial_table(model, second_width=table_widths[mode]))
                route = "bothfoldededge"
            if route in ("first", "folded", "both", "bothfolded", "bothfoldededge"):
                from mace_md_lab.grouped_first import install_grouped_first

                stack.enter_context(install_grouped_first(
                    model, atoms.numbers, fold_linear=route in ("folded", "bothfolded", "bothfoldededge")
                ))
            if route in ("second", "both", "bothfolded", "bothfoldededge"):
                from mace_md_lab.grouped_second import install_second_interaction

                stack.enter_context(install_second_interaction(model, atoms.numbers))
            if route == "bothfoldededge":
                from mace_md_lab.edge_virial import install_edge_virial

                stack.enter_context(install_edge_virial(model, None, {}))
            calc.reset()
            rows[mode] = check_model(atoms, calc, reference)
            calc.reset()
    passed = all(row["passed"] for row in rows.values())
    write_report(args.output, {"passed": passed, "model_sha256": file_hash(args.model),
                 "structure_sha256": file_hash(args.structure), "atoms": len(atoms),
                 "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__),
                 "dtype": "float64", "rows": rows})
    if not passed:
        raise SystemExit("force/stress derivative gates failed")


if __name__ == "__main__":
    main()
