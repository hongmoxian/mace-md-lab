"""Run conservative fixed-cell MD and write the final frame with energy, forces and stress."""

import argparse
from pathlib import Path

from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write

from .periodic import periodic_md
from .qualified_models import CHECKPOINTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", choices=tuple(CHECKPOINTS), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--structure", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="final extended XYZ")
    parser.add_argument("--dt-fs", type=float, default=0.5)
    parser.add_argument("--steps", type=int, required=True)
    args = parser.parse_args()
    if args.steps <= 0 or args.output.resolve() in (
        args.model.resolve(),
        args.structure.resolve(),
    ):
        parser.error("use a positive step count and a separate output file")
    atoms = read(args.structure, index=0)
    with periodic_md(args.framework, args.model, atoms, args.dt_fs) as trajectory:
        trajectory.advance(args.steps)
        atoms.positions = trajectory.positions
        atoms.set_momenta(trajectory.momenta)
        atoms.calc = SinglePointCalculator(
            atoms,
            energy=trajectory.energy,
            forces=trajectory.forces,
            stress=trajectory.stress,
        )
        write(args.output, atoms, format="extxyz")


if __name__ == "__main__":
    main()
