"""Write isolated benzene or a disconnected molecular size control."""

import argparse
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import molecule
from ase.io import read, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--from-geometry", type=Path)
    p.add_argument("--copies", type=int, default=1)
    args = p.parse_args()
    if args.output.exists() or args.copies < 1:
        p.error("choose a new output file and a positive number of copies")
    one = read(args.from_geometry, index=0) if args.from_geometry else molecule("C6H6")
    if any(one.pbc):
        p.error("supply one isolated molecule")
    if args.copies == 1:
        atoms = one
    else:
        atoms = Atoms()
        pitch = float(np.ptp(one.positions[:, 0])) + 20.0
        for i in range(args.copies):
            fragment = one.copy()
            fragment.translate([i * pitch, 0, 0])
            atoms += fragment
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write(args.output, atoms, format="extxyz")


if __name__ == "__main__":
    main()
