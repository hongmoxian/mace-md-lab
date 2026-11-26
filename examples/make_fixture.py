"""Build deterministic, untrained MACE checkpoints and a moving ASE/EMT Cu example."""

import argparse
from pathlib import Path

import numpy as np
import torch
from ase import units
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.io import write
from ase.md.verlet import VelocityVerlet
from mace_md_lab._e3nn import o3
from mace import modules


def structure_and_frames(repeat=2, lattice_a=3.615):
    atoms = bulk("Cu", "fcc", a=lattice_a, cubic=True).repeat((repeat, repeat, repeat))
    atoms.rattle(stdev=0.035, seed=11)
    rng = np.random.default_rng(17)
    velocities = rng.normal(size=(len(atoms), 3))
    velocities *= np.sqrt(units.kB * 300 / atoms.get_masses()[:, None])
    velocities -= np.average(velocities, axis=0, weights=atoms.get_masses())
    atoms.set_velocities(velocities)
    atoms.calc = EMT()
    md = VelocityVerlet(atoms, timestep=0.5 * units.fs)
    frames = []
    for _ in range(8):
        frames.append(atoms.copy())
        md.run(2)
    return frames


def model(seed, channels):
    torch.manual_seed(seed)
    return modules.MACE(
        r_max=5.0,
        num_bessel=8,
        num_polynomial_cutoff=5,
        max_ell=2,
        interaction_cls=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        interaction_cls_first=modules.interaction_classes[
            "RealAgnosticInteractionBlock"
        ],
        num_interactions=2,
        num_elements=1,
        hidden_irreps=o3.Irreps(f"{channels}x0e + {channels}x1o"),
        MLP_irreps=o3.Irreps("16x0e"),
        atomic_energies=np.zeros(1),
        avg_num_neighbors=42.0,
        atomic_numbers=[29],
        correlation=3,
        gate=torch.nn.functional.silu,
    ).eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("fixtures"))
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument(
        "--repeat",
        type=int,
        default=2,
        help="conventional FCC cells along each axis; 2 gives 32 atoms",
    )
    parser.add_argument(
        "--lattice-a",
        type=float,
        default=3.615,
        help="FCC lattice parameter in angstrom",
    )
    parser.add_argument(
        "--frames-only",
        action="store_true",
        help="generate Cu frames for benchmarking existing checkpoints",
    )
    args = parser.parse_args()
    if (
        min(args.channels, args.repeat) < 1
        or not np.isfinite(args.lattice_a)
        or args.lattice_a <= 0
    ):
        parser.error("channels, cell repeat, and lattice parameter must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    seeds = () if args.frames_only else (17, 29, 43)
    outputs = [args.out / f"member-{seed}.model" for seed in seeds]
    trajectory = args.out / "cu.extxyz"
    if any(p.exists() for p in [*outputs, trajectory]):
        parser.error("choose a fresh directory for the example files")
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    write(trajectory, structure_and_frames(args.repeat, args.lattice_a))
    for seed, path in zip(seeds, outputs):
        torch.save(model(seed, args.channels), path)
    print(
        f"Generated {len(outputs)} test weights and eight {4*args.repeat**3}-atom frames in {args.out}."
    )


if __name__ == "__main__":
    main()
