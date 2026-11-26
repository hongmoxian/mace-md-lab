"""Generate an idealized, non-equilibrated, periodic 192-atom Au(111)/H2O fixture."""

import argparse
import json
from pathlib import Path

import numpy as np


def minimum_image(delta, lengths):
    return delta - lengths * np.rint(delta / lengths)


def water_geometry(oxygen, rng):
    """Rotate a rigid water with a 0.9572 Å O-H distance and a 104.52 degree angle."""
    bond = 0.9572
    half = np.deg2rad(104.52 / 2)
    bisector = rng.normal(size=3)
    bisector /= np.linalg.norm(bisector)
    side = rng.normal(size=3)
    side -= np.dot(side, bisector) * bisector
    side /= np.linalg.norm(side)
    midpoint = oxygen + bond * np.cos(half) * bisector
    arm = bond * np.sin(half) * side
    return np.array([oxygen, midpoint + arm, midpoint - arm])


def build_coordinates(seed):
    rng = np.random.default_rng(seed)
    lattice = 4.078  # Å; reproducible FCC lattice for the test slab
    nn = lattice / np.sqrt(2)
    row = nn * np.sqrt(3) / 2
    dz = lattice / np.sqrt(3)
    lengths = np.array([4 * nn, 4 * row, 2 * dz + 3.0 + 3 * 3.15 + 13.0])
    gold = []
    for layer in range(3):
        for y in range(4):
            for x in range(4):
                position = [
                    (x + 0.5 * (y % 2) + 0.5 * layer) * nn,
                    (y + layer / 3) * row,
                    2.0 + layer * dz,
                ]
                gold.append(position)
    gold = np.asarray(gold) + rng.normal(scale=0.012, size=(48, 3))
    gold[:, :2] %= lengths[:2]
    top = 2.0 + 2 * dz
    oxygen_sites = []
    for layer in range(4):
        for j in range(3):
            for i in range(4):
                oxygen_sites.append(
                    [
                        (i + 0.25 + 0.50 * (layer % 2)) * lengths[0] / 4,
                        (j + 0.30 + 0.46 * (layer % 2)) * lengths[1] / 3,
                        top + 3.0 + 3.15 * layer,
                    ]
                )
    oxygen_sites = np.asarray(oxygen_sites)
    oxygen_sites += rng.uniform(-0.065, 0.065, size=oxygen_sites.shape)
    oxygen_sites[:, :2] %= lengths[:2]
    water = np.empty((len(oxygen_sites), 3, 3))
    # Random sequential orientations, with molecule identities retained. Check
    # against all O sites (including future waters) and previously placed H atoms.
    for n, oxygen in enumerate(oxygen_sites):
        others = np.delete(oxygen_sites, n, axis=0)
        placed_h = water[:n, 1:].reshape(-1, 3)
        for _ in range(2000):
            molecule = water_geometry(oxygen, rng)
            h = molecule[1:]

            def closest(group):
                delta = minimum_image(h[:, None, :] - group[None, :, :], lengths)
                return np.linalg.norm(delta, axis=-1).min(initial=np.inf)

            if (
                closest(others) >= 1.75
                and closest(placed_h) >= 1.48
                and closest(gold) >= 2.13
            ):
                water[n] = molecule
                break
        else:
            raise RuntimeError(f"failed to orient water {n} for seed {seed}")
    positions = np.vstack([gold, water.reshape(-1, 3)])
    elements = ["Au"] * 48 + ["O", "H", "H"] * 48
    return elements, positions, lengths


def diagnostics(elements, positions, lengths):
    d = np.linalg.norm(
        minimum_image(positions[:, None, :] - positions[None, :, :], lengths), axis=-1
    )
    au = np.arange(48)
    ox = np.arange(48, len(positions), 3)
    h = np.sort(np.r_[ox + 1, ox + 2])

    def minimum(a, b):
        return float(d[np.ix_(a, b)].min())

    iu = np.triu_indices(len(ox), 1)
    arms = positions[h].reshape(-1, 2, 3) - positions[ox, None]
    bond_lengths = np.linalg.norm(arms, axis=-1)
    angle = np.rad2deg(
        np.arccos(
            np.clip((arms[:, 0] * arms[:, 1]).sum(1) / bond_lengths.prod(1), -1, 1)
        )
    )
    oh = d[np.ix_(ox, h)].copy()
    idx = np.arange(len(ox))
    oh[idx, 2 * idx] = oh[idx, 2 * idx + 1] = np.inf
    hh = d[np.ix_(h, h)].copy()
    hh[np.repeat(idx, 2)[:, None] == np.repeat(idx, 2)[None, :]] = np.inf
    return dict(
        n_atoms=len(elements),
        n_gold=len(au),
        n_waters=len(ox),
        cell_A=lengths.tolist(),
        min_Au_Au_A=float(d[np.ix_(au, au)][np.triu_indices(len(au), 1)].min()),
        min_Au_O_A=minimum(au, ox),
        min_Au_H_A=minimum(au, h),
        min_O_O_A=float(d[np.ix_(ox, ox)][iu].min()),
        min_inter_O_H_A=float(oh.min()),
        min_inter_H_H_A=float(hh.min()),
        OH_range_A=[float(bond_lengths.min()), float(bond_lengths.max())],
        HOH_range_deg=[float(angle.min()), float(angle.max())],
    )


def main():
    from ase import Atoms, units
    from ase.io import write

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=17)
    a = p.parse_args()
    if a.seed < 0 or a.output.exists():
        p.error("use a nonnegative seed and a fresh output file")
    elements, positions, lengths = build_coordinates(a.seed)
    atoms = Atoms(elements, positions=positions, cell=lengths, pbc=True)
    masses = atoms.get_masses()
    rng = np.random.default_rng(a.seed + 1)
    v = rng.normal(size=(len(atoms), 3)) * np.sqrt(units.kB * 300 / masses[:, None])
    v -= np.average(v, axis=0, weights=masses)
    atoms.set_velocities(v)
    atoms.set_velocities(
        atoms.get_velocities() * np.sqrt(300 / atoms.get_temperature())
    )
    atoms.info["fixture"] = "idealized_Au111_water"
    atoms.info["generator_seed"] = a.seed
    a.output.parent.mkdir(parents=True, exist_ok=True)
    write(a.output, atoms, format="extxyz")
    print(json.dumps(diagnostics(elements, positions, lengths), indent=2))


if __name__ == "__main__":
    main()
