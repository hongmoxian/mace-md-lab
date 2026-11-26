"""Rigid normal translation of one interfacial water on a frozen Au/water geometry."""

import argparse
from pathlib import Path

import numpy as np
import torch
from ase.geometry import get_distances
from ase.io import read
from mace.calculators import MACECalculator

from .bench import file_hash, package_version, write_report
from .runtime import ResidentMACE
from .transforms import MODES, optimized_model


def interfacial_water(atoms):
    """Choose a complete O-H-H unit nearest either outer Au plane under PBC."""
    symbols = np.asarray(atoms.get_chemical_symbols())
    au, oxy, hyd = (np.flatnonzero(symbols == s) for s in ("Au", "O", "H"))
    if (
        not np.all(atoms.pbc)
        or atoms.cell.rank != 3
        or min(len(au), len(oxy), len(hyd)) == 0
    ):
        raise ValueError(
            "requires a fully periodic Au slab with water; cell vectors a and b span the slab"
        )
    cell = atoms.cell.array
    normal = np.cross(cell[0], cell[1])
    normal /= np.linalg.norm(normal)
    if np.dot(normal, cell[2]) < 0:
        normal *= -1
    period = float(np.dot(normal, cell[2]))
    _, oh = get_distances(
        atoms.positions[oxy], atoms.positions[hyd], cell=cell, pbc=atoms.pbc
    )
    # Assign each H to its nearest O under PBC. A water must have exactly two
    # distinct assigned H atoms within the dilute molecular O-H distance window.
    nearest_oxygen = oh.argmin(axis=0)
    molecular = {}
    for oi, o in enumerate(oxy):
        bonds = np.flatnonzero((nearest_oxygen == oi) & (oh[oi] < 1.25))
        if len(bonds) == 2 and np.min(oh[oi, bonds]) > 0.65:
            bonds = bonds[np.argsort(oh[oi, bonds])]
            molecular[int(o)] = (hyd[bonds], oh[oi, bonds])
    if not molecular:
        raise ValueError(
            "no complete water found using nearest oxygen assignment and 0.65–1.25 Å O-H bonds"
        )
    candidates = []
    for sign in (1.0, -1.0):
        axis = sign * normal
        z = atoms.positions @ axis
        projected_au = np.sort(np.mod(z[au], period))
        gaps = np.diff(np.r_[projected_au, projected_au[0] + period])
        # Unwrap the metal slab across the largest gap between metal planes, so
        # a wrapped slab cannot appear to have two surfaces on opposite boundaries.
        origin = projected_au[(int(np.argmax(gaps)) + 1) % len(au)]
        metal = np.mod(np.mod(z[au], period) - origin, period)
        outer = au[metal >= metal.max() - 0.6]
        top = float(origin + np.mean(metal[metal >= metal.max() - 0.6]))
        for o, (h, lengths) in molecular.items():
            height = float((z[o] - top) % period)
            candidates.append((height, o, h, lengths, axis, outer, top))
    height, o, h, lengths, axis, outer, top = min(
        candidates, key=lambda x: (x[0], x[1])
    )
    selected = np.r_[o, h].astype(int)
    metadata = {
        "indices_zero_based": selected.tolist(),
        "roles": ["O", "H", "H"],
        "OH_lengths_A": lengths.tolist(),
        "unit_axis_cartesian": axis.tolist(),
        "Au_surface_indices_zero_based": outer.tolist(),
        "Au_surface_coordinate_mod_period_A": float(top % period),
        "normal_period_A": period,
        "oxygen_height_at_q0_A": height,
        "surface_Au_layer_width_A": 0.6,
        "positive_q": "rigid translation away from the selected outer Au plane into the water",
        "selection": "Closest intact water to either outer Au plane by signed periodic normal height; "
        "surface plane is the mean of Au atoms within 0.6 A of the outermost Au.",
    }
    return selected, axis, metadata


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, type=Path, help="trusted local model")
    p.add_argument("--structure", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    p.add_argument(
        "--backend", choices=["auto", "eager", "raw", "padded"], default="auto"
    )
    p.add_argument("--adjoint", choices=MODES, default="none")
    p.add_argument("--head")
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--skin", type=float, default=0.5)
    p.add_argument("--fd-h", type=float, help="centered half displacement in A")
    p.add_argument("--q", type=float, nargs="+", default=[-0.2, -0.1, 0.0, 0.1, 0.2])
    a = p.parse_args(argv)
    if (
        not a.model.is_file()
        or not a.structure.is_file()
        or a.output.resolve() in {a.model.resolve(), a.structure.resolve()}
    ):
        p.error("supply two local input files and a separate result path")
    if a.fd_h is None:
        a.fd_h = 0.001 if a.dtype == "float64" else 0.025
    if (
        0.0 not in a.q
        or len(a.q) != len(set(a.q))
        or not np.isfinite([a.skin, a.fd_h, *a.q]).all()
        or min(a.fd_h, a.skin) <= 0
    ):
        p.error(
            "q values must be finite, unique and include zero; skin and fd-h must be positive"
        )
    backend = a.backend
    device = torch.device(a.device)
    if device.type not in ("cpu", "cuda") or (
        device.type == "cuda" and not torch.cuda.is_available()
    ):
        raise ValueError("unsupported or unavailable device")
    if device.type == "cuda":
        torch.cuda.set_device(
            device.index if device.index is not None else torch.cuda.current_device()
        )
    if backend == "auto":
        backend = "padded" if device.type == "cuda" else "eager"
    if device.type == "cpu" and backend != "eager":
        raise ValueError("use CUDA for replay")
    atoms = read(a.structure, index=a.frame)
    atoms.set_constraint()
    selected, axis, meta = interfacial_water(atoms)
    report = {
        "structure": a.structure.name,
        "structure_sha256": file_hash(a.structure),
        "checkpoint": a.model.name,
        "checkpoint_sha256": file_hash(a.model),
        "frame": a.frame,
        "formula": atoms.get_chemical_formula(),
        "backend": backend,
        "adjoint": a.adjoint,
        "device": a.device,
        "dtype": a.dtype,
        "fd_h_A": a.fd_h,
        "selection": meta,
        "torch": str(torch.__version__),
        "mace": package_version("mace-torch"),
        "points": [],
        "scope": "Instantaneous fixed-geometry potential-energy slice for a rigid translation of one water. "
        "Other atoms, the cell and water orientation are fixed. Adsorption free energy requires "
        "ensemble sampling and a thermodynamic reference.",
    }
    try:
        opts = {
            "model_paths": [a.model],
            "default_dtype": a.dtype,
            "device": str(device),
        }
        if a.head:
            opts["head"] = a.head
        native = MACECalculator(**opts)
        candidate = MACECalculator(**opts)
        if a.head and (native.head != a.head or candidate.head != a.head):
            raise ValueError("requested head differs from selected head")
        report["head"] = native.head
        ref = atoms.copy()
        ref.calc = native
        x0 = atoms.positions.copy()

        def moved(q):
            x = x0.copy()
            x[selected] += q * axis
            return x

        issues = []
        ef_atol, fq_atol, fd_abs, fd_rel = (
            (1e-8, 1e-6, 1e-4, 1e-4)
            if a.dtype == "float64"
            else (2e-4, 0.005, 0.02, 0.01)
        )
        report["tolerances"] = {
            "energy_eV_atom": ef_atol,
            "projected_force_eV_A": fq_atol,
            "directional_FD_atol_eV_A": fd_abs,
            "directional_FD_rtol": fd_rel,
        }
        with optimized_model(
            candidate.models[0], a.adjoint, atomic_numbers=atoms.numbers
        ) as metadata:
            report["optimization"] = metadata
            runner = ResidentMACE(
                atoms,
                candidate,
                skin=a.skin,
                certify_edges=True,
                capture_forces=backend != "eager",
                pad_capture=backend == "padded",
            )
            cache = {}

            def evaluate(q):
                if q not in cache:
                    x = moved(q)
                    ref.set_positions(x)
                    e_ref = ref.get_potential_energy()
                    f_ref = ref.get_forces(apply_constraint=False)
                    e, f = runner.evaluate(x)
                    e_opt = e.item()
                    f_opt = f.detach().cpu().numpy()
                    fq_ref = float(f_ref[selected].sum(axis=0) @ axis)
                    fq_opt = float(f_opt[selected].sum(axis=0) @ axis)
                    data = [e_ref, e_opt, fq_ref, fq_opt]
                    if not np.isfinite(data).all():
                        raise FloatingPointError(f"nonfinite output at q={q}")
                    cache[q] = data
                return cache[q]

            e0_ref, e0_opt, _, _ = evaluate(0.0)
            for q in sorted(a.q):
                er, eo, fr, fo = evaluate(q)
                left = evaluate(q - a.fd_h)
                right = evaluate(q + a.fd_h)
                fd_r = (left[0] - right[0]) / (2 * a.fd_h)
                fd_o = (left[1] - right[1]) / (2 * a.fd_h)
                de = (eo - e0_opt) - (er - e0_ref)
                row = {
                    "q_A": q,
                    "O_height_A": meta["oxygen_height_at_q0_A"] + q,
                    "native_relative_E_eV": er - e0_ref,
                    "optimized_relative_E_eV": eo - e0_opt,
                    "native_Fq_eV_A": fr,
                    "optimized_Fq_eV_A": fo,
                    "native_minus_dE_dq_FD_eV_A": fd_r,
                    "optimized_minus_dE_dq_FD_eV_A": fd_o,
                    "absolute_relative_energy_error_eV": abs(de),
                    "absolute_projected_force_error_eV_A": abs(fo - fr),
                    "native_FD_absolute_error_eV_A": abs(fd_r - fr),
                    "optimized_FD_absolute_error_eV_A": abs(fd_o - fo),
                }
                report["points"].append(row)
                if (
                    max(abs(eo - er), abs(de)) / len(atoms) > ef_atol
                    or abs(fo - fr) > fq_atol
                ):
                    issues.append(f"energy or projected-force parity failed at q={q}")
                if max(abs(fd_r - fr), abs(fd_o - fo)) > fd_abs + fd_rel * abs(fr):
                    issues.append(f"centered derivative check failed at q={q}")
        report["passed"] = not issues
        report["issues"] = issues
    except Exception as exc:
        report.update(
            passed=False, error={"type": type(exc).__name__, "message": str(exc)}
        )
        write_report(a.output, report)
        raise
    write_report(a.output, report)
    if issues:
        raise SystemExit("Scan did not pass; see " + str(a.output))
    print(f"Wrote the rigid water potential-energy slice to {a.output}")
    return 0


if __name__ == "__main__":
    main()
