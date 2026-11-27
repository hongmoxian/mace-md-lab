"""Fixed-cell field-free DeepPot energy, forces and ASE-ordered stress."""

from ase.stress import full_3x3_to_voigt_6_stress


def deep_eval(atoms, calc):
    if any(k in atoms.info for k in ("fparam", "aparam", "charge_spin")):
        raise ValueError("requires a field-free fixture")
    types = [calc.type_dict[k] for k in atoms.get_chemical_symbols()]
    cell = atoms.cell.array.reshape(1, 9)
    volume = atoms.get_volume()

    def evaluate(pos):
        e, f, v = calc.dp.eval(pos.reshape(1, -1), cell, types)[:3]
        return (
            float(e[0, 0]),
            f[0],
            full_3x3_to_voigt_6_stress(-v[0].reshape(3, 3) / volume),
        )

    return evaluate
