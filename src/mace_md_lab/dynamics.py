"""Fixed-cell velocity Verlet shared by periodic potential adapters."""

import numpy as np
from ase import units


class ArrayVerlet:
    """Propagate one unconstrained NVE state with host-returned native E/F/stress."""

    def __init__(self, atoms, evaluate, dt_fs):
        if atoms.constraints or not np.all(atoms.pbc):
            raise ValueError("this integrator supports unconstrained periodic NVE")
        self.positions = atoms.positions.copy()
        self.momenta = atoms.get_momenta().copy()
        self.inverse_mass = 1.0 / atoms.get_masses()[:, None]
        self.evaluate = evaluate
        self.dt = dt_fs * units.fs
        self.energy, self.forces, self.stress = evaluate(self.positions)

    def advance(self, steps):
        half_dt = 0.5 * self.dt
        for _ in range(steps):
            self.momenta += half_dt * self.forces
            self.positions += self.dt * self.inverse_mass * self.momenta
            self.energy, self.forces, self.stress = self.evaluate(self.positions)
            self.momenta += half_dt * self.forces

    def snapshot(self):
        velocity = self.momenta * self.inverse_mass
        kinetic = 0.5 * np.sum(self.momenta * velocity)
        return (
            self.positions.copy(),
            velocity.copy() * units.fs,
            self.energy,
            self.forces.copy(),
            float(kinetic),
            self.stress.copy(),
        )
