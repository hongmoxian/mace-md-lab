"""Fixed-cell velocity Verlet on the prepared-device API of Symmetrix-XL 0.2.0."""

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress

from .dynamics import ArrayVerlet  # Backward-compatible import for existing callers.


class PreparedXL:
    """Own the native evaluator's graph for one fully periodic, fixed-cell system.

    This experimental path reads energy, all atomic forces and full stress on
    every call. It requires a direct, fixed-weight MACE evaluator with no field
    or corrections. Use a dedicated evaluator while its graph token is live.
    """

    def __init__(self, atoms, engine, skin):
        self.engine = engine
        self.native = engine.evaluator
        if (engine.execution_plan["model_type"] != "MACE"
                or engine.execution_plan["selected_id"] != "mh0-direct-speed"
                or engine.alchemical is not None or engine._correction_pipeline
                or engine._is_mace_les or engine._has_native_field_coupling()
                or self.native.streamed_edges_mode != "direct"
                or self.native.factorized_execution_profile != "direct_fixed_weight"):
            raise ValueError("requires an uncorrected, field-free direct MACE speed plan")
        self.cell = np.array(atoms.cell, dtype=np.float64, order="C", copy=True)
        self.inverse = np.ascontiguousarray(np.linalg.inv(self.cell))
        self.volume = float(atoms.get_volume())
        self.skin = float(skin)
        self.cutoff = engine.cutoff
        shortest_height = 1.0 / np.linalg.norm(self.inverse, axis=0).max()
        if (not np.all(atoms.pbc) or len(atoms) == 0 or atoms.constraints
                or not np.isfinite(self.skin) or not 0.0 < self.skin < shortest_height / 2):
            raise ValueError("requires a positive skin below half the shortest cell height")
        species = {z: i for i, z in enumerate(self.native.atomic_numbers)}
        self.types = np.array([species[z] for z in atoms.numbers], dtype=np.int32)
        self.node_count = len(atoms)
        self.index_stub = np.empty(0, dtype=np.int32)
        self.xyz_stub = np.empty(0, dtype=np.float64)
        self.reference = None
        self.rebuilds = 0
        self.generation = 0

    def _valid(self, positions):
        if self.reference is None:
            return False
        delta = positions - self.reference
        squares = np.einsum("ij,ij->i", delta, delta)
        # A self-image keeps its distance in a fixed cell. For distinct atoms
        # the sum of their displacement norms bounds the change in distance.
        if self.node_count == 1:
            return float(np.sqrt(squares[0])) < self.skin / 2
        top = np.partition(squares, -2)[-2:]
        bound = np.sqrt(top).sum()
        margin = 32 * np.finfo(float).eps * max(
            1.0, self.cutoff, np.abs(positions).max(), np.abs(self.reference).max()
        )
        return bool(bound + margin < self.skin)

    def _build(self, positions):
        arguments = (self.types, positions.ravel(), self.cell.ravel(),
                     self.inverse.ravel(), self.cutoff + self.skin)
        graph = self.native._prepare_periodic_factorized_graph(*arguments)
        # Native launch-policy changes invalidate prepared schedules in XL.
        if self.engine._apply_kernel_launch_tuning(
            self.node_count, int(graph["num_edges"]), ["forces", "stress"]
        ):
            graph = self.native._prepare_periodic_factorized_graph(*arguments)
        self.generation = int(graph["generation"])
        if self.generation <= 0:
            raise RuntimeError("native graph preparation returned no token")
        self.reference = positions.copy()
        self.rebuilds += 1

    def evaluate(self, positions):
        positions = np.asarray(positions, dtype=np.float64, order="C")
        if positions.shape != (self.node_count, 3) or not np.isfinite(positions).all():
            raise ValueError("supply three finite coordinates for each installed atom")
        if not self._valid(positions):
            self._build(positions)
        generation = self.generation
        native = self.native
        native._compute_prepared_factorized_positions(generation, positions.ravel())
        energy = np.sum(native.node_energies, dtype=np.float64)
        forces = np.asarray(native._reduce_atom_forces(
            self.node_count, self.index_stub, self.index_stub, generation
        ), dtype=np.float64).reshape(self.node_count, 3)
        stress = np.asarray(native._reduce_stress(
            self.volume, self.xyz_stub, generation
        ), dtype=np.float64).reshape(3, 3)
        return float(energy), forces, full_3x3_to_voigt_6_stress(stress)


class PreparedCalculator(Calculator):
    """ASE control using the same prepared evaluation and ASE's integrator."""
    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, initial, prepared):
        super().__init__()
        self.initial_numbers = initial.numbers.copy()
        self.prepared = prepared

    def calculate(self, atoms=None, properties=("forces",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        if (not np.array_equal(atoms.numbers, self.initial_numbers)
                or not np.array_equal(atoms.cell.array, self.prepared.cell)
                or not np.all(atoms.pbc)):
            raise ValueError("composition, periodicity and cell must stay fixed")
        e, f, s = self.prepared.evaluate(atoms.positions)
        self.results = {"energy": e, "forces": f, "stress": s}
