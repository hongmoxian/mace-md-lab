"""Fixed-capacity force replay with exact physical cutoff and two Verlet buffers.

This benchmark qualifies the short-range MACE-MP-0 Small energy. Both interaction
layers multiply every message by the explicit physical cutoff mask. Padding edges are
finite, exterior self-images. Group and receiver offsets follow the sorted
candidate edges and are refreshed in-place before the force graph is replayed.
"""

from contextlib import contextmanager

import numpy as np
import torch

from .bench import ResidentPath
from .runtime import ResidentMACE, evaluate_prepared


@contextmanager
def explicit_cutoff(model, cutoff):
    if hasattr(model, "pair_repulsion") or hasattr(model, "joint_embedding"):
        raise ValueError("candidate replay requires the qualified short-range MP-0 Small checkpoint")

    def mask(_module, inputs, output):
        features, envelope = output
        active = (inputs[0] < cutoff).to(inputs[0].dtype)
        return features, active if envelope is None else envelope * active

    handle = model.radial_embedding.register_forward_hook(mask)
    try:
        yield
    finally:
        handle.remove()


class CandidateRunner(ResidentMACE):
    def __init__(self, atoms, calculator, skin, layout, capture, macro_extension):
        if min(skin, macro_extension) <= 0:
            raise ValueError("Verlet buffers must be positive")
        super().__init__(atoms, calculator, skin=skin)
        self.layout = layout
        self.micro_skin = skin
        self.macro_extension = macro_extension
        self.capture = capture
        self._graph = self._micro_reference = None
        self._capacity = self._captures = self._replays = 0
        self._batch = None
        self._species = torch.tensor(np.unique(atoms.numbers, return_inverse=True)[1],
                                     dtype=torch.long, device=self.device)
        self._nspecies = len(np.unique(atoms.numbers))
        self._receiver_keys = torch.arange(len(atoms) + 1, device=self.device)
        self._group_keys = torch.arange(len(atoms) * self._nspecies + 1, device=self.device)
        cell = atoms.cell.array
        multiple = int(np.ceil(2 * (self.cutoff + skin) / np.linalg.norm(cell[0]))) + 1
        self._padding_unit = torch.tensor([multiple, 0, 0], device=self.device, dtype=self._shift_dtype)
        self._padding_shift = self._state_tensor(cell[0] * multiple)

    @property
    def captures(self):
        return self._captures

    @property
    def replays(self):
        return self._replays

    def _action(self, pos):
        if self._reference is None:
            return 2
        displacement = (pos - self._micro_reference).square().sum(1).amax()
        macro_displacement = (pos - self._reference).square().sum(1).amax()
        # Strict bound retains a guard against roundoff at the list boundary.
        margin = 64 * torch.finfo(pos.dtype).eps * (1 + self.cutoff + pos.abs().amax())
        refresh = 2 * displacement.sqrt() + margin >= self.micro_skin
        rebuild = 2 * macro_displacement.sqrt() + margin >= self.macro_extension
        return int(torch.maximum(refresh.int(), 2 * rebuild.int()))

    def _allocate(self, count, pos):
        # Headroom is charged to each force call. The warm graph can survive a
        # change of physical or candidate edge count without a fresh capture.
        if self._batch is not None and count <= self._capacity:
            return
        if self._graph is not None:
            torch.cuda.synchronize(self.device)
            self._graph = self._out = None
        cap = max(128, ((count + max(16, count // 64) + 127) // 128) * 128)
        self._capacity = cap
        self._batch = dict(self._static, positions=pos.detach().clone(),
                           edge_index=torch.empty((2, cap), device=self.device, dtype=torch.long),
                           shifts=pos.new_empty((cap, 3)),
                           unit_shifts=torch.empty((cap, 3), device=self.device, dtype=self._shift_dtype))
        self.layout.groups = torch.empty(cap, device=self.device, dtype=torch.long)
        self.layout.order = torch.arange(cap, device=self.device)
        self.layout.group_offsets = torch.empty_like(self._group_keys)
        self.layout.receiver_offsets = torch.empty_like(self._receiver_keys)

    @torch.no_grad()
    def prepare(self, positions=None):
        pos = self.positions if positions is None else torch.as_tensor(
            positions, device=self.device, dtype=self.dtype
        )
        if pos.shape != self.positions.shape:
            raise ValueError("invalid coordinate shape")
        action = self._action(pos)
        if action == 2:
            self.skin = self.micro_skin + self.macro_extension
            try:
                super()._rebuild(pos)
            finally:
                self.skin = self.micro_skin
        if action:
            src, dst = self._candidates
            vectors = pos[dst] - pos[src] + self._candidate_shifts
            keep = torch.nonzero(vectors.square().sum(1) < (self.cutoff + self.micro_skin)**2).flatten()
            count = keep.numel()
            self._allocate(count, pos)
            index = self._batch["edge_index"]
            shifts, units = self._batch["shifts"], self._batch["unit_shifts"]
            index[:, :count] = self._candidates.index_select(1, keep)
            shifts[:count] = self._candidate_shifts.index_select(0, keep)
            units[:count] = self._candidate_units.index_select(0, keep)
            index[:, count:] = len(pos) - 1
            shifts[count:] = self._padding_shift
            units[count:] = self._padding_unit
            group = index[1] * self._nspecies + self._species.index_select(0, index[0])
            sorted_group, order = torch.sort(group, stable=True)
            # Physical tensors are sorted in receiver/species order, which is
            # also valid receiver order for the second layer. Sort once per list.
            index.copy_(index.index_select(1, order))
            shifts.copy_(shifts.index_select(0, order))
            units.copy_(units.index_select(0, order))
            self.layout.groups.copy_(sorted_group)
            torch.searchsorted(sorted_group, self._group_keys, out=self.layout.group_offsets)
            torch.searchsorted(index[1], self._receiver_keys, out=self.layout.receiver_offsets)
            self._micro_reference = pos.detach().clone()
            self.edge_refreshes += 1
        else:
            self.certificate_hits += 1
        self._batch["positions"].copy_(pos)
        return self._batch

    def verify_physical_edges(self):
        """Compare directed endpoints and integer periodic images at this exact frame."""
        from mace.data.neighborhood import get_neighborhood

        batch = self._batch
        positions = self.positions.detach().cpu().numpy()
        edge = batch["edge_index"].cpu().numpy()
        shift = batch["shifts"].cpu().numpy()
        unit = batch["unit_shifts"].cpu().numpy()
        vectors = positions[edge[1]] - positions[edge[0]] + shift
        active = np.linalg.norm(vectors, axis=1) < self.cutoff
        reference, _, ref_unit, _ = get_neighborhood(
            positions=positions, cutoff=self.cutoff, pbc=self._pbc, cell=self._cell.copy()
        )

        def keys(index, image):
            rows = np.column_stack((index.T, image)).astype(np.int64)
            return rows[np.lexsort(rows.T[::-1])]

        actual = keys(edge[:, active], unit[active])
        expected = keys(reference, ref_unit)
        if not np.array_equal(actual, expected):
            raise ValueError("active periodic edge identities disagree with freshly rebuilt MACE graph")
        return {"active": len(actual), "padded_candidate_capacity": self._capacity}

    def evaluate(self, positions=None):
        batch = self.prepare(positions)
        if self.capture:
            if self._graph is None:
                current = torch.cuda.current_stream(self.device)
                stream = torch.cuda.Stream(device=self.device)
                stream.wait_stream(current)
                with torch.cuda.stream(stream):
                    for _ in range(2):
                        evaluate_prepared(self.calculator, batch)
                current.wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    self._out = evaluate_prepared(self.calculator, batch)
                self._graph = graph
                self._captures += 1
            self._graph.replay()
            self._replays += 1
            result = self._out[0].clone(), self._out[1].clone()
        else:
            result = evaluate_prepared(self.calculator, batch)
        if positions is None or positions is self.positions:
            self._force = result[1]
            self._force_version = self.positions._version
        return result


class CandidatePath(ResidentPath):
    def __init__(self, initial, calculator, dt, skin, layout, capture, macro_extension=1.0):
        self.runner = CandidateRunner(initial, calculator, skin, layout, capture, macro_extension)
        self.dt = dt
        self.energy, self.forces = self.runner.evaluate()
