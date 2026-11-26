"""Velocity Verlet and fixed-cell neighbor maintenance for an exact candidate replay."""

import math

import torch
import triton
import triton.language as tl
from ase import units

from .bench import ResidentPath
from .candidate_replay import CandidateRunner
from .runtime import ResidentMACE


@triton.jit
def _advance_and_check(X, V, M, F, MICRO, MACRO, TRIAL, HALF, CODE,
                       n: tl.constexpr, dt: tl.constexpr, skin: tl.constexpr,
                       extension: tl.constexpr, cutoff: tl.constexpr, tile: tl.constexpr):
    atom = tl.arange(0, tile)
    atom_ok = atom < n
    mass = tl.load(M + atom, atom_ok, other=1.0)
    displacement = tl.full((tile,), 0, tl.float64)
    macro_displacement = tl.full((tile,), 0, tl.float64)
    magnitude = tl.full((tile,), 0, tl.float64)
    finite = tl.full((tile,), 1, tl.int32)
    # Vector components are stored atom-major in ASE conventions.
    for axis in tl.static_range(3):
        offset = 3 * atom + axis
        x = tl.load(X + offset, atom_ok, other=0)
        v = tl.load(V + offset, atom_ok, other=0)
        f = tl.load(F + offset, atom_ok, other=0)
        half = v + (0.5 * dt) * f / mass
        trial = x + dt * half
        tl.store(TRIAL + offset, trial, atom_ok)
        tl.store(HALF + offset, half, atom_ok)
        micro = tl.load(MICRO + offset, atom_ok, other=0)
        macro = tl.load(MACRO + offset, atom_ok, other=0)
        displacement += (trial - micro) * (trial - micro)
        macro_displacement += (trial - macro) * (trial - macro)
        magnitude = tl.maximum(magnitude, tl.maximum(tl.abs(trial), tl.maximum(tl.abs(micro), tl.abs(macro))))
        finite = finite & ((tl.abs(trial) < float("inf")) & (trial == trial)).to(tl.int32)
    # A pair of distinct atoms can move apart or together by at most the sum
    # of the two largest displacement norms. Self-image distances are constant.
    winner_micro = tl.argmax(displacement, axis=0)
    winner_macro = tl.argmax(macro_displacement, axis=0)
    bound_micro = tl.sqrt(tl.max(displacement, axis=0)) + tl.sqrt(
        tl.max(tl.where(atom != winner_micro, displacement, 0.0), axis=0)
    )
    bound_macro = tl.sqrt(tl.max(macro_displacement, axis=0)) + tl.sqrt(
        tl.max(tl.where(atom != winner_macro, macro_displacement, 0.0), axis=0)
    )
    guard = 64.0 * 2.220446049250313e-16 * (1.0 + cutoff + tl.max(magnitude, axis=0))
    refresh = (bound_micro + guard >= skin).to(tl.int32)
    rebuild = (bound_macro + guard >= extension) | (tl.min(finite, axis=0) == 0)
    tl.store(CODE, tl.where(rebuild, 2, refresh))


@triton.jit
def _finish(TRIAL, HALF, M, F, X, V, count, dt: tl.constexpr, tile: tl.constexpr):
    idx = tl.program_id(0) * tile + tl.arange(0, tile)
    ok = idx < count
    x = tl.load(TRIAL + idx, ok, other=0)
    half = tl.load(HALF + idx, ok, other=0)
    force = tl.load(F + idx, ok, other=0)
    mass = tl.load(M + idx // 3, ok, other=1)
    tl.store(X + idx, x, ok)
    tl.store(V + idx, half + 0.5 * dt * force / mass, ok)


@triton.jit
def _active(POS, IDX, SHIFT, KEEP, count, cutoff2: tl.constexpr, tile: tl.constexpr):
    slot = tl.program_id(0) * tile + tl.arange(0, tile)
    ok = slot < count
    source = tl.load(IDX + slot, ok, other=0)
    target = tl.load(IDX + count + slot, ok, other=0)
    length2 = tl.full((tile,), 0, tl.float64)
    for axis in tl.static_range(3):
        r = tl.load(POS + 3 * target + axis, ok, other=0) - tl.load(POS + 3 * source + axis, ok, other=0)
        r += tl.load(SHIFT + 3 * slot + axis, ok, other=0)
        length2 += r * r
    tl.store(KEEP + slot, length2 < cutoff2, ok)


@triton.jit
def _pack(KEEP, CANDIDATE, SHIFT, UNIT, GROUP, PAD_SHIFT, PAD_UNIT, OUT, OUT_SHIFT, OUT_UNIT,
          OUT_GROUP, macro_edges, count, capacity, last_atom, last_group, tile: tl.constexpr):
    slot = tl.program_id(0) * tile + tl.arange(0, tile)
    selected = slot < count
    valid = slot < capacity
    src_row = tl.load(KEEP + slot, selected, other=0)
    src = tl.load(CANDIDATE + src_row, selected, other=0)
    dst = tl.load(CANDIDATE + macro_edges + src_row, selected, other=0)
    group = tl.load(GROUP + src_row, selected, other=0)
    tl.store(OUT + slot, tl.where(selected, src, last_atom), valid)
    tl.store(OUT + capacity + slot, tl.where(selected, dst, last_atom), valid)
    # Padding only contributes zero after the strict physical-cutoff mask.
    # Assign the final group slot so the CSR array stays nondecreasing.
    tl.store(OUT_GROUP + slot, tl.where(selected, group, last_group), valid)
    for axis in tl.static_range(3):
        shift = tl.load(SHIFT + 3 * src_row + axis, selected, other=0)
        unit = tl.load(UNIT + 3 * src_row + axis, selected, other=0)
        tl.store(OUT_SHIFT + 3 * slot + axis, tl.where(selected, shift, tl.load(PAD_SHIFT + axis)), valid)
        tl.store(OUT_UNIT + 3 * slot + axis, tl.where(selected, unit, tl.load(PAD_UNIT + axis)), valid)


class ScheduledRunner(CandidateRunner):
    """Caches macro-edge order and fuses the fixed-cell integrator for small cells."""

    def __init__(self, atoms, calculator, skin, layout, capture=True, macro_extension=1.0):
        super().__init__(atoms, calculator, skin, layout, capture, macro_extension)
        self._trial = torch.empty_like(self.positions)
        self._half = torch.empty_like(self.velocities)
        self._action_code = torch.empty((), dtype=torch.int32, device=self.device)
        self._scheduled_action = False

    def _rebuild_sorted(self, pos):
        self.skin = self.micro_skin + self.macro_extension
        try:
            ResidentMACE._rebuild(self, pos)
        finally:
            self.skin = self.micro_skin
        src, dst = self._candidates
        group = dst * self._nspecies + self._species[src]
        self._sorted_groups, order = torch.sort(group, stable=True)
        self._candidates = self._candidates.index_select(1, order).contiguous()
        self._candidate_shifts = self._candidate_shifts.index_select(0, order).contiguous()
        self._candidate_units = self._candidate_units.index_select(0, order).contiguous()
        self._keep = torch.empty(len(order), dtype=torch.bool, device=self.device)

    @torch.no_grad()
    def prepare(self, positions=None):
        pos = self.positions if positions is None else torch.as_tensor(positions, device=self.device, dtype=self.dtype)
        if pos.shape != self.positions.shape:
            raise ValueError("invalid coordinate shape")
        if self._scheduled_action and pos is self._trial:
            action = int(self._action_code)
        else:
            action = self._action(pos)
        if action == 2:
            self._rebuild_sorted(pos)
        if action:
            macro_edges = self._candidates.shape[1]
            if macro_edges:
                _active[(triton.cdiv(macro_edges, 128),)](
                    pos, self._candidates, self._candidate_shifts, self._keep, macro_edges,
                    (self.cutoff + self.micro_skin)**2, 128, enable_fp_fusion=False,
                )
            selected = torch.nonzero(self._keep).flatten()
            self._allocate(selected.numel(), pos)
            index = self._batch["edge_index"]
            capacity = index.shape[1]
            _pack[(triton.cdiv(capacity, 128),)](
                selected, self._candidates, self._candidate_shifts, self._candidate_units, self._sorted_groups,
                self._padding_shift, self._padding_unit, index, self._batch["shifts"], self._batch["unit_shifts"],
                self.layout.groups, macro_edges, selected.numel(), capacity, len(pos) - 1,
                len(pos) * self._nspecies - 1, 128,
            )
            torch.searchsorted(self.layout.groups, self._group_keys, out=self.layout.group_offsets)
            torch.searchsorted(index[1], self._receiver_keys, out=self.layout.receiver_offsets)
            self._micro_reference = pos.detach().clone()
            self.edge_refreshes += 1
        else:
            self.certificate_hits += 1
        self._batch["positions"].copy_(pos)
        return self._batch

    @torch.no_grad()
    def step(self, dt_fs):
        if len(self.positions) > 1024:
            return super().step(dt_fs)
        dt = float(dt_fs) * units.fs
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("invalid time step")
        if self._force is None or self._force_version != self.positions._version:
            self.evaluate()
        n = len(self.positions)
        _advance_and_check[(1,)](
            self.positions, self.velocities, self._masses, self._force,
            self._micro_reference, self._reference, self._trial, self._half, self._action_code,
            n, dt, self.micro_skin, self.macro_extension, self.cutoff,
            triton.next_power_of_2(n), num_warps=4, enable_fp_fusion=False,
        )
        self._scheduled_action = True
        try:
            energy, force = self.evaluate(self._trial)
        finally:
            self._scheduled_action = False
        _finish[(triton.cdiv(3 * n, 128),)](
            self._trial, self._half, self._masses, force, self.positions, self.velocities,
            3 * n, dt, 128, enable_fp_fusion=False,
        )
        self._force = force
        self._force_version = self.positions._version
        return energy, force


class ScheduledPath(ResidentPath):
    def __init__(self, initial, calculator, dt, skin, layout, capture=True, macro_extension=1.0):
        self.runner = ScheduledRunner(initial, calculator, skin, layout, capture, macro_extension)
        self.dt = dt
        self.energy, self.forces = self.runner.evaluate()
