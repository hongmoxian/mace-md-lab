"""Single isolated molecule, exact radius graph, optional force-call CUDA recording."""

import torch
from ase import units


class MolecularGraph:
    """An all-pairs cutoff enumerator for the small nonperiodic molecular examples."""

    def __init__(self, positions, cutoff, include_self=False):
        ids = torch.arange(len(positions), device=positions.device)
        dst, src = ids.repeat_interleave(len(ids)), ids.repeat(len(ids))
        keep = (
            (src != dst) if not include_self else torch.ones_like(src, dtype=torch.bool)
        )
        self.pairs = torch.stack([src[keep], dst[keep]])
        self.cutoff = float(cutoff)
        self.origin = None
        self.gap = 0.0
        self.edges = None
        self.refreshes = 0

    def update(self, x):
        x = x.detach()
        if self.origin is not None:
            d = torch.linalg.vector_norm(x - self.origin, dim=1).max().item()
            if 2 * d < self.gap:
                return self.edges
        src, dst = self.pairs
        r = torch.linalg.vector_norm(x[dst] - x[src], dim=1)
        self.edges = self.pairs[:, r < self.cutoff].contiguous()
        self.gap = (
            torch.min(torch.abs(r - self.cutoff)).item() if r.numel() else float("inf")
        )
        self.gap -= 32 * torch.finfo(x.dtype).eps * max(1.0, self.cutoff)
        self.origin = x.clone()
        self.refreshes += 1
        return self.edges


class ForceCall:
    def __init__(self, adapter, mode, capture):
        self.adapter, self.mode, self.capture = adapter, mode, capture
        self.graph = None
        self.captures = self.replays = 0

    def _evaluate(self, x, edges):
        with torch.enable_grad():
            leaf = x.detach().requires_grad_(True)
            e = self.adapter.energy(leaf, edges, self.mode).reshape(())
            f = -torch.autograd.grad(e, leaf)[0]
        return e.detach(), f.detach()

    def __call__(self, positions, edges):
        if not self.capture:
            return self._evaluate(positions, edges)
        if positions.device.type != "cuda":
            raise ValueError("CUDA recording requires a CUDA device")
        if self.graph is None or self.static_edges.shape != edges.shape:
            active = torch.cuda.current_stream(positions.device)
            active.synchronize()
            self.graph = None
            self.static_x, self.static_edges = positions.clone(), edges.clone()
            side = torch.cuda.Stream(device=positions.device)
            side.wait_stream(active)
            with torch.cuda.stream(side):
                for _ in range(3):
                    self._evaluate(self.static_x, self.static_edges)
            active.wait_stream(side)
            record = torch.cuda.CUDAGraph()
            with torch.cuda.graph(record, stream=side):
                self.outputs = self._evaluate(self.static_x, self.static_edges)
            self.graph = record
            self.captures += 1
        else:
            self.static_x.copy_(positions)
            self.static_edges.copy_(edges)
        self.graph.replay()
        self.replays += 1
        return tuple(t.clone() for t in self.outputs)


class MolecularNVE:
    def __init__(self, atoms, adapter, dtype, device, mode, capture, dt_fs):
        if any(atoms.pbc) or atoms.constraints:
            raise ValueError("supply a finite, unconstrained molecule")
        self.x = torch.as_tensor(
            atoms.positions, dtype=torch.float64, device=device
        ).clone()
        self.v = torch.as_tensor(
            atoms.get_velocities(), dtype=torch.float64, device=device
        ).clone()
        self.m = torch.as_tensor(
            atoms.get_masses(), dtype=torch.float64, device=device
        )[:, None]
        self.dtype, self.dt = dtype, dt_fs * units.fs
        self.graph = MolecularGraph(
            self.x.to(dtype), adapter.cutoff, getattr(adapter, "include_self", False)
        )
        self.call = ForceCall(adapter, mode, capture)
        self.e, self.f = self.evaluate()

    def evaluate(self):
        x = self.x.to(self.dtype)
        return self.call(x, self.graph.update(x))

    def kinetic(self):
        return 0.5 * (self.m * self.v.square()).sum()

    def step(self):
        vhalf = self.v + 0.5 * self.dt * self.f / self.m
        self.x = self.x + self.dt * vhalf
        self.e, self.f = self.evaluate()
        self.v = vhalf + 0.5 * self.dt * self.f / self.m
