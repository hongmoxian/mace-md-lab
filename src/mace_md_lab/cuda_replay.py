"""Capture stock MACE energy and autograd forces for one real structure."""

from functools import partial

import torch
from mace.data.padding_tools import build_fake_padding_graph
from mace.tools import torch_geometric


class ForceReplay:
    """Keep one CUDA Graph, recapturing only when the required shape changes.

    Raw replay uses the exact edge count. Optional padded replay adds a separate,
    disconnected graph with a single fake atom, following the official MACE ASE
    padding implementation. No padding edge ever has a real atom as an endpoint.
    """

    def __init__(self, device, padded=False):
        if device.type != "cuda":
            raise ValueError("capture_forces requires a CUDA model")
        self.device = device
        self.padded = bool(padded)
        self._graph = None
        self._generation = None
        self._stream = None
        self.capacity = 0
        self.captures = 0
        self.replays = 0
        self.fallback_evaluations = 0

    def _padded_inputs(self, batch, capacity, cutoff):
        natoms = batch["positions"].shape[0]
        nedges = batch["edge_index"].shape[1]
        # The dictionary is already collated but holds exactly one structure.
        # Its tensor shapes thus coincide with one complete graph, with scalar
        # fields carrying a leading singleton. Batch.from_data_list concatenates
        # them back along the same dimensions as the official calculator.
        real = torch_geometric.data.Data.from_dict(
            {k: v for k, v in batch.items() if k not in ("batch", "ptr")}
        )
        real.num_nodes = natoms
        fake = build_fake_padding_graph(
            real, num_atoms=1, num_edges=capacity - nedges, r_max=cutoff
        )
        # num_nodes is explicit, so Batch increments all fake edge endpoints to
        # index natoms even if the physical graph contains no edges.
        self._fake_shift = max(2.0 * cutoff, 1.0)
        inputs = torch_geometric.Batch.from_data_list([real, fake]).to_dict()
        return inputs

    def _copy_edges(self, batch):
        n = batch["edge_index"].shape[1]
        edges = self._inputs["edge_index"]
        edges[:, :n].copy_(batch["edge_index"])
        for key in ("shifts", "unit_shifts"):
            self._inputs[key][:n].copy_(batch[key])
        if self.padded and n < self.capacity:
            # Overwrite *all* trailing entries, including edges that were real
            # one step earlier. Their true graph is always the dummy graph.
            edges[:, n:].fill_(batch["positions"].shape[0])
            shifts = self._inputs["shifts"][n:]
            units = self._inputs["unit_shifts"][n:]
            shifts.zero_()
            units.zero_()
            shifts[:, 0].fill_(self._fake_shift)
            units[:, 0].fill_(1)

    def _capture(self, calculator, batch, evaluate):
        n = batch["edge_index"].shape[1]
        if self.padded:
            # Add 6.25% headroom and round up to eight edges. The minimum is 8.
            self.capacity = max(8, ((n + (n + 15) // 16 + 7) // 8) * 8)
            self._inputs = self._padded_inputs(batch, self.capacity, calculator.r_max)
            forward = partial(evaluate, num_real_atoms=batch["positions"].shape[0])
        else:
            self.capacity = n
            self._inputs = batch.copy()
            for key in ("positions", "edge_index", "shifts", "unit_shifts"):
                self._inputs[key] = batch[key].detach().clone()
            forward = evaluate

        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                forward(calculator, self._inputs)
        torch.cuda.current_stream(self.device).wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph, stream=stream):
                outputs = forward(calculator, self._inputs)
        except (RuntimeError, ValueError) as exc:
            raise RuntimeError(
                "CUDA Graph capture of this MACE energy and autograd force path failed. "
                f"Real edges {n}, capacity {self.capacity}. "
                "Run without capture_forces for eager execution."
            ) from exc
        self._graph = graph
        self._outputs = outputs
        self.captures += 1

    def __call__(self, calculator, batch, generation, evaluate):
        with torch.cuda.device(self.device):
            current = torch.cuda.current_stream(self.device)
            if self._stream is None:
                self._stream = current
            elif self._stream != current:
                raise RuntimeError("use a captured MACE runtime from one CUDA stream")
            n = batch["edge_index"].shape[1]
            recapture = n > self.capacity if self.padded else n != self.capacity
            if self._graph is None or recapture:
                if self._graph is not None:
                    current.synchronize()
                    self._graph = self._inputs = self._outputs = None
                self._capture(calculator, batch, evaluate)
            else:
                self._inputs["positions"][: batch["positions"].shape[0]].copy_(
                    batch["positions"].detach()
                )
                if generation != self._generation:
                    self._copy_edges(batch)
            self._generation = generation
            self._graph.replay()
            self.replays += 1
            # Values survive the next replay, including evaluate(other_x) while
            # velocity Verlet still owns the original-position cached force.
            energy, forces = self._outputs
            return energy.clone(), forces.clone()
