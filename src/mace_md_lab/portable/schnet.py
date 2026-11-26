"""Use PyG SchNet's published QM9 U0 checkpoint for conservative force timings.

Prepare the model with `python -m mace_md_lab.portable.schnet schnet-u0.pt`.
Coordinates are in angstroms. The model predicts molecular U0 in eV and was
trained on equilibrium energies; its gradients here test computational fidelity.
"""

import argparse
import hashlib
import io
import pickle
import types
import urllib.request
import zipfile
from pathlib import Path

import torch
from torch import nn

from .geometry import edge_geometry_from_positions

SOURCE_URL = "https://www.quantum-machine.org/datasets/trained_schnet_models.zip"
MEMBER = "trained_schnet_models/qm9_energy_U0/best_model"
SOURCE_SHA256 = "ca567abbd43b5d0740a558d708761e232cf6d5e732f3ec498bc608e67105c62d"


def _backbone():
    from torch_geometric.datasets.qm9 import atomrefs
    from torch_geometric.nn import SchNet

    reference = torch.zeros(100, 1)
    reference[[1, 6, 7, 8, 9], 0] = torch.as_tensor(atomrefs[7])
    return SchNet(
        hidden_channels=128,
        num_filters=128,
        num_interactions=6,
        num_gaussians=50,
        cutoff=10.0,
        atomref=reference,
    )


class _LegacyCheckpoint(pickle.Unpickler):
    """Recover parameters of the SHA-pinned historical SchNetPack container."""

    def find_class(self, module, name):
        if module.startswith("schnetpack."):
            # Only its state is read. No historical module executes inference.
            return nn.Module
        if module == "torch.nn.backends.thnn" and name == "_get_thnn_function_backend":
            return _legacy_backend
        return super().find_class(module, name)


def _legacy_backend():
    return None


def convert_checkpoint(destination, archive=None):
    """Map the official U0 model to PyG following SchNet.from_qm9_pretrained."""
    if archive is None:
        with urllib.request.urlopen(SOURCE_URL, timeout=90) as response:
            archive = io.BytesIO(response.read())
    with zipfile.ZipFile(archive) as source:
        data = source.read(MEMBER)
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA256:
        raise ValueError("Official QM9 U0 model checksum mismatch")
    legacy = types.ModuleType("legacy_qm9_pickle")
    legacy.Unpickler = _LegacyCheckpoint
    legacy.load = pickle.load
    old = torch.load(
        io.BytesIO(data), pickle_module=legacy, weights_only=False, map_location="cpu"
    )
    net = _backbone()
    net.embedding.weight = old.representation.embedding.weight
    for dst, src in zip(net.interactions, old.representation.interactions, strict=True):
        for a, b in [
            (dst.mlp[0], src.filter_network[0]),
            (dst.mlp[2], src.filter_network[1]),
            (dst.lin, src.dense),
            (dst.conv.lin1, src.cfconv.in2f),
            (dst.conv.lin2, src.cfconv.f2out),
        ]:
            a.load_state_dict({name: getattr(b, name) for name in a.state_dict()})
    head = old.output_modules[0]
    for dst, src in zip((net.lin1, net.lin2), head.out_net[1].out_net, strict=True):
        dst.load_state_dict({name: getattr(src, name) for name in dst.state_dict()})
    if head.atom_pool.average or head.atomref is None:
        raise ValueError("Expected the additive QM9 U0 head with atomic references")
    net.atomref.weight = head.atomref.weight
    payload = dict(
        state_dict=net.state_dict(),
        mean=float(head.standardize.mean.item()),
        std=float(head.standardize.stddev.item()),
        source_sha256=SOURCE_SHA256,
    )
    torch.save(payload, destination)


class _AllPairsGraph(nn.Module):
    """Exact isolated-graph reference when the optional torch-cluster is absent."""

    def __init__(self, cutoff):
        super().__init__()
        self.cutoff = cutoff

    def forward(self, pos, batch):
        delta = pos[:, None] - pos[None, :]
        active = delta.detach().square().sum(-1) < self.cutoff**2
        active.fill_diagonal_(False)
        active &= batch[:, None] == batch[None, :]
        source, target = active.nonzero(as_tuple=True)
        index = torch.stack((source, target))
        return index, (pos[source] - pos[target]).norm(dim=-1)


class SchNetAdapter(nn.Module):
    def __init__(self, model, numbers, metadata):
        super().__init__()
        self.model = model
        self.register_buffer("numbers", numbers)
        self.register_buffer("batch", torch.zeros_like(numbers))
        self.metadata = metadata
        self.cutoff = model.cutoff

    def native_energy(self, positions):
        """Independent PyG SchNet.forward, including its interaction-graph call."""
        return self.model(self.numbers, positions, self.batch).reshape(())

    def energy(self, positions, edge_index, mode):
        """Use the original PyG learned layers on supplied physical directed edges."""
        m = self.model
        r = edge_geometry_from_positions(
            positions, edge_index, mode=mode, quantities="distance"
        )
        basis = m.distance_expansion(r)
        h = m.embedding(self.numbers)
        for interaction in m.interactions:
            h = h + interaction(h, edge_index, r, basis)
        h = m.lin2(m.act(m.lin1(h)))
        h = h * m.std + m.mean + m.atomref(self.numbers)
        # Supplying the known graph count avoids a GPU scalar read during capture.
        return m.readout(h, self.batch, dim=0, dim_size=1).reshape(())


def load_adapter(weights, device, dtype, atomic_numbers):
    """Load a converted U0 model; weights=None uses the local PyTorch cache."""
    import torch_geometric
    from torch_geometric.typing import WITH_TORCH_CLUSTER

    if weights is None:
        weights = Path(torch.hub.get_dir()) / "checkpoints" / "pyg-schnet-qm9-u0.pt"
        if not weights.exists():
            weights.parent.mkdir(parents=True, exist_ok=True)
            convert_checkpoint(weights)
    weights = Path(weights)
    saved = torch.load(weights, map_location="cpu", weights_only=True)
    if saved["source_sha256"] != SOURCE_SHA256:
        raise ValueError("Expected the published SchNet QM9 U0 checkpoint")
    model = _backbone()
    model.load_state_dict(saved["state_dict"], strict=True)
    model.mean, model.std = saved["mean"], saved["std"]
    model.scale = 1.0  # PyG target 7 has no post-conversion energy factor.
    # On a single molecule use the complete directed graph within cutoff.
    # For a matching reference avoid clipping the larger molecules at 32 neighbors.
    model.interaction_graph.max_num_neighbors = len(atomic_numbers)
    if not WITH_TORCH_CLUSTER:
        model.interaction_graph = _AllPairsGraph(model.cutoff)
    numbers = torch.as_tensor(atomic_numbers, dtype=torch.long, device=device)
    if not set(numbers.tolist()) <= {1, 6, 7, 8, 9}:
        raise ValueError("QM9 SchNet supports H/C/N/O/F")
    metadata = dict(
        framework="torch_geometric",
        version=torch_geometric.__version__,
        model="SchNet",
        target="QM9 U0",
        pretrained=True,
        source=SOURCE_URL,
        source_sha256=SOURCE_SHA256,
        weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
        hidden_channels=128,
        interactions=6,
        energy_unit="eV",
        length_unit="angstrom",
        native_edges=(
            "PyG radius_graph"
            if WITH_TORCH_CLUSTER
            else "all-pairs exact cutoff supplied to PyG SchNet.forward"
        ),
    )
    return (
        SchNetAdapter(model, numbers, metadata)
        .to(device=device, dtype=dtype)
        .eval()
        .requires_grad_(False)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--archive", type=Path, help="previously downloaded official ZIP"
    )
    args = parser.parse_args()
    convert_checkpoint(args.output, args.archive)
