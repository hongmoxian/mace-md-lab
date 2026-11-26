"""SchNetPack's published rMD17 ethanol PaiNN, with exchangeable edge geometry."""

import hashlib
from importlib.metadata import version
from pathlib import Path

import torch
from ase import units

from .geometry import edge_geometry_from_positions

CHECKPOINT_SHA256 = "260f262f4021215dc65edec62c2c3128b910b413e3070024db3b21510ef30b29"
UPSTREAM = "https://raw.githubusercontent.com/atomistic-machine-learning/schnetpack/69d26f3842a012f1f1e461ad7f1984d353843c8d"
CHECKPOINT_URL = f"{UPSTREAM}/examples/trained_models/rmd17_ethanol/painn_1/best_model"
STRUCTURE_URL = f"{UPSTREAM}/tests/testdata/md_ethanol.xyz"


class MolecularPaiNN:
    include_self = False

    def __init__(self, weights, device, dtype, atomic_numbers):
        from schnetpack import properties, transform

        filename = Path(weights)
        if hashlib.sha256(filename.read_bytes()).hexdigest() != CHECKPOINT_SHA256:
            raise ValueError(
                "supply the published SchNetPack rMD17 ethanol painn_1/best_model"
            )
        if version("schnetpack") != "2.3.0":
            raise ValueError("use SchNetPack 2.3.0 for this pretrained checkpoint")
        self.dtype = dtype
        self.device = torch.device(device)
        self.z = torch.as_tensor(atomic_numbers, device=device, dtype=torch.long)
        if self.z.ndim != 1 or not len(self.z) or not set(self.z.tolist()) <= {1, 6, 8}:
            raise ValueError("expected an isolated molecule containing H, C or O")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("use float32 or float64")

        # The checksum above authenticates the full-module pickle before loading.
        self.model = torch.load(filename, weights_only=False, map_location="cpu")
        self.model.to(device).eval().requires_grad_(False)
        self.rep = self.model.representation.to(dtype=dtype)
        self.head = self.model.output_modules[0].to(dtype=dtype)
        self.cutoff = float(self.rep.cutoff_fn.cutoff)
        self.neighbors = transform.MatScipyNeighborList(self.cutoff)
        self.prop = properties
        self.batch = torch.zeros_like(self.z)
        self.count = self.z.new_tensor([len(self.z)])
        self.unit = units.kcal / units.mol
        self.metadata = {
            "family": "PaiNN",
            "framework": "SchNetPack",
            "schnetpack_version": version("schnetpack"),
            "checkpoint": "rMD17 ethanol painn_1",
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "checkpoint_url": CHECKPOINT_URL,
            "training_molecule": "ethanol",
            "n_atom_basis": self.rep.n_atom_basis,
            "n_interactions": self.rep.n_interactions,
            "native_energy_unit": "kcal/mol",
            "energy_unit": "eV",
            "native_energy_to_eV": self.unit,
            "length_unit": "angstrom",
            "cutoff_A": self.cutoff,
            "periodic": False,
            "geometry": "radial distances and unit vectors for every PaiNN interaction",
        }

    def _static_inputs(self):
        p = self.prop
        return {p.Z: self.z, p.idx_m: self.batch, p.n_atoms: self.count}

    def _official_inputs(self, positions):
        p = self.prop
        data = {
            p.Z: self.z.detach().cpu(),
            p.R: positions.detach().cpu(),
            p.cell: torch.zeros(3, 3, dtype=positions.dtype),
            p.pbc: torch.zeros(3, dtype=torch.bool),
        }
        data = {k: v.to(positions.device) for k, v in self.neighbors(data).items()}
        data.update(self._static_inputs())
        data[p.R] = positions
        return data

    def native_energy(self, positions):
        """Unmodified upstream neighbor policy, representation, energy head and offsets."""
        data = self._official_inputs(positions)
        for module in self.model.input_modules:
            data = module(data)
        data = self.head(self.rep(data))
        return self.model.postprocess(data)["energy"].reshape(()) * self.unit

    def energy(self, positions, edge_index, mode):
        from schnetpack.nn import scatter_add

        if (
            positions.shape != (len(self.z), 3)
            or positions.dtype != self.dtype
            or positions.device != self.z.device
        ):
            raise ValueError(
                "positions must match the adapter molecule, dtype and device"
            )
        if (
            edge_index.ndim != 2
            or edge_index.shape[0] != 2
            or edge_index.dtype != torch.long
            or edge_index.device != self.z.device
        ):
            raise ValueError(
                "expected a device-local (2, E) tensor of directed integer edges"
            )
        sender, center = edge_index.unbind(0)
        distance, unit = edge_geometry_from_positions(positions, edge_index, mode=mode)
        # SchNetPack uses R_j - R_i, pointing from receiving center i toward sender j.
        direction = -unit
        d = distance.unsqueeze(-1)
        rep = self.rep
        filters = rep.filter_net(rep.radial_basis(d)) * rep.cutoff_fn(d).unsqueeze(-1)
        by_layer = filters.split(3 * rep.n_atom_basis, dim=-1)
        q = rep.embedding(self.z).unsqueeze(1)
        mu = q.new_zeros((len(self.z), 3, rep.n_atom_basis))
        for interaction, mixing, weights in zip(
            rep.interactions, rep.mixing, by_layer, strict=True
        ):
            q, mu = interaction(q, mu, weights, direction, center, sender, len(self.z))
            q, mu = mixing(q, mu)
        # One molecule per call. The upstream readout infers this known size via
        # int(batch[-1]), which synchronizes the GPU during CUDA Graph capture.
        per_atom = self.head.outnet(q.squeeze(1))
        energy = scatter_add(per_atom, self.batch, dim_size=1).squeeze(-1)
        data = self._static_inputs()
        data["energy"] = energy
        return self.model.postprocess(data)["energy"].reshape(()) * self.unit


def load_adapter(weights, device, dtype, atomic_numbers):
    return MolecularPaiNN(weights, device, dtype, atomic_numbers)
