"""Test first-layer receiver/species grouping in trained molecular SchNet and PaiNN."""

import argparse
import hashlib
from importlib import import_module
import math
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import torch
from ase import units
from ase.io import read
from torch.nn import functional as F

from mace_md_lab.portable.benchmark import dump, reference, validate
from mace_md_lab.portable.geometry import edge_geometry_from_positions
from mace_md_lab.portable.runner import MolecularNVE


def species_projection(weight, bias, source):
    """Fold each element's source features into the last affine radial map."""
    if bias is None:
        bias = weight.new_zeros(weight.shape[0])
    augmented = torch.cat((weight.T, bias[None, :]), dim=0)
    return (augmented[None, :, :] * source[:, None, :]).flatten(0, 1).contiguous()


class GroupedFirst:
    """Fixed composition and weights. Every call consumes current directed indices."""

    def __init__(self, adapter, family):
        self.adapter = adapter
        self.family = family
        self.cutoff = adapter.cutoff
        numbers = adapter.numbers if family == "schnet" else adapter.z
        elements, self.species = torch.unique(numbers, sorted=True, return_inverse=True)
        self.n = len(numbers)
        self.s = len(elements)
        if family == "schnet":
            model = adapter.model
            if (adapter.metadata.get("version") != "2.6.1" or adapter.metadata.get("target") != "QM9 U0"
                    or model.dipole or model.scale != 1.0):
                raise ValueError("expected the pinned PyG 2.6.1 QM9 U0 additive SchNet adapter")
            block = model.interactions[0]
            if (len(block.mlp) != 3 or block.conv.nn is not block.mlp
                    or not isinstance(block.mlp[2], torch.nn.Linear)):
                raise ValueError("expected a shared first-block MLP with a final affine projection")
            last = block.mlp[2]
            source = block.conv.lin1(model.embedding(elements))
            self.scalar = species_projection(last.weight, last.bias, source)
        else:
            rep = adapter.rep
            if (rep.share_filters or len(rep.electronic_embeddings)
                    or not isinstance(rep.filter_net.activation, torch.nn.Identity)
                    or rep.filter_net.bias is None):
                raise ValueError("expected the unshared affine filter and elemental embedding of ethanol PaiNN")
            width = rep.n_atom_basis
            last = rep.filter_net
            source = rep.interactions[0].interatomic_context_net(rep.embedding(elements))
            self.scalar = species_projection(last.weight[:width], last.bias[:width], source[:, :width])
            self.vector = species_projection(
                last.weight[width:2*width], last.bias[width:2*width], source[:, width:2*width]
            )

    def moments(self, values, edge_index):
        sender, center = edge_index
        groups = center * self.s + self.species[sender]
        sums = values.new_zeros(self.n * self.s, *values.shape[1:]).index_add(0, groups, values)
        return sums.reshape(self.n, self.s, *values.shape[1:])

    def energy(self, positions, edge_index, mode):
        if self.family == "schnet":
            return self.schnet(positions, edge_index, mode)
        return self.painn(positions, edge_index, mode)

    def schnet(self, positions, edge_index, mode):
        adapter = self.adapter
        model = adapter.model
        r = edge_geometry_from_positions(positions, edge_index, mode=mode, quantities="distance")
        basis = model.distance_expansion(r)
        block = model.interactions[0]
        hidden = block.mlp[1](block.mlp[0](basis))
        scale = 0.5 * (torch.cos(r * math.pi / block.conv.cutoff) + 1.0)
        augmented = torch.cat((hidden, torch.ones_like(scale[:, None])), dim=-1)
        sums = self.moments(augmented * scale[:, None], edge_index).reshape(self.n, -1)
        msg = sums @ self.scalar
        h = model.embedding(adapter.numbers) + block.lin(block.act(block.conv.lin2(msg)))
        for block in model.interactions[1:]:
            h = h + block(h, edge_index, r, basis)
        h = model.lin2(model.act(model.lin1(h)))
        h = h * model.std + model.mean + model.atomref(adapter.numbers)
        return model.readout(h, adapter.batch, dim=0, dim_size=1).reshape(())

    def painn(self, positions, edge_index, mode):
        from schnetpack.nn import scatter_add

        adapter = self.adapter
        rep = adapter.rep
        r, unit = edge_geometry_from_positions(positions, edge_index, mode=mode)
        direction = -unit
        d = r[:, None]
        radial = rep.radial_basis(d)
        cutoff = rep.cutoff_fn(d)
        values = torch.cat((radial[:, 0], torch.ones_like(d)), dim=-1) * cutoff
        # Moment channel zero carries the scalar term; the other three are Cartesian.
        angular = torch.cat((torch.ones_like(d), direction), dim=-1)
        sums = self.moments(angular[:, :, None] * values[:, None, :], edge_index)
        by_component = sums.transpose(1, 2).reshape(self.n, 4, -1)
        q = rep.embedding(adapter.z).unsqueeze(1) + by_component[:, :1] @ self.scalar
        mu = by_component[:, 1:] @ self.vector
        q, mu = rep.mixing[0](q, mu)
        width = 3 * rep.n_atom_basis
        filt = rep.filter_net
        remaining = F.linear(radial, filt.weight[width:], filt.bias[width:]) * cutoff[..., None]
        sender, center = edge_index
        for block, mix, weight in zip(
            rep.interactions[1:], rep.mixing[1:], remaining.split(width, dim=-1), strict=True
        ):
            q, mu = block(q, mu, weight, direction, center, sender, self.n)
            q, mu = mix(q, mu)
        per_atom = adapter.head.outnet(q.squeeze(1))
        data = adapter._static_inputs()
        data["energy"] = scatter_add(per_atom, adapter.batch, dim_size=1).squeeze(-1)
        return adapter.model.postprocess(data)["energy"].reshape(()) * adapter.unit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--framework", choices=("schnet", "painn"), required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--structure", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=6)
    p.add_argument("--dt-fs", type=float, default=0.25)
    args = p.parse_args()
    if (min(args.steps, args.warmup, args.repeats) < 1 or not 0 < args.dt_fs < float("inf")
            or args.output.resolve() in (args.model.resolve(), args.structure.resolve())):
        p.error("positive step counts, finite time step and a separate output file required")
    atoms = read(args.structure, index=0)
    if len(atoms) < 2 or np.any(atoms.pbc) or atoms.constraints:
        p.error("supply a finite unconstrained molecule")
    device, dtype = torch.device(args.device), torch.float64
    rng = np.random.default_rng(17)
    mass = atoms.get_masses()
    velocity = rng.normal(size=(len(atoms), 3)) * np.sqrt(units.kB * 300 / mass[:, None])
    atoms.set_velocities(velocity - np.average(velocity, axis=0, weights=mass))
    module = import_module("mace_md_lab.portable." + args.framework)
    stock = module.load_adapter(args.model, device, dtype, torch.as_tensor(atoms.numbers.copy(), device=device))
    grouped = GroupedFirst(stock, args.framework)
    ref = reference(stock, atoms, device, dtype, args.dt_fs, args.steps + args.warmup)
    direction = rng.normal(size=(len(atoms), 3))
    direction -= direction.mean(axis=0)
    direction = torch.as_tensor(direction / np.linalg.norm(direction), device=device, dtype=dtype)
    capture = device.type == "cuda"
    def sync():
        if capture:
            torch.cuda.synchronize(device)
    def build(adapter):
        return MolecularNVE(atoms, adapter, dtype, device, "analytic", capture, args.dt_fs)
    check, results = {}, {}
    variants = {"stock": stock, "grouped": grouped}
    for name, adapter in variants.items():
        md = build(adapter)
        check[name] = validate(md, ref, stock, len(atoms), dtype, direction)
        if not check[name]["passed"]:
            raise RuntimeError(f"{name} numerical gate failed: {check[name]}")
        results[name] = []
        del md
    for trial in range(args.repeats):
        for name in (list(variants) if trial % 2 == 0 else list(reversed(variants))):
            md = build(variants[name])
            for _ in range(args.warmup):
                md.step()
            n0, cap0 = md.graph.refreshes, md.call.captures
            sync()
            start = perf_counter()
            for _ in range(args.steps):
                md.step()
            sync()
            elapsed = 1000 * (perf_counter() - start) / args.steps
            results[name].append(dict(ms_step=elapsed, graph_refreshes=md.graph.refreshes - n0,
                                      graph_captures=md.call.captures - cap0))
            del md
    init = build(stock)
    sender, center = init.graph.edges
    counts = len(torch.unique(center * grouped.s + grouped.species[sender]))
    medians = {k: median(v["ms_step"] for v in rows) for k, rows in results.items()}
    dump(args.output, dict(
        model=stock.metadata, framework=args.framework, device=str(device), dtype="float64",
        torch=str(torch.__version__), gpu=torch.cuda.get_device_name(device) if capture else None,
        cpu_threads=torch.get_num_threads(), structure=args.structure.name,
        structure_sha256=hashlib.sha256(args.structure.read_bytes()).hexdigest(),
        atoms=len(atoms), directed_edges=len(sender), occupied_groups=counts,
        allocated_groups=grouped.s * grouped.n, input_source_types=grouped.s,
        steps=args.steps, warmup=args.warmup, dt_fs=args.dt_fs, repeats=args.repeats,
        velocity_seed=17, cuda_graph=capture, geometry="analytic", passed=True,
        scope="Full FP64 molecular velocity-Verlet using identical graph policy. Only first interaction differs.",
        checks=check, trials=results, median_ms_step=medians,
        relative_step_time_reduction=1-medians["grouped"]/medians["stock"],
    ))
    print(args.framework, medians, flush=True)


if __name__ == "__main__":
    main()
