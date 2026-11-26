"""Evaluate reciprocal-edge reuse in a trained MACE on a periodic system.

Run with --model, --structure and --output. The timed region is a full energy and
coordinate-force evaluation on a prepared graph. Use --md-steps and --prime-md
for a full moving NVE comparison. The pair map is valid only for the ordered
edge list and image shifts used to build it.
"""

import argparse
from copy import copy
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import torch
from ase.io import read
import mace_md_lab  # noqa: F401 -- install scoped e3nn compatibility before import.
from e3nn.nn import FullyConnectedNet
from mace.calculators import MACECalculator
from mace.modules.blocks import (
    RadialEmbeddingBlock,
    RealAgnosticInteractionBlock,
    RealAgnosticResidualInteractionBlock,
)
from mace.modules.models import MACE, ScaleShiftMACE

from mace_md_lab.bench import file_hash, package_version, sync, write_report
from mace_md_lab.reverse_profile import energy_output
from mace_md_lab.transforms import optimized_model
from mace_md_lab.runtime import ResidentMACE


def make_pair_map(edge_index, unit_shifts):
    """Return one representative per reciprocal pair and a directed-to-pair map."""
    index = edge_index.detach().cpu().numpy()
    image = unit_shifts.detach().cpu().numpy()
    if index.ndim != 2 or index.shape[0] != 2 or image.shape != (index.shape[1], 3):
        raise ValueError("edge_index and integer lattice shifts must agree")
    if len(image) == 0 or len(image) % 2 or not np.array_equal(image, np.rint(image)):
        raise ValueError("expected a nonempty reciprocal graph with integral shifts")
    src, dst = index
    image = image.astype(np.int64)
    # Select one orientation by endpoints, then by the first nonzero image
    # component for periodic images of the same atom.
    reverse = src > dst
    tied = src == dst
    for axis in image.T:
        reverse |= tied & (axis < 0)
        tied &= axis == 0
    if np.any(tied):
        raise ValueError("zero-displacement self edges are unsupported")
    keys = np.column_stack(
        (
            np.minimum(src, dst),
            np.maximum(src, dst),
            np.where(reverse[:, None], -image, image),
        )
    )
    order = np.lexsort((reverse, *keys.T[::-1]))
    first, second = order[::2], order[1::2]
    if not (
        np.array_equal(keys[first], keys[second])
        and not reverse[first].any()
        and reverse[second].all()
    ):
        raise ValueError("each directed periodic edge must have exactly one reciprocal")
    if np.any(np.all(keys[first[1:]] == keys[first[:-1]], axis=1)):
        raise ValueError("duplicate periodic edge")
    inverse = np.empty(len(src), dtype=np.int64)
    inverse[first] = inverse[second] = np.arange(len(first))
    return tuple(torch.as_tensor(v, device=edge_index.device) for v in (first, inverse))


class PairedRadialWeights(torch.nn.Module):
    """Apply a distance-only rowwise MLP once per reciprocal edge pair."""

    def __init__(self, net, representative, inverse):
        super().__init__()
        self.net = net
        self.register_buffer("representative", representative)
        self.register_buffer("inverse", inverse)

    def forward(self, edge_features):
        values = self.net(edge_features.index_select(0, self.representative))
        return values.index_select(0, self.inverse)


def prepared_view(model, pair_map):
    if (
        package_version("mace-torch") != "0.3.16"
        or type(model) not in (MACE, ScaleShiftMACE)
        or type(model.radial_embedding) is not RadialEmbeddingBlock
        or hasattr(model.radial_embedding, "distance_transform")
    ):
        raise ValueError("expected ordinary MACE 0.3.16 with distance-only radial features")
    interactions = []
    for layer in model.interactions:
        if (
            type(layer) not in (
                RealAgnosticInteractionBlock, RealAgnosticResidualInteractionBlock
            )
            or not isinstance(layer.conv_tp_weights, FullyConnectedNet)
        ):
            raise ValueError("unsupported interaction or radial network")
        view = copy(layer)
        view._modules = {
            **layer._modules,
            "conv_tp_weights": PairedRadialWeights(layer.conv_tp_weights, *pair_map),
        }
        interactions.append(view)
    result = copy(model)
    result._modules = {**model._modules, "interactions": torch.nn.ModuleList(interactions)}
    return result


def evaluate(model, batch):
    positions = batch["positions"].detach().requires_grad_(True)
    energy = energy_output(model, {**batch, "positions": positions})
    force = -torch.autograd.grad(energy, positions)[0]
    return energy.detach(), force.detach()


class PairedTrajectory(ResidentMACE):
    """Rebuild the pair map on every change of the exact ordered physical graph."""

    def __init__(self, atoms, calculator, skin, certify_edges):
        local = copy(calculator)
        self.original = calculator.models[0]
        local.models = [self.original]
        self.pair_revision = -1
        super().__init__(atoms, local, skin=skin, certify_edges=certify_edges)

    def prepare(self, positions=None):
        batch = super().prepare(positions)
        if self.edge_refreshes != self.pair_revision:
            pair_map = make_pair_map(batch["edge_index"], batch["unit_shifts"])
            if self.pair_revision < 0:
                self.calculator.models = [prepared_view(self.original, pair_map)]
            else:
                for layer in self.calculator.models[0].interactions:
                    weights = layer.conv_tp_weights
                    weights.representative, weights.inverse = pair_map
            self.pair_revision = self.edge_refreshes
        return batch


DELTA_UNITS = ["position_A", "velocity_A_per_ASE_time", "energy_eV", "force_eV_A"]


def validate_md_delta(delta, n_atoms):
    limits = [1e-8, 1e-7, 1e-8 * n_atoms, 1e-6]
    if not np.isfinite(delta).all() or np.any(np.asarray(delta) > limits):
        raise ValueError(f"moving trajectory comparison failed: {delta}")


def time_md(atoms, calculator, args, device):
    if not atoms.has("momenta") or not np.any(atoms.get_velocities()):
        raise ValueError("supply a periodic moving fixture with initial momenta")
    paths = {"directed": ResidentMACE, "paired": PairedTrajectory}
    timings = {name: [] for name in paths}
    snapshots, refreshes = {}, {}
    prime_delta = None
    if args.prime_md:
        # Exercise the entire range of graph shapes outside the clock on both paths.
        # The timed trials still restart from the same positions and velocities.
        trajectories = {}
        for name, path in paths.items():
            runner = path(atoms, calculator, skin=0.12, certify_edges=True)
            runner.evaluate()
            snapshots_at_each_step = []
            for _ in range(args.md_warmup + args.md_steps):
                energy, force = runner.step(0.5)
                snapshots_at_each_step.append(
                    (runner.positions.clone(), runner.velocities.clone(), energy, force)
                )
            trajectories[name] = snapshots_at_each_step
        differences = [
            [float((x - y).abs().max().item()) for x, y in zip(lhs, rhs)]
            for lhs, rhs in zip(trajectories["directed"], trajectories["paired"])
        ]
        maximum_delta = np.max(differences, axis=0)
        validate_md_delta(maximum_delta, len(atoms))
        prime_delta = dict(zip(DELTA_UNITS, maximum_delta.tolist()))
    for rep in range(args.md_repeats):
        for name in (paths if rep % 2 == 0 else reversed(paths)):
            runner = paths[name](atoms, calculator, skin=0.12, certify_edges=True)
            runner.evaluate()
            for _ in range(args.md_warmup):
                runner.step(0.5)
            start_count = runner.edge_refreshes
            sync(device)
            start = perf_counter()
            for _ in range(args.md_steps):
                energy, force = runner.step(0.5)
            sync(device)
            timings[name].append((perf_counter() - start) * 1000 / args.md_steps)
            refreshes[name] = runner.edge_refreshes - start_count
            if rep == 0:
                snapshots[name] = (
                    runner.positions.clone(), runner.velocities.clone(), energy, force
                )
    delta = [
        float((x - y).abs().max().item())
        for x, y in zip(snapshots["directed"], snapshots["paired"])
    ]
    validate_md_delta(delta, len(atoms))
    return {
        "scope": "moving NVE including topology refresh and pair-map refresh",
        "dt_fs": 0.5,
        "skin_A": 0.12,
        "warmup_steps": args.md_warmup,
        "preflight_trajectory_each_route": args.prime_md,
        "preflight_max_deviation": prime_delta,
        "timed_steps": args.md_steps,
        "raw_ms_step": timings,
        "median_ms_step": {k: median(v) for k, v in timings.items()},
        "directed_over_paired": median(timings["directed"]) / median(timings["paired"]),
        "graph_refreshes_timed": refreshes,
        "final_max_deviation": dict(zip(DELTA_UNITS, delta)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--structure", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--adjoint", choices=("none", "both"), default="none")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--md-steps", type=int, default=0)
    parser.add_argument("--md-repeats", type=int, default=3)
    parser.add_argument("--md-warmup", type=int, default=3)
    parser.add_argument("--prime-md", action="store_true")
    args = parser.parse_args()
    if (
        min(args.md_warmup, args.md_steps) < 0
        or min(args.repeats, args.md_repeats) < 1
        or args.output.resolve() in {args.model.resolve(), args.structure.resolve()}
    ):
        parser.error("positive repeats and a separate output path are required")
    device = torch.device(args.device)
    atoms = read(args.structure, index=args.frame)
    calculator = MACECalculator(
        model_paths=[str(args.model)], device=args.device, default_dtype="float64"
    )
    model = calculator.models[0].eval()
    batch = calculator._atoms_to_batch(atoms).to_dict()
    batch = {
        k: v.double() if torch.is_tensor(v) and v.is_floating_point() else v
        for k, v in batch.items()
    }
    start = perf_counter()
    pair_map = make_pair_map(batch["edge_index"], batch["unit_shifts"])
    sync(device)
    setup_ms = (perf_counter() - start) * 1000
    timings = {"directed": [], "paired": []}
    reference_e, reference_f = evaluate(model, batch)
    with optimized_model(model, args.adjoint, atoms.numbers):
        variants = {"directed": model, "paired": prepared_view(model, pair_map)}
        errors = {}
        for name, variant in variants.items():
            en, force = evaluate(variant, batch)
            de = float((en - reference_e).abs().item()) / len(atoms)
            df = float((force - reference_f).abs().max().item())
            if not np.isfinite([de, df]).all() or de > 1e-8 or df > 1e-6:
                raise ValueError(f"{name} differs from stock MACE E/F: {de}, {df}")
            errors[name] = {"energy_eV_atom": de, "max_force_eV_A": df}
        for rep in range(args.repeats):
            for name in (variants if rep % 2 == 0 else reversed(variants)):
                sync(device)
                start = perf_counter()
                evaluate(variants[name], batch)
                sync(device)
                timings[name].append((perf_counter() - start) * 1000)
        md = time_md(atoms, calculator, args, device) if args.md_steps else None
    report = {
        "scope": (
            "Energy and conservative forces on a prepared graph; "
            "excludes pair construction; optional moving_md is separate"
        ),
        "moving_md": md,
        "interaction_specs": [
            {
                "type": type(layer).__name__,
                "node_feats": str(layer.node_feats_irreps),
                "edge_feats": str(layer.edge_feats_irreps),
                "output": str(layer.target_irreps),
                "radial_mlp": [
                    list(parameter.shape)
                    for parameter in layer.conv_tp_weights.parameters()
                    if parameter.ndim == 2
                ],
            }
            for layer in model.interactions
        ],
        "checkpoint_sha256": file_hash(args.model),
        "structure_sha256": file_hash(args.structure),
        "checkpoint": args.model.name,
        "structure": args.structure.name,
        "frame": args.frame,
        "atoms": len(atoms),
        "directed_edges": len(pair_map[1]),
        "pairs": len(pair_map[0]),
        "periodic_directed_edges": int((batch["unit_shifts"] != 0).any(1).sum().item()),
        "torch": str(torch.__version__),
        "mace": package_version("mace-torch"),
        "device": args.device,
        "cpu_threads": torch.get_num_threads(),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "adjoint": args.adjoint,
        "dtype": "float64",
        "pair_setup_ms": setup_ms,
        "errors": errors,
        "raw_ms": timings,
        "median_ms": {k: median(v) for k, v in timings.items()},
        "directed_over_paired": median(timings["directed"]) / median(timings["paired"]),
    }
    write_report(args.output, report)


if __name__ == "__main__":
    main()
