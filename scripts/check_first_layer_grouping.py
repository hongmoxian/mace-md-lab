"""Check receiver/species grouping in MACE-MP-0 Small, on CPU in FP64.

Count arithmetic on the complete input graph. Check messages and coordinate VJPs
on all incoming edges of the first atom of each element by default, or every
atom with --all-receivers. Batching bounds the memory used by the reference.
"""

import argparse
from pathlib import Path

import torch
from ase.io import read
from mace.calculators import MACECalculator
from mace.modules.utils import get_edge_vectors_and_lengths

from mace_md_lab.bench import file_hash, package_version, write_report


SMALL_SHA256 = "2ddb079cee0e131eaaf6912ba581b394551ead283e95c99cfe78c605d10b5736"


def compare_receivers(
    centers, probe, species, batch, model, source_features, radial, weight, angular_dimensions
):
    layer = model.interactions[0]
    attrs = batch["node_attrs"]
    width, channels = weight.shape[0], source_features.shape[1]
    final_projection = radial[-1]
    selected = torch.isin(batch["edge_index"][1], centers)
    edge_index = batch["edge_index"][:, selected]
    source, receiver = edge_index
    center_index = torch.full((len(species),), -1, dtype=torch.long)
    center_index[centers] = torch.arange(len(centers))
    receiver_local = center_index[receiver]
    groups, edge_group = torch.unique(
        torch.stack((receiver_local, species[source]), dim=1), dim=0, return_inverse=True
    )
    species_example = torch.tensor([(species == z).nonzero()[0, 0] for z in groups[:, 1]], dtype=torch.long)

    positions = batch["positions"].detach().requires_grad_(True)
    vectors, lengths = get_edge_vectors_and_lengths(positions, edge_index, batch["shifts"][selected])
    harmonics = model.spherical_harmonics(vectors)
    radial_features, cutoff = model.radial_embedding(lengths, attrs, edge_index, model.atomic_numbers)
    hidden = radial_features
    for transform in radial[:-1]:
        hidden = transform(hidden)

    weights = final_projection(hidden)
    if cutoff is not None:
        weights = weights * cutoff
        hidden = hidden * cutoff
    messages = layer.conv_tp(source_features[source], harmonics, weights)
    direct = messages.new_zeros(len(centers), messages.shape[1]).index_add(0, receiver_local, messages)

    # Each scalar input channel couples diagonally to each spherical-harmonic
    # component of one angular order. The two schedules preserve that layout.
    products = harmonics.unsqueeze(-1) * hidden.unsqueeze(1)
    moments = products.new_zeros(len(groups), harmonics.shape[1], width).index_add(0, edge_group, products)
    pieces = []
    for order, dimension in enumerate(angular_dimensions):
        by_order = moments[:, order**2 : order**2 + dimension]
        projection = weight[:, order * channels : (order + 1) * channels]
        grouped = (by_order @ projection).transpose(1, 2) * source_features[species_example, :, None]
        summed = grouped.new_zeros(len(centers), channels, dimension).index_add(0, groups[:, 0], grouped)
        pieces.append(summed.flatten(1))
    reordered = torch.cat(pieces, dim=1)
    original_vjp = torch.autograd.grad((direct * probe).sum(), positions, retain_graph=True)[0]
    grouped_vjp = torch.autograd.grad((reordered * probe).sum(), positions)[0]
    return (
        len(source), len(groups), float((direct - reordered).detach().abs().max()),
        float(direct.detach().abs().max()), original_vjp, grouped_vjp,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--structure", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--all-receivers", action="store_true")
    parser.add_argument("--batch-receivers", type=int, default=8)
    args = parser.parse_args()
    if args.batch_receivers < 1:
        parser.error("batch size must be positive")
    if args.output.resolve() in (args.model.resolve(), args.structure.resolve()):
        parser.error("input and output files must differ")
    if file_hash(args.model) != SMALL_SHA256:
        parser.error("this check implements the UVU layout of MACE-MP-0 Small")

    torch.set_num_threads(1)
    dtype = torch.float64
    calc = MACECalculator(model_paths=[str(args.model)], device="cpu", default_dtype="float64")
    model = calc.models[0].eval().requires_grad_(False)
    layer = model.interactions[0]
    radial = list(layer.conv_tp_weights.children())
    final_projection = radial[-1]
    width = 64
    channels = 128
    angular_dimensions = (1, 3, 5, 7)
    weight = final_projection(torch.eye(width, dtype=dtype))
    assert weight.shape == (width, channels * len(angular_dimensions))
    assert not torch.count_nonzero(final_projection(torch.zeros(1, width, dtype=dtype)))

    atoms = read(args.structure, index=0)
    batch = calc._atoms_to_batch(atoms).to_dict()
    species = torch.as_tensor(atoms.numbers.copy())
    source, receiver = batch["edge_index"]
    keys = torch.stack((receiver, species[source]), dim=1)
    group_count = len(torch.unique(keys, dim=0))
    edge_count = len(source)
    centers = torch.tensor([(species == z).nonzero()[0, 0] for z in torch.unique(species)])
    if args.all_receivers:
        centers = torch.arange(len(atoms))
    source_features = layer.linear_up(model.node_embedding(batch["node_attrs"]))
    probe = torch.randn(
        len(centers), channels * sum(angular_dimensions), dtype=dtype,
        generator=torch.Generator().manual_seed(79),
    )
    ref_vjp = torch.zeros_like(batch["positions"])
    test_vjp = torch.zeros_like(ref_vjp)
    msg_error = ref_msg_max = partial_vjp_error = 0.0
    total_edges = total_groups = 0
    for start in range(0, len(centers), args.batch_receivers):
        section = slice(start, start + args.batch_receivers)
        checked_edges, checked_groups, diff, magnitude, reference, candidate = compare_receivers(
            centers[section], probe[section], species, batch, model,
            source_features, radial, weight, angular_dimensions,
        )
        total_edges += checked_edges
        total_groups += checked_groups
        msg_error = max(msg_error, diff)
        ref_msg_max = max(ref_msg_max, magnitude)
        partial_vjp_error = max(partial_vjp_error, float((reference - candidate).abs().max()))
        ref_vjp += reference
        test_vjp += candidate
    vjp_error = float((ref_vjp - test_vjp).abs().max())
    passed = msg_error < 1e-10 and max(vjp_error, partial_vjp_error) < 1e-9
    if args.all_receivers:
        passed = passed and total_edges == edge_count and total_groups == group_count

    report = {
        "checkpoint_sha256": SMALL_SHA256, "structure_sha256": file_hash(args.structure),
        "torch": str(torch.__version__), "mace": package_version("mace-torch"), "dtype": "float64",
        "full_graph": {
            "atoms": len(atoms), "directed_edges": edge_count, "occupied_receiver_species_groups": group_count,
            "radial_dimensions": [10, 64, 64, 64, 512], "angular_dimensions": list(angular_dimensions),
            "direct_final_projection_MACs": edge_count * width * channels * len(angular_dimensions),
            "grouped_final_projection_MACs": group_count * sum(angular_dimensions) * width * channels,
            "grouped_edge_moment_MACs": edge_count * sum(angular_dimensions) * width,
            "count_scope": "Nominal dense forward multiply-accumulates for named operations only. No MD timing.",
        },
        "selected_receivers": {
            "indices_zero_based": centers.tolist(), "atomic_numbers": species[centers].tolist(),
            "incoming_edges": total_edges, "occupied_receiver_species_groups": total_groups,
            "probe_seed": 79, "max_message_error": msg_error, "max_coordinate_VJP_error": vjp_error,
            "reference_max_abs_message": ref_msg_max,
            "reference_max_abs_coordinate_VJP": float(ref_vjp.abs().max()),
            "scope": "Layer-one message before receiver linear projection; VJP with seeded random message cotangent.",
        },
        "passed": passed,
    }
    if args.all_receivers:
        report["selected_receivers"].update(
            all_receivers=True, batch_receivers=args.batch_receivers,
            max_batch_coordinate_VJP_error=partial_vjp_error,
        )
    write_report(args.output, report)
    if not passed:
        raise SystemExit("message or coordinate VJP verification failed")


if __name__ == "__main__":
    main()
