"""Isolate the frozen product's fused value and feature-derivative CUDA kernel."""
import argparse
from pathlib import Path
from statistics import median

import torch
from ase.io import read
from mace.calculators import MACECalculator

from mace_md_lab.bench import file_hash, write_report
from mace_md_lab.frozen_product import _launch
from mace_md_lab.transforms import optimized_model


def compare(block, types):
    n, ne = types.shape
    channels = block.table.shape[-1]
    x = torch.randn(n, channels, block.dimension, device="cuda", dtype=types.dtype)
    out = {name: (torch.empty_like(x[:, :, 0]), torch.empty_like(x))
           for name in ("original", "channel_tile")}

    def run(name, attrs):
        _launch("value_and_derivative", x, attrs, block.table,
                *out[name], block.terms, tiled=name == "channel_tile")

    errors = {}
    mixed = types.clone()
    mixed[0].zero_()
    mixed[1].zero_()
    mixed[1, :2] = mixed.new_tensor([0.25, 0.75])
    mixed[2].mul_(-0.4)
    for label, attrs in (("physical", types), ("zero_mixed_scaled", mixed)):
        for name in out:
            run(name, attrs)
        errors[label] = {kind: float((a - b).abs().max())
                         for kind, a, b in zip(("value", "jacobian"), *out.values())}

    graphs = {}
    for name in out:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(32):
                run(name, types)
        graph.replay()
        graphs[name] = graph
    torch.cuda.synchronize()
    samples = {name: [] for name in out}
    for r in range(12):
        for name in (tuple(graphs) if r % 2 == 0 else tuple(reversed(graphs))):
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            graphs[name].replay()
            stop.record()
            stop.synchronize()
            samples[name].append(start.elapsed_time(stop) * 1000 / 32)
    return {"node_count": n, "elements": ne, "channels": channels, "components": block.dimension,
            "monomials": len(block.terms), "errors": errors, "us_per_kernel": samples,
            "median_us": {name: median(vals) for name, vals in samples.items()}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--structure", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    torch.manual_seed(17)
    atoms = read(a.structure, index=0)
    if len(atoms) < 3:
        raise ValueError("the mixed-attribute control needs at least three nodes")
    model = MACECalculator(model_paths=[str(a.model)], device="cuda", default_dtype="float64").models[0]
    z = model.atomic_numbers.cpu().tolist()
    ix = torch.tensor([z.index(int(number)) for number in atoms.numbers], device="cuda")
    types = torch.nn.functional.one_hot(ix, len(z)).double()
    with optimized_model(model, "product", atoms.numbers):
        results = [compare(block.symmetric_contractions, types) for block in model.products]
    passed = all(max(control.values()) < 1e-12 for layer in results for control in layer["errors"].values())
    write_report(a.output, {"passed": passed, "model_sha256": file_hash(a.model),
                            "structure_sha256": file_hash(a.structure), "device": torch.cuda.get_device_name(),
                            "dtype": "float64", "seed": 17, "layers": results,
                            "scope": "CUDA events around 32 captured fused polynomial value/Jacobian launches. "
                                     "Actual frozen weights and species; seeded normal feature probes. "
                                     "No integration, topology, or downstream VJP scaling."})
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
