"""Measure both interactions in a complete MACE energy-to-coordinate derivative.

Run on a CUDA device. CUDA event ranges cover the active stream including
launch gaps. A single identity autograd node joins the differentiable inputs
or outputs at each module boundary, including tensors passed by keyword.
"""

import argparse
from contextlib import ExitStack
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import torch
from mace.calculators import MACECalculator
from torch.utils._pytree import tree_flatten, tree_unflatten

from mace_md_lab.bench import file_hash, package_version, starting_structure, write_report
from mace_md_lab.reverse_profile import energy_output
from mace_md_lab.runtime import ResidentMACE
from mace_md_lab.transforms import optimized_model


class BackwardBoundary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, event, *values):
        ctx.event = event
        ctx.set_materialize_grads(False)
        return values

    @staticmethod
    def backward(ctx, *gradients):
        ctx.event.record()
        return (None, *gradients)


def wrap_boundary(tree, event):
    leaves, spec = tree_flatten(tree)
    active = [i for i, leaf in enumerate(leaves) if torch.is_tensor(leaf) and leaf.requires_grad]
    if not active:
        raise ValueError("interaction has no differentiable tensors at boundary")
    marked = BackwardBoundary.apply(event, *(leaves[i] for i in active))
    for i, value in zip(active, marked, strict=True):
        leaves[i] = value
    return tree_unflatten(leaves, spec)


def cuda_event():
    return torch.cuda.Event(enable_timing=True)


def install_boundaries(module, stack):
    enter, leave, backward_start, backward_end = (cuda_event() for _ in range(4))

    def before(_mod, args, kwargs):
        inputs = wrap_boundary((args, kwargs), backward_end)
        enter.record()
        return inputs

    def after(_mod, args, kwargs, output):
        leave.record()
        return wrap_boundary(output, backward_start)

    stack.callback(module.register_forward_pre_hook(before, with_kwargs=True).remove)
    stack.callback(module.register_forward_hook(after, with_kwargs=True).remove)
    return enter, leave, backward_start, backward_end


def measure_force(model, batch, events, boundaries):
    start, energy_ready, end = events
    positions = batch["positions"].detach().requires_grad_(True)
    inputs = {**batch, "positions": positions}
    torch.cuda.synchronize()
    start.record()
    energy = energy_output(model, inputs)
    energy_ready.record()
    force = -torch.autograd.grad(energy, positions)[0]
    end.record()
    torch.cuda.synchronize()
    times = {
        "energy_forward": start.elapsed_time(energy_ready),
        "energy_to_force_reverse": energy_ready.elapsed_time(end),
        "energy_and_force": start.elapsed_time(end),
    }
    # Verify the two backwards are nonoverlapping and fall inside the total reverse window.
    if boundaries:
        cursor = energy_ready
        for i in reversed(range(len(boundaries))):
            f0, f1, b0, b1 = boundaries[i]
            if cursor.elapsed_time(b0) < 0 or b0.elapsed_time(b1) < 0 or b1.elapsed_time(end) < 0:
                raise RuntimeError("overlapping or invalid backward boundary")
            times[f"interaction_{i+1}_forward"] = f0.elapsed_time(f1)
            times[f"interaction_{i+1}_reverse"] = b0.elapsed_time(b1)
            cursor = b1
    return energy.detach(), force.detach(), times


def check(energy, force, reference, n_atoms):
    e0, f0 = reference
    delta = [abs(float(energy) - e0) / n_atoms, float(np.max(np.abs(force.cpu().numpy() - f0)))]
    if not np.isfinite(delta).all() or delta[0] > 1e-8 or delta[1] > 1e-6:
        raise ValueError(f"disagreement with stock MACE on the same coordinates: {delta}")
    return {"energy_eV_atom": delta[0], "force_eV_A": delta[1]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--structure", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--variant", choices=("base", "first", "folded", "second", "both", "bothfolded"), default="base")
    p.add_argument("--frames", type=int, default=6)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--steps", type=int, default=48)
    p.add_argument("--repeats", type=int, default=6)
    p.set_defaults(frame=0, seed=17, temperature=300.)
    args = p.parse_args()
    if min(args.frames, args.stride, args.steps, args.repeats, args.warmup) < 1:
        p.error("frame/step counts and repeats must be positive")
    if args.output.resolve() in (args.model.resolve(), args.structure.resolve()):
        p.error("supply a separate output path")
    initial, momentum_source = starting_structure(args)
    options = dict(model_paths=[str(args.model)], device="cuda", default_dtype="float64")
    stock, candidate = (MACECalculator(**options) for _ in range(2))
    model = candidate.models[0]
    samples, raw_md = [], []
    with ExitStack() as transforms:
        transforms.enter_context(optimized_model(model, "both", initial.numbers))
        if args.variant != "base":
            model.eval().requires_grad_(False)
        if args.variant in ("first", "folded", "both", "bothfolded"):
            from mace_md_lab.grouped_first import install_grouped_first

            transforms.enter_context(install_grouped_first(
                model, initial.numbers, fold_linear=args.variant in ("folded", "bothfolded")
            ))
        if args.variant in ("second", "both", "bothfolded"):
            from mace_md_lab.grouped_second import install_second_interaction

            transforms.enter_context(install_second_interaction(model))
        runner = ResidentMACE(initial, candidate, skin=0.12, certify_edges=True)
        runner.evaluate()
        for _ in range(args.warmup):
            runner.step(0.5)
        events = [cuda_event() for _ in range(3)]
        for index in range(args.frames):
            for _ in range(args.stride):
                runner.step(0.5)
            batch = runner.prepare()
            frame = initial.copy()
            frame.positions = runner.positions.cpu().numpy()
            frame.calc = stock
            reference = frame.get_potential_energy(), frame.get_forces()
            sample = {"md_step": args.warmup + (index + 1) * args.stride,
                      "directed_edges": int(batch["edge_index"].shape[1])}
            # Swap measurement order on successive geometries.
            for label in (("plain", "marked") if index % 2 == 0 else ("marked", "plain")):
                with ExitStack() as stack:
                    marks = [install_boundaries(layer, stack) for layer in model.interactions] if label == "marked" else []
                    measure_force(model, batch, events, marks)  # Prime event creation and operators.
                    energy, force, timings = measure_force(model, batch, events, marks)
                    sample[label] = timings
                    sample[label + "_errors"] = check(energy, force, reference, len(initial))
            samples.append(sample)

        for _ in range(args.repeats):
            runner = ResidentMACE(initial, candidate, skin=0.12, certify_edges=True)
            runner.evaluate()
            for _ in range(args.warmup):
                runner.step(.5)
            torch.cuda.synchronize()
            t0 = perf_counter()
            for _ in range(args.steps):
                runner.step(.5)
            torch.cuda.synchronize()
            raw_md.append(1000 * (perf_counter() - t0) / args.steps)
    medians = {label: {key: median(row[label][key] for row in samples)
                       for key in samples[0][label]} for label in ("plain", "marked")}
    report = {
        "passed": True, "checkpoint_sha256": file_hash(args.model),
        "structure_sha256": file_hash(args.structure), "atoms": len(initial),
        "formula": initial.get_chemical_formula(), "mace": package_version("mace-torch"),
        "torch": str(torch.__version__), "gpu": torch.cuda.get_device_name(),
        "dtype": "float64", "adjoint": "both", "variant": args.variant, "mode": "eager", "skin_A": .12,
        "dt_fs": .5, "velocity_source": momentum_source,
        "force_scope": "Prepared graphs from moving MD frames. Marked ranges are in situ CUDA event spans across first/second interaction forward and coordinate-adjoint boundaries on the current stream. They include launch gaps and internal reductions. Their disjoint reverse order is checked. Hooks may change autograd scheduling. Plain timings omit boundaries. Neither includes graph preparation or integration.",
        "step_scope": "Independent hook-free E/F moving velocity-Verlet trials with endpoint synchronization, neighbor updates and integration. One warmup sequence before each trial.",
        "timed_steps": args.steps, "warmup_steps": args.warmup,
        "samples": samples, "median_force_ms": medians, "raw_md_ms_step": raw_md,
        "median_md_ms_step": median(raw_md),
    }
    write_report(args.output, report)
    print(report["formula"], medians, report["median_md_ms_step"])


if __name__ == "__main__":
    main()
