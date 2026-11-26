"""Replay individual force-path VJPs using activations and cotangents from stock MACE."""

import argparse
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from statistics import median

import numpy as np
import torch
from ase.io import iread
from mace.calculators import MACECalculator
from torch.utils._pytree import tree_flatten, tree_unflatten

from .bench import file_hash, package_version, sync, write_report
from .profile import Mark, layer_groups
from .runtime import validate_calculator


@dataclass
class LayerCall:
    name: str
    arguments: tuple
    outputs: list
    tree: object
    cotangents: list


def observe_calls(groups, stack):
    """Insert identity boundaries for each output use, including aliased tuple leaves."""
    calls = []

    def observe(name):
        def after(module, args, kwargs, output):
            leaves, spec = tree_flatten(output)
            call = LayerCall(name, (args, kwargs), leaves, spec, [None] * len(leaves))
            calls.append(call)
            exported = []
            for index, value in enumerate(leaves):
                if torch.is_tensor(value) and value.requires_grad:
                    boundary = value.view_as(value)

                    def save(grad, index=index):
                        call.cotangents[index] = grad.detach()

                    boundary.register_hook(save)
                    exported.append(boundary)
                else:
                    exported.append(value)
            return tree_unflatten(exported, spec)

        return after

    for name, module in groups.items():
        stack.callback(
            module.register_forward_hook(observe(name), with_kwargs=True).remove
        )
    return calls


def energy_output(model, batch):
    return model(
        batch,
        training=False,
        compute_force=False,
        compute_stress=False,
        compute_virials=False,
    )["energy"].sum()


def capture_calls(model, batch):
    inputs = batch.copy()
    pos = batch["positions"].detach().requires_grad_(True)
    inputs["positions"] = pos
    with ExitStack() as stack:
        calls = observe_calls(layer_groups(model), stack)
        energy = energy_output(model, inputs)
        grad = (
            torch.autograd.grad(energy, pos, allow_unused=True)[0]
            if energy.requires_grad
            else None
        )
    force = torch.zeros_like(pos) if grad is None else -grad.detach()
    return energy.detach(), force, calls


def fresh_inputs(arguments):
    # The same tensor passed twice is one independent leaf. Views that arrive
    # as distinct arguments are separate leaves at the local VJP boundary.
    leaves, spec = tree_flatten(arguments)
    detached, variables, aliases = [], [], {}
    for val in leaves:
        if not torch.is_tensor(val):
            detached.append(val)
            continue
        if id(val) not in aliases:
            leaf = val.detach()
            if val.requires_grad:
                leaf.requires_grad_(True)
                variables.append(leaf)
            aliases[id(val)] = leaf
        detached.append(aliases[id(val)])
    return tree_unflatten(detached, spec), variables


def local_vjp(module, call, device, markers, check_output=False):
    (args, kwargs), variables = fresh_inputs(call.arguments)
    outs, spec = tree_flatten(module(*args, **kwargs))
    if spec != call.tree:
        raise ValueError(f"output tree mismatch in {call.name}")
    error = 0.0
    if check_output:
        for ref, cur in zip(call.outputs, outs):
            if torch.is_tensor(ref):
                if not torch.is_tensor(cur) or cur.shape != ref.shape:
                    raise ValueError(f"output shape mismatch in {call.name}")
                if cur.is_floating_point() and cur.numel():
                    magnitude = float((ref.detach() - cur.detach()).abs().max().item())
                    if not np.isfinite(magnitude):
                        raise FloatingPointError(
                            f"nonfinite replay output in {call.name}"
                        )
                    error = max(error, magnitude)
    active, grads = [], []
    for y, upstream in zip(outs, call.cotangents):
        if upstream is not None:
            if not torch.is_tensor(y) or not y.requires_grad:
                raise ValueError(f"force path disappeared in {call.name}")
            active.append(y)
            grads.append(upstream)
    if not active or not variables:
        return None, [], error
    # Synchronize before the backward window: isolate it from local forward
    # dispatch or outstanding work. One CUDA event pair suffices per sample.
    sync(device)
    start, end = markers
    start.record()
    local = torch.autograd.grad(
        active, variables, grad_outputs=grads, allow_unused=True
    )
    end.record()
    sync(device)
    return (
        start.to_ms(end),
        [x.detach() if x is not None else None for x in local],
        error,
    )


def maximum_delta(left, right):
    if len(left) != len(right):
        raise ValueError("unequal local gradient lists")
    err, scale = 0.0, 0.0
    for a, b in zip(left, right):
        if a is None and b is None:
            continue
        if a is None or b is None:
            other = b if a is None else a
            diff = other.abs()
        else:
            if a.shape != b.shape:
                raise ValueError("local gradient shape mismatch")
            diff = (a - b).abs()
        if diff.numel():
            val = float(diff.max().item())
            if not np.isfinite(val):
                raise FloatingPointError("nonfinite local gradient")
            err = max(err, val)
        if a is not None and a.numel():
            scale = max(scale, float(a.abs().max().item()))
    return err, scale


def full_reverse(model, batch, device, markers):
    inputs = batch.copy()
    leaf = inputs["positions"].detach().requires_grad_(True)
    inputs["positions"] = leaf
    energy = energy_output(model, inputs)
    sync(device)
    start, end = markers
    start.record()
    grad = (
        torch.autograd.grad(energy, leaf, allow_unused=True)[0]
        if energy.requires_grad
        else None
    )
    end.record()
    sync(device)
    force = torch.zeros_like(leaf) if grad is None else -grad.detach()
    return energy.detach(), force, start.to_ms(end)


def profile_frame(native, candidate, atoms, device, reps, mode):
    ref = atoms.copy()
    ref.calc = native
    official_e, official_f = ref.get_potential_energy(), ref.get_forces(
        apply_constraint=False
    )
    batch = native._atoms_to_batch(atoms).to_dict()
    dt = next(native.models[0].parameters()).dtype
    batch = {
        k: v.to(dtype=dt) if torch.is_tensor(v) and v.is_floating_point() else v
        for k, v in batch.items()
    }
    atol, rtol = (1e-8, 1e-7) if dt == torch.float64 else (5e-4, 5e-4)
    marks = Mark(device), Mark(device)
    e, f, calls = capture_calls(native.models[0], batch)
    ef_lim = 1e-6 if dt == torch.float64 else 3e-3
    e_lim = 1e-8 if dt == torch.float64 else 2e-4

    def gate(e, f):
        de = abs(e.item() - official_e) / len(atoms)
        df = float(np.abs(f.cpu().numpy() - official_f).max())
        if not np.isfinite([de, df]).all() or de > e_lim or df > ef_lim:
            raise ValueError(
                f"full-checkpoint E/F disagreement (eV/atom, eV/A): {de}, {df}"
            )
        return {"energy_error_eV_atom": de, "max_force_error_eV_A": df}

    out = {
        "atoms": len(atoms),
        "edges": batch["edge_index"].shape[1],
        "reference_instrumentation": gate(e, f),
        "whole_reverse": {},
        "layers": {},
    }
    modes = {"stock": native.models[0]}
    if mode != "none":
        modes[mode] = candidate.models[0]
    for label, model in modes.items():
        e, f, _ = full_reverse(model, batch, device, marks)
        value = gate(e, f)
        raw = [full_reverse(model, batch, device, marks)[2] for _ in range(reps)]
        out["whole_reverse"][label] = {**value, "raw_ms": raw, "median_ms": median(raw)}
    groups = {label: layer_groups(model) for label, model in modes.items()}
    for call in calls:
        key = call.name
        if key in out["layers"]:
            raise ValueError(
                f"{key} invoked more than once; use a call-indexed adapter"
            )
        local_result = {
            "actual_output_cotangents": int(sum(x is not None for x in call.cotangents))
        }
        out["layers"][key] = local_result
        ref_grads = None
        for label in modes:
            layer = groups[label][key]
            value, grads, forward_error = local_vjp(
                layer, call, device, marks, check_output=True
            )
            if value is None:
                local_result[label] = {"on_position_gradient_path": False}
                continue
            if ref_grads is None:
                ref_grads = grads
            delta, scale = maximum_delta(ref_grads, grads)
            if delta > atol + rtol * scale:
                raise ValueError(
                    f"isolated VJP mismatch in {key}: {delta}, reference max {scale}"
                )
            out_scale = max(
                (
                    float(x.detach().abs().max().item())
                    for x in call.outputs
                    if torch.is_tensor(x) and x.numel() and x.is_floating_point()
                ),
                default=0.0,
            )
            if (
                not np.isfinite(forward_error)
                or forward_error > atol + rtol * out_scale
            ):
                raise ValueError(f"isolated forward mismatch in {key}: {forward_error}")
            raw = [local_vjp(layer, call, device, marks)[0] for _ in range(reps)]
            local_result[label] = {
                "on_position_gradient_path": True,
                "median_ms": median(raw),
                "raw_ms": raw,
                "maximum_local_output_error": forward_error,
                "maximum_local_VJP_error": delta,
                "reference_maximum_local_VJP": scale,
                "requested_input_gradients": len(grads),
                "used_input_gradients": sum(x is not None for x in grads),
            }
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--structure", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    p.add_argument(
        "--adjoint", choices=["none", "angular", "product", "both"], default="none"
    )
    p.add_argument("--frames", type=int, default=3)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--head")
    args = p.parse_args(argv)
    if (
        min(args.frames, args.repeats) < 1
        or not args.model.is_file()
        or not args.structure.is_file()
    ):
        p.error("positive frames/repeats and existing local input files are required")
    if args.output.resolve() in (args.model.resolve(), args.structure.resolve()):
        p.error("output must differ from inputs")
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda") or (
        device.type == "cuda" and not torch.cuda.is_available()
    ):
        raise ValueError("unsupported or unavailable device")
    if device.type == "cuda":
        torch.cuda.set_device(
            device.index if device.index is not None else torch.cuda.current_device()
        )
    frames = list(islice(iread(args.structure, index=":"), args.frames))
    if not frames:
        raise ValueError("trajectory has no frames")
    opts = dict(model_paths=[args.model], default_dtype=args.dtype, device=str(device))
    if args.head:
        opts["head"] = args.head
    report = {
        "schema_version": 1,
        "checkpoint": args.model.name,
        "checkpoint_sha256": file_hash(args.model),
        "structure": args.structure.name,
        "structure_sha256": file_hash(args.structure),
        "mode": args.adjoint,
        "device": str(device),
        "dtype": args.dtype,
        "torch": str(torch.__version__),
        "mace": package_version("mace-torch"),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "repeats": args.repeats,
        "frames": [],
        "scope": "Isolated module VJPs use stock activations and cotangents intercepted from the full "
        "energy-to-position derivative. Local inputs are detached boundary leaves; only inputs that required grad in "
        "stock inference are differentiated. Each local backward uses a fresh untimed forward. Independent module "
        "timings include launch gaps and cannot be added to recover whole backward or MD time. Whole reverse is "
        "measured separately without module hooks. Repeated source indices reuse the same geometry.",
    }
    try:
        native = MACECalculator(**opts)
        candidate = MACECalculator(**opts) if args.adjoint != "none" else native
        validate_calculator(native)
        validate_calculator(candidate)
        if args.head and (native.head != args.head or candidate.head != args.head):
            raise ValueError("requested head differs from selected head")
        context = nullcontext()
        if args.adjoint != "none":
            from .transforms import optimized_model

            species = np.unique(np.concatenate([frame.numbers for frame in frames]))
            context = optimized_model(
                candidate.models[0], args.adjoint, atomic_numbers=species
            )
        with context as metadata:
            report["optimization"] = metadata or {
                "mode": "none",
                "specialized_product_layers": 0,
            }
            for i in range(args.frames):
                j = i % len(frames)
                result = profile_frame(
                    native, candidate, frames[j], device, args.repeats, args.adjoint
                )
                report["frames"].append({"source_index": j, **result})
        report["passed"] = True
    except Exception as exc:
        report.update(
            passed=False, error={"type": type(exc).__name__, "message": str(exc)}
        )
        write_report(args.output, report)
        raise
    write_report(args.output, report)
    print(
        f"Wrote isolated VJP profile for {len(report['frames'])} samples to {args.output}"
    )
    return 0


if __name__ == "__main__":
    main()
