"""Attribute force-evaluation time on prepared MACE graphs for local structures."""

import argparse
from contextlib import ExitStack
from itertools import islice
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np
import torch
from ase.io import iread
from mace.calculators import MACECalculator

from .bench import file_hash, package_version, write_report
from .runtime import validate_calculator


class Mark:
    def __init__(self, device):
        self.event = (
            torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        )
        self.cpu = None

    def record(self):
        if self.event is None:
            self.cpu = perf_counter()
        else:
            self.event.record()

    def to_ms(self, end):
        if self.event is None:
            return 1000 * (end.cpu - self.cpu)
        return self.event.elapsed_time(end.event)


def layer_groups(model):
    groups = {}
    for name in ("spherical_harmonics", "radial_embedding"):
        if hasattr(model, name):
            groups[name] = getattr(model, name)
    for name in ("interactions", "products"):
        for index, layer in enumerate(getattr(model, name, ())):
            groups[f"{name}[{index}]"] = layer
    if not groups:
        raise ValueError("checkpoint exposes no supported layers")
    layers = list(groups.values())
    if len(set(layers)) != len(layers):
        raise ValueError("profiling groups share a module")
    for layer in layers:
        descendants = set(layer.modules()) - {layer}
        if descendants.intersection(layers):
            raise ValueError("profiling groups overlap")
    return groups


class LayerTimer:
    def __init__(self, model, device):
        self.groups = layer_groups(model)
        self.pool = {name: [] for name in self.groups}
        self.device = device
        self.counts = {}
        self.active = False

    def reset(self):
        self.counts = dict.fromkeys(self.pool, 0)

    def install(self, stack):
        for name, module in self.groups.items():

            def before(_module, _args, name=name):
                if self.active:
                    index = self.counts[name]
                    if index == len(self.pool[name]):
                        self.pool[name].append((Mark(self.device), Mark(self.device)))
                    self.counts[name] += 1
                    self.pool[name][index][0].record()

            def after(_module, _args, _out, name=name):
                if self.active:
                    self.pool[name][self.counts[name] - 1][1].record()

            stack.callback(module.register_forward_pre_hook(before).remove)
            stack.callback(module.register_forward_hook(after).remove)

    def elapsed(self):
        return {
            name: sum(a.to_ms(b) for a, b in slots[: self.counts[name]])
            for name, slots in self.pool.items()
        }


def energy_and_force(model, inputs, leaf, t0, t1, t2, hooks=None):
    if hooks:
        hooks.reset()
        hooks.active = True
    t0.record()
    try:
        energy = model(
            inputs,
            training=False,
            compute_force=False,
            compute_stress=False,
            compute_virials=False,
        )["energy"].reshape(())
    finally:
        if hooks:
            hooks.active = False
    t1.record()
    gradient = (
        torch.autograd.grad(energy, leaf, allow_unused=True)[0]
        if energy.requires_grad
        else None
    )
    force = torch.zeros_like(leaf) if gradient is None else -gradient
    t2.record()
    return energy.detach(), force.detach()


def evaluate(batch, model, device, markers, hooks=None):
    inputs = batch.copy()
    leaf = batch["positions"].detach().requires_grad_(True)
    inputs["positions"] = leaf
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    with torch.enable_grad():
        e, f = energy_and_force(model, inputs, leaf, *markers, hooks=hooks)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start, middle, end = markers
    timing = {
        "energy_forward_ms": start.to_ms(middle),
        "position_backward_ms": middle.to_ms(end),
        "forward_and_backward_ms": start.to_ms(end),
    }
    if hooks:
        parts = hooks.elapsed()
        parts["other_forward_ms"] = timing["energy_forward_ms"] - sum(parts.values())
        timing["forward_parts_ms"] = parts
        timing["forward_module_calls"] = hooks.counts.copy()
    return e, f, timing


def run(args, report):
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda") or (
        device.type == "cuda" and not torch.cuda.is_available()
    ):
        raise ValueError(f"unavailable device {device}")
    if device.type == "cuda":
        torch.cuda.set_device(
            device.index if device.index is not None else torch.cuda.current_device()
        )
    if device.type != "cuda" and (args.enable_oeq or args.enable_cueq):
        raise ValueError("GPU acceleration requires a CUDA device")
    options = dict(
        model_paths=[args.model],
        device=str(device),
        default_dtype=args.dtype,
        enable_oeq=args.enable_oeq,
        enable_cueq=args.enable_cueq,
    )
    if args.head is not None:
        options["head"] = args.head
    calc = MACECalculator(**options)
    validate_calculator(calc)
    if args.head is not None and args.head != calc.head:
        raise ValueError("requested MACE head was not selected")
    for flag in ("enable_oeq", "enable_cueq"):
        if getattr(args, flag) and not getattr(calc, "_" + flag, False):
            raise ValueError(f"requested {flag} was not activated")
    model = calc.models[0]
    dtype = next(model.parameters()).dtype
    report["model"] = {
        "class": type(model).__name__,
        "head": calc.head,
        "cutoff_A": float(calc.r_max),
    }
    report["gpu"] = (
        torch.cuda.get_device_name(device) if device.type == "cuda" else None
    )
    source = list(islice(iread(args.structure, index=":"), args.frames))
    if not source:
        raise ValueError("structure file contains no frames")
    indices = [i % len(source) for i in range(args.frames)]
    report["source_frame_indices"] = indices
    batches, errors = [], []
    markers = tuple(Mark(device) for _ in range(3))
    for index, atoms in enumerate(source):
        if not len(atoms):
            raise ValueError(f"empty source frame {index}")
        atoms.calc = calc
        ref_e = atoms.get_potential_energy()
        ref_f = atoms.get_forces(apply_constraint=False)
        batch = calc._atoms_to_batch(atoms).to_dict()
        batch = {
            k: v.to(dtype=dtype) if torch.is_tensor(v) and v.is_floating_point() else v
            for k, v in batch.items()
        }
        batches.append(batch)
        e, f, _ = evaluate(batch, model, device, markers)
        energy_error = abs(e.item() - ref_e) / len(atoms)
        force_error = float(np.max(np.abs(f.cpu().numpy() - ref_f)))
        if not np.isfinite([energy_error, force_error]).all():
            raise ValueError(
                f"nonfinite reference or prediction on source frame {index}"
            )
        errors.append(
            {
                "energy_eV_atom": energy_error,
                "force_eV_A": force_error,
                "atoms": len(atoms),
                "directed_edges": int(batch["edge_index"].shape[1]),
            }
        )
    e_tol, f_tol = (1e-8, 1e-6) if args.dtype == "float64" else (2e-4, 3e-3)
    passed = all(
        row["energy_eV_atom"] <= e_tol and row["force_eV_A"] <= f_tol for row in errors
    )
    report["validation"] = {
        "passed": passed,
        "reference": "unmodified MACECalculator",
        "energy_tolerance_eV_atom": e_tol,
        "force_tolerance_eV_A": f_tol,
        "frames": errors,
    }
    if not passed:
        raise ValueError("fixed-geometry energy/force validation failed")

    timer = LayerTimer(model, device)
    with ExitStack() as hooks:
        timer.install(hooks)
        for batch in batches:
            evaluate(
                batch, model, device, markers, timer
            )  # Prime kernels and hook events.
        hooked = [
            evaluate(batches[i], model, device, markers, timer)[2] for i in indices
        ]
    plain = [evaluate(batches[i], model, device, markers)[2] for i in indices]
    report["samples"] = [
        {"source_index": i, "with_hooks": h, "without_hooks": p}
        for i, h, p in zip(indices, hooked, plain)
    ]
    report["mean_ms"] = {
        "forward_parts": {
            k: mean(h["forward_parts_ms"][k] for h in hooked)
            for k in hooked[0]["forward_parts_ms"]
        },
        "with_hooks": {k: mean(h[k] for h in hooked) for k in plain[0]},
        "without_hooks": {k: mean(p[k] for p in plain) for k in plain[0]},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, type=Path, help="trusted local .model checkpoint"
    )
    parser.add_argument("--structure", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float64", choices=("float64", "float32"))
    parser.add_argument(
        "--frames",
        type=int,
        default=3,
        help="number of samples; cycle available input frames if fewer are present",
    )
    parser.add_argument("--enable-oeq", action="store_true")
    parser.add_argument("--enable-cueq", action="store_true")
    parser.add_argument("--head")
    args = parser.parse_args(argv)
    if args.frames < 1 or not args.model.is_file() or not args.structure.is_file():
        parser.error(
            "a positive sample count and two existing input files are required"
        )
    if args.output.resolve() in (args.model.resolve(), args.structure.resolve()):
        parser.error("output path must differ from the inputs")
    report = {
        "checkpoint": args.model.name,
        "checkpoint_sha256": file_hash(args.model),
        "structure": args.structure.name,
        "structure_sha256": file_hash(args.structure),
        "torch": str(torch.__version__),
        "mace_torch": package_version("mace-torch"),
        "device": args.device,
        "dtype": args.dtype,
        "enable_oeq": args.enable_oeq,
        "enable_cueq": args.enable_cueq,
        "scope": "Prepared real-geometry graphs only. CUDA events measure time between markers on the "
        "current stream, including launch gaps. CPU uses wall time. Forward parts are disjoint "
        "and sum to energy_forward_ms. Position backward is measured once as a whole. Excludes "
        "neighbor construction, data transfers, integration and stress. Hookless totals quantify "
        "instrumentation effects. Repeated indices are repeated evaluations of the same geometry.",
    }
    try:
        run(args, report)
    except Exception as exc:
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        write_report(args.output, report)
        raise
    write_report(args.output, report)
    print(f"Wrote {args.frames} prepared-graph samples to {args.output}")
    return 0


if __name__ == "__main__":
    main()
