"""Paired moving-atom ablations for conservative first-derivative implementations."""

import argparse
from contextlib import ExitStack
from pathlib import Path
from statistics import median
from time import perf_counter

import torch
from mace.calculators import MACECalculator

from .bench import (
    arguments as md_arguments,
    compare_trial,
    file_hash,
    numerical_gates,
    package_version,
    sample_trajectory,
    starting_structure,
    sync,
    time_trial,
    write_report,
)
from .transforms import BASE_MODES, MODES, optimized_model


def main(argv=None):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--modes", nargs="+", choices=MODES, default=list(BASE_MODES))
    p.add_argument("--replay", choices=("padded", "exact", "eager"), default="padded")
    options, rest = p.parse_known_args(argv)
    args = md_arguments(rest)
    if args.enable_cueq or args.enable_oeq:
        raise ValueError("specialized adjoints require stock e3nn operator inputs")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(
            device.index if device.index is not None else torch.cuda.current_device()
        )
    elif options.replay != "eager":
        raise ValueError("CPU execution requires --replay eager")
    mode_names = list(dict.fromkeys(["none", *options.modes]))
    graph_path = {
        "padded": "resident_padded",
        "exact": "resident_capture",
        "eager": "resident_certified" if args.certify_edges else "resident_cached",
    }[options.replay]
    initial, velocity_source = starting_structure(args)
    config = dict(
        model_paths=[args.model], device=str(device), default_dtype=args.dtype
    )
    if args.head:
        config["head"] = args.head
    native = MACECalculator(**config)
    choices = ["ase_ef", *mode_names]
    records = {s: [] for s in choices}
    gates = numerical_gates(args)
    report = {
        "checkpoint": args.model.name,
        "checkpoint_sha256": file_hash(args.model),
        "structure": args.structure.name,
        "structure_sha256": file_hash(args.structure),
        "formula": initial.get_chemical_formula(),
        "atoms": len(initial),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch": str(torch.__version__),
        "mace": package_version("mace-torch"),
        "dtype": args.dtype,
        "torch_threads": torch.get_num_threads(),
        "ase": package_version("ase"),
        "replay": options.replay,
        "certify_edges": args.certify_edges,
        "dt_fs": args.dt_fs,
        "steps": args.steps,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "skin_A": args.skin,
        "initial_velocities": velocity_source,
        "selected_head": native.head,
        "scope": "Same input, frozen checkpoint and precision. Setup and initial force are separate. "
        "Timed Verlet steps include the topology check, all refreshes, recaptures and CUDA synchronization. "
        "The ASE E/F path copies outputs to the host. One-time model transformation is measured separately.",
        "force_tolerances": gates,
        "validation": {},
        "transforms": {},
        "trials": records,
    }
    try:
        ref, drift, _, _ = sample_trajectory("ase_native", initial, native, args)
        with ExitStack() as contexts:
            calculators = {"ase_ef": native, "none": native}
            for name in mode_names:
                calc = native if name == "none" else MACECalculator(**config)
                calculators[name] = calc
                sync(device)
                start = perf_counter()
                info = contexts.enter_context(
                    optimized_model(calc.models[0], name, initial.numbers)
                )
                sync(device)
                report["transforms"][name] = {
                    **info,
                    "transform_ms": 1000 * (perf_counter() - start),
                }
            for name in choices:
                path = "ase_ef" if name == "ase_ef" else graph_path
                metrics, errors = compare_trial(
                    path, initial, calculators[name], args, ref, drift, gates
                )
                report["validation"][name] = {"metrics": metrics, "errors": errors}
                if errors:
                    raise ValueError(f"{name} did not match stock MACE: {errors}")
            for repeat in range(args.repeats):
                for name in (
                    choices[repeat % len(choices) :] + choices[: repeat % len(choices)]
                ):
                    path = "ase_ef" if name == "ase_ef" else graph_path
                    if device.type == "cuda":
                        sync(device)
                        allocated = torch.cuda.memory_allocated(device)
                        torch.cuda.reset_peak_memory_stats(device)
                    row = time_trial(
                        path, initial, calculators[name], args, device, repeat
                    )
                    if device.type == "cuda":
                        row["peak_extra_torch_allocated_bytes_including_setup"] = (
                            torch.cuda.max_memory_allocated(device) - allocated
                        )
                    if path in ("resident_padded", "resident_capture") and (
                        row["replays"] != args.steps or row["fallback_evaluations"]
                    ):
                        raise RuntimeError(f"{name} did not replay every measured step")
                    records[name].append(row)
        times = {
            name: median(t["ms_step"] for t in rows) for name, rows in records.items()
        }
        report.update(
            passed=True,
            median_ms_step=times,
            speedup_vs_stock_same_replay={
                s: times["none"] / t for s, t in times.items() if s != "ase_ef"
            },
        )
    except Exception as exc:
        report.update(passed=False, failure=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_report(args.output, report)
    print(report["median_ms_step"])


if __name__ == "__main__":
    main()
