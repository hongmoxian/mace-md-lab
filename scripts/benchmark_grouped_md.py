"""Compare grouped interactions to optimized MACE and tuned XL in moving NVE.

The resident MACE E/F controls use the original public driver. E/F/stress controls
add the same native MACE stress request on every force evaluation. Native XL
always supplies all three observables. Supply its compact JSON to measure it in
the same set of interleaved trials.
"""
import argparse
from contextlib import ExitStack, contextmanager
from itertools import product
from pathlib import Path
from statistics import median
from time import perf_counter
from unittest.mock import patch
from types import SimpleNamespace
import os

import numpy as np
import torch
from ase.stress import full_3x3_to_voigt_6_stress
from mace.calculators import MACECalculator

from mace_md_lab.bench import (
    ASEPath, ResidentPath, file_hash, numerical_gates, package_version,
    starting_structure, sync, topology_counts, write_report,
)
from mace_md_lab.transforms import optimized_model


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--structure", type=Path, required=True)
    parser.add_argument("--symmetrix-json", type=Path)
    parser.add_argument("--include-prepared-xl", action="store_true",
                        help="also time the prepared native device graph with array Verlet")
    parser.add_argument("--xl-product-dir", type=Path,
                        help="also time the native compiled product with the prepared driver")
    parser.add_argument("--xl-stock-product-dir", type=Path,
                        help="use XL's upstream-generated product cubin in the stock reference")
    parser.add_argument("--xl-stock-r0-dir", type=Path,
                        help="also install XL's upstream-generated R0 in the stock and prepared references")
    parser.add_argument("--xl-stock-m0-dir", type=Path,
                        help="also install XL's upstream-generated M0 in all native evaluators")
    parser.add_argument("--xl-factored-r0-dir", type=Path,
                        help="also time the prepared driver with upstream M1 and the factored R0 reverse")
    parser.add_argument("--xl-launch-policy", choices=("automatic", "static"), default="automatic")
    parser.add_argument("--host-observables", action="store_true",
                        help="deliver energy, forces and stress to host memory on every timed step")
    parser.add_argument("--selector", default="cuda12-sm70")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--variants", nargs="+",
        choices=["base", "first", "folded", "second", "both", "bothfolded", "candidate", "replay", "sched", "schedtable", "schedpoly", "packed", "tangent", "schedpacked", "schedtangent", "table64", "table512", "replaytable", "replaytable512"],
        default=["base", "first"],
    )
    parser.add_argument("--outputs", nargs="+", choices=["ef", "efs", "edgeefs"], default=["ef", "efs"])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--skin", type=float, default=0.12)
    parser.add_argument(
        "--resident-skins", type=float, nargs="+",
        help="compare multiple neighbor skins in the same invocation",
    )
    parser.add_argument("--check-candidate-edges", action="store_true", help="verify current physical periodic image membership in candidate paths during validation")
    parser.add_argument("--macro-extension", type=float, default=1.0)
    parser.add_argument("--spline-points", type=int, default=8192)
    parser.add_argument("--xl-skin", type=float, default=1.0)
    parser.add_argument("--dt-fs", type=float, default=0.5)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--temperature", type=float, default=300.0)
    args = parser.parse_args()
    args.dtype = "float64"
    if args.xl_stock_r0_dir and not args.xl_stock_product_dir:
        parser.error("stock R0 comparisons require the upstream stock M1")
    if args.xl_factored_r0_dir and not args.xl_stock_r0_dir:
        parser.error("the factored R0 comparison requires the matched upstream R0 reference")
    if args.xl_stock_m0_dir and not args.xl_stock_r0_dir:
        parser.error("the generated M0 comparison requires the matched upstream R0/M1 reference")
    if (args.include_prepared_xl or args.xl_product_dir or args.xl_stock_product_dir) and not args.symmetrix_json:
        parser.error("the combined native routes require the XL compact export")
    if args.host_observables and "ef" in args.outputs:
        parser.error("host-observables compares all three quantities; select efs or edgeefs")
    args.resident_skins = args.resident_skins or [args.skin]
    if (min(args.steps, args.repeats) < 1 or args.warmup < 0
            or min(*args.resident_skins, args.xl_skin, args.dt_fs) <= 0):
        parser.error("positive steps, repeats, skins and dt and a nonnegative warmup are required")
    if any(len(values) != len(set(values)) for values in (args.outputs, args.variants, args.resident_skins)):
        parser.error("each variant, output and skin may appear once")
    if args.output.resolve() in {p.resolve() for p in (args.model, args.structure, args.symmetrix_json) if p}:
        parser.error("inputs and output must differ")
    return args


def load_xl(args):
    from symmetrix import load_backend, selected_backend
    load_backend(args.selector)
    if selected_backend().get("selector") != args.selector or not args.selector.startswith("cuda"):
        raise RuntimeError("expected the requested native CUDA backend")
    from symmetrix import Symmetrix
    return Symmetrix(str(args.symmetrix_json), dtype="float64", execution_profile="speed",
                     streamed_edges="direct", neighbor_skin=args.xl_skin, dispersion=False,
                     kernel_launch_policy=args.xl_launch_policy)


def stress_request(model, stress, stack):
    def request(_module, _inputs, kwargs):
        kwargs["compute_stress"] = True

    def collect(_module, _inputs, output):
        stress["tensor"] = output["stress"].detach()

    stack.callback(model.register_forward_pre_hook(request, with_kwargs=True).remove)
    stack.callback(model.register_forward_hook(collect).remove)


def read_snapshot(path, stress):
    snap = path.snapshot()
    if hasattr(path, "prepared_xl"):
        return snap
    if stress is None:
        return (*snap, None)
    if isinstance(path, ASEPath):
        if "stress" not in path.atoms.calc.results:
            raise ValueError("energy/force evaluation did not compute stress")
        value = path.atoms.get_stress().copy()
    else:
        value = full_3x3_to_voigt_6_stress(stress["tensor"].cpu().numpy().reshape(3, 3))
    return (*snap, value)


def count_updates(path):
    if hasattr(path, "prepared_xl"):
        return {"neighbor_rebuilds": path.prepared_xl.rebuilds}
    if isinstance(path, ASEPath):
        return {"neighbor_rebuilds": path.atoms.calc.neighbor_cache_build_count}
    return topology_counts(path)


def change_per_atom(frames, atoms):
    totals = np.array([snap[2] + snap[4] for snap in frames])
    return (totals - totals[0]) / atoms


def main():
    args = arguments()
    if "edgeefs" in args.outputs:
        from mace_md_lab.edge_virial import verify_edge_checkpoint

        verify_edge_checkpoint(args.model)
    dev = torch.device("cuda")
    atoms, velocities = starting_structure(args)
    calc = MACECalculator(model_paths=[str(args.model)], device="cuda", default_dtype=args.dtype)
    model = calc.models[0].eval().requires_grad_(False)
    configs = {}
    for variant, observable, skin in product(args.variants, args.outputs, args.resident_skins):
        name = f"{variant}_{observable}"
        if len(args.resident_skins) > 1:
            name += f"_skin{skin:g}"
        configs[name] = (variant, observable, skin)
    paths = list(configs)
    candidate_variants = {"candidate", "replay", "sched", "schedtable", "schedpoly", "replaytable", "replaytable512", "packed", "tangent", "schedpacked", "schedtangent"}
    table_width = {"table64": 64, "table512": 512, "replaytable": 64, "replaytable512": 512, "schedtable": 64}
    if candidate_variants.intersection(args.variants) and set(args.outputs) != {"edgeefs"}:
        raise ValueError("candidate variants require edgeefs and a hash-qualified checkpoint")
    tables = {}
    if table_width.keys() & set(args.variants):
        from mace_md_lab.radial_table import prepare_radial_tables
        for width in sorted({table_width[v] for v in args.variants if v in table_width}):
            tables[width] = prepare_radial_tables(model, knots=args.spline_points, second_width=width)
    xl = product_xl = product_info = stock_product_info = None
    factored_xl = stock_r0_info = factored_r0_info = stock_m0_info = None
    if args.symmetrix_json:
        with patch.dict(os.environ, SYMMETRIX_NEIGHBOR_BACKEND="kokkos"):
            xl = load_xl(args)
            if args.xl_stock_product_dir:
                from mace_md_lab.xl_product import install_xl_product
                stock_product_info = install_xl_product(xl, args.xl_stock_product_dir, variant="upstream")
            if args.xl_stock_r0_dir:
                from mace_md_lab.xl_product import install_xl_r0
                stock_r0_info = install_xl_r0(xl, args.xl_stock_r0_dir, variant="upstream")
            if args.xl_stock_m0_dir:
                stock_m0_info = install_xl_product(xl, args.xl_stock_m0_dir, stage="m0", variant="upstream")
        paths.append("xl_efs")
        if args.include_prepared_xl or args.xl_product_dir:
            paths.append("xl_prepared_efs")
        if args.xl_product_dir:
            from mace_md_lab.xl_product import install_xl_product
            with patch.dict(os.environ, SYMMETRIX_NEIGHBOR_BACKEND="kokkos"):
                product_xl = load_xl(args)
                product_info = install_xl_product(product_xl, args.xl_product_dir)
                if args.xl_stock_r0_dir:
                    install_xl_r0(product_xl, args.xl_stock_r0_dir, variant="upstream")
                if args.xl_stock_m0_dir:
                    install_xl_product(product_xl, args.xl_stock_m0_dir, stage="m0", variant="upstream")
            paths.append("xl_prepared_product_efs")
        if args.xl_factored_r0_dir:
            with patch.dict(os.environ, SYMMETRIX_NEIGHBOR_BACKEND="kokkos"):
                factored_xl = load_xl(args)
                install_xl_product(factored_xl, args.xl_stock_product_dir, variant="upstream")
                factored_r0_info = install_xl_r0(factored_xl, args.xl_factored_r0_dir, variant="factored")
                if args.xl_stock_m0_dir:
                    install_xl_product(factored_xl, args.xl_stock_m0_dir, stage="m0", variant="upstream")
            paths.append("xl_prepared_factored_efs")
    transform_info = {}

    @contextmanager
    def new_path(name, structure):
        with ExitStack() as stack:
            if name.startswith("xl_prepared_"):
                from mace_md_lab.xl_fixed_step import ArrayVerlet, PreparedXL
                stack.enter_context(patch.dict(os.environ, SYMMETRIX_NEIGHBOR_BACKEND="kokkos"))
                engine = {"xl_prepared_efs": xl, "xl_prepared_product_efs": product_xl,
                          "xl_prepared_factored_efs": factored_xl}[name]
                engine.reset()
                engine._neighbor_cache = None
                prepared = PreparedXL(structure, engine, args.xl_skin)
                path = ArrayVerlet(structure, prepared.evaluate, args.dt_fs)
                path.prepared_xl = prepared
                yield path, {}
                return
            if name == "xl_efs":
                stack.enter_context(patch.dict(os.environ, SYMMETRIX_NEIGHBOR_BACKEND="kokkos"))
                xl.reset()
                xl._neighbor_cache = None
                path = ASEPath(structure, xl, args.dt_fs)
                if xl.neighbor_graph_backend != "kokkos":
                    raise ValueError(f"unexpected XL neighbor builder {xl.neighbor_graph_backend}")
                yield path, {}
                return
            calc.reset()
            stack.callback(calc.reset)
            if name == "official":
                yield ASEPath(structure, calc, args.dt_fs), {}
                return
            variant, observable, skin = configs[name]
            product_mode = "both_channel_tile" if variant == "schedpoly" else "both"
            stack.enter_context(optimized_model(model, product_mode, structure.numbers))
            layout = SimpleNamespace() if variant in candidate_variants else None
            if variant in ("packed", "tangent", "schedpacked", "schedtangent"):
                from mace_md_lab.exact_packed_radial import install_exact_packed_radials
                stack.enter_context(install_exact_packed_radials(model, tangent=variant in ("tangent", "schedtangent")))
            if variant in table_width:
                from mace_md_lab.radial_table import install_radial_table
                stack.enter_context(install_radial_table(model, tables=tables[table_width[variant]]))
            if layout is not None:
                from mace_md_lab.candidate_replay import explicit_cutoff
                stack.enter_context(explicit_cutoff(model, calc.r_max))
            if variant in ("first", "folded", "both", "bothfolded") or layout is not None or variant in table_width:
                from mace_md_lab.grouped_first import install_grouped_first
                first_options = {"fold_linear": variant in ("folded", "bothfolded") or layout is not None or variant in table_width, "layout": layout}
                transform_info[variant] = stack.enter_context(
                    install_grouped_first(model, structure.numbers, **first_options)
                )
            if variant in ("second", "both", "bothfolded") or layout is not None or variant in table_width:
                from mace_md_lab.grouped_second import install_second_interaction
                transform_info["second"] = stack.enter_context(install_second_interaction(model, layout=layout))
            stress = {} if observable != "ef" else None
            if observable == "edgeefs":
                from mace_md_lab.edge_virial import install_edge_virial
                stack.enter_context(install_edge_virial(model, structure.get_volume(), stress))
            elif observable == "efs":
                stress_request(model, stress, stack)
            if layout is not None:
                from mace_md_lab.candidate_replay import CandidatePath
                path_type = CandidatePath
                if variant in ("sched", "schedtable", "schedpoly", "schedpacked", "schedtangent"):
                    from mace_md_lab.candidate_schedule import ScheduledPath
                    path_type = ScheduledPath
                yield path_type(structure, calc, args.dt_fs, skin, layout, variant != "candidate", args.macro_extension), stress
            else:
                yield ResidentPath(structure, calc, args.dt_fs, skin, certified=True), stress

    edge_audit = {}

    def trajectory(name):
        with new_path(name, atoms) as (path, stress):
            frames = [read_snapshot(path, stress)]
            audit = args.check_candidate_edges and hasattr(getattr(path, "runner", None), "verify_physical_edges")
            if audit:
                edge_audit[name] = [path.runner.verify_physical_edges()]
            for _ in range(args.warmup + args.steps):
                path.advance(1)
                frames.append(read_snapshot(path, stress))
                if audit:
                    edge_audit[name].append(path.runner.verify_physical_edges())
        if not all(np.isfinite(v).all() for snap in frames for v in snap if v is not None):
            raise ValueError(f"nonfinite values on {name}")
        return frames

    print("Checking freely moving trajectories and common frames", flush=True)
    ref = trajectory("official")
    if np.array_equal(ref[0][0], ref[-1][0]):
        raise ValueError("stationary trajectory")
    ref_drift = change_per_atom(ref, len(atoms))
    gates = {**numerical_gates(args), "stress_eV_A3": 1e-7}
    checks = {}
    for name in paths:
        print("Validating", name, flush=True)
        frames = trajectory(name)
        common = []
        for idx in sorted({0, len(ref) // 2, len(ref) - 1}):
            fixed = atoms.copy()
            fixed.set_positions(ref[idx][0])
            with new_path(name, fixed) as (p, s):
                common.append((ref[idx], read_snapshot(p, s)))
        drift = change_per_atom(frames, len(atoms))
        errors = {
            "energy_eV_atom": max(abs(a[2] - b[2]) / len(atoms) for a, b in common),
            "force_eV_A": max(np.max(np.abs(a[3] - b[3])) for a, b in common),
            "position_A": max(np.max(np.abs(a[0] - b[0])) for a, b in zip(ref, frames)),
            "velocity_A_fs": max(np.max(np.abs(a[1] - b[1])) for a, b in zip(ref, frames)),
            "nve_difference_eV_atom": np.max(np.abs(drift - ref_drift)),
        }
        if name.startswith("xl_") or configs[name][1] != "ef":
            errors["stress_eV_A3"] = max(np.max(np.abs(a[5] - b[5])) for a, b in common)
        checks[name] = {
            "errors": {k: float(v) for k, v in errors.items()},
            "passed": all(np.isfinite(v) and v <= gates[k] for k, v in errors.items()),
            "nve_excursion_meV_atom": 1000.0 * float(np.max(np.abs(drift))),
        }

    passed = all(item["passed"] for item in checks.values())
    times = {name: [] for name in paths}
    updates = {name: [] for name in paths}
    orders = []
    if passed:
        for rep in range(args.repeats):
            # Each neighboring pair of rounds traverses the same rotation in reverse.
            shift = (rep // 2) % len(paths)
            order = paths[shift:] + paths[:shift]
            if rep % 2:
                order = list(reversed(order))
            orders.append(order)
            print("Timing trial", rep + 1, flush=True)
            for name in order:
                with new_path(name, atoms) as (path, stress):
                    path.advance(args.warmup)
                    before = count_updates(path)
                    sync(dev)
                    start = perf_counter()
                    if args.host_observables and isinstance(path, ResidentPath):
                        for _step in range(args.steps):
                            path.advance(1)
                            torch.cat((path.energy.reshape(-1), path.forces.reshape(-1),
                                       stress["tensor"].reshape(-1))).cpu()
                    else:
                        path.advance(args.steps)
                    sync(dev)
                    times[name].append(1000.0 * (perf_counter() - start) / args.steps)
                    after = count_updates(path)
                    updates[name].append({k: v - before[k] for k, v in after.items()})
    result = {
        "passed": passed, "gates": gates, "checks": checks,
        "median_ms_step": {k: median(v) for k, v in times.items() if v},
        "raw_ms_step": times, "timed_graph_updates": updates, "orders": orders,
        "source_sha256": file_hash(args.model), "structure_sha256": file_hash(args.structure),
        "xl_json_sha256": file_hash(args.symmetrix_json) if xl else None,
        "untimed_candidate_edge_audit": edge_audit,
        "candidate_macro_extension_A": args.macro_extension,
        "table_knots": args.spline_points if tables else None,
        "host_efs_each_step": args.host_observables,
        "model": args.model.name, "formula": atoms.get_chemical_formula(), "atoms": len(atoms),
        "device": torch.cuda.get_device_name(dev), "torch": str(torch.__version__),
        "mace": package_version("mace-torch"), "symmetrix": package_version("symmetrix-xl") if xl else None,
        "xl_plan": xl.execution_plan if xl else None,
        "xl_r0": xl.r0_diagnostics if xl else None,
        "xl_prepared_product": product_info,
        "xl_stock_product": stock_product_info,
        "xl_stock_r0": stock_r0_info,
        "xl_stock_m0": stock_m0_info,
        "xl_factored_r0": factored_r0_info,
        "xl_launch_policy": args.xl_launch_policy if xl else None,
        "xl_r1": getattr(xl.evaluator, "r1_implementation", None) if xl else None,
        "xl_neighbor_backend": "kokkos" if xl else None, "xl_selector": args.selector if xl else None,
        "dtype": args.dtype,
        "resident_skin_A": args.resident_skins[0] if len(args.resident_skins) == 1 else args.resident_skins,
        "xl_skin_A": args.xl_skin if xl else None,
        "dt_fs": args.dt_fs, "velocity_source": velocities, "cpu_threads": torch.get_num_threads(),
        "steps": args.steps, "warmup": args.warmup, "repeats": args.repeats,
        "frame": args.frame, "transform_info": transform_info,
        "scope": (
            "Synchronized complete fixed-cell velocity-Verlet steps. Each path first ran the entire "
            "trajectory untimed. Resident controls retain tensors on GPU. Native XL returns energy, "
            "forces and stress to ASE. The efs controls request native MACE stress, edgeefs derives "
            "stress via edge vectors, and ef requests energy and force. Both stress paths compute "
            "stress in each evaluation and validate it on common frames."
        ),
    }
    write_report(args.output, result)
    print("Passed:", passed, "ms/step:", result["median_ms_step"], flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
