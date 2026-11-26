"""Application wall time across two different pretrained potentials on the same input."""
import argparse
from importlib.resources import files
import os
from pathlib import Path
from statistics import median

from ase.stress import full_3x3_to_voigt_6_stress
import numpy as np
import torch

from benchmark_mace_xl_direct import measure
from benchmark_xl_algorithms import read_routes, setup_engine
from mace_md_lab.bench import file_hash, package_version, starting_structure, write_report
from mace_md_lab.xl_fixed_step import ArrayVerlet, PreparedXL


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("symmetrix-json", "product-dir", "structure", "output"):
        p.add_argument("--" + key, required=True, type=Path)
    p.add_argument("--m0-dir", type=Path)
    p.add_argument("--routes-json", type=Path, default=Path("examples/native-xl-locked.json"))
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--repeats", type=int, default=6)
    p.add_argument("--warmup", type=int, default=3)
    p.set_defaults(seed=17, temperature=300., dt_fs=0.5, skin=1., frame=0,
                   harmonic_storage="y-only-direct-v1")
    a = p.parse_args()
    if min(a.warmup, a.steps, a.repeats) < 1 or a.output.resolve() in {
            q.resolve() for q in (a.symmetrix_json, a.structure, a.routes_json)}:
        p.error("require positive counts and separate output")
    os.environ["SYMMETRIX_NEIGHBOR_BACKEND"] = "kokkos"
    from symmetrix import load_backend
    from chgnet.model.dynamics import CHGNetCalculator
    from chgnet.model.model import CHGNet

    load_backend("cuda12-sm70")
    atoms, velocities = starting_structure(a)
    compact = file_hash(a.symmetrix_json)
    native, _ = setup_engine(a, read_routes(a.routes_json)["locked"], compact,
                             sorted(set(map(int, atoms.numbers))))
    torch.set_default_dtype(torch.float32)
    model = CHGNet.load(model_name="0.3.0", use_device="cuda")
    chg = CHGNetCalculator(model=model, use_device="cuda")
    weights = next((files("chgnet") / "pretrained" / "0.3.0").glob("*.pth.tar"))

    def make_eval(name):
        if name == "xl_locked":
            native.reset()
            native._neighbor_cache = None
            return PreparedXL(atoms, native, a.skin).evaluate
        sample = atoms.copy()
        def evaluate(pos):
            sample.set_positions(pos)
            chg.calculate(sample, properties=["energy", "forces", "stress"], task="efs")
            out = chg.results
            return out["energy"], out["forces"], full_3x3_to_voigt_6_stress(out["stress"])
        return evaluate

    reference = ArrayVerlet(atoms, make_eval("xl_locked"), a.dt_fs)
    frames = [reference.positions.copy()]
    for _ in range(a.steps + a.warmup):
        reference.advance(1)
        frames.append(reference.positions.copy())
    if any(np.array_equal(x, y) for x, y in zip(frames, frames[1:])):
        raise RuntimeError("expected successive frames with different positions")
    names = ["xl_locked", "chgnet_efs"]
    md = {name: [] for name in names}
    inference = {name: [] for name in names}
    orders = []
    for trial in range(a.repeats):
        order = names if trial % 2 == 0 else names[::-1]
        orders.append(order)
        for name in order:
            path = ArrayVerlet(atoms, make_eval(name), a.dt_fs)
            path.advance(a.warmup)
            md[name].append(measure(lambda: path.advance(a.steps), a.steps))
            if not all(np.isfinite(x).all() for x in path.snapshot()):
                raise ValueError("nonfinite trajectory")
            evaluate = make_eval(name)
            for pos in frames[:a.warmup + 1]:
                evaluate(pos)
            def replay():
                for pos in frames[a.warmup + 1:]:
                    evaluate(pos)
            inference[name].append(measure(replay, a.steps))

    write_report(a.output, {"atoms": len(atoms), "formula": atoms.get_chemical_formula(),
        "structure_sha256": file_hash(a.structure), "structure": a.structure.name,
        "compact_sha256": compact, "chgnet_weights_sha256": file_hash(weights),
        "chgnet_weights_version": "0.3.0", "chgnet_task": "efs", "chgnet": package_version("chgnet"),
        "chgnet_parameter_dtype": str(next(model.parameters()).dtype),
        "chgnet_graph_algorithm": str(model.graph_converter.algorithm),
        "xl": package_version("symmetrix-xl"), "xl_dtype": "float64",
        "torch": str(torch.__version__), "gpu": torch.cuda.get_device_name(),
        "ase": package_version("ase"), "warmup": a.warmup, "steps": a.steps, "orders": orders,
        "dt_fs": a.dt_fs, "temperature_K": a.temperature, "seed": a.seed, "velocities": velocities,
        "common_frame_source": "locked XL trajectory", "md_ms_per_step": md,
        "md_medians": {n: median(v) for n,v in md.items()}, "common_frames_efs_ms": inference,
        "common_frames_efs_medians": {n: median(v) for n,v in inference.items()},
        "scope": "runtime of different pretrained potentials. No DFT or cross-model accuracy test."})


if __name__ == "__main__":
    main()
