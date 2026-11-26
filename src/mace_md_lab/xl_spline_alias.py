"""Reuse byte-identical radial spline rows of one frozen Symmetrix-XL export."""
import argparse
import ctypes
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

from .xl_product import factored_r0_source

QUALIFIED_COMPACT = "80c2ee186373a08e538018a44127782d3f960a3d3520841a391abe4fc89c3a42"


def alias_source(source, stage):
    """Preserve all FP64 coefficients and change only the selected coefficient row."""
    if stage == "r0":
        pattern = r"receiver_type\s*\*\s*args\.active_type_count\s*\+\s*(neighbor_type|sender_type)"
        transformed, count = re.subn(pattern, r"\1", source)
        if count != 2:
            raise ValueError("unknown R0 radial table indexing")
        return transformed
    key = "static_cast<std::size_t>(edge_type) * radial.intervals + interval"
    if stage != "r1" or source.count(key) != 1:
        raise ValueError("unknown R1 radial table indexing")
    return source.replace(key, "static_cast<std::size_t>(interval)")


def probe_source(source, stage):
    # A separate, untimed audit binary exposes the actual device table and extents.
    # Synchronize and read it once while that evaluator and its graph remain live.
    ptr = "coefficients_raw" if stage == "r0" else "coefficients"
    marker = "symmetrix_r0_forward_v1" if stage == "r0" else "symmetrix_factorized_(?:tiled_|projected_)?forward_v2"
    pattern = re.compile(r"(void " + marker + r"\([^()]+ args\)\s*\{)")
    command = ('\n    if (blockIdx.x == 0 && threadIdx.x == 0) '
               'printf("MDLAB_TABLE_' + stage + ' %llu %u %u %u %u\\n", '
               '(unsigned long long)args.radial.' + ptr + ', (unsigned int)args.radial.edge_types, '
               'args.radial.intervals, args.radial.functions, (unsigned int)args.active_type_count);\n')
    updated, count = pattern.subn(lambda m: m[0] + command, source)
    if count != (1 if stage == "r0" else 3):
        raise ValueError("could not instrument the expected kernel entry points")
    return updated


def build(compact, directory):
    from symmetrix.mace_jit_operator_codegen import render_jit_r0_device_module
    from symmetrix.mace_jit_codegen import render_jit_r1_cuda_module
    compact = Path(compact)
    contents = compact.read_bytes()
    if version("symmetrix-xl") != "0.2.0" or sha256(contents).hexdigest() != QUALIFIED_COMPACT:
        raise ValueError("requires the four-element MP-0 Small FP64 compact export at 8192 knots")
    model = json.loads(contents)
    folder = Path(directory)
    folder.mkdir(exist_ok=True, parents=True)
    r0, _ = render_jit_r0_device_module(model["execution_contracts"]["R0"], precision="float64",
                                         target="sm_70", schedule="receiver-edge16")
    r1 = render_jit_r1_cuda_module(model["execution_contracts"]["R1"], 70, precision="float64")
    for stage, source in [("r0", factored_r0_source(r0)), ("r1", r1)]:
        for mode in ("alias", "probe", "control"):
            code = folder / f"{stage}-{mode}.cu"
            transformed = source
            if mode == "alias":
                transformed = alias_source(source, stage)
            elif mode == "probe":
                transformed = probe_source(source, stage)
            code.write_text(transformed)
            binary = code.with_suffix(".cubin")
            subprocess.run(["nvcc", "-O3", "-std=c++17", "-arch=sm_70", "-cubin", str(code), "-o", str(binary)],
                           check=True, timeout=90)
            details = {"stage": stage, "mode": mode, "compact_sha256": QUALIFIED_COMPACT,
                       "source_sha256": sha256(code.read_bytes()).hexdigest(),
                       "cubin_sha256": sha256(binary.read_bytes()).hexdigest()}
            code.with_suffix(".json").write_text(json.dumps(details, indent=2) + "\n")


def load_module(calc, folder, stage, mode):
    binary = (Path(folder)/f"{stage}-{mode}.cubin").resolve()
    details = json.loads(binary.with_suffix(".json").read_text())
    if (details["compact_sha256"] != QUALIFIED_COMPACT or details["mode"] != mode or details["stage"] != stage
            or sha256(binary.read_bytes()).hexdigest() != details["cubin_sha256"]):
        raise ValueError("invalid qualified module artifact")
    native = calc.evaluator
    if stage == "r0":
        native._load_r0_device_module(str(binary), 4, False, "receiver-edge16")
        if not native.r0_device_module_ready or Path(native.r0_device_module_path).resolve() != binary:
            raise RuntimeError("R0 module unavailable")
    else:
        native._load_jit_cuda_plugin(str(binary), 8)
        if not native.jit_cuda_plugin_ready:
            raise RuntimeError("R1 module unavailable")
    return details


def qualify(calc, atoms, folder, skin=1.0):
    import cupy as cp
    import torch
    from .xl_fixed_step import PreparedXL

    for stage in ("r0", "r1"):
        load_module(calc, folder, stage, "probe")
    prepared = PreparedXL(atoms, calc, skin)
    libc = ctypes.CDLL(None)
    with tempfile.TemporaryFile(mode="w+b") as output:
        libc.fflush(None)
        original = os.dup(1)
        try:
            os.dup2(output.fileno(), 1)
            prepared.evaluate(atoms.positions)
            torch.cuda.synchronize()
            libc.fflush(None)
        finally:
            os.dup2(original, 1)
            os.close(original)
        output.seek(0)
        log = output.read().decode(errors="replace")
    rows = {}
    for stage in ("r0", "r1"):
        matches = re.findall(r"MDLAB_TABLE_" + stage + r" (\d+) (\d+) (\d+) (\d+) (\d+)", log)
        if not matches or len(set(matches)) != 1:
            raise RuntimeError(f"unable to identify a unique active {stage} table: {log[:250]}")
        address, types, intervals, funcs, active = map(int, matches[0])
        count = intervals*4*funcs
        if (address == 0 or count < 1 or active < 1 or
                types != (active*active if stage == "r0" else active*(active+1)//2)):
            raise ValueError(f"unexpected {stage} table dimensions {types}, {intervals}, {funcs}, {active}")
        # Compare every coefficient, including signed zero, in the actual FP64 allocation.
        memory = cp.cuda.UnownedMemory(address, types*count*8, calc)
        data = cp.ndarray((types, count), dtype=cp.uint64, memptr=cp.cuda.MemoryPointer(memory, 0))
        counts = []
        for i in range(types):
            representative = i % active if stage == "r0" else 0
            diff = 0
            if i != representative:
                for start in range(0, count, 1 << 20):
                    sl = slice(start, min(count, start + (1 << 20)))
                    diff += int(cp.count_nonzero(data[i, sl] != data[representative, sl]).item())
            counts.append(diff)
        rows[stage] = dict(rows=types, intervals=intervals, functions=funcs, row_bytes=count*8,
                           representative_rows=active if stage == "r0" else 1,
                           nonidentical_coefficients_per_row=counts, passed=not any(counts))
    result = dict(compact_sha256=QUALIFIED_COMPACT, atomic_numbers=sorted(set(map(int, atoms.numbers))),
                  stages=rows, passed=all(row["passed"] for row in rows.values()))
    return result


def install_algorithm(calc, compact, config):
    """Install after reading the passed native audit for the identical species and compact file."""
    folder = Path(config["artifact_dir"])
    if sha256(Path(compact).read_bytes()).hexdigest() != QUALIFIED_COMPACT:
        raise ValueError("unsupported compact model")
    audit = json.loads(Path(config["audit"]).read_text())
    if (audit["compact_sha256"] != QUALIFIED_COMPACT
            or audit["atomic_numbers"] != sorted(set(config["atomic_numbers"]))):
        raise ValueError("this composition requires its own native coefficient audit")
    stages = config.get("stages", ["r0","r1"])
    if not stages or not set(stages) <= {"r0","r1"}:
        raise ValueError("expected r0 or r1")
    info = []
    for stage in stages:
        if not audit["stages"][stage]["passed"]:
            raise ValueError(f"duplicate-row proof failed for {stage}")
        info.append(load_module(calc, folder, stage, "alias"))
    return info


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--compact", required=True, type=Path)
    p.add_argument("--artifacts", required=True, type=Path)
    args = p.parse_args()
    build(args.compact,args.artifacts)
