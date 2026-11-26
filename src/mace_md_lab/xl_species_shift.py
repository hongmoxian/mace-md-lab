"""Fold a constant first-layer residual into an XL scalar product for fixed species.

For H = M0 P + b[z], solve d[z] P = b[z] once and emit M0 + d[z].
The remaining learned readout, skip and linear-up maps then act on the single
shifted product. This experiment uses FP64 and fixed atom types.
"""
from importlib.metadata import version
import hashlib
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import numpy as np


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def compile_shifted_product(source_compact, directory):
    from symmetrix.h1_affine import h1_affine_mode_of
    from symmetrix.mace_jit_operator_codegen import render_jit_m0_device_module

    payload = Path(source_compact).read_bytes()
    data = json.loads(payload)
    k, n = data["num_channels"], data["num_elements"]
    if (version("symmetrix-xl") != "0.2.0" or data["model_type"] != "MACE"
            or h1_affine_mode_of(data) != "pre_linear_up"
            or data["L_max"] != 0 or data["num_interactions"] != 2
            or data["has_field_coupling"] or not data["first_interaction_residual"]
            or data["radial_representation"] != "compact" or "heads" in data
            or "head_payloads" in data or data.get("has_zbl")):
        raise ValueError("expected a field-free scalar, two-interaction compact MACE with fixed species")
    p = np.asarray(data.pop("H1_product_weights"), dtype=np.float64).reshape(k, k)
    u = np.asarray(data.pop("H1_linear_up_weights"), dtype=np.float64).reshape(k, k)
    c = np.asarray(data["H1_weights"], dtype=np.float64).reshape(k, k)
    b = np.asarray(data.pop("H1_first_residual_weights"), dtype=np.float64).reshape(n, k)
    r = np.asarray(data.pop("readout_1_weights_for_H1_pre_linear_up"), dtype=np.float64)
    skip = np.asarray(data.pop("H2_weights_for_H1_pre_linear_up"), dtype=np.float64).reshape(n, k, k)
    shift = np.linalg.solve(p.T, b.T).T
    residual = float(np.max(np.abs(shift @ p - b)))
    cond = float(np.linalg.cond(p))
    if (not np.isfinite(shift).all() or not np.isfinite(cond) or cond > 1e5
            or residual > 1e-10 or not np.allclose(c, p @ u, rtol=1e-13, atol=1e-13)):
        raise ValueError("ill-conditioned or incompatible frozen affine maps")
    data["first_interaction_residual"] = False
    data["h1_affine_mode"] = "m0_composite"
    data["readout_1_weights_for_M0"] = (p @ r).tolist()
    data["H2_weights_for_M0"] = (p @ skip).reshape(n, k * k).tolist()
    data.pop("h1_affine_provenance", None)
    if h1_affine_mode_of(data) != "m0_composite":
        raise ValueError("unexpected composed affine basis")
    contract = data["execution_contracts"]["M0"]
    if contract["output_components"] != 1 or n * k * 8 > 48000:
        raise ValueError("expected one scalar product and a small table of species shifts")
    code, meta = render_jit_m0_device_module(contract, precision="float64", target="sm_70")
    anchor = 'extern "C" __global__ __launch_bounds__(256)\nvoid symmetrix_m0_forward_v1'
    store = '= out[component];'
    if code.count(anchor) != 1 or code.count(store) != 1:
        raise ValueError("unexpected XL generated M0 source")
    constants = ",".join(format(float(x), ".17g") for x in shift.ravel())
    code = code.replace(anchor, f"__device__ __constant__ Scalar mdlab_species_shift[] = {{{constants}}};\n" + anchor)
    code = code.replace(store, "= out[component] + mdlab_species_shift[type * channel_count + channel];")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    cu = directory / "shift.cu"
    binary = directory / "shift.cubin"
    cu.write_text(code)
    subprocess.run(["nvcc", "-std=c++17", "-arch=sm_70", "-cubin", "-O3", str(cu), "-o", str(binary)], check=True, timeout=90)
    intermediate = json.dumps(data, separators=(",", ":"), allow_nan=False).encode()
    metadata = dict(source_compact_sha256=_hash(payload), cubin_sha256=_hash(binary.read_bytes()),
                    input_compact_sha256=_hash(intermediate), cuda_source_sha256=_hash(code.encode()),
                    artifact_id=meta["artifact_id"], condition=cond, max_species_residual=residual)
    # The compact and its constant-shift kernel form an inseparable pair.
    # A generic XL model reader must never consume the compact on its own.
    (directory / "shift.payload").write_bytes(intermediate)
    (directory / "shift.manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def make_shifted_calculator(source_compact, directory, **calculator_kwargs):
    from symmetrix import Symmetrix
    if calculator_kwargs.get("dtype") != "float64":
        raise ValueError("affine residual fold is qualified for FP64")
    directory = Path(directory)
    meta = json.loads((directory / "shift.manifest.json").read_text())
    payload = (directory / "shift.payload").read_bytes()
    binary = directory / "shift.cubin"
    if (meta["input_compact_sha256"] != _hash(payload)
            or meta["source_compact_sha256"] != _hash(Path(source_compact).read_bytes())
            or meta["cubin_sha256"] != _hash(binary.read_bytes())):
        raise ValueError("shifted payload/kernel/original model mismatch")
    with TemporaryDirectory() as tmp:
        transient = Path(tmp) / "model.json"
        transient.write_bytes(payload)
        calc = Symmetrix(str(transient), **calculator_kwargs)
    ev = calc.evaluator
    ev._load_m0_device_module(str(binary.resolve()), "chunk32", 2)
    if (not ev.m0_device_module_ready or ev.m0_device_module_artifact_id != meta["artifact_id"]
            or Path(ev.m0_device_module_path).resolve() != binary.resolve()):
        raise RuntimeError("native shifted M0 was not admitted")
    return calc, meta
