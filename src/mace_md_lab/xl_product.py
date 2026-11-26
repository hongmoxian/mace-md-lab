"""Build experimental FP64 native M1 and R0 kernels for Symmetrix-XL 0.2.0."""

import argparse
from collections import Counter, defaultdict
from importlib.metadata import version
import hashlib
import json
from pathlib import Path
import subprocess


def factor_polynomial(terms):
    """Factor one frozen scalar polynomial while preserving its dynamic coefficients."""
    constants = [w for powers, w in terms if not powers]
    others = [(powers, w) for powers, w in terms if powers]
    while others:
        counts = Counter(component for powers, _ in others for component in set(powers))
        component = max(counts, key=lambda c: (counts[c], -c))
        selected = []
        remainder = []
        for powers, w in others:
            if component in powers:
                reduced = list(powers)
                reduced.remove(component)
                selected.append((reduced, w))
            else:
                remainder.append((powers, w))
        constants.append(f"(x[{component}] * {factor_polynomial(selected)})")
        others = remainder
    return "(" + " + ".join(constants) + ")" if constants else "0.0"


def horner_source(contract, metadata):
    from symmetrix.generated_operator_codegen import _contraction_device_prefix

    if (contract["output_components"] != 1 or contract["input_components"] != 16
            or contract["channels"] != 128 or contract["correlation"] != 3):
        raise ValueError("expected a 128-channel scalar M1 output with 16 input components")
    groups = contract["monomial_groups"]
    if len(groups) != 1 or groups[0]["output_component"] != 0:
        raise ValueError("expected exactly one scalar output")
    terms = groups[0]["terms"]
    value_terms = []
    derivatives = defaultdict(list)
    for index, powers in enumerate(terms):
        weight = f"w[{index} * channel_count]"
        value_terms.append((powers, weight))
        for component, multiplicity in Counter(powers).items():
            reduced = list(powers)
            reduced.remove(component)
            coeff = weight if multiplicity == 1 else f"({multiplicity}.0 * {weight})"
            derivatives[component].append((reduced, coeff))
    head = _contraction_device_prefix(metadata, operator="M1")
    owner = """
    for (long long owner = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         owner < args.num_nodes * channel_count; owner += (long long)blockDim.x * gridDim.x) {
        const long long node = owner / channel_count;
        const int c = owner % channel_count;
        const Scalar *w = (const Scalar*)args.weights_raw
            + (long long)args.node_types[node] * term_count * channel_count + c;
        Scalar x[input_components];
        #pragma unroll
        for (int j = 0; j < input_components; ++j)
            x[j] = ((const Scalar*)args.input_raw)[node * args.input_node_stride + j * channel_count + c];
    """
    fwd = 'extern "C" __global__ __launch_bounds__(256) void symmetrix_m1_forward_v1(M1Args args) {'
    fwd += owner + "((Scalar*)args.output_raw)[node * args.output_node_stride + c] = " + factor_polynomial(value_terms) + ";\n}}\n"
    rev = 'extern "C" __global__ __launch_bounds__(256) void symmetrix_m1_reverse_v1(M1Args args) {'
    rev += owner + """
        const Scalar adj = ((const Scalar*)args.output_adjoint_raw)[node * args.output_adjoint_node_stride + c];
        Scalar *dst = (Scalar*)args.input_adjoint_raw + node * args.input_adjoint_node_stride + c;
        Scalar scale = 0.0;
    """
    for j in range(contract["input_components"]):
        rev += f"""
        {{
            const Scalar v = adj * {factor_polynomial(derivatives[j])};
            if (args.reserved & 1u) dst[{j} * channel_count] += v;
            else dst[{j} * channel_count] = v;
            if (args.capture_input_scale_adjoint) scale += x[{j}] * v;
        }}
        """
    rev += "if (args.capture_input_scale_adjoint) atomicAdd(args.input_scale_adjoint + node, scale);\n}}\n"
    return head + fwd + rev


def factored_r0_source(source):
    """Pull the radial direction and per-order spline factors past their sums."""
    start_mark = "        Scalar force_x = Scalar(0);"
    end_mark = "        force_x = subgroup_sum(force_x, edge_vector_length);"
    if source.count(start_mark) != 1 or source.count(end_mark) != 1:
        raise ValueError("unsupported XL R0 reverse source")
    start, end = source.index(start_mark), source.index(end_mark)
    body = """
        Scalar force_x = 0, force_y = 0, force_z = 0, force_r = 0;
        if (edge_active) {
            #pragma unroll
            for (int l = 0; l <= r0_l_max; ++l) {
                #pragma unroll
                for (int channel = lane; channel < channel_count; channel += edge_vector_length) {
                    Scalar spline_dr;
                    const Scalar spline = radial_spline_from_coordinates(
                        args.radial, edge_type, l * channel_count + channel,
                        spline_coordinates, &spline_dr);
                    Scalar alpha = 0, beta_x = 0, beta_y = 0, beta_z = 0;
                    #pragma unroll
                    for (int component = l*l; component < (l+1)*(l+1); ++component) {
                        const Scalar adj = output_adjoint[
                            (receiver * harmonic_count + component) * channel_count + channel];
                        alpha += adj * harmonics[edge * harmonic_count + component];
                        beta_x += adj * edge_gradients[component];
                        beta_y += adj * edge_gradients[harmonic_count + component];
                        beta_z += adj * edge_gradients[2 * harmonic_count + component];
                    }
                    force_r += spline_dr * alpha;
                    force_x += spline * beta_x;
                    force_y += spline * beta_y;
                    force_z += spline * beta_z;
                }
            }
            force_x = weight * (force_r * direction_x + force_x);
            force_y = weight * (force_r * direction_y + force_y);
            force_z = weight * (force_r * direction_z + force_z);
        }
"""
    return source[:start] + body + source[end:]


def build(compact, directory, stage="m1"):
    from symmetrix.mace_jit_operator_codegen import (
        render_jit_m0_device_module, render_jit_m1_device_module, render_jit_r0_device_module,
    )

    if version("symmetrix-xl") != "0.2.0":
        raise ValueError("this experiment requires Symmetrix-XL 0.2.0")
    payload = Path(compact).read_bytes()
    contract = json.loads(payload)["execution_contracts"][stage.upper()]
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if stage == "m0":
        standard, metadata = render_jit_m0_device_module(contract, precision="float64", target="sm_70")
        variants = [("upstream", standard)]
    elif stage == "m1":
        standard, metadata = render_jit_m1_device_module(contract, precision="float64", target="sm_70")
        variants = [("upstream", standard), ("horner", horner_source(contract, metadata))]
    else:
        standard, metadata = render_jit_r0_device_module(
            contract, precision="float64", target="sm_70", schedule="receiver-edge16"
        )
        variants = [("upstream", standard), ("factored", factored_r0_source(standard))]
    # XL derives this ABI identifier from the scalar contract, precision,
    # target and launch schedule. The content hashes distinguish binaries.
    for name, source in variants:
        code = directory / f"{stage}-{name}.cu"
        code.write_text(source)
        binary = code.with_suffix(".cubin")
        subprocess.run(["nvcc", "-std=c++17", "-arch=sm_70", "-cubin", "-O3",
                        str(code), "-o", str(binary)], check=True, timeout=90)
        details = {"source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                   "cubin_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                   "compact_sha256": hashlib.sha256(payload).hexdigest(),
                   "artifact_id": metadata["artifact_id"], "variant": name,
                   "structure": metadata["structure_fingerprint"]}
        code.with_suffix(".json").write_text(json.dumps(details, indent=2) + "\n")


def install_xl_product(calc, artifact_dir, *, variant="horner", stage="m1"):
    """Load one compiled product into a dedicated native evaluator before stepping."""
    allowed = {"m0": {"upstream"}, "m1": {"upstream", "horner"}}
    if variant not in allowed.get(stage, set()) or version("symmetrix-xl") != "0.2.0":
        raise ValueError("unsupported product variant or XL version")
    stem = Path(artifact_dir) / f"{stage}-{variant}.cubin"
    details = json.loads(stem.with_suffix(".json").read_text())
    if hashlib.sha256(stem.read_bytes()).hexdigest() != details["cubin_sha256"]:
        raise ValueError("product cubin hash mismatch")
    evaluator = calc.evaluator
    loader = getattr(evaluator, f"_load_{stage}_device_module")
    loader(str(stem.resolve()), "chunk32", 2)
    if (not getattr(evaluator, f"{stage}_device_module_ready")
            or getattr(evaluator, f"{stage}_device_module_artifact_id") != details["artifact_id"]
            or Path(getattr(evaluator, f"{stage}_device_module_path")).resolve() != stem.resolve()):
        raise ValueError("native evaluator did not admit the expected product")
    return details


def install_xl_r0(calc, artifact_dir, *, variant="factored"):
    """Install a compiled receiver-edge16 R0 forward and coordinate reverse pair."""
    if variant not in {"upstream", "factored"} or version("symmetrix-xl") != "0.2.0":
        raise ValueError("unsupported R0 variant or XL version")
    stem = Path(artifact_dir) / f"r0-{variant}.cubin"
    details = json.loads(stem.with_suffix(".json").read_text())
    if hashlib.sha256(stem.read_bytes()).hexdigest() != details["cubin_sha256"]:
        raise ValueError("R0 cubin hash mismatch")
    evaluator = calc.evaluator
    evaluator._load_r0_device_module(str(stem.resolve()), 4, False, "receiver-edge16")
    if (not evaluator.r0_device_module_ready
            or evaluator.r0_device_module_artifact_id != details["artifact_id"]
            or Path(evaluator.r0_device_module_path).resolve() != stem.resolve()):
        raise ValueError("native evaluator did not admit the expected R0 module")
    return details


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symmetrix-json", required=True, type=Path)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--stage", choices=("m0", "m1", "r0"), default="m1")
    args = parser.parse_args()
    build(args.symmetrix_json, args.artifact_dir, args.stage)
