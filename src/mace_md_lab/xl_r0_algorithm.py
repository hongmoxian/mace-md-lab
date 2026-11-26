"""Native channel-tiled neighbor reductions for the XL 0.2.0 MP-0 Small R0 operator."""

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess

from mace_md_lab.xl_product import factored_r0_source


VARIANTS = {"tile1": 1, "child4": 4, "child8": 8, "warp": 1,
            "channelgrad": 1, "tilegrad": 1, "childgrad": 4,
            "staged": 1, "stagedgrad": 1, "endpoints": 1, "tangentproj": 1, "ambientwarp": 1, "lowerproj": 1}


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"unexpected XL R0 source at {old!r}")
    return source.replace(old, new)


def channel_first_gradient(source):
    """Contract the radial value/slope over channels before angular projection."""
    begin = source.index("        Scalar force_x = Scalar(0);")
    end = source.index("        force_x = subgroup_sum(force_x, edge_vector_length);", begin)
    body = """
        Scalar force_r = 0, force_x = 0, force_y = 0, force_z = 0;
        if (edge_active) {
    """
    for order in range(4):
        size = 2*order+1
        body += f"""
            {{
                Scalar val[{size}] = {{}}, slope[{size}] = {{}};
                #pragma unroll
                for (int c=lane; c<channel_count; c+=edge_vector_length) {{
                    Scalar dr;
                    const Scalar v = radial_spline_from_coordinates(
                        args.radial,edge_type,{order}*channel_count+c,spline_coordinates,&dr);
                    #pragma unroll
                    for (int m=0; m<{size}; ++m) {{
                        const Scalar b = output_adjoint[
                            (receiver*harmonic_count+{order**2}+m)*channel_count+c];
                        val[m] += v*b;
                        slope[m] += dr*b;
                    }}
                }}
                #pragma unroll
                for (int m=0; m<{size}; ++m) {{
                    const int lm = {order**2}+m;
                    force_r += slope[m]*harmonics[edge*harmonic_count+lm];
                    force_x += val[m]*edge_gradients[lm];
                    force_y += val[m]*edge_gradients[harmonic_count+lm];
                    force_z += val[m]*edge_gradients[2*harmonic_count+lm];
                }}
            }}
        """
    body += """
            force_x = weight*(force_r*direction_x+force_x);
            force_y = weight*(force_r*direction_y+force_y);
            force_z = weight*(force_r*direction_z+force_z);
        }
    """
    return source[:begin]+body+source[end:]


def forward_source(source, variant):
    """Distribute edge shards among warps, with a single final block reduction."""
    warps = VARIANTS[variant]
    begin = source.index('extern "C" __global__ __launch_bounds__(forward_threads_per_block)')
    end = source.index('extern "C" __global__ __launch_bounds__(', begin + 1)
    forward = """
template<int L, int WARPS>
__device__ __forceinline__ void receiver_channel_tile(
        R0Args args, long long receiver, int tile, Scalar* shared) {
    constexpr int components = 2*L+1;
    const int lane = threadIdx.x & 31;
    const int shard = threadIdx.x >> 5;
    const int channel = tile * 32 + lane;
    const int receiver_type = args.type_to_active[args.node_types[receiver]];
    const int begin = args.first_neigh[receiver];
    const int count = args.num_neigh[receiver];
    const Scalar* harmonics = static_cast<const Scalar*>(args.harmonics_raw);
    Scalar values[components] = {};
    for (int local=shard; local<count; local+=WARPS) {
        const int edge = begin+local;
        const Scalar r = args.radius[edge];
        if (!edge_is_active(args.cutoff, r)) continue;
        const int sender_type = args.type_to_active[args.neigh_types[edge]];
        const int pair = receiver_type*args.active_type_count+sender_type;
        const RadialSplineCoordinates coords = radial_spline_coordinates(args.radial, r);
        Scalar unused;
        const Scalar radial = radial_spline_from_coordinates(
            args.radial, pair, L*channel_count+channel, coords, &unused);
        #pragma unroll
        for (int m=0; m<components; ++m)
            values[m] += radial*harmonics[edge*harmonic_count+L*L+m];
    }
    if constexpr (WARPS > 1) {
        #pragma unroll
        for (int m=0; m<components; ++m)
            shared[(shard*components+m)*32+lane] = values[m];
        __syncthreads();
    }
    if (shard == 0) {
        const Scalar scale = args.apply_density_scale ? args.density_state[receiver] : Scalar(1);
        Scalar* output = static_cast<Scalar*>(args.output_raw)
            +(receiver*harmonic_count+L*L)*channel_count+channel;
        #pragma unroll
        for (int m=0; m<components; ++m) {
            #pragma unroll
            for (int s=1; s<WARPS; ++s)
                values[m] += shared[(s*components+m)*32+lane];
            output[m*channel_count] = scale*values[m];
        }
    }
}

extern "C" __global__ __launch_bounds__(forward_threads_per_block)
void symmetrix_r0_forward_v1(R0Args args) {
    constexpr int warps = forward_threads_per_block/32;
    __shared__ Scalar partial[warps*7*32];
    constexpr int tiles = channel_count/32;
    for (long long task=blockIdx.x; task<args.num_nodes*4*tiles; task+=gridDim.x) {
        const long long receiver = task/(4*tiles);
        const int tile = task%tiles;
        switch ((task/tiles)%4) {
        case 0: receiver_channel_tile<0,warps>(args,receiver,tile,partial); break;
        case 1: receiver_channel_tile<1,warps>(args,receiver,tile,partial); break;
        case 2: receiver_channel_tile<2,warps>(args,receiver,tile,partial); break;
        case 3: receiver_channel_tile<3,warps>(args,receiver,tile,partial); break;
        }
        if constexpr (warps > 1) __syncthreads();
    }
}

"""
    if variant == "warp":
        # Each warp processes the same receiver/order while the four channel
        # tiles are walked sequentially. The small register tile avoids
        # dynamically indexed shared accumulators.
        forward = forward.replace("for (long long task=blockIdx.x; task<args.num_nodes*4*tiles; task+=gridDim.x) {",
                  "for (long long owner=blockIdx.x; owner<args.num_nodes*4; owner+=gridDim.x) "
                  "for (long long task=owner*tiles; task<(owner+1)*tiles; ++task) {")
    elif warps > 1:
        # XL verifies the entry point's 32-thread launch contract. A single
        # thread dispatches the larger edge-sharded grid with the same args.
        forward = forward.replace('extern "C" __global__ __launch_bounds__(forward_threads_per_block)\n'
                                  'void symmetrix_r0_forward_v1(R0Args args)',
                                  f'__global__ __launch_bounds__({32*warps}) void r0_tiled_grid(R0Args args)')
        forward = forward.replace("constexpr int warps = forward_threads_per_block/32;",
                                  f"constexpr int warps = {warps};")
        forward += f'''
extern "C" __global__ __launch_bounds__(forward_threads_per_block)
void symmetrix_r0_forward_v1(R0Args args) {{
    if (blockIdx.x == 0 && threadIdx.x == 0)
        r0_tiled_grid<<<args.num_nodes*16,{32*warps}>>>(args);
}}
'''
    return source[:begin]+forward+source[end:]



def project_angular_sum(source):
    """Apply the unit-vector Jacobian after contracting all angular adjoints."""
    begin = source.index("        const Scalar radial = dx[lm]*x + dy[lm]*y + dz[lm]*z;")
    end = source.index("    }", begin)
    source = source[:begin]+"""
        gradients[lm] = dy[lm]*scale;
        gradients[16+lm] = dz[lm]*scale;
        gradients[32+lm] = dx[lm]*scale;
""" + source[end:]
    source = replace_once(source, """            force_x = weight*(force_r*direction_x+force_x);
            force_y = weight*(force_r*direction_y+force_y);
            force_z = weight*(force_r*direction_z+force_z);""", """
            if (gradients == nullptr) {
                force_r *= weight;
                force_x *= weight;
                force_y *= weight;
                force_z *= weight;
            } else {
                force_x = weight*(force_r*direction_x+force_x);
                force_y = weight*(force_r*direction_y+force_y);
                force_z = weight*(force_r*direction_z+force_z);
            }""")
    end_sum = "        force_z = subgroup_sum(force_z, edge_vector_length);"
    return replace_once(source, end_sum, end_sum + """
        if (gradients == nullptr) {
            force_r = subgroup_sum(force_r, edge_vector_length);
            if (lane == 0 && edge_active) {
                const Scalar norm2 = direction_x*direction_x
                    +direction_y*direction_y+direction_z*direction_z;
                const Scalar parallel = (force_x*direction_x
                    +force_y*direction_y+force_z*direction_z)/norm2;
                const Scalar delta = force_r-parallel;
                force_x += delta*direction_x;
                force_y += delta*direction_y;
                force_z += delta*direction_z;
            }
        }""")



def parallel_ambient_components(source):
    """Assign one spherical-harmonic component to each of the 16 edge lanes."""
    import re

    start = source.index("    dx[0] = dy[0] = dz[0] = Scalar(0);")
    end = source.index("    const Scalar scale = Scalar(3.5449077018110318)", start)
    field = re.compile(r"d[xyz]\[\d+\]")
    expressions = {}
    for statement in source[start:end].split(";"):
        if not statement.strip():
            continue
        parts = [part.strip() for part in statement.split("=")]
        rhs = field.sub(lambda m: "("+expressions[m.group()]+")", parts[-1])
        for name in parts[:-1]:
            if not field.fullmatch(name):
                raise ValueError("unsupported angular derivative expression")
            expressions[name] = rhs
    if len(expressions) != 48:
        raise ValueError("requires all real harmonics through degree three")
    # Each case uses the upstream expression, with exact constants and the
    # same signed cyclic axis permutation as direct_harmonic_gradients.
    assignments = []
    for m in range(16):
        components = [expressions[f"d{axis}[{m}]"] for axis in "xyz"]
        assignments.append(f"        case {m}: gx={components[0]}; gy={components[1]}; gz={components[2]}; break;")
    original_begin = source.index("__device__ __forceinline__ void direct_harmonic_gradients(")
    header_end = source.index("    sph[9] =", original_begin)
    header = source[original_begin:header_end].replace(
        "void direct_harmonic_gradients(", "void direct_ambient_component("
    ).replace("    Scalar* gradients)", "    Scalar* gradients, int lm)")
    header = header.replace("    Scalar dx[16];\n    Scalar dy[16];\n    Scalar dz[16];\n", "")
    function = header + """
    Scalar gx=0, gy=0, gz=0;
    switch (lm) {
""" + "\n".join(assignments) + """
    }
    const Scalar scale = Scalar(3.5449077018110318)*inverse_norm/static_cast<Scalar>(radius);
    gradients[lm] = gy*scale;
    gradients[16+lm] = gz*scale;
    gradients[32+lm] = gx*scale;
}
"""
    original_end = source.index("\n}", end)+len("\n}")
    source = source[:original_begin]+function+source[original_end:]
    source = replace_once(source, "if (lane == 0 && edge_active && gradients == nullptr)",
                          "if (lane < harmonic_count && edge_active && gradients == nullptr)")
    source = replace_once(source, "            direct_harmonic_gradients(\n",
                          "            direct_ambient_component(\n")
    return replace_once(source, "                args.radius[edge], direct_gradients);",
                        "                args.radius[edge], direct_gradients, lane);")


def lower_ambient_components(source):
    from mace_md_lab.xl_harmonic_lowering import cuda_helper

    begin = source.index("__device__ __forceinline__ void direct_harmonic_gradients(")
    end = source.index("\n}", begin)+len("\n}")
    source = source[:begin]+cuda_helper(ambient_only=True)+source[end:]
    call = source.index("        if (lane == 0 && edge_active && gradients == nullptr)")
    after_call = source.index("        subgroup_sync(edge_vector_length);", call)
    source = source[:call]+"""
        if (edge_active && gradients == nullptr)
            cooperative_gradients(harmonics+edge*harmonic_count,args.radius[edge],lane,direct_gradients);
"""+source[after_call:]
    # Stored Y contains unit-sphere harmonics. The direct compact geometry
    # may store a rounded unit direction u. Its normalization contributes
    # 1/|u| to dY/dx in addition to the physical inverse radius.
    return replace_once(source, """                const Scalar delta = force_r-parallel;
                force_x += delta*direction_x;
                force_y += delta*direction_y;
                force_z += delta*direction_z;""", """
                const Scalar inverse_norm = Scalar(1)/sqrt(norm2);
                force_x = (force_x-parallel*direction_x)*inverse_norm + force_r*direction_x;
                force_y = (force_y-parallel*direction_y)*inverse_norm + force_r*direction_y;
                force_z = (force_z-parallel*direction_z)*inverse_norm + force_r*direction_z;""")


def stage_radials(source, channelgrad):
    """Stage each active edge's radial values and derivatives in the forward grid.

    The workspace belongs to this cubin loaded by one serialized evaluator.
    Edges beyond capacity use the original checked interpolation in reverse.
    """
    marker = "template<int L, int WARPS>"
    source = replace_once(source, marker, """
constexpr int saved_edge_capacity = 24576;
__device__ Scalar saved_r0[saved_edge_capacity][4][2][channel_count];
__device__ double saved_radius[saved_edge_capacity];
__device__ int saved_pair[saved_edge_capacity];
__device__ const void* saved_table[saved_edge_capacity];

__device__ __forceinline__ Scalar staged_vjp(
        R0Args args, long long edge, int order, int channel, int pair,
        RadialSplineCoordinates coords, Scalar* dr) {
    if (edge < saved_edge_capacity && saved_table[edge] == args.radial.coefficients_raw
            && saved_radius[edge] == args.radius[edge] && saved_pair[edge] == pair) {
        *dr = saved_r0[edge][order][1][channel];
        return saved_r0[edge][order][0][channel];
    }
    return radial_spline_from_coordinates(
        args.radial, pair, order*channel_count+channel, coords, dr);
}

""" + marker)
    source = replace_once(source, "        #pragma unroll\n        for (int m=0; m<components; ++m)\n"
                          "            values[m] += radial*", """
        if (edge < saved_edge_capacity) {
            saved_r0[edge][L][0][channel] = radial;
            saved_r0[edge][L][1][channel] = unused;
            if constexpr (L == 0) {
                if (channel == 0) {
                    saved_table[edge] = args.radial.coefficients_raw;
                    saved_pair[edge] = pair;
                    saved_radius[edge] = r;
                }
            }
        }
        #pragma unroll
        for (int m=0; m<components; ++m)
            values[m] += radial*""")
    if channelgrad:
        for order in range(4):
            source = replace_once(source,
                f"radial_spline_from_coordinates(\n                        args.radial,edge_type,{order}*channel_count+c,spline_coordinates,&dr)",
                f"staged_vjp(args,edge,{order},c,edge_type,spline_coordinates,&dr)")
    else:
        source = replace_once(source,
            "radial_spline_from_coordinates(\n                        args.radial, edge_type, l * channel_count + channel,\n"
            "                        spline_coordinates, &spline_dr)",
            "staged_vjp(args,edge,l,channel,edge_type,spline_coordinates,&spline_dr)")
    return source


def endpoint_spline(source):
    """Evaluate a Hermite cubic from the adjacent endpoint values and slopes."""
    marker = "    const Scalar c2 = coefficients[base+2*spline.functions];"
    pos = source.index(marker)
    source = source[:pos] + """
    if (coordinates.interval + 1 < static_cast<int>(spline.intervals)) {
        const Scalar next_y = coefficients[base+4*spline.functions];
        const Scalar next_d = coefficients[base+5*spline.functions];
        const Scalar t = coordinates.x/spline.h;
        const Scalar dy = next_y-c0;
        const Scalar d0h = c1*spline.h;
        const Scalar d1h = next_d*spline.h;
        const Scalar a = d0h+d1h-2*dy;
        const Scalar b = 3*dy-2*d0h-d1h;
        *derivative = (d0h+t*(2*b+3*t*a))/spline.h;
        return c0+t*(d0h+t*(b+t*a));
    }
""" + source[pos:]
    return source


def build(compact, directory, variant):
    from symmetrix.mace_jit_operator_codegen import render_jit_r0_device_module

    if version("symmetrix-xl") != "0.2.0":
        raise ValueError("expected XL 0.2.0")
    payload = Path(compact).read_bytes()
    contract = json.loads(payload)["execution_contracts"]["R0"]
    source, meta = render_jit_r0_device_module(
        contract, precision="float64", target="sm_70", schedule="receiver-edge16"
    )
    for expected in ("constexpr int channel_count = 128;", "constexpr int r0_l_max = 3;",
                     "symmetrix_operator_r0_edge_weight_mode_v1 = 0;"):
        if expected not in source:
            raise ValueError(f"unsupported frozen R0 contract: {expected}")
    # Select an exact ordering of the radial and angular contractions.
    channelgrad = variant in {"channelgrad", "tilegrad", "childgrad", "stagedgrad", "tangentproj", "ambientwarp", "lowerproj"}
    source = channel_first_gradient(source) if channelgrad else factored_r0_source(source)
    if variant != "channelgrad":
        source = forward_source(source, variant)
    if variant in {"tangentproj", "ambientwarp", "lowerproj"}:
        source = project_angular_sum(source)
    if variant == "ambientwarp":
        source = parallel_ambient_components(source)
    if variant == "lowerproj":
        source = lower_ambient_components(source)
    if variant in {"staged", "stagedgrad"}:
        source = stage_radials(source, channelgrad)
    if variant == "endpoints":
        source = endpoint_spline(source)
    directory = Path(directory)
    directory.mkdir(exist_ok=True, parents=True)
    code = directory/f"r0-{variant}.cu"
    code.write_text(source)
    cubin = code.with_suffix(".cubin")
    if VARIANTS[variant] > 1:
        obj = code.with_suffix(".o")
        # The CUDA runtime headers declare std::size_t for relocatable code.
        subprocess.run(["nvcc","-std=c++17","-arch=sm_70","-dc","-O3",str(code),
                        "-o",str(obj)], timeout=90, check=True)
        subprocess.run(["nvcc","-arch=sm_70","-dlink","-cubin",str(obj),"-lcudadevrt",
                        "-o",str(cubin)], timeout=90, check=True)
    else:
        subprocess.run(["nvcc","-std=c++17","-arch=sm_70","-cubin","-O3",str(code),
                        "-o",str(cubin)], timeout=90, check=True)
    data = dict(source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                cubin_sha256=hashlib.sha256(cubin.read_bytes()).hexdigest(),
                compact_sha256=hashlib.sha256(payload).hexdigest(), artifact_id=meta["artifact_id"],
                structure=meta["structure_fingerprint"], variant=variant, forward_shards=VARIANTS[variant])
    cubin.with_suffix(".json").write_text(json.dumps(data,indent=2)+"\n")


def _load_compiled(calc, directory, *, variant):
    if version("symmetrix-xl") != "0.2.0" or variant not in VARIANTS:
        raise ValueError("unsupported XL version or R0 variant")
    path = (Path(directory)/f"r0-{variant}.cubin").resolve()
    data = json.loads(path.with_suffix(".json").read_text())
    if data["variant"] != variant or hashlib.sha256(path.read_bytes()).hexdigest() != data["cubin_sha256"]:
        raise ValueError("unverified R0 module")
    evaluator = calc.evaluator
    evaluator._load_r0_device_module(str(path),4,False,"receiver-edge16")
    if (not evaluator.r0_device_module_ready
            or evaluator.r0_device_module_artifact_id != data["artifact_id"]
            or Path(evaluator.r0_device_module_path).resolve() != path):
        raise ValueError("native evaluator did not load the requested R0 variant")
    return data


def install(calc, directory, *, variant):
    if variant in {"staged", "stagedgrad"}:
        raise ValueError("staging requires the isolated single-evaluator sensitivity benchmark")
    return _load_compiled(calc, directory, variant=variant)


def install_algorithm(calc, compact, config):
    directory, variant = Path(config["directory"]), config["variant"]
    if variant not in VARIANTS:
        raise ValueError("unknown native R0 variant")
    metadata = json.loads((directory/f"r0-{variant}.json").read_text())
    if hashlib.sha256(Path(compact).read_bytes()).hexdigest() != metadata["compact_sha256"]:
        raise ValueError("R0 cubin was built for a different compact export")
    return install(calc, directory, variant=variant)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symmetrix-json", required=True, type=Path)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=["tile1"])
    args = parser.parse_args()
    for name in args.variants:
        build(args.symmetrix_json,args.artifact_dir,name)
