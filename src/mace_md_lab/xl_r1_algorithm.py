"""Warp-owned edges and scalar-source adjoints for the frozen XL MP-0 Small R1."""
import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
import subprocess


CONTRACT = "sha256:963963a85b9316a79f901a9b9e63eb953bcd4abccfe80e74687305e58a492c0f"
SCALAR_PROGRAM = "sha256:a686e71e5d35d53c00b3140ba5b69c67c0c287a7a14b339b58fbb25241a74504"
WARPS = {"edgewarp2": 2, "edgewarp4": 4, "edgewarp8": 8, "edgewarp8forward": 8, "edgewarp8pairs": 8,
         "edgewarp8lower": 8}


def forward_source(paired):
    # The native owner grid counts (receiver, channel). Each warp remains
    # coalesced along channel. Optionally split one receiver between all
    # block warps and reduce disjoint chunks of its incoming neighbor list.
    launch = """
    constexpr int splits = 2;
    __shared__ Scalar partial[splits][16][128];
    const int warp = threadIdx.x / 32;
    const int group = warp / 4;
    const int c = threadIdx.x % 128;
    for (int recv = blockIdx.x; recv < args.num_nodes; recv += gridDim.x) {
""" if paired else """
    constexpr int splits = 1;
    const int group = 0;
    for (long long owner = blockIdx.x*blockDim.x+threadIdx.x; owner < args.num_nodes*128;
         owner += static_cast<long long>(blockDim.x)*gridDim.x) {
        int c = owner % 128;
        int recv = owner / 128;
"""
    storage = """
        #pragma unroll
        for (int k=0; k<16; ++k) partial[group][k][c] = acc[k];
        __syncthreads();
        if (group==0) {
            #pragma unroll
            for (int k=0;k<16;++k)
                output[(recv*16+k)*128+c] = partial[0][k][c] + partial[1][k][c];
        }
        __syncthreads();
""" if paired else """
        #pragma unroll
        for (int k=0; k<16; ++k) output[(recv*16+k)*128+c]=acc[k];
"""
    return """
extern "C" __global__ void symmetrix_factorized_forward_v2(SymmetrixJitCudaR1ForwardArgsV2 args)
{
    const Scalar* features = static_cast<const Scalar*>(args.neighbor_features);
    const Scalar* Y = static_cast<const Scalar*>(args.harmonics_values);
    Scalar* output = static_cast<Scalar*>(args.output);
""" + launch + """
        Scalar acc[16] = {};
        int t = args.type_to_active[args.node_types[recv]];
        int begin = args.first_neigh[recv], end = begin + args.num_neigh[recv];
        for (int e=begin+group; e<end; e+=splits) {
            double r=args.radius[e];
            if (!(r<args.cutoff)) continue;
            int src=args.neigh_indices[e];
            int pair=pair_type(t,args.type_to_active[args.neigh_types[e]],args.active_type_count);
            EvaluationPoint p=evaluation_point(args.radial,r);
            Scalar h=features[src*128+c];
            #pragma unroll
            for (int l=0;l<4;++l) {
                Scalar v=h*evaluate_radial(args.radial,pair,p,l*128+c);
                #pragma unroll
                for (int m=l*l;m<(l+1)*(l+1);++m) acc[m] += v*Y[e*16+m];
            }
        }
""" + storage + "    }\n}\n"


def reverse_source(warps):
    return r"""
extern "C" __global__ __launch_bounds__(WARPS * 32)
void symmetrix_factorized_reverse_fused_v2(
    SymmetrixJitCudaR1SourceArgsV2 src, SymmetrixJitCudaR1EdgeArgsV2 edges)
{
    constexpr int nwarps = WARPS;
    constexpr int channels = 128;
    __shared__ Scalar partial_src[nwarps][channels];
    __shared__ Scalar fallback_grad[nwarps][48];
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const Scalar* adjoint = static_cast<const Scalar*>(src.output_adjoint);
    const Scalar* features = static_cast<const Scalar*>(edges.neighbor_features);
    const Scalar* harmonic = static_cast<const Scalar*>(edges.harmonics_values);
    const Scalar* grad = static_cast<const Scalar*>(edges.harmonics_gradients);
    for (int source = blockIdx.x; source < src.num_nodes; source += gridDim.x) {
        Scalar source_acc[4] = {};
        Scalar feature[4];
        #pragma unroll
        for (int q = 0; q < 4; ++q)
            feature[q] = features[source*channels + q*32 + lane];
        int source_type = src.type_to_active[src.node_types[source]];
        int begin = src.source_offsets[source], end = src.source_offsets[source + 1];
        // A warp owns one edge, all 128 channels, and its directed-force slot.
        for (int slot = begin + warp; slot < end; slot += nwarps) {
            int e = src.source_edges[slot];
            double r = src.radius[e];
            if (!(r < src.cutoff)) continue;
            int recv = src.edge_receivers[e];
            int pair = pair_type(src.type_to_active[src.node_types[recv]],
                                 source_type, src.active_type_count);
            EvaluationPoint point = evaluation_point(src.radial, r);
            const Scalar* Y = harmonic + e*16;
            const Scalar* G = grad ? grad + e*48 : fallback_grad[warp];
            if (!grad) {
                if (lane == 0) {
                    if (!edges.coordinates_are_unit && edges.coordinate_scalar_size == sizeof(float)) {
                        // XL's Cartesian gradient helper accepts a double vector.
                        const float* p = static_cast<const float*>(edges.xyz) + 3*e;
                        double cartesian[3] = {p[0], p[1], p[2]};
                        direct_harmonic_gradients(cartesian, sizeof(double), 0, 0, r, fallback_grad[warp]);
                    } else {
                        direct_harmonic_gradients(edges.xyz, edges.coordinate_scalar_size,
                                                  edges.coordinates_are_unit, 3*e, r, fallback_grad[warp]);
                    }
                }
                __syncwarp();
            }
            Scalar radial = 0, ax = 0, ay = 0, az = 0;
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int c = q*32 + lane;
                Scalar dr = 0, tx = 0, ty = 0, tz = 0;
                const Scalar* recv_grad = adjoint + recv*2048 + c;
                // One radial value multiplies every magnetic component at fixed l.
                #pragma unroll
                for (int l = 0; l < 4; ++l) {
                    Scalar dot = 0, gx = 0, gy = 0, gz = 0;
                    #pragma unroll
                    for (int k = l*l; k < (l+1)*(l+1); ++k) {
                        Scalar a = recv_grad[k*channels];
                        dot += a*Y[k];
                        gx += a*G[k];
                        gy += a*G[16+k];
                        gz += a*G[32+k];
                    }
                    Scalar v, dv;
                    evaluate_radial(src.radial, pair, point, l*channels + c, v, dv);
                    source_acc[q] += v*dot;
                    dr += dv*dot;
                    tx += v*gx;
                    ty += v*gy;
                    tz += v*gz;
                }
                radial += feature[q]*dr;
                ax += feature[q]*tx;
                ay += feature[q]*ty;
                az += feature[q]*tz;
            }
            #pragma unroll
            for (int s=16; s>0; s/=2) {
                radial += __shfl_down_sync(0xffffffffu, radial, s);
                ax += __shfl_down_sync(0xffffffffu, ax, s);
                ay += __shfl_down_sync(0xffffffffu, ay, s);
                az += __shfl_down_sync(0xffffffffu, az, s);
            }
            if (lane == 0) {
                Scalar angular[3] = {ax, ay, az};
                #pragma unroll
                for (int k=0; k<3; ++k) {
                    const int index = 3*e + k;
                    Scalar unit = edges.coordinate_scalar_size == sizeof(float)
                        ? static_cast<const float*>(edges.xyz)[index]
                        : static_cast<const double*>(edges.xyz)[index];
                    if (!edges.coordinates_are_unit) unit /= r;
                    edges.directed_forces[index] -= radial*unit + angular[k];
                }
            }
        }
        #pragma unroll
        for (int q = 0; q < 4; ++q) partial_src[warp][q*32 + lane] = source_acc[q];
        __syncthreads();
        for (int c = threadIdx.x; c < channels; c += blockDim.x) {
            Scalar a = 0;
            #pragma unroll
            for (int w=0; w<nwarps; ++w) a += partial_src[w][c];
            static_cast<Scalar*>(src.source_adjoint)[source*channels + c] += a;
        }
        __syncthreads();
    }
}
""".replace("WARPS", str(warps))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(compact, directory, variant):
    from symmetrix.mace_jit_codegen import render_jit_r1_cuda_module
    if version("symmetrix-xl") != "0.2.0":
        raise ValueError("Requires Symmetrix-XL 0.2.0")
    compact, directory = Path(compact), Path(directory)
    c = json.loads(compact.read_text())["execution_contracts"]["R1"]
    if (c["structure_fingerprint"] != CONTRACT or c["fingerprint"] != SCALAR_PROGRAM
            or variant not in (*WARPS, "upstream")):
        raise ValueError("Unsupported scalar source, coupling layout, or R1 variant")
    source = render_jit_r1_cuda_module(c, 70, precision="float64")
    if variant in WARPS:
        begin = source.index('extern "C" __global__ __launch_bounds__(reverse_threads_per_block)\nvoid symmetrix_factorized_reverse_fused_v2(')
        end = source.index('extern "C" __global__', begin + 15)
        kernel = reverse_source(WARPS[variant])
        if variant == "edgewarp8lower":
            from .xl_harmonic_lowering import cuda_helper
            branch = kernel.index("                if (lane == 0) {")
            sync = kernel.index("                __syncwarp();", branch)
            # Rounded unit-vector inputs carry a norm distinct from the physical
            # radius. Retain XL's original fallback on that separate input layout.
            ordinary = kernel[branch:sync]
            kernel = (cuda_helper() + "\n" + kernel[:branch]
                      + "                if (!edges.coordinates_are_unit) {\n"
                      + "                    cooperative_gradients(Y, r, lane, fallback_grad[warp]);\n"
                      + "                } else {\n" + ordinary + "                }\n" + kernel[sync:])
        source = source[:begin] + kernel + source[end:]
        marker = 'symmetrix_factorized_reverse_threads_per_block =\n    64;'
        if source.count(marker) != 1:
            raise ValueError("Unknown XL launch metadata")
        source = source.replace(marker, marker.replace("64;",f"{32*WARPS[variant]};"))
    if variant in ("edgewarp8forward","edgewarp8pairs"):
        begin = source.index('extern "C" __global__ void symmetrix_factorized_forward_v2(')
        end = source.index('extern "C" __global__', begin + 15)
        source = source[:begin] + forward_source(variant == "edgewarp8pairs") + source[end:]
    directory.mkdir(exist_ok=True, parents=True)
    out = directory / f"r1-{variant}.cu"
    out.write_text(source)
    cubin = out.with_suffix(".cubin")
    subprocess.run(["nvcc", "-O3", "-std=c++17", "-arch=sm_70", "-cubin", str(out), "-o", str(cubin)],
                   check=True, timeout=90)
    metadata = dict(variant=variant, compact_sha256=digest(compact), structure=CONTRACT,
                    source_sha256=digest(out), cubin_sha256=digest(cubin))
    out.with_suffix(".json").write_text(json.dumps(metadata, indent=2)+"\n")
    return metadata


def install_algorithm(calc, compact, config):
    """Load an R1 candidate into a dedicated stock XL evaluator."""
    variant = config["variant"]
    if variant not in (*WARPS, "upstream") or version("symmetrix-xl") != "0.2.0":
        raise ValueError("Unsupported R1 module")
    if (calc.alchemical is not None or calc._has_native_field_coupling()
            or calc.execution_plan["selected_id"] != "mh0-direct-speed"):
        raise ValueError("R1 requires the unweighted static-composition direct MACE speed graph")
    cubin = Path(config["directory"]) / f"r1-{variant}.cubin"
    info = json.loads(cubin.with_suffix(".json").read_text())
    if (info["structure"] != CONTRACT or info["variant"] != variant
            or info["cubin_sha256"] != digest(cubin) or info["compact_sha256"] != digest(compact)):
        raise ValueError("Mismatch between model, R1 module, and build metadata")
    evaluator = calc.evaluator
    evaluator._load_jit_cuda_plugin(str(cubin.resolve()), 8)
    admitted = Path(evaluator.jit_cuda_plugin_path).resolve()
    if not evaluator.jit_cuda_plugin_ready or admitted != cubin.resolve():
        raise ValueError("Native evaluator did not admit the requested R1 cubin path")
    return dict(info, native_artifact=evaluator.jit_cuda_plugin_artifact_id,
                native_module_basename=admitted.name)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--compact", type=Path, required=True)
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--variants", nargs="+", choices=["upstream", *WARPS], required=True)
    args = p.parse_args()
    for v in args.variants:
        print(build(args.compact, args.directory, v), flush=True)
