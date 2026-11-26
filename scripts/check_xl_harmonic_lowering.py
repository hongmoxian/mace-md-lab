"""Check the XL l<=3 lowering identity in 16- and 32-lane CUDA subgroups."""
import argparse
import hashlib
from pathlib import Path

import cupy as cp
import numpy as np
import torch

# The installed e3nn constants contain Python slices in their lookup indexes.
with torch.serialization.safe_globals([slice]):
    from e3nn import o3

from mace_md_lab.bench import write_report
from mace_md_lab.xl_harmonic_lowering import cuda_helper


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(421)
    dirs = rng.normal(size=(257, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs = np.concatenate((dirs, np.eye(3), -np.eye(3)))
    radii = np.array([0.2, 1, 2.67, 5.999])
    xyz = np.concatenate([r * dirs for r in radii])
    x = torch.tensor(xyz, device="cuda", dtype=torch.float64, requires_grad=True)
    def harmonics(q):
        return o3.spherical_harmonics(list(range(4)), q, normalize=True, normalization="component")
    y = harmonics(x)
    ref = torch.stack(
        [torch.autograd.grad(y[:, lm].sum(), x, retain_graph=True)[0] for lm in range(16)], dim=2
    ).detach()
    fds = {}
    for h in (2e-5, 1e-5):
        projections = []
        with torch.no_grad():
            for v in h * torch.eye(3, device=x.device, dtype=x.dtype):
                numeric = (harmonics(x-2*v) - 8*harmonics(x-v)
                           + 8*harmonics(x+v) - harmonics(x+2*v)) / (12*h)
                projections.append(numeric)
        fds[str(h)] = float((torch.stack(projections, dim=1)-ref).abs().max())
    full = "namespace full {\n" + cuda_helper() + "}\n"
    ambient = "namespace ambient {\n" + cuda_helper(ambient_only=True) + "}\n"
    source = "using Scalar = double;\n" + full + ambient + """
extern "C" __global__ void evaluate(const double* y, const double* r, double* out, int n, int width, int ambient_only) {
    const int thread = blockIdx.x*blockDim.x + threadIdx.x;
    const int group = thread/width, lane=thread % width;
    if (group >= n) return;
    if (ambient_only) ambient::cooperative_gradients(y + 16*group, r[group], lane, out+48*group);
    else full::cooperative_gradients(y + 16*group, r[group], lane, out+48*group);
}
"""
    kernel = cp.RawKernel(source, "evaluate", options=("--std=c++17",))
    # Copies precede launch and avoid depending on DLPack stream conventions in this diagnostic.
    features = cp.asarray(y.detach().cpu().numpy())
    rdev = cp.asarray(np.linalg.norm(xyz, axis=1))
    refcpu = ref.cpu().numpy()
    errors = {}
    unit = xyz / np.linalg.norm(xyz, axis=1, keepdims=True)
    for width in (16, 32):
        threads = 128
        blocks = (len(xyz)*width + threads-1) // threads
        for ambient_only in (False, True):
            out = cp.full(refcpu.shape, np.nan, dtype=cp.float64)
            args_kernel = (features, rdev, out, np.int32(len(xyz)), np.int32(width), np.int32(ambient_only))
            kernel((blocks,), (threads,), args_kernel)
            result = cp.asnumpy(out)
            if ambient_only:
                result -= unit[:, :, None] * np.einsum("na,nak->nk", unit, result)[:, None, :]
            delta = np.abs(result - refcpu)
            key = f"{width}-" + ("ambient-projected" if ambient_only else "tangent")
            errors[key] = {"all": float(delta.max()), "by_xyz": delta.max(axis=(0, 2)).tolist(),
                           "by_l": [float(delta[:, :, l*l:(l+1)**2].max()) for l in range(4)]}
    passed = all(row["all"] < 2e-12 for row in errors.values()) and max(fds.values()) < 3e-8
    result = dict(passed=bool(passed), device=torch.cuda.get_device_name(), dtype="float64", e3nn_basis="component",
                  radii_A=radii.tolist(), vectors=len(xyz), random_directions=257, axes_directions=6, l_max=3,
                  cuda_sha256=hashlib.sha256(source.encode()).hexdigest(), subgroup_max_errors=errors,
                  centered_fourth_order_fd_vs_autograd=fds)
    write_report(args.output, result)
    print(result, flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
