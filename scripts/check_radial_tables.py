"""Qualify tabulated frozen radial networks at sub-knot radii and at the cutoff."""

import argparse
from pathlib import Path

import torch
from mace.calculators import MACECalculator

from mace_md_lab.bench import file_hash, write_report
from mace_md_lab.edge_virial import verify_edge_checkpoint
from mace_md_lab.radial_table import _exact_radial, prepare_radial_tables


def check_layer(model, tables, index):
    layer = model.interactions[index]
    net = list(layer.conv_tp_weights.children())
    if tables.lookup[index].coeff.shape[-1] == 64:
        net = net[:-1]
    lookup = tables.lookup[index]
    cutoff, width = tables.cutoff, lookup.coeff.shape[-1]
    device = next(model.parameters()).device
    dtype = torch.float64
    step = cutoff / (tables.knots - 1)
    middle = (torch.arange(tables.knots - 1, device=device, dtype=dtype) + 0.5) * step
    boundary = torch.tensor([0., 1e-8, 1e-6, 0.01, 0.1, 0.25, 0.5, cutoff - 1e-6,
                             cutoff - 1e-10, cutoff, cutoff + 1e-8, cutoff + 1.0],
                            device=device, dtype=dtype)
    # Project the component slopes with an alternating-sign vector of unit norm.
    probe = (((torch.arange(width, device=device) % 2) * 2 - 1) / width**0.5).to(dtype)
    metrics = {}
    for label, radii in [("all_interval_midpoints", middle), ("origin_cutoff_and_shell", boundary)]:
        max_value = max_proj = max_reference = grad_scale = fd_error = shell = 0.
        for chunk in radii.split(256):
            r = chunk[:, None].detach().requires_grad_(True)
            exact = _exact_radial(model.radial_embedding, net, r)
            interpolated = lookup(r)
            grad = torch.autograd.grad((interpolated * probe).sum(), r)[0]
            reference_grad = torch.autograd.grad((exact * probe).sum(), r)[0]
            max_value = max(max_value, (interpolated - exact).abs().max().item())
            max_proj = max(max_proj, (grad - reference_grad).abs().max().item())
            max_reference = max(max_reference, exact.abs().max().item())
            grad_scale = max(grad_scale, reference_grad.abs().max().item())
            if (r >= cutoff).any():
                outer = (r >= cutoff).flatten()
                shell = max(shell, interpolated[outer].abs().max().item(), grad[outer].abs().max().item())
            interior = (r[:, 0] > 1e-4) & (r[:, 0] < cutoff - 1e-4)
            if interior.any():
                offset = 1e-6
                pts = r[interior].detach()
                central = ((lookup(pts+offset) - lookup(pts-offset)) * probe).sum(1) / (2 * offset)
                fd_error = max(fd_error, (central - grad[interior].flatten()).abs().max().item())
        metrics[label] = {"max_value_abs_difference": max_value,
                          "max_reference_abs_value": max_reference,
                          "max_projected_derivative_abs_difference": max_proj,
                          "max_reference_abs_projected_derivative": grad_scale,
                          "max_interpolated_fd_abs_difference": fd_error,
                          "max_abs_exterior_value_or_derivative": shell}
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--knots", type=int, default=8192)
    args = p.parse_args()
    if args.output.resolve() == args.model.resolve():
        p.error("output overlaps input")
    verify_edge_checkpoint(args.model)
    model = MACECalculator(model_paths=[str(args.model)], default_dtype="float64",
                           device="cuda").models[0].eval().requires_grad_(False)
    cases = {}
    for width in (64, 512):
        tables = prepare_radial_tables(model, args.knots, width)
        if width == 64:
            cases["first_hidden64"] = check_layer(model, tables, 0)
        cases["second_" + str(width)] = check_layer(model, tables, 1)
    # Scale internal-unit errors by the largest reference response in this probe.
    # Moving-MD cases separately gate physical energies, forces and stresses.
    value_rtol, slope_rtol, fd_rtol = 1e-8, 1e-6, 5e-6
    passed = all(
        all(torch.isfinite(torch.as_tensor(list(m.values()))).tolist())
        and m["max_abs_exterior_value_or_derivative"] == 0
        and m["max_value_abs_difference"] <= value_rtol * max(1, m["max_reference_abs_value"])
        and m["max_projected_derivative_abs_difference"] <= slope_rtol * max(1, m["max_reference_abs_projected_derivative"])
        and m["max_interpolated_fd_abs_difference"] <= fd_rtol * max(1, m["max_reference_abs_projected_derivative"])
        for rows in cases.values() for m in rows.values()
    )
    write_report(args.output, dict(
        passed=passed, knots=args.knots, checkpoint_sha256=file_hash(args.model),
        gpu=torch.cuda.get_device_name(), torch=str(torch.__version__), dtype="float64",
        derivative_probe="alternating component signs, unit 2-norm",
        scope="internal units, every interval midpoint and explicit origin/cutoff/shell probes",
        cases=cases, tolerances={"value_scaled": value_rtol, "projected_slope_scaled": slope_rtol,
                                 "slope_fd_scaled": fd_rtol, "exterior_absolute": 0.0},
    ))
    if not passed:
        raise SystemExit("internal interpolation test failed")


if __name__ == "__main__":
    main()
