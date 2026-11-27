"""Compute time-origin averaged MSDs and replica uncertainties for unwrapped liquid-Cu trajectories."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def time_origin_msd(xyz):
    """Einstein MSD in A^2, averaged over atoms and all available time origins."""
    n, atoms, _ = xyz.shape
    paths = np.asarray(xyz, dtype=np.float64).copy()
    paths -= paths[:1].copy()
    paths -= paths.mean(axis=1, keepdims=True)
    r2 = np.einsum("tac,tac->t", paths, paths)
    prefix = np.r_[0.0, np.cumsum(r2)]
    fft_length = 1 << (2 * n - 1).bit_length()
    corr = np.zeros(n)
    for chunk in np.array_split(paths, max(1, (atoms + 31) // 32), axis=1):
        freq = np.fft.rfft(chunk, n=fft_length, axis=0)
        spectrum = (freq.real**2 + freq.imag**2).sum(axis=(1, 2))
        corr += np.fft.irfft(spectrum, n=fft_length)[:n]
    lag = np.arange(n)
    return np.maximum(
        (prefix[n] - prefix[lag] + prefix[n - lag] - 2 * corr) / ((n - lag) * atoms),
        0.0,
    )


def fit_diffusion(t, msd, low=3.0, high=15.0):
    mask = (t >= low) & (t <= high)
    # 1 A^2 / ps = 10 * 10^-9 m^2/s. D = slope/6.
    slope, intercept = np.polyfit(t[mask], msd[mask], 1)
    beta = np.polyfit(np.log(t[mask]), np.log(msd[mask]), 1)[0]
    return float(slope * 10 / 6), float(beta)


def analyze(directory):
    info = json.loads((directory / "result.json").read_text())
    with np.load(directory / "samples.npz") as z:
        t, xyz = z["t_ps"], z["xyz"]
        msd = time_origin_msd(xyz)
        diffusion, beta = fit_diffusion(t, msd)
        halves = np.array_split(xyz, 2)
        half_d = [fit_diffusion(t[: len(x)], time_origin_msd(x))[0] for x in halves]
        info.update(
            D=diffusion,
            beta=beta,
            D_2_10ps=fit_diffusion(t, msd, 2, 10)[0],
            D_5_20ps=fit_diffusion(t, msd, 5, 20)[0],
            D_halves=half_d,
            D_unit="1e-9 m2/s",
            fcc_order_p95=float(np.quantile(z["fcc_order"], 0.95)),
            sampling_ps=float(t[1] - t[0]),
        )
    return info, t, msd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--blocks",
        action="store_true",
        help="compute consecutive 100 ps intervals for one continuous trajectory",
    )
    a = p.parse_args()
    a.out.mkdir(exist_ok=True, parents=True)
    files = sorted(a.root.glob("T*-N*-r*/result.json"))
    if not files:
        p.error("no completed liquid-Cu trajectories found")
    if a.blocks and len(files) != 1:
        p.error("--blocks expects one continuous trajectory")
    runs = []
    with (a.out / "msd.csv").open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["temperature_K", "atoms", "seed", "lag_ps", "msd_A2"])
        for file in files:
            info, t, msd = analyze(file.parent)
            runs.append(info)
            for i in range(0, len(t) // 2 + 1, 10):
                w.writerow(
                    [
                        info["T"],
                        info["N"],
                        info["seed"],
                        round(t[i], 8),
                        round(msd[i], 8),
                    ]
                )
    groups = []
    for key in sorted(set((r["T"], r["N"]) for r in runs)):
        group = [r for r in runs if (r["T"], r["N"]) == key]
        vals = np.array([r["D"] for r in group])
        groups.append(
            {
                "T": key[0],
                "N": key[1],
                "replicas": len(group),
                "D_mean": float(vals.mean()),
                "D_SD": float(vals.std(ddof=1)) if len(vals) > 1 else None,
                "D_SEM": (
                    float(vals.std(ddof=1) / np.sqrt(len(vals)))
                    if len(vals) > 1
                    else None
                ),
                "replica_ps": [r["ps"] for r in group],
                "production_s": sum(r["production_s"] for r in group),
                "mean_observed_T": float(np.mean([r["mean_T"] for r in group])),
                "mean_log_slope": float(np.mean([r["beta"] for r in group])),
                "mean_msd_2_10_D": float(np.mean([r["D_2_10ps"] for r in group])),
                "mean_msd_5_20_D": float(np.mean([r["D_5_20ps"] for r in group])),
            }
        )
    (a.out / "report.json").write_text(
        json.dumps({"groups": groups, "runs": runs}, indent=2, allow_nan=False) + "\n"
    )
    if a.blocks:
        with np.load(files[0].parent / "samples.npz") as trajectory:
            positions, times = trajectory["xyz"], trajectory["t_ps"]
        width = round(100.0 / (times[1] - times[0]))
        if len(times) <= width or (len(times) - 1) % width:
            p.error("continuous run length must be an integer multiple of 100 ps")
        block_d = [
            fit_diffusion(
                times[: width + 1], time_origin_msd(positions[i : i + width + 1])
            )[0]
            for i in range(0, len(times) - 1, width)
        ]
        (a.out / "blocks.json").write_text(
            json.dumps({"block_ps": 100, "D_100ps_blocks": block_d}, indent=2)
        )
    print(json.dumps(groups, indent=2))


if __name__ == "__main__":
    main()
