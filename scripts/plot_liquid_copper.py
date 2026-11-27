"""Plot experimental and simulated Cu diffusivities and directly timed ten-ps trajectories."""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "data", type=Path, help="directory containing report.json and matched-10ps.json"
    )
    p.add_argument("output", type=Path)
    a = p.parse_args()
    groups = json.loads((a.data / "report.json").read_text())["groups"]
    runs = json.loads((a.data / "matched-10ps.json").read_text())["routes"]
    temperatures = np.array([1420, 1520, 1620])
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(7.0, 3.05), gridspec_kw={"width_ratios": [1.1, 1.0]}
    )
    fig.subplots_adjust(left=0.09, right=0.98, top=0.86, bottom=0.20, wspace=0.56)
    left.errorbar(
        temperatures,
        [3.70, 4.50, 5.20],
        yerr=[0.06, 0.07, 0.10],
        fmt="s--",
        ms=5,
        color="#2a3342",
        lw=1,
        capsize=2,
        label="Neutron scattering",
    )
    for n, marker, color in [(108, "o", "#007d73"), (256, "D", "#b76115")]:
        rows = sorted((g for g in groups if g["N"] == n), key=lambda g: g["T"])
        left.errorbar(
            [r["T"] for r in rows],
            [r["D_mean"] for r in rows],
            yerr=[r["D_SD"] for r in rows],
            marker=marker,
            ls="-" if n == 108 else "none",
            ms=5,
            color=color,
            lw=1.3,
            capsize=3,
            label=f"MACE · {n} atoms",
        )
    continuous = json.loads((a.data / "continuous" / "report.json").read_text())[
        "runs"
    ][0]
    left.plot(
        continuous["T"],
        continuous["D"],
        marker="x",
        color="#52366f",
        markersize=7,
        markeredgewidth=1.4,
        ls="none",
        label="MACE · 1 ns (108)",
    )
    left.set_xticks(temperatures)
    left.set_xlabel("Temperature (K)")
    left.set_ylabel(r"$D$ ($10^{-9}$ m$^2$ s$^{-1}$)")
    left.legend(frameon=False, fontsize=7, loc="best")
    left.set_title(
        "(a)   Liquid Cu self-diffusion", loc="left", weight="bold", fontsize=9.5
    )
    tags = ["official", "independent", "xl", "xl_plus"]
    values = [runs[k]["seconds"] for k in tags]
    left.grid(axis="y", alpha=0.18)
    left.set_axisbelow(True)
    right.barh(
        np.arange(4),
        values,
        color=["#65727e", "#259186", "#9ca9b2", "#005f57"],
        height=0.54,
    )
    right.set_yticks(
        np.arange(4),
        ["Official MACE", "Independent", "XL alone", "Ours + XL"],
        fontsize=8,
    )
    right.invert_yaxis()
    right.set_xlim(0, max(values) * 1.30)
    right.set_xlabel("Measured wall time (s)")
    for i, v in enumerate(values):
        right.text(v + max(values) * 0.023, i, f"{v:.1f}", va="center", fontsize=8.2)
    right.set_title("(b)   10 ps at 1520 K", loc="left", weight="bold", fontsize=9.5)
    fig.savefig(a.output, dpi=220, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


if __name__ == "__main__":
    main()
