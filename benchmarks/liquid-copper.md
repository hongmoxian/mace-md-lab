# Liquid copper diffusion with accelerated MACE

Use the full-precision MACE-MP-0 Small checkpoint and its 8,192-knot XL compact export with the prepared Verlet driver and the locked first- and second-interaction kernels. Each integration step computes host energy, all atomic forces and the six configurational stress components. Experimental diffusion coefficients are from Table I in [Meyer, Phys. Rev. B 81, 012102](https://doi.org/10.1103/PhysRevB.81.012102).

## Runtime and diffusion

A continuous 1 ns trajectory (108 atoms, 1520 K, 500,000 steps) ran in **17.74 min** on one Tesla V100-SXM2 (16 GB), with coordinates and observables sampled every 50 fs. Session setup took 5.95 s and the separate 13 ps preparation took 14.17 s. The measured time across setup, preparation, production, writing the trajectory and six high-precision official checks was 18.16 min. The long-run self-diffusivity was 3.584 × 10⁻⁹ m²/s, compared with 4.50 ± 0.07 experimentally. Its 100 ps block estimates ranged from 3.35 to 3.87. [Uninterrupted run](raw/liquid-cu/continuous/report.json) · [Block estimates](raw/liquid-cu/continuous/blocks.json) · [Download full unwrapped trajectories and checksums](https://github.com/hongmoxian/mace-md-lab/tree/data/liquid-cu-v1).

| 1520 K, 108 atoms | Measured seconds / 10 ps | Speedup vs official |
|---|---:|---:|
| Official MACE + ASE | 133.22 | 1.00× |
| Independent MACE | 18.39 | 7.24× |
| Tuned upstream XL + ASE | 17.06 | 7.81× |
| Prepared driver and locked kernels on XL | 9.82 | 13.57× |

Each route ran a complete separate 5,000-step window from the same liquid coordinates and momenta with the same thermostat seed. Preparation and ten warmup steps are excluded. These four windows supplement the interleaved 4 × 50-step measurements in the per-replica report.

| T (K) | Atoms | Replicas × ps | D ± replica SD | Neutron D | Production GPU-min (sum) |
|---:|---:|---:|---:|---:|---:|
| 1420 | 108 | 3 × 100 | 2.927 ± 0.224 | 3.70 | 5.19 |
| 1520 | 108 | 3 × 100 | 3.428 ± 0.217 | 4.50 | 5.13 |
| 1620 | 108 | 3 × 100 | 4.227 ± 0.054 | 5.20 | 5.26 |
| 1520 | 256 | 3 × 50 | 3.758 ± 0.232 | 4.50 | 3.38 |

D is in units of 10⁻⁹ m²/s. The 108-atom estimates differ from the corresponding neutron results by -20.9%, -23.8%, -18.7%. The 256-atom control at 1520 K measures the sensitivity to cell size. Absolute agreement with experiment also depends on the frozen potential and density; the comparisons above are for finite cells.

Across the 13 accelerated trajectories, 2.05 ns of production required 36.69 GPU-min (sum of measured production intervals). For the 78 official recalculations, the largest energy, force and stress component discrepancies were 1.98e-09 eV/atom, 8.91e-08 eV/Å and 5.23e-10 eV/Å³. The 2 fs, 2 ps NVE pilot had a total energy span of 0.0769 meV/atom.

## Reproduce the simulations

Prepare the compact export, the SM70 XL backend, and the M0/M1/R0/R1 modules described in [the direct comparison](direct-comparison.md#reproduce). Run from the repository root. Set `model`, `compact`, `mods` and `m0` to the frozen checkpoint, matching compact model, product module directory and first-product module directory, respectively. The locked route configuration locates its R0 and R1 kernels relative to the working directory.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export SYMMETRIX_NEIGHBOR_BACKEND=kokkos
args=(--model "$model" --symmetrix-json "$compact" --product-dir "$mods" --m0-dir "$m0")
for temp in 1520 1420 1620; do
  for replica in 0 1 2; do
    extra=()
    if [[ "$temp" == 1520 && "$replica" == 0 ]]; then extra+=(--pilot); fi
    python scripts/liquid_copper_md.py "${args[@]}" --temperature "$temp" --atoms 108 \
      --seed "$((temp*1000 + 108 + replica))" --ps 100 --out "runs/liquid/T$temp-N108-r$replica" "${extra[@]}"
  done
done
for replica in 0 1 2; do
  python scripts/liquid_copper_md.py "${args[@]}" --temperature 1520 --atoms 256 \
    --seed "$((1520000 + 256 + replica))" --ps 50 --out "runs/liquid/T1520-N256-r$replica"
done
python scripts/liquid_copper_md.py "${args[@]}" --temperature 1520 --atoms 108 \
  --seed 1520111 --ps 1000 --out runs/liquid/continuous/T1520-N108-r3
python scripts/analyze_liquid_copper.py runs/liquid/continuous --out runs/liquid/analysis/continuous --blocks
python scripts/analyze_liquid_copper.py runs/liquid --out runs/liquid/analysis
python scripts/benchmark_liquid_copper_paths.py "${args[@]}" --structure runs/liquid/T1520-N108-r0/start.extxyz \
  --output runs/liquid/analysis/matched-10ps.json
python scripts/plot_liquid_copper.py runs/liquid/analysis runs/liquid/analysis/liquid-copper.pdf
```

Each replica starts independently from an FCC crystal with Maxwell-distributed momenta and zero net momentum. The fixed volume follows the [Brillo and Egry](https://elib.dlr.de/15881/) linear liquid density in g/cm³, `7.90 − 0.000765 × (T − 1357.15)`. The model melts the cell at 2800 K for 5 ps, followed by 8 ps at the target temperature. The FCC order parameter is the average squared Bragg amplitude of three original {200} modes. The production trajectories use a 2 fs time step and [canonical stochastic velocity rescaling](https://doi.org/10.1063/1.2408420) with a 1 ps relaxation time. Unwrapped coordinates are saved every 0.05 ps. The pilot also checks 2 ps NVE at 1 and 2 fs. Six common frames per trajectory, including the first and last, are checked against the source MACE forces, stress and energy.

The analysis removes center-of-mass motion, averages MSD over all atoms and all available time origins and divides the linear 3–15 ps slope by six. Replicas supply the SD and SEM. Two alternative lag windows and fits to both trajectory halves are included in the [per-replica report](raw/liquid-cu/report.json). [MSD curves](raw/liquid-cu/msd.csv) · [Ten-ps control](raw/liquid-cu/matched-10ps.json) · [Trajectory checksums](raw/liquid-cu/trajectory-sha256.txt).
