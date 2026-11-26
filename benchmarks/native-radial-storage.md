# Native radial coefficient storage controls

An XL evaluation at 8,192 radial grid points reads four FP64 cubic coefficients per interval and output channel. The generated R0 looks up a row by the directed receiver/sender species pair. The generated R1 looks up a row by the unordered pair. For the tested checkpoint, the input Bessel embedding has no species-dependent distance transform. The first interaction has an element-dependent sender embedding, while the second applies its radial function to dynamic sender features. These structures suggest sharing equivalent coefficient rows.

The qualification probe uses the standard native forward to locate the actual R0 and R1 device arrays on the running evaluator. For every row and every cubic coefficient across the entire table it checks the emitted FP64 bits against the candidate representative. R0 uses the corresponding sender row in the first receiver block. R1 uses row zero. This check is composition-specific. All original allocations remain live. A qualifying route gathers from the representative row in forward and reverse. It preserves the grid, polynomial arithmetic, checkpoint and weights. The possible benefit is a smaller coefficient working set in cache. The regular plugin and each alternative are compiled with the same nvcc settings.

## Same-grid result

In the 192-atom Au/water development case, R0 had nine 134,201,344-byte rows covered by three validated representatives. R1 had six equally sized rows with one representative. Every coefficient matched bitwise. Referencing four rows instead of fifteen lowers the total address span of the used representative rows from 1.875 GiB to 0.500 GiB (rounded). The number of loads per interacting edge and the upstream memory allocation stay fixed.

| Canonical coefficient rows used | Median ms/MD step | Paired wins against control |
|---|---:|---:|
| Compiler-matched original indexing | 3.250 | — |
| R0 only | 3.243 | 4/6 |
| R1 only | 3.191 | 5/6 |
| R0 and R1 | 3.289 | 3/6 |

All four routes passed the common-frame MACE energy, force and stress checks and moving trajectory comparison. The R1-only saving was 1.8%. Using canonical rows in both stages increased the median by 1.2%. The algebraic redundancy in the allocated table therefore provides limited end-to-end benefit through canonical indexing alone on this input. [Bitwise table audit](raw/native-spline-row-audit-au.json) · [Raw paired step times](raw/native-spline-row-md-au.json)

A separate CUPTI test collected three two-step windows per route and counted the R0 and R1 forward and reverse kernels. The median of their summed device durations was 1.752 ms/step for original indexing, 1.737 for R0-only canonical indexing, 1.700 for R1 only and 1.755 for both. The largest saving across the four kernels was 0.051 ms/step. [Dispatch counts and kernel durations](raw/native-spline-row-trace-au.json)

## Reproduce the coefficient audit and timing

Prepare the periodic fixtures, compact model and stock M1 as described in the acceptance protocol. Each MD evaluation uses a 1 Å neighbor buffer and the same prepared driver. R0 uses the previously developed factored reverse in all routes. [Initial setup](acceptance.md#reproduction)

```bash
python -m mace_md_lab.xl_spline_alias --compact runs/compact-small.json --artifacts runs/alias
python scripts/benchmark_xl_spline_alias.py --compact runs/compact-small.json \
    --artifacts runs/alias --m1 runs/xl-compiled --model checkpoints/mace-mp-0-small.model \
    --structure fixtures-interface/au-water.extxyz --audit runs/alias-audit.json \
    --output runs/alias-md.json --steps 48 --repeats 6 --warmup 5
```

For the independent kernel timing, run the coefficient benchmark again with `--trace-only --repeats 3` and a separate output path.

## Radial grid sensitivity

Changing the number of interpolation points changes the interpolated scalar energy. This control holds the weights, static kernel policy, R0 and M1 cubins and all MD driver work fixed. It retains the original analytic derivative of each new interpolant. At every validation position it compares energy, Cartesian forces and global stress with the original FP64 MACE model. It also compares NVE changes, positions and velocities to the 8,192-point trajectory. Wall times are recorded for every grid, with a separate gate for each grid's accuracy.

| Grid points | Median ms/step | Maximum force difference vs MACE (eV/Å) | Accuracy gates |
|---:|---:|---:|---|
| 8,192 | 3.157 | 1.45 × 10⁻⁷ | Pass |
| 4,096 | 3.123 | 6.87 × 10⁻⁷ | Pass |
| 2,048 | 3.079 | 8.44 × 10⁻⁶ | Force gate failed |

The fourfold reduction in grid size improved wall time by 2.5% and exceeded the 10⁻⁶ eV/Å force tolerance. Halving the grid size reduced time by 1.1% on this trajectory. Retaining 8,192 points for all primary routes gives an identical interpolated energy throughout the main comparison. [Per-grid timing, accuracy and rebuilds](raw/native-radial-grid-au.json)

```bash
python scripts/benchmark_xl_grid_control.py --compact runs/compact-small.json \
    --artifacts runs/xl-compiled --model checkpoints/mace-mp-0-small.model \
    --structure fixtures-interface/au-water.extxyz --output runs/grid-control.json \
    --grids 8192 4096 2048 --steps 48 --repeats 6 --warmup 5
```
