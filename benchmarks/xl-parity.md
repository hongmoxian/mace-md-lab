# MACE and Symmetrix-XL on the Au/water interface

These experiments use one V100-SXM2 16 GB, FP64, MACE-MP-0 Small and a fixed cell containing 48 Au atoms and 48 waters. The moving-atom tests use the same initial momenta and a 0.5 fs velocity-Verlet step. The existing benchmark gives the checkpoint, starting structure and converted-model hashes. [Inputs and setup](adjoints.md#reproduce-the-tables)

## End-to-end time with the Symmetrix-XL force engine

Each row reports median milliseconds per complete ASE step in six interleaved trials. Both columns use Symmetrix-XL 0.2.0 compiled for SM70, the direct speed plan and 8,192 radial spline points. Both return energy, conservative forces and stress. Validation runs before timing. All configurations passed checks on matched frames and freely moving trajectories. The two runs below use different timing windows.

| Timed steps after warmup | Neighbor builder | Skin (Å) | Stock XL (ms) | XL with fixed-cell check (ms) |
|---:|---|---:|---:|---:|
| 16 after 3 | Automatic (host) | 0.12 | 8.128 | 7.580 |
| 16 after 3 | Kokkos | 0.30 | 5.982 | 5.826 |
| 48 after 5 | Kokkos | 0.30 | 6.237 | 6.083 |
| 48 after 5 | Kokkos | 0.60 | 5.848 | 5.698 |
| 48 after 5 | Kokkos | 1.00 | 5.840 | 5.682 |

For a fixed cell, let the two largest atomic displacement norms since the last list build be $d_1$ and $d_2$. Every distinct pair can change separation by at most $d_1+d_2$, and a periodic self-image retains its separation. The experimental check accepts reuse below the skin, allowing for roundoff. It uses unwrapped displacements and defers to the stock predicate when the cell or system identity changes. An image jump can trigger an extra rebuild. This is a fixed-cell implementation of an established Verlet-list criterion. [NPL report](https://resource.npl.co.uk/docs/science_technology/scientific_computing/ssfm/documents/demes017.pdf)

At skin 1.00 Å the new check reduced median time by 2.70%. Both paths rebuilt three times per 48-step trial. The maximum force-component difference across this scan was $3.05\times10^{-13}$ eV/Å. The JSON files include all trials, rebuild counts, numerical gates and boundary cases. The wheel emitted a Kokkos finalization message at process shutdown after saving the results. Both jobs exited successfully. [16-step scan](raw/xl-pair-skin-interface.json) · [48-step scan](raw/xl-pair-skin-interface-longer.json)

## Changes to the standalone force calculation

**Shared radial weights on reciprocal edges.** The two directed edges between the same atoms and opposite periodic images have equal lengths. The probe computes the distance-only radial MLP once per pair, retains the two directed messages and differentiates the resulting energy. The strict active graph has 13,306 directed edges and 6,653 reciprocal pairs at the starting frame.

The prepared-graph energy and force call took 16.26 ms using directed edges and 16.33 ms with sharing. In moving MD, the full step time rose from 19.66 to 22.96 ms when sharing was enabled. Both routes refreshed the active graph on all 12 timed steps. Initial pair-map construction took 2.98 ms. These medians come from a second run that primed both complete trajectories and used six alternating trials. The two routes agreed within 9.93 × 10⁻¹⁴ eV/Å in force over 18 moving frames. [Repeat measurements](raw/paired-radial-interface.json) · [Pilot measurements](raw/paired-radial-interface-pilot.json)

**First-layer projection after neighbor aggregation.** The input feature of each source atom in this checkpoint's first interaction depends only on its element. For each receiver and source element the probe first sums products of the radial hidden features and spherical harmonics. It then applies the learned final radial projection to each angular block and multiplies by the source-element features. This moves a linear operation across a sum. Later interaction layers have environment-dependent source features.

The 13,306 edges occupy 444 receiver/species groups. The original final projection uses an estimated 436.0 million multiply-accumulates. The alternative uses 58.2 million for the projection and 13.6 million for the new edge aggregation. On all 192 receivers the largest message difference was $2.27\times10^{-13}$. The largest difference in the coordinate derivative of a seeded linear combination of all messages was $5.46\times10^{-12}$. These are internal layer values. The grouped probe measures numerical agreement and counts operations. Its MD wall time has yet to be measured. XL already folds the learned radial projection into native spline tables, which changes the cost of testing this reordering in that engine. [Full-graph check](raw/first-layer-grouping-interface-all.json) · [Symmetrix-XL, Appendix B](https://arxiv.org/html/2610.01036v1)

## Reproduce

Use the checkpoint, converted JSON, interface fixture and environment described in the baseline setup. These commands run from the repository root. [Setup](adjoints.md#reproduce-the-tables)

```bash
small=checkpoints/mace-mp-0-small.model
interface=fixtures-interface/au-water.extxyz
compact=runs/compact-small.json
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export SYMMETRIX_JIT_CACHE="$(mktemp -d)"

python scripts/benchmark_xl_step.py --symmetrix-json "$compact" --structure "$interface" \
  --skins 0.12 0.30 --steps 16 --warmup 3 --repeats 6 --output runs/xl-16.json
python scripts/benchmark_xl_step.py --symmetrix-json "$compact" --structure "$interface" \
  --builders kokkos --skins 0.30 0.60 1.00 --steps 48 --warmup 5 --repeats 6 --output runs/xl-48.json
python scripts/benchmark_paired_radial.py --model "$small" --structure "$interface" \
  --device cuda --adjoint both --repeats 8 --md-steps 12 --md-warmup 6 --md-repeats 6 \
  --prime-md --output runs/paired-radial.json
python scripts/check_first_layer_grouping.py --model "$small" --structure "$interface" \
  --all-receivers --batch-receivers 8 --output runs/first-layer-grouping.json
```
