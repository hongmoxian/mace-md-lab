# Full-step conservative MD acceptance

A subsequent same-driver study isolates the speed contributed by the native R0 and R1 force algorithms, with the official on-demand angular derivative enabled on both sides. [Native algorithm comparison](native-algorithms.md)

The independent MACE path evaluates the frozen MACE-MP-0 Small checkpoint directly. Its primary timing criterion is at most 1.10 times the median step time of the stock Symmetrix-XL Python/ASE driver on the same GPU and trajectory. The native combination uses the same XL export, our prepared fixed-cell driver and our factored first-layer coordinate reverse. Its criterion is at most 0.90 times stock. Both conditions apply to synchronized full velocity-Verlet steps, including integration, periodic neighbor maintenance and host-visible energy, forces and global stress at every step. Setup, model export, offline compilation and graph capture at initialization are excluded. A separate cubic-interpolation route is reported against its own energy derivative and the source model.

## Locked periodic results

Median milliseconds per complete step. The independent/stock ratio passes at 1.10 or below. Combined time reduction passes at 10% or above. The prepared-only column retains the upstream R0 and isolates the driver contribution.

| Case | Steps × repeats | Stock XL | Independent exact | Independent/stock | Prepared only | Prepared + factored R0 | Combined time reduction |
|---|---:|---:|---:|---:|---:|---:|---:|
| [Au/water, holdout seed 73](raw/locked-r0-au73.json) | 48 × 6 | 4.364 | 4.530 | 1.038 | 3.409 | 3.330 | 23.7% |
| [Au/water, development seed 17](raw/locked-r0-au17.json) | 48 × 6 | 4.502 | 4.517 | 1.003 | 3.449 | 3.308 | 26.5% |
| [Cu, 108 atoms](raw/locked-m0-cu108.json) | 24 × 6 | 3.320 | 2.889 | 0.870 | 2.505 | 2.394 | 27.9% |

All six paired trials met both criteria in each case (18 of 18 for each criterion). Across the three systems, the independent route differed from upstream MACE by at most 5.33 × 10⁻¹⁵ eV/Å in a Cartesian force component. The native combination, using the XL export, differed by at most 8.49 × 10⁻⁸ eV/Å. The largest differences in the change of NVE total energy were 1.06 × 10⁻¹⁵ and 5.78 × 10⁻¹¹ eV/atom respectively. On the held-out Au/water structure, the optional independent spline route took 4.104 ms/step, with a maximum force difference of 5.78 × 10⁻⁸ eV/Å relative to the original MACE model.

## Protocol

All periodic cases use FP64, MACE 0.3.16, Symmetrix-XL 0.2.0 and a Tesla V100-SXM2-16GB. A 0.5 fs time step and five warmup steps precede each timed window. Route order rotates and reverses on alternate repeats. For each route the upstream MACE calculator provides common-frame energy, Cartesian force and stress references. All propagated frames are checked for position, velocity and the change in NVE total energy relative to step zero. Physical edges and periodic images of the independent paths are checked against a fresh upstream graph at every validation frame. Each JSON records the raw per-repeat timings, route order, numerical errors, rebuild counts, checkpoint hash, fixture hash and native kernel identifiers.

At each point force and stress tolerances are 10⁻⁶ eV/Å and 10⁻⁷ eV/Å³, the energy and NVE-difference tolerance is 10⁻⁸ eV/atom, and position/velocity tolerances are 10⁻⁶ Å and 10⁻⁷ Å/fs. An independent central-difference test perturbs positions and strains the cell. At displacements of 10⁻⁴ Å the largest Au/water projected-force error is 1.70 × 10⁻⁶ eV/Å (O–H stretch), decreasing fourfold from the 2 × 10⁻⁴ Å result. At strain 10⁻⁵, the largest stress error across the two tested cells is 1.02 × 10⁻¹⁰ eV/Å³. [Au/water finite differences](raw/tangent-fd-interface.json) · [Cu finite differences](raw/tangent-fd-cu108.json)

The development structure is an idealized periodic Au(111) slab with 48 Au atoms and 48 water molecules (192 atoms, generator seed 17). Qualifying the independent analytic radial tangent on this case fixed the solver and its inner/outer neighbor extensions at 0.15/1.0 Å. The independent test uses seed 73 for a new slab displacement, water orientations and initial velocities. A 108-atom periodic copper cell tests transfer at smaller system size. XL uses a 1 Å neighbor buffer in all three cells. These are latency and arithmetic-fidelity tests on short fixed-cell trajectories.

Before final scoring, separate within-run controls compared stock M0 with its upstream generated alternative while R0 and M1 were fixed. The generated M0 increased Au/water latency from 4.364 to 4.616 ms/step and reduced Cu latency from 3.578 to 3.486 ms/step. Accordingly, the official generated R0/M1 and the static launch policy are active for every primary comparison, and the Cu case also enables generated M0 in all native routes. R1 keeps its native factorized implementation. Stock XL uses an 8,192-point radial export. The independent analytic MACE route evaluates the original neural radial functions. [Au/water M0 control](raw/xl-r0-m0-interface.json) · [Cu M0 control](raw/xl-r0-m0-cu108.json) · [Cu without generated M0](raw/locked-r0-cu108.json)

## Implementations

1. **Independent exact MACE.** Both frozen radial MLPs take the same scalar edge length. The implementation propagates the feature values and their exact derivatives together through each Bessel, matrix and SiLU operation. In reverse it contracts both output adjoints with the stored derivatives, giving one radial gradient per edge. The original learned weights, activation functions and radial dependence enter each evaluation. A fixed-capacity sorted edge graph replays the force calculation across neighbor changes. A two-level displacement certificate triggers refreshes, and a strict physical cutoff zeros all buffered, inactive and padding edges. [Radial tangent](../src/mace_md_lab/exact_packed_radial.py) · [Graph replay](../src/mace_md_lab/candidate_replay.py) · [Scheduling](../src/mace_md_lab/candidate_schedule.py)
2. **Native XL combination.** The prepared driver keeps a device neighbor graph and fixed-cell metadata across steps. In the native R0 reverse, channel contributions to the radial derivative are summed before projection on the bond direction. Angular derivatives are accumulated separately. These terms result from the product rule for a radial coefficient times a spherical harmonic. Stock and candidate share the same M1, R1, weights and table export. The custom R0 forward uses the upstream formula. CUPTI separately verifies dispatch of the compiled functions. [Prepared driver](prepared-xl.md) · [Native R0 trace](raw/xl-r0-factored-interface.json)
3. **Portable conservative radial derivatives.** On trained SchNet and PaiNN checkpoints, the adapters compute the Gaussian basis and cosine cutoff and contract their analytic derivatives with the learned filter adjoints. Each model keeps its original scalar energy. The tests use matched molecular MD drivers, 9- and 72-atom inputs and eight timing repeats. [Molecular results and code](portable-radial-filters.md)

## Reproduction

Use the documented installation and model export for the CUDA 12 / SM70 wheel. Build the upstream generated modules for the compact export with nvcc; the same commands also build the factored R0 and the separate M1 Horner ablation. [Environment, checkpoint and fixtures](adjoints.md#reproduce-the-tables) · [XL configuration](prepared-xl.md)

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export SYMMETRIX_JIT_CACHE="$(mktemp -d)"
model=checkpoints/mace-mp-0-small.model
compact=runs/compact-small.json
mods=runs/xl-compiled
python -m mace_md_lab.xl_product --symmetrix-json "$compact" --artifact-dir "$mods" --stage m0
python -m mace_md_lab.xl_product --symmetrix-json "$compact" --artifact-dir "$mods" --stage m1
python -m mace_md_lab.xl_product --symmetrix-json "$compact" --artifact-dir "$mods" --stage r0
python examples/make_interface.py --seed 73 --output fixtures-interface/au-water-seed73.extxyz
run_case() {
    python scripts/benchmark_grouped_md.py --model "$model" --symmetrix-json "$compact" \
      --xl-stock-product-dir "$mods" --xl-stock-r0-dir "$mods" --xl-factored-r0-dir "$mods" \
      --xl-launch-policy static --include-prepared-xl --check-candidate-edges --host-observables \
      --resident-skins 0.15 --macro-extension 1.0 --xl-skin 1 --outputs edgeefs \
      --variants schedtangent schedtable --warmup 5 --structure "$1" --steps "$2" --repeats "$3" --output "$4" "${@:5}"
    python scripts/score_acceptance.py "$4" --exact schedtangent_edgeefs \
      --combined xl_prepared_factored_efs --interpolated schedtable_edgeefs
}
run_case fixtures-interface/au-water.extxyz        48 6 runs/locked-r0-au17.json
run_case fixtures-interface/au-water-seed73.extxyz 48 6 runs/locked-r0-au73.json
run_case fixtures108/cu.extxyz                   24 6 runs/locked-m0-cu108.json --xl-stock-m0-dir "$mods"
```

## Additional controls

The inner graph was explicitly tested across removals, additions and periodic reimaging. A 0.0003 Å excursion through the cutoff changed the directed physical-edge count from 13,306 to 13,304 and back. With the final analytic tangent, all six perturbations shared one CUDA Graph. Its maximum force and stress deviations were 4.44 × 10⁻¹⁵ eV/Å and 5.55 × 10⁻¹⁷ eV/Å³. The optional Hermite radial approximation uses 8,192 knots and differentiates the same interpolant in reverse. [Final tangent boundary tests](raw/tangent-crossings.json) · [Earlier replay boundary tests](raw/candidate-boundaries.json) · [Radial table qualification](raw/radial-table-qualification.json) · [Table finite differences](raw/radial-table-fd-interface.json)

The reproducibility checks also executed all 22 repository tests in the GPU benchmark environment with the official PaiNN validation checkpoint available.

Earlier interleaved comparisons used the upstream generated M1 and the standard R0. Their results show the effect of enabling graph replay, interpolation and the prepared driver before enabling the faster generated first-layer R0. [Au/water](raw/acceptance-v1-interface.json) · [Cu](raw/acceptance-v1-cu108.json) · [M1 Horner control](raw/xl-m1-horner-interface.json)

A CUDA activity trace identifies the operations that motivated the exact radial tangent, and a separate channel-layout experiment measures a smaller scalar-product optimization. [Captured force trace](candidate-replay.md#where-the-original-captured-replay-spends-time) · [Scalar product ablation](product-tiles.md)
