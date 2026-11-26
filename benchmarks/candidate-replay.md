# Conservative candidate-graph replay for MACE

The input is a single fixed-cell periodic structure and the FP64 MACE-MP-0 Small checkpoint. The driver integrates velocity Verlet and returns the potential energy, Cartesian forces and cell stress on every step. It also compares common-frame observables and independently integrated trajectories against the original checkpoint. The grouped kernels and radial implementation in this benchmark specialize the two scalar-input interactions of MP-0 Small.

## Run the exact radial path

Install the CUDA dependencies, verify the public checkpoint and generate the Au/water fixture using the [input preparation commands](adjoints.md#reproduce-the-tables). From the repository root, run

```bash
PYTHONPATH=src python scripts/benchmark_grouped_md.py \
  --model checkpoints/mace-mp-0-small.model \
  --structure fixtures-interface/au-water.extxyz \
  --variants sched schedtangent --outputs edgeefs --resident-skins 0.15 \
  --macro-extension 1.0 --dt-fs 0.5 --temperature 300 --seed 17 \
  --warmup 5 --steps 32 --repeats 5 --host-observables \
  --check-candidate-edges --output runs/compare-exact-radials.json
```

The `sched` variant uses the source model's radial networks. The `schedtangent` variant runs the two trained radial prefixes together. For each radial distance it propagates the analytical Bessel and cutoff derivatives, then applies the ordinary chain rule through the normalized SiLU layers. If the two prefixes return vectors `h₀` and `h₁`, their joint reverse contribution to the distance derivative equals the sum of the inner products of each incoming adjoint with its corresponding distance tangent. No training or radial grid is needed. Both variants use identical grouped interactions, the edge virial and the neighbor-rebuild policy.

During [development timing on Au/water](raw/packed-timing.json) with a 0.15 Å skin, the scheduled trajectory took 5.421 ms/step with the source radial networks and 4.515 ms/step with the exact radial tangents (1.201×). Each timing was the median of five 32-step windows, including host delivery of all three observables. All 38 validation frames passed the independent periodic-edge check. This same-fixture comparison isolates the radial change.

## Graph validity

The GPU keeps a candidate list out to the physical cutoff plus an inner skin. A second list encloses that shell by the macro extension. At each evaluation the interaction masks the candidates at the physical cutoff. A displacement certificate triggers reconstruction before an initially omitted pair can enter the corresponding shell. Rebuilds update the pre-sorted candidate indices, periodic shifts and receiver-group offsets in graph-owned buffers. Growth beyond capacity triggers a new CUDA Graph capture. The check flag compares all directed physical pairs and their integer periodic images against a fresh MACE graph on every untimed freely moving reference frame.

Use the targeted boundary regression as follows

```bash
PYTHONPATH=src python scripts/check_candidate_boundaries.py \
  --model checkpoints/mace-mp-0-small.model \
  --structure fixtures-interface/au-water.extxyz \
  --output runs/candidate-boundaries.json
```

The stored [boundary regression](raw/candidate-boundaries.json) moves one pair across the 6 Å cutoff twice, forces an inner-shell refresh, and moves an atom by one lattice vector. The active directed edge count changes from 13,306 to 13,304 and back without a refresh. Across these configurations the largest force difference from a fresh upstream calculation is 3.33 × 10⁻¹⁵ eV/Å. Repeating the six perturbations on the locked exact tangent route (`--route schedtangent --skin 0.15`) produced the same active counts, one capture over all six configurations, and a largest force difference of 4.44 × 10⁻¹⁵ eV/Å. [Exact radial and scheduled crossing regression](raw/tangent-crossings.json)

The [initial skin sweep](raw/candidate-sweep.json) times the original two-list runner and a separately qualified interpolation option. Compare host-visible runs using a common configuration when reporting per-step performance.

Check the transformed model's own force and stress derivatives with

```bash
PYTHONPATH=src python scripts/check_interaction_derivatives.py \
  --model checkpoints/mace-mp-0-small.model \
  --structure fixtures-interface/au-water.extxyz \
  --modes tangentedge --output runs/packed-radial-fd.json
```

At a 10⁻⁴ Å displacement, the Au/water force projection differs from the central energy difference by 3.52 × 10⁻⁹ eV/Å in the random direction and by 1.70 × 10⁻⁶ eV/Å along an O–H stretch. The stretch error decreases fourfold on halving the displacement. The Cu force projection differs by 2.70 × 10⁻¹⁰ eV/Å. The largest normal or shear stress error across both structures at strain 10⁻⁵ is 1.02 × 10⁻¹⁰ eV/Å³. [Au/water finite differences](raw/tangent-fd-interface.json) · [Cu108 finite differences](raw/tangent-fd-cu108.json)

## Where the original captured replay spends time

On the 192-atom Au/water fixture, the original `replay` variant spent 2.241 ms per step on general PyTorch elementwise kernels and 1.500 ms on dense linear algebra. Together they account for 73% of the 5.132 ms of recorded device activity. Two FP64 multiplication kernel names alone totaled 1.050 ms across 70 calls per step. The SiLU forward and reverse kernels accounted for another 0.370 ms. Kernel names distinguish these operations but cannot identify which model layer issued every multiplication. [Per-step trace and kernel names](raw/candidate-cuda-interface.json)

| Recorded GPU activity | Mean ms per moving step |
|---|---:|
| General elementwise kernels | 2.241 |
| Dense linear algebra | 1.500 |
| Grouped second interaction, including 0.410 ms for the message adjoint | 0.547 |
| Grouped first interaction | 0.278 |
| Frozen scalar product | 0.192 |
| Spherical harmonics and their adjoint | 0.019 |
| All remaining activity, including device copies and sets | 0.353 |

The trace covered ten moving FP64 steps on one V100 after five warmup steps, with a 0.25 Å skin, 1.0 Å macro extension and energy, forces and stress delivered to host memory every step. Ten graph replays required two candidate-list refreshes and zero recaptures. Each step averaged 383.4 GPU kernels, one host graph launch (0.524 ms) and 40.4 individual kernel-launch calls (0.199 ms). Host API durations overlap GPU work. Device copies and sets occupied 0.044 ms per step, while the host spent 4.551 ms inside asynchronous copy calls, including waits. The traced wall time averaged 7.265 ms. Three separate ten-step blocks without instrumentation took 6.135, 5.453 and 5.314 ms per step. The trace characterizes the original replay with the checkpoint's radial networks. [Profiler and arguments](../scripts/profile_candidate_cuda.py)

```bash
PYTHONPATH=src python scripts/profile_candidate_cuda.py \
  --model checkpoints/mace-mp-0-small.model --structure fixtures-interface/au-water.extxyz \
  --warmup 5 --steps 10 --trials 3 --output runs/candidate-cuda-interface.json
```

## Outer-list sensitivity

The exact scheduled control with its original radial networks also ran for three windows of 96 steps at a fixed 0.15 Å inner skin on the development Au/water fixture. A 1.8 Å outer extension led to four charged broad-list rebuilds in each window, whereas 3.2 Å required one. Both times included 47 inner refreshes in each window. The median decreased from 5.338 to 5.051 ms/step. All 102 periodic-edge audits passed for both extensions. The primary radial comparison used the earlier fixed 1.0 Å extension. [1.8 Å run](raw/exact-macro-1.8-interface.json) · [3.2 Å run](raw/exact-macro-3.2-interface.json)
