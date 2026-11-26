# Native affine residual folding and spherical-harmonic storage

Both comparisons keep the fixed-cell PreparedXL/ArrayVerlet driver, FP64 arithmetic, static launch policy and host-visible energy, atomic forces and stress on every step. Both evaluators use official generated R0 and M1. The Cu reference additionally uses official generated M0. Each row uses the same checkpoint, compact radial export, moving atoms, integrator, 1 Å skin, five warmup steps and alternating trial order.

## Complete MD step timings

Each comparison below comes from an independent run. Compare reference and alternative within a row. The Au/water trajectory uses 32 steps and Cu uses 24 steps in each of six trials.

| System | Change tested | Reference ms/step | Alternative ms/step | Time reduction | Faster pairs |
|---|---|---:|---:|---:|---:|
| [Au/water, 192 atoms](raw/algorithm-shift-au17.json) | Species-shifted M0 | 3.346 | 3.275 | 2.13% | 6/6 |
| [Cu, 108 atoms](raw/algorithm-shift-cu108.json) | Species-shifted M0 | 2.330 | 2.293 | 1.59% | 6/6 |
| [Au/water, 192 atoms](raw/algorithm-ylayout-au17.json) | Official Y-only policy | 3.351 | 3.128 | 6.67% | 6/6 |
| [Cu, 108 atoms](raw/algorithm-ylayout-cu108.json) | Official Y-only policy | 2.510 | 2.369 | 5.61% | 5/6 |

The largest Cartesian force-component deviations from the original XL evaluation are 2.87 × 10⁻¹³ eV/Å for the species shift and 2.79 × 10⁻¹³ eV/Å for the Y-only control. Both pass short-trajectory, common-frame energy, force and stress checks. Synchronized native traces confirm dispatch of the species-shifted M0 forward and reverse. The storage control favors pinning Y-only for the subsequent fixed-driver force-algorithm comparisons on both systems. The independent medians from the species-shift study measure the shift under retained harmonics.

## Species-constant residual

The checkpoint's first residual depends on the chemical species. Using row vectors, let its first scalar product be $m$, its learned output matrix $P$, and its per-element residual $b_z$. The pre-linear-up feature is $h=mP+b_z$. Solve $d_zP=b_z$ once for each supported element and use the shifted product $m'=m+d_z$. The second interaction receives $m'(PU)$. The readout and skip use $m'(Pr)$ and $m'(PS_z)$. With fixed species the shift has zero coordinate derivative. The generated M0 reverse therefore uses its original weight-and-coordinate formula. Both the readout and skip retain their original residual contribution.

In the benchmark compact $P$ is 128 × 128, its measured condition number is 401 and the largest component of $d_z P-b_z$ is 1.76 × 10⁻¹⁴. The builder rejects incompatible or ill-conditioned maps. The prepared artifact pairs its rewritten affine weights with the shifted generated M0. Both are required to reproduce the original potential. Their hashes and the source compact export hash are checked at load time. All subject weights come from the original export. [Implementation](../src/mace_md_lab/xl_species_shift.py)

## Native harmonic-storage control

XL's retained path computes spherical-harmonic values and gradients, normalizes and permutes their axes, then reads the gradients during the R0 and R1 reverses. XL's official `y-only-direct-v1` option stores the 16 harmonic values. R0 and R1 reconstruct the angular gradients from the current edge geometry during their coordinate reverses. In FP64 this option requires Cartesian FP64 geometry. The control asks whether its smaller intermediate arrays outweigh the extra computation. The Y-only switch is supplied by XL. [XL geometry implementation](https://github.com/bonan-group/symmetrix-xl/blob/95b1f33aaa7c8c918723ca475033ddd41939adef/libsymmetrix/source/mace_kokkos_first_interaction.cpp) · [XL policy admission](https://github.com/bonan-group/symmetrix-xl/blob/95b1f33aaa7c8c918723ca475033ddd41939adef/libsymmetrix/source/mace_kokkos_runtime.cpp)

## Lowering the retained harmonic values

For a Cartesian edge $R=r n$, let $H_{lm}(R)=r^lY_{lm}(n)$ be the regular solid harmonic in the XL real basis. Its Cartesian derivative is a degree-$(l-1)$ solid harmonic. Therefore a sparse linear map of the retained lower-degree values gives the first term below at the unit direction.

$$
\frac{\partial Y_{lm}}{\partial R_a}
=\frac{(\partial_a H_{lm})(n) - l n_aY_{lm}(n)}{r}.
$$

For example, the degree-two component at zero-based index four is $Y_4=\sqrt{15}n_xn_z$. Its homogeneous derivative in the $x$ direction is $\sqrt{5}Y_3$, so its edge derivative is $(\sqrt{5}Y_3-2n_xY_4)/r$. For $l\leq3$, every row of the lowering map has at most two coefficients. The retained degree-one vector gives $n=(Y_1,Y_2,Y_3)/\sqrt{3}$. A 16-lane subgroup evaluates one component per lane, uses shuffles to read the neighboring components, and writes the 48 Cartesian derivatives for the calling kernel. An optional variant returns the homogeneous derivative divided by $r$. Its caller must project the accumulated vector onto the plane perpendicular to the edge. These derivative checks are independent of complete MD performance. [Coefficient table and CUDA implementation](../src/mace_md_lab/xl_harmonic_lowering.py) · [Autograd, finite differences and CUDA subgroup test](../scripts/check_xl_harmonic_lowering.py)

Across 1,052 edge vectors spanning 257 random directions, six signed axes and four radii (0.2–5.999 Å), the maximum absolute Cartesian derivative difference from e3nn autograd was 1.43 × 10⁻¹⁴ Å⁻¹ for the projected derivative and 2.14 × 10⁻¹⁴ Å⁻¹ for projecting after lowering. The 16- and 32-lane CUDA tests gave the same bounds. A fourth-order finite difference at 2 × 10⁻⁵ Å agreed with autograd to 8.21 × 10⁻¹¹ Å⁻¹. [GPU and finite-difference data](raw/algorithm-harmonic-derivative.json)

An additional R1-only comparison on the development Au/water case enabled cooperative lowering inside the same warp-per-edge reverse, with identical Y-only storage on both sides. The R1 reverse took 0.335 ms with the original geometric helper and 0.373 ms with lowering. Full MD took 2.929 and 2.941 ms/step respectively (six matched 48-step trials). This variant did not accelerate the R1 reverse. [R1 experiments](native-r1.md)

## Reproduce

Use the checkpoint export and upstream compiled modules described in the acceptance benchmark. For Cu108, provide `--stock-m0-dir "$mods"` and use 24 timed steps. Generate the shifted artifact with the first run. The Y-only comparison uses standard calculators and needs no shifted artifact. [Inputs and upstream module preparation](acceptance.md#reproduction)

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
compact=runs/compact-small.json
mods=runs/xl-compiled
casefile=fixtures-interface/au-water.extxyz
python scripts/benchmark_xl_species_shift.py --compile --symmetrix-json "$compact" \
  --product-dir "$mods" --shift-dir runs/shifted-product --structure "$casefile" \
  --steps 32 --repeats 6 --output runs/algorithm-shift-au17.json
python scripts/benchmark_xl_species_shift.py --study harmonics --symmetrix-json "$compact" \
  --product-dir "$mods" --shift-dir runs/shifted-product --structure "$casefile" \
  --steps 32 --repeats 6 --output runs/algorithm-ylayout-au17.json
```
