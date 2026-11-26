# Fixed-cell MD with the native Symmetrix-XL evaluator

This path runs the native XL forward and coordinate reverse, rebuilds a periodic neighbor graph when its displacement certificate expires, and returns total energy, atom forces and global stress on the host every step. It supports one fixed set of atoms in a fully periodic, fixed cell, without constraints, fields, dispersion corrections, or variable charges. It uses private interfaces of Symmetrix-XL 0.2.0. This benchmark uses MACE-MP-0 Small, FP64, the direct speed plan, a single V100 16 GB and a 1 Å Kokkos neighbor buffer.

## What is reused

A `PreparedXL` instance owns the evaluator's current device graph. Atom types, cell, inverse cell, volume and empty host index buffers are prepared once. For positions that satisfy the existing two-largest-displacements skin bound, the evaluator gets new Cartesian coordinates and the same graph token. When a rebuild is required, XL prepares a fresh graph at the model cutoff plus skin. Rebuilds apply XL's launch-policy selection and refresh the graph if the new policy invalidates it. After each evaluation the implementation sums the native atom energies and calls XL's native atomic force and stress reductions. The optional host velocity-Verlet integrator advances the atoms using those returned forces. [Implementation](../src/mace_md_lab/xl_fixed_step.py)

| Route | Force evaluation | Integrator |
|---|---|---|
| `ase` | Standard XL calculator | ASE |
| `certificate` | Standard XL calculator with fixed-cell displacement check | ASE |
| `ase_prepared` | Prepared native geometry and reductions | ASE |
| `arrays_stock` | Standard XL calculator | Fixed-cell array Verlet |
| `arrays_prepared` | Prepared native geometry and reductions | Fixed-cell array Verlet |

## Timings

The interface contains 48 Au atoms and 48 waters. Each of six alternating trials measured 48 freely moving steps after five warmup steps. Cu108 used four trials of 24 steps. Both the independent original MACE calculator and the original XL calculator were checked on identical frames. All five routes passed. The host received stress, forces and energy on every step. Times are median ms per full step. Each column is a separate interleaved experiment using the indicated XL launch policy. [Au/water, automatic](raw/xl-faststep-interface.json) · [Au/water, static](raw/xl-faststep-static-interface.json) · [Cu108, automatic](raw/xl-faststep-cu108.json)

| Route | Au/water, automatic | Au/water, static | Cu108, automatic |
|---|---:|---:|---:|
| ASE with the standard XL calculator | 6.131 | 5.547 | 4.416 |
| ASE with the fixed-cell displacement certificate | 5.947 | 5.451 | 4.177 |
| ASE with prepared native geometry | 5.161 | 5.132 | 3.565 |
| Array Verlet with the standard XL calculator | 5.613 | 5.118 | 4.056 |
| Array Verlet with prepared native geometry | **4.580** | **4.579** | **3.002** |

On Au/water, all six prepared-Verlet trials rebuilt the neighbor list three times. The largest same-coordinate differences from stock XL were 1.66 × 10⁻¹³ eV/Å in force and 6.94 × 10⁻¹⁷ eV/Å³ in stress. Relative to the original MACE checkpoint, the corresponding differences were 8.49 × 10⁻⁸ eV/Å and 8.89 × 10⁻¹⁰ eV/Å³. The latter comparison includes XL's converted radial model.

## Combining the driver with XL's generated product kernel

XL can generate a specialized kernel for the second symmetric product, M1. The table uses XL's static launch policy throughout and compares the same native kernel with two host drivers. The Horner rows test a separate rearrangement of the product derivative. Trial counts, integrator parameters and observables match the preceding experiments. [Au/water](raw/xl-faststep-products-interface.json) · [Cu108](raw/xl-faststep-products-cu108.json)

| Host driver | Native M1 kernel | Au/water, ms/step | Cu108, ms/step |
|---|---|---:|---:|
| ASE + XL calculator | Built-in | 5.504 | 3.825 |
| ASE + XL calculator | XL generated | **4.728** | **3.752** |
| ASE + XL calculator | Horner | 4.712 | 3.742 |
| Prepared + array Verlet | Built-in | 4.564 | 2.967 |
| Prepared + array Verlet | XL generated | **3.863** | **2.828** |
| Prepared + array Verlet | Horner | 3.760 | 2.885 |

With the same XL-generated M1 kernel, the prepared driver reduced median wall time by **18.3% on Au/water** and **24.6% on Cu108**. On Au/water, all six prepared trials (3.762–3.893 ms/step) were faster than every ASE trial (4.655–4.797 ms/step). The Au/water trajectories agreed with stock XL to 1.43 × 10⁻¹⁴ Å in final position and 1.19 × 10⁻¹⁵ eV/atom in total energy change. Each Au/water trial triggered three graph rebuilds. Cu108 kept the initial graph for all 24 measured steps. These comparisons measure the Python/ASE driver boundary. A fully native MD integrator would have different host costs.

The Horner rearrangement had mixed end-to-end timings across the two systems. On Au/water its native M1 forward and reverse kernels took 13.62 and 22.11 µs, respectively, compared with 11.50 and 18.40 µs for XL's generated kernels. The host-driver speedups above use XL's original generated kernel on both sides. [Native kernel profile](raw/xl-m1-horner-interface.json)

## Factoring the R0 coordinate derivative

The first radial interaction uses one spline value $w_{lc}(r)$ for every magnetic component $m$ at a given degree $l$ and channel $c$. For one directed edge, let $a_{lmc}$ denote the incoming node adjoint and $Y_{lm}$ the real spherical harmonic. For each $l,c$, the new reverse kernel contracts the adjoint with the angular values and their Cartesian derivatives first. It then applies the spline factors once per contraction. The resulting derivative has a scalar radial part along the edge unit vector and a Cartesian angular part.

$$
\left(\nabla_{\mathbf r} E\right)_{\rm R0,edge}
= q\left[
\hat{\mathbf r}\sum_{lc} w'_{lc}(r)\sum_m a_{lmc}Y_{lm}
+ \sum_{lc} w_{lc}(r)\sum_m a_{lmc}\nabla_{\mathbf r}Y_{lm}
\right],
$$

where $q$ is the edge weight. The surrounding XL kernel supplies the spline derivative, handles any separate density derivative and applies the directed-force sign. Both variants use the same spline tables, radial forward kernel, edge schedule and M1 kernel. [CUDA code generator](../src/mace_md_lab/xl_product.py)

On Au/water the generated R0 reverse took 432.45 µs per step with XL's source and 306.85 µs after factoring (29.0% less GPU time in the two-step activity trace). The corresponding R0 forward times were 402.49 and 400.22 µs. The median ASE wall time with automatic launch tuning fell from 4.997 to 4.755 ms/step over six 48-step trials. The largest same-coordinate difference from native XL was $4.00\times10^{-15}$ eV/Å in force and $5.21\times10^{-18}$ eV/Å³ in stress. [Trace, numerical checks and raw trial times](raw/xl-r0-factored-interface.json)

With a fixed launch policy, the five routes below ran on the same hardware, starting coordinates, momenta and timestep. Each trial returned energy, forces and stress on every step. CUPTI recorded both passes through the generated R0 and M1 modules on every two-step trace of the four R0 routes. Times are median ms per step. [Au/water raw trials](raw/xl-prepared-r0-interface.json) · [Cu108 raw trials](raw/xl-prepared-r0-cu108.json)

| Host driver | Native modules | Au/water | Cu108 |
|---|---|---:|---:|
| ASE | Generated M1, built-in R0 | 4.740 | 3.719 |
| ASE | Generated M1 and R0 | 4.394 | 3.476 |
| ASE | Generated M1, factored R0 | 4.324 | 3.387 |
| Prepared + array Verlet | Generated M1 and R0 | 3.380 | 2.562 |
| Prepared + array Verlet | Generated M1, factored R0 | **3.273** | **2.398** |

Relative to ASE with the same generated M1 and R0, reusing the prepared geometry and replacing R0 reduced full-step time by 25.5% on Au/water and 31.0% on Cu108. Within the prepared driver, replacing R0 alone reduced the medians by 3.2% and 6.4%, respectively. In the Au/water prepared-driver traces, the R0 reverse fell from 430.83 to 304.70 µs/step.

The choice of M0 changed which stock configuration performed best. On Au/water, the static ASE route with built-in M0 and generated M1 and R0 took 4.364 ms/step. Enabling upstream generated M0 raised the median to 4.616 ms/step. On Cu108 it lowered the median from 3.578 to 3.486 ms/step. The generated M0 forward and reverse each ran twice in both two-step device traces. The five-route table above holds M0 constant at its built-in setting. [Au/water M0 sensitivity](raw/xl-r0-m0-interface.json) · [Cu108 M0 sensitivity](raw/xl-r0-m0-cu108.json)

## Reproduce

Use the model and fixture preparation instructions in the original benchmark. The XL artifact directory, when supplied, must have been built for the same compact model and GPU. [Fixture and compact-model setup](adjoints.md#reproduce-the-tables) · [Benchmark driver](../scripts/benchmark_xl_fixed_step.py)

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export SYMMETRIX_JIT_CACHE="$(mktemp -d)"
common=(--model checkpoints/mace-mp-0-small.model --symmetrix-json runs/compact-small.json)
python scripts/benchmark_xl_fixed_step.py "${common[@]}" \
  --structure fixtures-interface/au-water.extxyz --steps 48 --repeats 6 \
  --output runs/xl-faststep-interface.json
python scripts/benchmark_xl_fixed_step.py "${common[@]}" \
  --structure fixtures108/cu.extxyz --steps 24 --repeats 4 \
  --output runs/xl-faststep-cu108.json
python scripts/benchmark_xl_fixed_step.py "${common[@]}" \
  --structure fixtures-interface/au-water.extxyz --kernel-launch-policy static \
  --steps 48 --repeats 6 --output runs/xl-faststep-static-interface.json

python -m mace_md_lab.xl_product --symmetrix-json runs/compact-small.json \
  --artifact-dir runs/mdlab-product
products=(--kernel-launch-policy static --product-dir runs/mdlab-product
          --routes ase ase_upstream ase_product arrays_prepared arrays_upstream arrays_product)
python scripts/benchmark_xl_fixed_step.py "${common[@]}" "${products[@]}" \
  --structure fixtures-interface/au-water.extxyz --steps 48 --repeats 6 \
  --output runs/xl-faststep-products-interface.json
python scripts/benchmark_xl_fixed_step.py "${common[@]}" "${products[@]}" \
  --structure fixtures108/cu.extxyz --steps 24 --repeats 4 \
  --output runs/xl-faststep-products-cu108.json

python -m mace_md_lab.xl_product --stage r0 --symmetrix-json runs/compact-small.json \
  --artifact-dir runs/mdlab-product
r0=(--kernel-launch-policy static --product-dir runs/mdlab-product --routes
    ase_upstream ase_upstream_r0 ase_factored_r0 arrays_upstream_r0 arrays_factored_r0)
python scripts/benchmark_xl_fixed_step.py "${common[@]}" "${r0[@]}" \
  --structure fixtures-interface/au-water.extxyz --steps 48 --repeats 6 \
  --output runs/xl-prepared-r0-interface.json
python scripts/benchmark_xl_fixed_step.py "${common[@]}" "${r0[@]}" \
  --structure fixtures108/cu.extxyz --steps 24 --repeats 4 \
  --output runs/xl-prepared-r0-cu108.json

python -m mace_md_lab.xl_product --stage m0 --symmetrix-json runs/compact-small.json \
  --artifact-dir runs/mdlab-product
m0=(--stage m0 --kernel-launch-policy static --artifact-dir runs/mdlab-product
    --symmetrix-json runs/compact-small.json)
python scripts/benchmark_xl_product.py "${m0[@]}" \
  --structure fixtures-interface/au-water.extxyz --steps 48 --repeats 6 \
  --output runs/xl-r0-m0-interface.json
python scripts/benchmark_xl_product.py "${m0[@]}" \
  --structure fixtures108/cu.extxyz --steps 24 --repeats 4 \
  --output runs/xl-r0-m0-cu108.json
```

To use the default native message kernels in a standalone fixed-cell simulation, construct one dedicated XL calculator in the same configuration and use it only via the prepared path.

```python
from mace_md_lab.xl_fixed_step import ArrayVerlet, PreparedXL

prepared = PreparedXL(atoms, engine, skin=1.0)
md = ArrayVerlet(atoms, prepared.evaluate, dt_fs=0.5)
md.advance(48)
positions, velocity_A_fs, energy_eV, forces_eV_A, kinetic_eV, stress_voigt = md.snapshot()
```
