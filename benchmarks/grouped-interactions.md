# Grouped interactions in moving molecular dynamics

The subsequent independent and native-combination tests use a stronger stock XL configuration and return all three observables to the host every step. [Full-step acceptance results](acceptance.md)

**Final matched comparison.** All five routes below compute energy, forces and configurational stress on every moving velocity-Verlet step. Both resident MACE and Symmetrix-XL use a 1.0 Å neighbor buffer. The interface has 192 atoms and Cu has 108. Each column is a single interleaved experiment on one V100 with FP64 precision. Values are median ms per step.

| Force and stress implementation | Au/water, 48 steps × 6 | Cu108, 24 steps × 4 |
|---|---:|---:|
| Previous resident implementation | 19.26 | 14.48 |
| Previous implementation with a single edge derivative for forces and stress | 16.35 | 12.42 |
| Both optimized interactions, conventional stress derivative | 11.17 | 9.51 |
| Both optimized interactions, edge derivative for forces and stress | **8.45** | **7.63** |
| Tuned stock XL | **6.15** | **4.45** |

On Au/water the full implementation cuts resident time by 56.1% (2.28× speedup). Its six trial times range from 8.43 to 8.48 ms/step. XL is 1.38× faster in the same comparison. On Cu the resident speedup is 1.90× and XL is 1.72× faster. Matching the original MACE calculator at common coordinates gives maximum Au/water force and stress differences of 4.06 × 10⁻¹⁵ eV/Å and 3.13 × 10⁻¹⁷ eV/Å³. [Au/water matched timing and validation](raw/interaction-factorial-interface.json) · [Cu matched timing and validation](raw/interaction-factorial-cu108.json)

The edge derivative schedule is restricted to the tested checkpoint. Its energy depends on positions and the periodic cell through the directed Cartesian edge vectors. One derivative of the energy with respect to these vectors supplies both the endpoint forces and the conventional symmetric virial divided by volume. The stress calculation includes lattice-image shifts. A separate run compared 8.50 ms for energy and forces alone with 8.49 ms for all three outputs using the edge derivative. Centered differences of the modified model's energies checked O–H stretch forces and normal and shear stresses under cell deformation. [Implementation](../src/mace_md_lab/edge_virial.py) · [Force and strain checks](raw/edge-virial-fd-interface.json) · [Matched output costs](raw/edge-virial-md-interface.json)

These V100-SXM2 16 GB tests run frozen, pretrained models in FP64. The primary MACE case contains 48 Au atoms and 48 waters. Both periodic fixtures use a 0.5 fs velocity-Verlet step, five warmup steps and synchronized wall-clock timing. The Au/water trials each cover 48 steps and are repeated six times. The Cu108 trials each cover 24 steps and are repeated four times. Every route was checked on freely propagating trajectories and identical reference coordinates before timing. [Au/water results and trials](raw/grouped-md-interface.json) · [Cu108 results and trials](raw/grouped-md-cu108.json)

## MACE-MP-0 Small

The baseline already uses the angular derivative and frozen product optimizations. The first interaction groups edges by receiver and source element, sums radial hidden features times spherical harmonics, and applies the final radial projection after summation. The implementation allocates 576 group slots for Au/water, including the 444 occupied on the initial geometry. Folding additionally incorporates the frozen receiver linear map and neighbor normalization into the projected weights. The second interaction forms and sums directed scalar–harmonic tensor products inside a GPU kernel. Its backward recomputes the scalar factors and returns gradients for the sender, harmonic and radial inputs. Both kernels read the current edge list. [First interaction](../src/mace_md_lab/grouped_first.py) · [Second interaction](../src/mace_md_lab/grouped_second.py)

| Route | Au/water E/F | Au/water E/F/stress | Cu108 E/F | Cu108 E/F/stress |
|---|---:|---:|---:|---:|
| Resident baseline | 19.42 | 22.29 | 12.67 | 14.67 |
| Group first interaction | 15.72 | 18.60 | 10.28 | 12.31 |
| Group first and fold receiver linear map | 14.92 | 17.80 | 9.57 | 11.59 |
| Fuse second interaction | 15.62 | 18.51 | 10.51 | 12.54 |
| Group first and fuse second | 12.88 | 15.58 | 9.17 | 11.02 |
| Group first, fold receiver map and fuse second | **11.51** | **14.21** | **7.85** | **9.71** |
| Tuned Symmetrix-XL 0.2.0 | — | **6.10** | — | **4.35** |

Values are median ms per complete MD step. The E/F/stress columns compare the same requested outputs. The MACE resident routes keep state and outputs on the device inside the timing loop. The XL route returns results to ASE. The original ablation uses a 0.12 Å resident skin and a 1.0 Å XL skin. XL uses Kokkos neighbor construction, the direct speed profile and 8,192 radial spline knots. Its entries measure stock XL on the same trajectories. [Earlier native neighbor-list experiment](xl-parity.md)

For the combined MACE route on Au/water, the largest same-coordinate force and stress differences from the original checkpoint were 3.45 × 10⁻¹⁵ eV/Å and 2.95 × 10⁻¹⁷ eV/Å³. The largest difference in the change in total NVE energy was 5.93 × 10⁻¹⁶ eV/atom. At a displacement of 0.0001 Å, centered differences of the modified model's own energy reproduced its force projection to 3.53 × 10⁻⁹ eV/Å along a random direction and 1.71 × 10⁻⁶ eV/Å along an O–H stretch. Diagonal and shear strain differences also passed. The longest measurement window was 24 fs. [Force and stress derivatives](raw/interaction-fd-interface.json)

On six moving Au/water frames at a 0.12 Å skin, the median first-interaction forward/reverse spans fell from 2.37/4.25 ms to 0.91/0.47 ms with both optimizations and folding. The second-interaction spans fell from 2.29/5.26 to 1.23/1.97 ms. Without layer timing hooks the full energy-to-force reverse pass fell from 9.81 to 3.77 ms. These are elapsed GPU stream intervals including launch gaps. [Au/water baseline](raw/layer-ranges-interface.json) · [Optimized Au/water profile](raw/layer-ranges-bothfolded-interface.json) · [Cu108 baseline](raw/layer-ranges-cu108.json)

## Tuning the neighbor buffer and measuring the remaining gap

A separate six-trial Au/water test computes stress, energy and forces on every step. With resident skins of 0.12, 0.60, and 1.00 Å, the two-layer optimized time was 14.23, 11.64, and 11.22 ms/step. The candidate list rebuilt 24, 6, and 3 times per 48 timed steps. At 1.00 Å, the resident baseline took 19.32 ms/step and stock XL took 6.20 ms/step. Both resident implementations use the same skin in this comparison. [Skin ablation](raw/grouped-md-skin-interface.json)

A native XL CUDA activity trace on 12 moving steps measured 0.354 ms/step in the R0 forward kernel and 0.836 ms/step in its coordinate reverse. The R1 forward and fused reverse kernels took 0.273 and 0.670 ms/step. The second product's reverse kernel took 0.895 ms/step. The activity trace raised wall time from 6.124 to 7.208 ms/step. The union of recorded GPU activity occupied 4.293 ms/step during profiling. Overlap and CPU work limit the translation of individual kernel times to wall-clock savings. [Native activity and raw symbols](raw/xl-cuda-activity-interface.json)

The native compact-model diagnostic reconstructed all 512 R0 channels for the Au–O pair from 64 hidden channels on 8,192 spline knots. Maximum knot-value and slope differences were zero and 1.82 × 10⁻⁹ in radial-network units. Integrating receiver/species grouping with native XL requires exposing the hidden table and final projection to its R0 forward and coordinate reverse kernels. The full-step XL entries above use its original message kernels. [Native factorization report](raw/xl-r0-factorization-interface.json)

## First-layer grouping in SchNet and PaiNN

Both molecular adapters group the first-layer input by receiver and source element. The SchNet adapter moves the last affine filter projection past the sum and includes the cutoff and bias. The PaiNN adapter handles the scalar and directional messages separately and uses the zero initial vector feature of the supported checkpoint. Each comparison enables the same CUDA Graph and coordinate derivative on both sides. Timings use a 0.25 fs step, five warmup steps and six alternating trials of 16 steps. The larger case contains eight isolated copies of ethanol. Both sizes have three elements. [Molecular benchmark and adapters](../scripts/benchmark_grouped_molecules.py)

| Model | Atoms | Baseline ms/step | Grouped ms/step | Change in step time |
|---|---:|---:|---:|---:|
| PyG SchNet | 9 | 1.105 | 1.136 | +2.78% |
| PyG SchNet | 72 | 1.700 | 1.707 | +0.45% |
| SchNetPack PaiNN | 9 | 1.142 | 1.156 | +1.17% |
| SchNetPack PaiNN | 72 | 1.701 | 1.643 | −3.38% |

All four grouped paths matched their original model's energy, forces and short trajectories within the recorded gates. The measurable speed benefit depends on both architecture and size. [SchNet, 9 atoms](raw/grouped-schnet-ethanol.json) · [SchNet, 72 atoms](raw/grouped-schnet-ethanol8.json) · [PaiNN, 9 atoms](raw/grouped-painn-ethanol.json) · [PaiNN, 72 atoms](raw/grouped-painn-ethanol8.json)

## Reproduce

Use the environment, exported XL model, MACE checkpoint and fixture generation commands in the periodic setup. The molecular setup supplies the two trained checkpoints and the ethanol inputs. [Periodic setup](adjoints.md#reproduce-the-tables) · [Molecular setup](portable.md#reproduce-a-row)

```bash
small=checkpoints/mace-mp-0-small.model
interface=fixtures-interface/au-water.extxyz
compact=runs/compact-small.json
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export SYMMETRIX_JIT_CACHE="$(mktemp -d)"

python scripts/benchmark_grouped_md.py --model "$small" --symmetrix-json "$compact" \
  --structure "$interface" --variants base bothfolded --outputs efs edgeefs --resident-skins 1.0 \
  --xl-skin 1.0 --warmup 5 --steps 48 --repeats 6 --output runs/final-interface.json
python scripts/benchmark_grouped_md.py --model "$small" --symmetrix-json "$compact" \
  --structure fixtures108/cu.extxyz --variants base bothfolded --outputs efs edgeefs \
  --resident-skins 1.0 --xl-skin 1.0 --warmup 5 --steps 24 --repeats 4 --output runs/final-cu108.json
python scripts/benchmark_grouped_md.py --model "$small" --symmetrix-json "$compact" \
  --structure "$interface" --variants base first folded second both bothfolded \
  --outputs ef efs --warmup 5 --steps 48 --repeats 6 --output runs/grouped-interface.json
python scripts/benchmark_grouped_md.py --model "$small" --symmetrix-json "$compact" \
  --structure fixtures108/cu.extxyz --variants base first folded second both bothfolded \
  --outputs ef efs --warmup 5 --steps 24 --repeats 4 --output runs/grouped-cu108.json
python scripts/profile_moving_interactions.py --model "$small" --structure "$interface" \
  --output runs/profile-interface.json
python scripts/profile_moving_interactions.py --model "$small" --structure "$interface" \
  --variant bothfolded --output runs/profile-interface-grouped.json
python scripts/benchmark_grouped_md.py --model "$small" --symmetrix-json "$compact" \
  --structure "$interface" --variants base bothfolded --outputs efs --resident-skins 0.12 0.6 1.0 \
  --warmup 5 --steps 48 --repeats 6 --output runs/grouped-skins.json
python scripts/profile_xl_cuda.py --symmetrix-json "$compact" --structure "$interface" \
  --output runs/xl-activity.json
python scripts/check_xl_radial_factorization.py --symmetrix-json "$compact" \
  --output runs/xl-r0-gate.json
python scripts/check_interaction_derivatives.py --model "$small" --structure "$interface" \
  --modes first second bothfolded bothfoldededge --output runs/grouped-derivatives.json
python scripts/benchmark_grouped_molecules.py --framework painn --model checkpoints/painn-ethanol.model \
  --structure fixtures-molecules/ethanol8.extxyz --warmup 5 --steps 16 --repeats 6 \
  --dt-fs 0.25 --device cuda --output runs/grouped-painn-ethanol8.json
```
