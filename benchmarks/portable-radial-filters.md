# Frozen radial filters in SchNet and PaiNN

This comparison runs two trained molecular potentials in FP64 on a V100-SXM2 16 GB. Every path uses the same coordinate differentiation, CUDA Graph force-call replay, velocity-Verlet integrator, neighbor membership check and 0.25 fs step. Each case begins from the same seeded velocities. The 72-atom input consists of eight ethanol molecules separated by more than the cutoff. It has 576 directed edges. The 9-atom input has 72 directed edges. These are short execution and numerical fidelity tests. [Checkpoints, fixtures and previous implementation](portable.md)

## Schedules

- **Stock** runs the original learned interaction layers through the existing device-resident molecular adapter.
- **Direct control** evaluates each original learned filter with the original basis and cutoff. It uses the same explicit node-message path as the reordered schedules, which separates changes to message execution from changes inside the filter.
- **Exact batched filters** collect the SchNet layers' independent radial MLPs into a shared first projection and a batch of last-layer matrix products. The fixed last affine map is rearranged using the identity $c(r)[Wh(r)+b]=[W,b] [c(r)h(r),c(r)]^T$. All SchNet nonlinear hidden activations remain upstream of the last projection. PaiNN uses its single existing affine filter over the cutoff-weighted augmented Gaussian basis.
- **Exact fused Gaussian derivatives** evaluate the frozen Gaussian basis, cosine cutoff and their distance derivatives in a GPU kernel. For PaiNN the kernel supplies cutoff-weighted features including the cutoff contribution for the affine bias. For SchNet it preserves the original ordering across the nonlinear hidden activation and applies the cutoff before the last affine map. In reverse, a per-edge sum contracts incoming feature adjoints with the exact analytic derivatives. Both architectures retain their original learned energy.
- **Hermite filters** tabulate the learned radial filter of every interaction, including its cutoff, on a uniform distance grid of 4,097 knots from zero to the cutoff. Values and slopes at knots are evaluated by the exact weight program. Each interval stores four FP64 coefficients per channel. The GPU computes one cubic per channel and contracts the analytic derivative of the same cubic with all incoming adjoints for that edge. The resulting forces are derivatives of the interpolated energy. The 768 SchNet channels use 96 MiB and the 1,152 PaiNN channels use 144 MiB. Setup plus interior-grid validation takes about 0.33 s for SchNet and 0.48 s for PaiNN. Setup and capture are excluded from the step timings.

Every column uses 5 warmup steps, 16 timed steps and 8 interleaved repeats. All values below are median ms per full MD step.

| Model | Atoms | Stock | Direct control | Exact batched filters | Exact fused Gaussian derivatives | Hermite filters |
|---|---:|---:|---:|---:|---:|---:|
| SchNet | 9 | 1.118 | 1.232 | 0.867 | **0.824** | 0.583 |
| SchNet | 72 | 1.704 | 1.794 | 1.372 | **1.327** | 1.027 |
| PaiNN | 9 | 1.140 | 1.156 | 1.156 | **1.062** | 1.023 |
| PaiNN | 72 | 1.696 | 1.729 | 1.656 | **1.544** | 1.520 |

Against stock, the exact fused schedule reduces step time by 22.1–26.3% on SchNet and 6.8–9.0% on PaiNN. Hermite reduces time by 39.7–47.9% and 10.3–10.4%, respectively. Maximum force-component differences from upstream are 1.62 × 10⁻¹⁴ eV/Å for the exact fused schedule and 1.24 × 10⁻⁷ eV/Å for Hermite. The corresponding largest difference in the change of NVE energy for Hermite is 2.61 × 10⁻¹⁰ eV/atom. Validation also checks projected forces against displaced reference energies and each variant's own displaced energies. At one off-knot point in every interval, the largest absolute slope errors are 6.62 × 10⁻⁸ and 3.49 × 10⁻⁹ in the respective internal filter units. The raw reports record each trial, energy and coordinate errors, finite-difference displacements, and timed graph updates. [SchNet, 9 atoms](raw/radial-gaussian-schnet-ethanol.json) · [SchNet, 72 atoms](raw/radial-gaussian-schnet-ethanol8.json) · [PaiNN, 9 atoms](raw/radial-gaussian-painn-ethanol.json) · [PaiNN, 72 atoms](raw/radial-gaussian-painn-ethanol8.json)

Independent earlier runs give the three-way comparison before the Gaussian kernel and the separate message-code controls. [SchNet, 9 atoms](raw/radial-schnet-ethanol.json) · [SchNet, 72 atoms](raw/radial-schnet-ethanol8.json) · [PaiNN, 9 atoms](raw/radial-painn-ethanol.json) · [PaiNN, 72 atoms](raw/radial-painn-ethanol8.json) · [SchNet direct control](raw/radial-control-schnet-ethanol8.json) · [PaiNN direct control](raw/radial-control-painn-ethanol8.json)

## Reproduce the controls

Use the molecular setup to prepare the checkpoints and fixtures. Run the following from the repository root. Use `examples/structures/ethanol.xyz` for a single molecule. [Environment and inputs](portable.md#reproduce-a-row)

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

python scripts/benchmark_grouped_molecules.py --experiment radial --direct-control --fused-basis \
  --framework schnet --model checkpoints/pyg-u0.pt --structure fixtures-molecules/ethanol8.extxyz \
  --device cuda --warmup 5 --steps 16 --repeats 8 --dt-fs 0.25 --knots 4097 \
  --output runs/radial-gaussian-schnet-ethanol8.json
python scripts/benchmark_grouped_molecules.py --experiment radial --direct-control --fused-basis \
  --framework painn --model checkpoints/painn-ethanol.model --structure fixtures-molecules/ethanol8.extxyz \
  --device cuda --warmup 5 --steps 16 --repeats 8 --dt-fs 0.25 --knots 4097 \
  --output runs/radial-gaussian-painn-ethanol8.json
```
