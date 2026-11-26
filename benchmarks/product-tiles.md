# Coalesced channels in MACE's frozen scalar products

Two scalar product blocks in MACE-MP-0 Small evaluate a sparse polynomial in the angular components of each feature channel. The existing kernel assigns one CUDA thread to each channel, so neighboring threads load angular components from strided addresses. The experimental `both_channel_tile` mode uses each 32-thread warp to read a contiguous tile and transposes the tile in shared memory. It similarly coalesces the writes of the polynomial derivatives. A warp ballot identifies single-species inputs. For dense elemental attributes, the kernel uses the original weighted coefficient sum. The coefficients, polynomial term order and derivative accumulation order are identical. This experiment uses PyTorch and CuPy on a V100 in FP64. [Kernel implementation](../src/mace_md_lab/frozen_product.py)

| Measurement | Original | Channel tile | Reduction |
|---|---:|---:|---:|
| First product, value and feature Jacobian | 94.75 µs | 60.13 µs | 36.5% |
| Second product, value and feature Jacobian | 84.99 µs | 54.74 µs | 35.6% |
| Full Au/water MD with energy, force and stress | 5.499 ms/step | 5.457 ms/step | 0.76% |

The isolated measurement used actual frozen weights and the species of the 192-atom interface with seeded normal features. Each of 12 alternating-order samples timed 32 captured launches with CUDA events. Values and Jacobians matched elementwise for the physical one-hot inputs and for the zero, mixed and scaled attribute controls. [Isolated product timings](raw/product-tiles-kernel.json)

The full MD comparison used the same scheduled neighbor driver, 0.25 Å neighbor skin, starting coordinates and momenta, and 0.5 fs timestep. Each of six interleaved trials comprised five warmup and 48 timed steps. Energy, force and stress reached host memory at every step. The tile path was faster in every paired trial and took 5.454–5.465 ms/step over all six trials. The maximum same-frame force difference from original MACE was 4.00 × 10⁻¹⁵ eV/Å. The absolute full-step saving was 0.042 ms on this trajectory. Composition with the exact radial tangent optimization requires a separate paired measurement. [Moving MD timings and errors](raw/product-tiles-md.json)

## Reproduce

Set up the MACE-MP-0 Small checkpoint and Au/water input, then run these commands from the project root. [Fixture and model setup](adjoints.md#reproduce-the-tables)

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
common=(--model checkpoints/mace-mp-0-small.model --structure fixtures-interface/au-water.extxyz)
python scripts/benchmark_product_tiles.py "${common[@]}" --output runs/product-tiles-kernel.json
python scripts/benchmark_grouped_md.py "${common[@]}" --variants sched schedpoly --outputs edgeefs \
  --resident-skins 0.25 --warmup 5 --steps 48 --repeats 6 --host-observables \
  --check-candidate-edges --output runs/product-tiles-md.json
```
