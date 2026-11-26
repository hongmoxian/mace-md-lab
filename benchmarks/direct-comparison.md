# Direct MD and inference times on Au/water and Cu

These runs answer two practical questions. How long does an MD step take when the MACE weights stay fixed but the evaluator changes? How long does the same application take when one selects a different pretrained potential? All runs used one Tesla V100-SXM2 16 GB, ASE 3.29.0, one CPU thread, fixed periodic cells and a 0.5 fs velocity-Verlet step. The fixtures contain 48 Au atoms and 48 water molecules (192 atoms, seed 109), or 108 Cu atoms. The recorded inputs supply the same initial coordinates and momenta to each route.

## Same pretrained MACE weights

| Complete MD, median ms/step | Original MACE + ASE | Tuned upstream XL + ASE | Tuned upstream XL + prepared driver | Locked XL + prepared driver | Speedup vs original MACE | Time saved by switching only the kernels |
|---|---:|---:|---:|---:|---:|---:|
| [Au/water](raw/direct-mace-xl-au109.json) | 34.236 | 4.007 | 3.007 | 2.241 | 15.27× | 25.5% |
| [Cu108](raw/direct-mace-xl-cu108.json) | 28.344 | 3.377 | 2.505 | 2.158 | 13.13× | 13.8% |

The source is MACE-MP-0 Small, evaluated by MACE 0.3.16 in FP64 through its default PyTorch calculator. The three XL paths use the four-element, 8,192-point compact export of that checkpoint. All three use Symmetrix-XL 0.2.0, its direct speed plan, static launch policy, Kokkos neighbor builder with 1 Å skin, tuned generated upstream modules and on-demand angular derivatives. The two prepared paths share the same fixed-cell integration and neighbor handling. The locked variant changes the R0 and R1 kernels. Each MD step returns energy, Cartesian forces and global stress. The median includes the force call, neighbor handling and integration. Model setup, compilation and five warmup steps precede the 16 timed steps. Each of eight trials restarts from the same initial state and the order is balanced. The locked kernels beat the upstream kernels in all eight pairs on both structures. Earlier 48-step trials measured 18.6% lower time on Au/water and 12.7% lower time on Cu108. [Longer-window measurements](native-algorithms.md)

| Identical moving frames, median ms per energy + force + stress call | Original MACE + ASE | Tuned upstream XL + ASE | Tuned upstream XL + prepared driver | Locked XL + prepared driver | Speedup vs original MACE |
|---|---:|---:|---:|---:|---:|
| Au/water | 34.249 | 3.851 | 2.913 | 2.288 | 14.97× |
| Cu108 | 28.287 | 3.360 | 2.501 | 2.159 | 13.10× |

For inference, each route evaluates the same 16 successive geometries from a precomputed MACE trajectory in the same order. Timing includes neighbor handling, synchronization and returning all three observables to the host. Integration is timed in the first table only. All four implementations also evaluated all 22 reference frames, including the initial and warmup frames. The maximum force-component differences between the locked evaluator and source MACE were 1.34 × 10⁻⁷ eV/Å on Au/water and 3.83 × 10⁻⁹ eV/Å on Cu. The largest energy and stress differences over both cases were 5.70 × 10⁻⁹ eV/atom and 9.23 × 10⁻¹⁰ eV/Å³. These check numerical agreement with MACE. Absolute accuracy against DFT would require reference labels.

## Different pretrained potential on the same periodic inputs

CHGNet ran through its published calculator (package 0.4.2, pretrained model 0.3.0) with its fast graph converter, default FP32 weights and the explicit energy/forces/stress task. Both potentials used the same array-based host integrator. Each trial restarted both from the same initial state, after which each model supplied its own forces. For the inference-only measurement, both evaluated identical successive geometries from the locked MACE trajectory. Each of six alternating trials used three warmups and 12 timed steps. The paired XL column comes from this experiment. [CHGNet model and weights](https://github.com/CederGroupHub/chgnet)

| Case | CHGNet MD (ms/step) | MACE + locked XL MD (ms/step) | MD time ratio, CHGNet / MACE + XL | CHGNet same-frame E/F/stress (ms) | MACE + XL same-frame E/F/stress (ms) |
|---|---:|---:|---:|---:|---:|
| [Au/water](raw/direct-chgnet-xl-au109.json) | 126.281 | 2.430 | 51.96× | 127.398 | 2.354 |
| [Cu108](raw/direct-chgnet-xl-cu108.json) | 107.132 | 2.027 | 52.84× | 107.077 | 2.038 |

These timings cover each named implementation including its neighbor and graph construction. The models differ in architecture, learned energy, checkpoint and precision. The ratios quantify the cost of choosing these implementations for these inputs. They give no estimate of an optimization applied within CHGNet or of the accuracy of either model on Au/water. Within-model measurements on SchNet and PaiNN use their respective pretrained weights on separate molecular inputs. [SchNet and PaiNN timings](portable-radial-filters.md)

## Reproduce

Start from the same source model, 8,192-point compact export, upstream compiled modules, locked kernel artifacts and fixtures as in the native-algorithm comparison. Use a CUDA 12 SM70 XL installation. Install chgnet==0.4.2 for the second script. The official 0.3.0 checkpoint is included in that distribution. Paths inside the locked manifest are relative to the checkout root. [Model conversion, fixtures and kernel preparation](native-algorithms.md#reproduce-the-same-driver-comparison)

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export SYMMETRIX_JIT_CACHE="$(mktemp -d)"
model=checkpoints/mace-mp-0-small.model
compact=runs/compact-small.json
mods=runs/xl-compiled
compare() {
    local tag="$1" structure="$2"
    shift 2
    python scripts/benchmark_mace_xl_direct.py --model "$model" --symmetrix-json "$compact" \
      --product-dir "$mods" --structure "$structure" "$@" --output "runs/direct-mace-xl-$tag.json"
    python scripts/benchmark_chgnet_xl_direct.py --symmetrix-json "$compact" \
      --product-dir "$mods" --structure "$structure" "$@" --output "runs/direct-chgnet-xl-$tag.json"
}
compare au109 fixtures-interface/au-water-seed109.extxyz
compare cu108 fixtures108/cu.extxyz --m0-dir "$mods"
```
