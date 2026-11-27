# Portable fixed-cell MD against official calculators

The primary quantity is complete-step throughput relative to the model's own official ASE calculator and ASE velocity Verlet, `median(official ms/step) / median(independent ms/step)`. Each model loads its own checkpoint. Both routes start with identical positions and momenta and return energy, forces and all six ASE stress components. A 2× ratio is a working target for the independent adapters.

## Complete moving MD steps on one V100-SXM2 (16 GB)

The main times below are from the 16-step confirmation with reversed starting order. Each process measured four windows per route after three warmup steps. A second column gives the independent eight-step process for comparison. All step times are in milliseconds. These are warmed end-to-end measurements against each model's distributed Python ASE calculator and ASE velocity Verlet. An independently compiled production baseline would answer a different question.

| Checkpoint | System (atoms) | Official | Independent | Speedup | 8-step check | Raw windows |
|---|---|---:|---:|---:|---:|---|
| MACE-MP-0 Small | Au(111)/water (192) | 34.23 | 4.36 | **7.85×** | 7.88× | [primary](raw/official/mace-au-primary.json) / [confirm](raw/official/mace-au-confirm.json) |
| MACE-MP-0 Small | Cu (108) | 27.72 | 3.29 | **8.42×** | 8.61× | [primary](raw/official/mace-cu-primary.json) / [confirm](raw/official/mace-cu-confirm.json) |
| SevenNet-0 | Au(111)/water (192) | 43.66 | 14.40 | **3.03×** | 2.97× | [primary](raw/official/sevennet-au-primary.json) / [confirm](raw/official/sevennet-au-confirm.json) |
| SevenNet-0 | Cu (108) | 38.31 | 10.62 | **3.61×** | 3.57× | [primary](raw/official/sevennet-cu-primary.json) / [confirm](raw/official/sevennet-cu-confirm.json) |
| DPA4-Mini-OMat24 | Au(111)/water (192) | 36.85 | 15.53 | **2.37×** | 2.38× | [primary](raw/official/dpa4-au-primary.json) / [confirm](raw/official/dpa4-au-confirm.json) |
| DPA4-Mini-OMat24 | Cu (108) | 36.75 | 13.00 | **2.83×** | 2.76× | [primary](raw/official/dpa4-cu-primary.json) / [confirm](raw/official/dpa4-cu-confirm.json) |

### Optional MACE integration

Each entry is milliseconds per step with throughput relative to official MACE in parentheses. The final column divides tuned XL/ASE time by combined time; both use the same XL compact export.

| System | Official MACE | Independent | Tuned XL/ASE alone | Our driver and kernels on XL | Gain over XL alone |
|---|---:|---:|---:|---:|---:|
| Au(111)/water (192) | 34.23 (1.00×) | 4.36 (7.85×) | 3.71 (9.22×) | 2.18 (15.71×) | 1.70× |
| Cu (108) | 27.72 (1.00×) | 3.29 (8.42×) | 2.96 (9.35×) | 1.64 (16.92×) | 1.81× |

Across both processes and fixtures, maximum independent force-component discrepancies were 3.20e-14 eV/Å for MACE, 1.53e-5 for SevenNet, and 1.69e-5 for DPA4. The energy, force and stress gates passed in all twelve comparisons. The independent session's initialization and first force took 1.04–11.81 s for MACE, 2.53–2.93 s for SevenNet and 2.21–17.93 s for DPA4. These setup measurements may reuse compiled caches. They exclude process startup and the prior official reference and are outside the speedup ratio.

The total gain includes the driver, graph scheduling of forward and reverse operations, and model-specific kernels. Matched-driver operator ablations are reported separately for [MACE](native-algorithms.md), [SevenNet](sevennet-message-reverse.md) and [DPA4](dpa4-forward-replay.md).

## Run the comparisons

Use a single CUDA GPU, one CPU thread, a fully periodic fixture with momenta, and the checkpoint for the selected adapter. The Au/water cell contains 48 Au atoms and 48 water molecules (seed 109, 192 atoms). Cu is a 108-atom periodic crystal. Use the fixture generators and checkpoint sources in the [periodic transfer benchmark](periodic-transfer-memory.md), [MACE native baseline](native-algorithms.md) and [DPA4 replay benchmark](dpa4-forward-replay.md).

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export DP_INTRA_OP_PARALLELISM_THREADS=1 DP_INTER_OP_PARALLELISM_THREADS=1
python scripts/benchmark_periodic_official.py --framework sevennet --model "$checkpoint" --structure "$fixture" --steps 8 --repeats 4 --output result.json
```

Use `mace` or `dpa4` with its checkpoint for the other two models. To include both MACE-only XL comparison paths, use the upstream generated artifacts in [the direct comparison](direct-comparison.md#reproduce) and add `--symmetrix-json "$compact" --product-dir "$products"` (also `--m0-dir "$m0"` for the tuned Cu reference). Four columns then separate the official MACE calculator, the independent adapter, tuned upstream XL/ASE, and XL plus the prepared integrator and specialized kernels. The independent and XL adapters are alternative implementations of the same checkpoint; XL uses its compact radial export. The independent path evaluates the original MACE radial networks.

## Common API and model adapters

| Function | MACE-MP-0 Small | SevenNet-0 | DPA4-Mini-OMat24 |
|---|---|---|---|
| Current geometry and edges | Two Verlet buffers and strict cutoff | Skin list with physical cutoff and masked graph slots | Official DeepPot neighbor builder with masked graph slots |
| Energy and conservative derivative | Grouped interactions, exact radial tangents, analytic edge virial | e3nn tensor products, edge-local reverse contractions, edge-gradient force and virial | SO(2) rotations, attention contractions, and native force/virial assembly |
| GPU execution | Reusable graph with sorted candidate edges | Reusable energy and derivative graph with capacity headroom | Reusable full energy/force graph with capacity headroom |
| Dynamics | Fixed-cell 0.5 fs velocity Verlet with host E/F/stress | Same | Same |

For each model the script records raw paired windows, timing order, warmup, initialization and first independent force time, checkpoint and fixture hashes, GPU and package versions, and discrepancies at eight shared geometries (five moving frames, a perturbation, a periodic image change and a repeated frame). Setup is separate from warmed steady-state throughput. The qualified checkpoints and numerical gates limit interpretation to the tested models and fixed-cell MD.
