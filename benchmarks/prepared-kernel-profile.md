# Native force kernels inside a prepared XL step

The 192-atom Au/water development cell was profiled for 20 moving steps after five warmup steps. Both routes used the same fixed-cell Verlet driver and prepared device graph, the static launch policy, upstream generated M1, FP64, and the default native M0. Each trajectory rebuilt its actual graph once, returned energy, force and global stress to the host at every step, and executed each R0, R1 and M1 forward/reverse entry point 20 times. The trace records all 47 distinct CUDA kernel names on a single Tesla V100-SXM2-16GB. [Full per-step activities, kernel names and CUDA API durations](raw/prepared-allkernels-interface.json)

| GPU activity, mean ms/step | Upstream R0 | Factored R0 |
|---|---:|---:|
| R1 fused reverse | 0.672 | 0.673 |
| R1 forward | 0.273 | 0.273 |
| R0 coordinate reverse | 0.436 | 0.308 |
| R0 forward | 0.403 | 0.404 |
| compute_Y (Kokkos) | 0.200 | 0.199 |
| BLAS kernels | 0.120 | 0.120 |
| Generated M1 forward and reverse combined | 0.030 | 0.030 |
| GPU memory copy and set | 0.061 | 0.061 |
| All GPU activity, interval union | 2.882 | 2.755 |

R0 and R1 together occupy 1.78 ms/step in the upstream trace. The two R1 kernels account for 0.95 ms/step and remain unchanged when installing factored R0. The 19 steps without a neighbor rebuild averaged 2.668 ms of upstream device activity. The one rebuilding step took 6.939 ms of device activity. These numbers describe time spent on the device during a complete step. CPU-side runtime calls can overlap CUDA execution. Profiler instrumentation also alters wall time. Use the uninstrumented paired acceptance tests to quantify any speed change.

## Reproduction

Use the standard Au/water development fixture, compact model, upstream R0/M1 and factored R0 modules specified in the acceptance benchmark. The script measures an additional three uninstrumented 20-step blocks for context. [Setup](acceptance.md#reproduction)

```bash
python scripts/profile_prepared_xl.py --structure fixtures-interface/au-water.extxyz \
  --symmetrix-json runs/compact-small.json --module-dir runs/xl-compiled \
  --warmup 5 --steps 20 --trials 3 --output runs/prepared-allkernels-interface.json
```
