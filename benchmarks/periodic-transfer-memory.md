# Periodic checkpoint transfer and peak GPU memory

The transfer experiment measures three implementations of each pretrained model on identical starting structures and momenta. The official route uses its published ASE calculator with ASE velocity Verlet. The second retains the model operators and changes the fixed-cell driver. The third also replaces the radial expansion and first derivative. The 192-atom Au/water interface (seed 109) and perturbed 108-atom Cu cell use a 0.5 fs integration step. Each step returns energy, forces and stress. Each trial starts from the original state after independent initialization and warmup. Times are synchronized wall milliseconds per complete MD step on a Tesla V100-SXM2 16 GB.

SevenNet-0 uses the 11 July 2024 public checkpoint, Python package 0.13.0 and the default e3nn tensor product. The prepared driver builds a candidate graph with 1 Å skin, refreshes it when twice the maximum displacement reaches the skin, and filters edges at the model cutoff every step. The custom operator uses the learned Bessel frequencies and the checkpoint's XPLOR switching function. A second ablation also tests the existing MACE spherical-harmonics kernel on the qualified normalized degree-0 through degree-2 SevenNet embedding. DPA4-Mini uses the OMat24 v20260805 checkpoint and DeePMD-kit 3.2.0 in PyTorch eager mode. Its driver calls DeepPot with fixed atom types and cell, retaining the backend's own graph construction. Its kernel evaluates the checkpoint radial basis and its C3 envelope and also fuses the separate edge envelope. In both models the reverse contracts incoming feature gradients with analytic derivatives of the same basis. No force regression or training is performed. [SevenNet checkpoint](https://github.com/MDIL-SNU/SevenNet/tree/main/sevenn/pretrained_potentials/SevenNet_0__11Jul2024) · [DPA4 checkpoint and deployment modes](https://huggingface.co/deepmodelingcommunity/DPA4-OMat24)

| Model | System | Official | Driver | Driver + radial | Kernel time saved vs driver | Faster pairs |
|---|---|---:|---:|---:|---:|---:|
| SevenNet-0 | [Au/water](raw/port-angular-sevennet-au.json) | 42.878 | 41.872 | 41.674 | +0.5% | 7/8 |
| DPA4-Mini | [Au/water](raw/port-dpa4-au.json) | 36.800 | 34.872 | 35.266 | -1.1% | 1/6 |
| SevenNet-0 | [Cu108](raw/port-angular-sevennet-cu.json) | 38.247 | 37.184 | 36.810 | +1.0% | 8/8 |
| DPA4-Mini | [Cu108](raw/port-dpa4-cu.json) | 36.297 | 35.613 | 35.584 | +0.1% | 3/6 |

Positive time savings indicate a shorter full step. Each route also evaluates five common frames from the official trajectory. Maximum absolute differences are reported in eV/atom for energy, eV/Å for any force component and eV/Å³ for stress. The acceptance limits were 2e-5, 2e-4 and 2e-5 respectively. Each kernel route has a dispatch counter. These runs evaluate execution and fidelity to each published checkpoint. They have no DFT accuracy or long-trajectory stability labels.

SevenNet on Au/water used 41.368 ms/step for radial plus angular, 41.674 for radial only and 41.872 for the stock operators on the same driver. Signed savings for both operators relative to stock operators were +1.2%. [JSON](raw/port-angular-sevennet-au.json)

SevenNet on Cu108 used 36.342 ms/step for radial plus angular, 36.810 for radial only and 37.184 for the stock operators on the same driver. Signed savings for both operators relative to stock operators were +2.3%. [JSON](raw/port-angular-sevennet-cu.json)


For a separate transfer at the message-passing level, see [SevenNet convolution reverse](sevennet-message-reverse.md).

## GPU memory

Each route uses a fresh subprocess. An independent parent queries NVIDIA NVML for the worker PID, including CUDA contexts and native allocations that the PyTorch allocator does not own. It samples throughout initialization, the first force call, warmup and MD, sleeping 10 ms between polls. This is a sampled high-water mark; short allocations between samples can be missed. Raw reports also give the whole-device sampled peak, capacity and other visible compute PIDs. PyTorch allocated and reserved allocator maxima are reported separately. A failed allocation has no successful peak or step time. A wall-time limit is a distinct outcome. MACE wall times collected with monitoring are ancillary to the dedicated paired timing runs.

For size scaling, replicate the Au/water interface equally in both lateral directions, at fixed density. Every copy carries its original 300 K momenta. The replicated inputs measure capacity and execution on a periodic workload. The native MACE-MP-0 Small routes all use the same FP64 export and on-demand angular derivatives. The tuned XL ASE and prepared routes retain upstream R0/R1. The locked route installs the two project kernels. Only routes that complete all requested steps count as demonstrated capacity. [XL configuration](native-algorithms.md)

| Atoms | Source MACE | Tuned XL + ASE | Prepared XL + upstream kernels | Prepared XL + locked kernels | Measurements |
|---:|---:|---:|---:|---:|---|
| 192 | 1,360 | 2,536 | 2,536 | 2,536 | [JSON](raw/gpu-mem-au-1.json) |
| 768 | 4,490 | 2,608 | 2,608 | 2,608 | [JSON](raw/gpu-mem-au-2.json) |
| 3,072 | 15,332 | 2,864 | 2,864 | 2,864 | [JSON](raw/gpu-mem-au-4.json) |
| 4,800 | OOM | — | — | — | [JSON](raw/gpu-mem-au-5-mace.json) |
| 12,288 | OOM | 3,804 | 3,804 | 3,804 | [JSON](raw/gpu-mem-au-8.json) · [JSON](raw/gpu-mem-au-8-locked.json) · [JSON](raw/gpu-mem-au-8-prepared.json) |
| 49,152 | — | — | 7,240 | 7,240 | [JSON](raw/gpu-mem-au-16-locked.json) · [JSON](raw/gpu-mem-au-16-prepared.json) |

Entries are per-process peak MiB sampled with NVML. A dash indicates an unavailable completed measurement; OOM identifies an explicit CUDA allocation error. Separate-route reports can also contain post-warmup and post-integration resident memory. The 49,152-atom capacity protocol uses one warmup and two single-step timing intervals per completed native route.

On the 192-atom interface, sevennet peak process memory for official / driver / driver with radial replacement was 1,472 / 1,472 / 1,472 MiB. [JSON](raw/port-mem-sevennet-au.json)

On the 192-atom interface, dpa4 peak process memory for official / driver / driver with radial replacement was 1,016 / 1,016 / 1,008 MiB. [JSON](raw/port-mem-dpa4-au.json)


Both the 3,072-atom upstream and locked runs had 2,864 MiB resident after warmup and after the last MD step, equal to the sampled peak. The 4,800-atom source MACE run failed on a 5.05 GiB allocation. At 49,152 atoms, with matching drivers, the upstream and locked variants both peaked at 7,240 MiB; the corresponding step times were 480.59 and 428.12 ms. The largest completed native input was a short capacity test. The 81.3% reduction at 3,072 atoms relative to source MACE was already present in upstream XL. [Resident memory](raw/gpu-mem-steady-4.json)

On the small periodic systems the radial and angular replacements together shortened the SevenNet step by 1.2% on Au/water and 2.3% on Cu vs its prepared driver (8/8 faster pairs on each). The DPA4 radial replacement gave a 1.1% slowdown on Au/water and a 0.1% reduction on Cu (1/6 and 3/6 faster pairs). These comparisons use the tested eager PyTorch backends; they do not benchmark compiled DPA4 or optional SevenNet acceleration packages.

SevenNet with the angular and radial replacements peaked at 1,472 MiB versus 1,472 MiB with original operators on the prepared driver. [Four-route memory](raw/port-mem-angular-sevennet-au.json)

## Reproduce

Use the benchmark inputs and compiled XL artifacts from the native comparison. The scripts are inference-only. Install each model in a separate CUDA environment and install the repository and Triton there; the memory monitor also needs nvidia-ml-py. Keep the model's original floating-point precision. Set OMP, MKL and OpenBLAS thread counts to one.

```bash
python scripts/benchmark_periodic_transfer.py --framework sevennet --model "$sevennet_checkpoint" \
  --structure fixtures-interface/au-water-seed109.extxyz --output runs/port-sevennet-au.json
python scripts/benchmark_periodic_transfer.py --framework dpa4 --model "$dpa4_checkpoint" \
  --structure fixtures-interface/au-water-seed109.extxyz --output runs/port-dpa4-au.json
```

For Cu use `fixtures108/cu.extxyz`. The three-route comparison uses three warmup steps and six repeats of eight timed steps. For the SevenNet angular ablation add `--angular --repeats 8`. For fresh-process memory measurements of either periodic GNN use `--memory --warmup 2 --steps 4`. Add `--angular` for the four-route SevenNet memory comparison.

```bash
python scripts/benchmark_peak_memory.py --model "$mace_checkpoint" --symmetrix-json "$compact" \
  --product-dir "$compiled" --structure fixtures-interface/au-water-seed109.extxyz \
  --tiles 4 --steps 4 --output runs/gpu-mem-au-4.json
```

The default uses two warmups and three repeats of four measured steps. `--tiles k` tiles the base cell by k × k × 1. Individual routes can be isolated using `--routes prepared` or `--routes locked`. The former accepts `--save-initial reference.npz`, which can be compared in the latter with `--initial-reference reference.npz`.
