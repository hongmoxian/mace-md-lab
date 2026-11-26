# SevenNet message-passing reverse on complete periodic MD steps

An edge-owned reverse for SevenNet-0's tensor product and receiver sum reduced full-step latency by 35.5–37.0% on 192-atom Au/water and 29.6–29.8% on Cu108 across two fresh runs per case. Each comparison uses the same prepared driver and checkpoint. The e3nn forward, radial MLP, spherical harmonics and neighbor policy are shared. All steps return energy, forces and stress. This experiment uses SevenNet's default e3nn path on a single 16 GB V100. The optional accelerated SevenNet backends were outside the experiment.

## Where the time goes

CUDA events on the active stream bracket each convolution's reverse from its receiver-output adjoint to its differentiable inputs (node features, harmonics and radial features). This covers the complete convolution reverse including the radial MLP. It excludes separate gates, node-linear blocks, final readout and geometric projection outside the convolution. Intervals from all five layers are merged before summation. The denominator is synchronized wall time for a complete instrumented step, including graph and integration. Each run used four trials of four steps, three warmup steps per route/trial and a separate process warmup. The event spans measure elapsed stream time, including possible waits for host dispatch.

| System | Reverse, original → new (ms/step) | Whole instrumented step, original → new (ms) | Reverse share, original → new | Uninstrumented control, original → new (ms) |
|---|---:|---:|---:|---:|
| Au/water | 22.58 → 2.86 | 43.09 → 27.78 | 52.4% → 10.3% | 41.89 → 26.52 |
| Cu108 | 14.74 → 2.83 | 38.41 → 26.81 | 38.4% → 10.6% | 36.89 → 25.63 |

Times are trial medians and shares are medians of within-trial ratios. Original middle convolution layers 2–4 together occupied about 20.0 ms/step on the interface and 12.4 ms/step on Cu. [Original interface profile](raw/seven-reverse-profile-au.json) · [Original Cu profile](raw/seven-reverse-profile-cu.json) · [Optimized interface profile](raw/seven-reverse-profile-optimized-au.json) · [Optimized Cu profile](raw/seven-reverse-profile-optimized-cu.json)

## Paired acceptance runs without instrumentation

| System | Independent run | Original e3nn (ms/step) | Fused reverse (ms/step) | Time saved | Faster pairs |
|---|---|---:|---:|---:|---:|
| Au/water | [1](raw/seven-reverse-md-au-1.json) | 42.83 | 27.65 | 35.5% | 8/8 |
| Au/water | [2](raw/seven-reverse-md-au-2.json) | 41.86 | 26.37 | 37.0% | 8/8 |
| Cu108 | [1](raw/seven-reverse-md-cu-1.json) | 37.01 | 25.99 | 29.8% | 7/8 |
| Cu108 | [2](raw/seven-reverse-md-cu-2.json) | 37.06 | 26.10 | 29.6% | 7/8 |

Each fresh process tested both routes at the same initial coordinates and momenta, in alternating order. Each trial used three warmup steps and eight timed 0.5 fs steps. Both routes completed preliminary evaluations and compilation before the timed trials. The first Cu pair in each run favored the original path; both are included. All four run medians pass the prespecified 10% full-step time-saving target. Five moving common frames from the stock driver were separately evaluated with the official ASE calculator. Across these runs, the largest absolute energy, force or stress differences for the fused route were 3.18e-7 eV/atom, 1.20e-6 eV/Å and 3.73e-9 eV/Å³. A third, four-pair timing run started each system with the new reverse. Savings were 36.7% on the interface and 29.8% on Cu; both won 4/4 pairs. Across all three runs the maximum force component error was 1.32e-6 eV/Å and stress error was 4.66e-9 eV/Å³. [Interface order control](raw/seven-reverse-order-au.json) · [Cu order control](raw/seven-reverse-order-cu.json)

Independent random tensor tests for all five layers check node, angular and radial vector-Jacobian products against e3nn autograd. The largest absolute component discrepancy was 1.15e-5 in float32.

## Transfer from MACE

SevenNet's channel-preserving weighted tensor-product paths have different irreducible representations and learn different weights from MACE. The port extracts SevenNet's actual path map, real Clebsch–Gordan coefficients and e3nn normalization. For each directed edge and path, one kernel reads the **receiver** adjoint, **sender** features, harmonics and radial weights. It contracts receiver adjoints with sender features once and uses the contraction for both the radial-weight and angular derivatives. It accumulates the sender-feature derivative across incident edges. A custom autograd function retains node features, angular and radial inputs, and edge indices for this reverse. Its forward uses the existing tensor product and scatter; the generic backward tape for those operations is omitted. This ports the MACE edge-ownership and contraction ordering to SevenNet's five convolutions. It leaves the radial MLP reverse to PyTorch. Focused qualifications are CUDA float32, local default e3nn SevenNet-0 and first derivatives in inference.

## Reproduce both measurements

Use the environment and fixtures in [periodic transfer](periodic-transfer-memory.md). The checkpoint is SevenNet-0 released 11 July 2024, SHA-256 `7052cb42b7b3be42b40b97fa0d21077a48c54b5548948fc4dcf346629f813c36`. Use SevenNet 0.13.0 and one CPU thread for OMP, OpenBLAS and MKL.

```bash
python scripts/profile_sevennet_reverse.py --model "$sevennet_checkpoint" --structure "$fixture" --output profile.json
python scripts/profile_sevennet_reverse.py --model "$sevennet_checkpoint" --structure "$fixture" --adjoint --output profile-optimized.json
python scripts/benchmark_sevennet_reverse.py --model "$sevennet_checkpoint" --structure "$fixture" --output paired.json
```

Set `fixture` to `fixtures-interface/au-water-seed109.extxyz` or `fixtures108/cu.extxyz`. Repeat the last command in a fresh process to reproduce the paired runs. Add `--adjoint-first --repeats 4` for the initial-order control.
