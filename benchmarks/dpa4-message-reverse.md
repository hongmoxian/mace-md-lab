# DPA4-Mini message-passing reverse in periodic MD

This experiment tests whether a conservative edge-owned reverse and a matching SO(2) forward implementation reduce the complete step time by 10% at fixed driver and precision. It uses the public FP32 DPA4-Mini-OMat24 checkpoint (SHA-256 `13edeea63448c0f8a0b38e08fd8ba196fb1e2fad5f12486b1cd23320829da268`), DeePMD-kit 3.2.0, PyTorch 2.12.0 and one 16 GB Tesla V100. All comparisons retain the DeepPot eager evaluator, its neighbor graph construction, a 0.5 fs velocity-Verlet step and the same initial atomic positions and momenta. Energy, conservative forces and stress are evaluated at each step. The exported .pt2 and official optional accelerated deployment paths are outside the comparison.

The full two-rotation route saved 10.8–12.4% on Au/water (192) and 11.3–12.3% on Cu (108). Both systems cleared 10% in both run medians. The matched-original-forward receiver adjoint is reported separately below.

## Time in the two SO(2) message-passing blocks

CUDA events on the active stream bracket each convolution from the arrival of its output adjoint to its differentiable input boundary, including the per-edge attention, mixing and the convolution's own node projections. Per-node FFNs outside this module, global geometry construction and integration are outside the numerator. The denominator is synchronized wall time for the complete instrumented step. The two elapsed-stream intervals are merged before summing. Each profile uses four trials of four measured steps. Uninstrumented timings on the same driver quantify profiling overhead. Times are medians; fractions are medians of the within-trial ratios.

| System | Kernel route | Reverse (ms/step) | Instrumented step (ms) | Reverse share | Uninstrumented step (ms) |
|---|---|---:|---:|---:|---:|
| Au/water (192) | [original](raw/dpa-rev-profile-au.json) | 9.04 | 35.91 | 25.2% | 35.06 |
| Au/water (192) | [receiver rotation + attention](raw/dpa-rev-profile-transport-au.json) | 5.48 | 32.40 | 17.0% | 31.89 |
| Au/water (192) | [both rotations + attention](raw/dpa-rev-profile-local-au.json) | 4.71 | 31.43 | 15.0% | 31.77 |
| Cu (108) | [original](raw/dpa-rev-profile-cu.json) | 7.77 | 36.66 | 21.2% | 36.43 |
| Cu (108) | [receiver rotation + attention](raw/dpa-rev-profile-transport-cu.json) | 5.33 | 33.30 | 16.0% | 32.98 |
| Cu (108) | [both rotations + attention](raw/dpa-rev-profile-local-cu.json) | 4.65 | 32.04 | 14.5% | 30.74 |

## Same-driver end-to-end comparisons

Time savings are one minus the ratio of route medians. Every trial starts from the same state; all reported pairs, including slower ones, are retained. Route setup and priming occur before timed integration. Each fresh process compares its own stock route with its candidates. The independent confirmation gives the candidate first in the initial timing pair, then alternates order. Use the raw records for full orders, lengths, timings, checks and dispatch counters.

| System | Run and candidate | Stock | Candidate | Time saved | Faster pairs |
|---|---|---:|---:|---:|---:|
| Au/water (192) | [attention adjoint + radial mixer](raw/dpa-rev-ablation-au.json) | 35.43 | 34.01 | +4.0% | 5/5 |
| Au/water (192) | [linear blocks only](raw/dpa-rev-ablation-au.json) | 35.43 | 34.49 | +2.7% | 5/5 |
| Au/water (192) | [edge + linear](raw/dpa-rev-ablation-au.json) | 35.43 | 33.47 | +5.6% | 5/5 |
| Au/water (192) | [add channel gates](raw/dpa-rev-ablation-au.json) | 35.43 | 32.16 | +9.2% | 5/5 |
| Au/water (192) | [cache frozen weights only](raw/dpa-rev-fused-au.json) | 35.50 | 35.47 | +0.1% | 3/8 |
| Au/water (192) | [original forward operations + receiver-side VJP](raw/dpa-rev-native-control-au.json) | 35.35 | 34.65 | +2.0% | 6/6 |
| Au/water (192) | [receiver rotation + attention](raw/dpa-rev-fused-au.json) | 35.50 | 31.50 | +11.3% | 8/8 |
| Au/water (192) | [receiver-side confirmation](raw/dpa-rev-confirm-transport-au.json) | 35.77 | 32.25 | +9.9% | 8/8 |
| Au/water (192) | [both rotations + attention](raw/dpa-rev-local-au.json) | 35.56 | 31.16 | +12.4% | 6/6 |
| Au/water (192) | [two-rotation confirmation](raw/dpa-rev-confirm-local-au.json) | 35.36 | 31.56 | +10.8% | 8/8 |
| Cu (108) | [attention adjoint + radial mixer](raw/dpa-rev-ablation-cu.json) | 34.66 | 33.68 | +2.8% | 4/5 |
| Cu (108) | [linear blocks only](raw/dpa-rev-ablation-cu.json) | 34.66 | 35.49 | -2.4% | 1/5 |
| Cu (108) | [edge + linear](raw/dpa-rev-ablation-cu.json) | 34.66 | 34.38 | +0.8% | 4/5 |
| Cu (108) | [add channel gates](raw/dpa-rev-ablation-cu.json) | 34.66 | 32.44 | +6.4% | 5/5 |
| Cu (108) | [cache frozen weights only](raw/dpa-rev-fused-cu.json) | 35.45 | 35.47 | -0.1% | 4/8 |
| Cu (108) | [original forward operations + receiver-side VJP](raw/dpa-rev-native-control-cu.json) | 35.31 | 34.86 | +1.3% | 6/6 |
| Cu (108) | [receiver rotation + attention](raw/dpa-rev-fused-cu.json) | 35.45 | 31.67 | +10.7% | 8/8 |
| Cu (108) | [receiver-side confirmation](raw/dpa-rev-confirm-transport-cu.json) | 35.63 | 31.13 | +12.6% | 8/8 |
| Cu (108) | [both rotations + attention](raw/dpa-rev-local-cu.json) | 35.00 | 31.03 | +11.3% | 6/6 |
| Cu (108) | [two-rotation confirmation](raw/dpa-rev-confirm-local-cu.json) | 35.41 | 31.07 | +12.3% | 8/8 |

The three-route experiment also measured the receiver-only alternative on the same schedule as the two-rotation version. [Interface trials](raw/dpa-rev-local-au.json) · [Cu trials](raw/dpa-rev-local-cu.json).

## Operator and checks

The checkpoint uses two single-focus, single-head SO(2) convolutions (32 channels, angular degree 2, retained order 1). A local-rotation Jacobian contracts the incoming edge gradient with the sender's node features to accumulate orientation gradients. At the receiver, with local features $u_e$, inverse rotation $R_e$, fixed rescaling $S$, attention $a_e$ and receiver gradient $g_i$, the local-feature adjoint is $a_e R_e^T S g_i$. The softmax logit adjoint is $a_e g_i^T(m_e-o_i)$ for message $m_e$ and aggregate $o_i$. The rotation adjoint uses the same receiver gradient and local features. The null softmax mass and squared cutoff envelope are retained. The implementation reuses these contractions within each edge, propagates gradients to the sender and receiver, and handles the rank-one radial mixer, degree gates and fixed SO(2) linear maps explicitly. Receiver-grouped normalization and summation run on the current directed graph. Source and destination indices, geometry and activations are refreshed every call. The mathematical forward is preserved; its GPU implementation changes in the combined route.

The original-forward control replaces only the inverse-rotation/attention reverse. The combined timing also includes the changed forward schedule and other message suboperators. All routes are specialized for first coordinate/strain derivatives in inference on the qualified single-focus model.

On five common moving frames, the largest two-rotation discrepancies from the official ASE calculator across the selection and confirmation runs were 4.9e-08 eV/atom, 7.8e-06 eV/Å and 1.07e-08 eV/Å³ for energy, force and stress. Acceptance limits were 2e-5, 2e-4 and 2e-5 in those units. A 48-edge replay of each real convolution separately checked the output and adjoints for node features, radial features, cutoff envelopes, rotation and inverse rotation against ordinary autograd. The largest component difference across these checks was 3.34e-06.

## Reproduce

Use the [periodic transfer inputs](periodic-transfer-memory.md) and the CUDA DeePMD eager environment. Pin OMP, MKL, OpenBLAS and the two DeePMD thread counts to one; leave optional DeepMD inference acceleration flags at their defaults. Select either `fixtures-interface/au-water-seed109.extxyz` or `fixtures108/cu.extxyz`. Pre-create the output directory.

```bash
python scripts/profile_dpa4_reverse.py --model "$checkpoint" --structure "$fixture" --output profile.json
python scripts/profile_dpa4_reverse.py --model "$checkpoint" --structure "$fixture" --variant local --output profile-local.json
python scripts/benchmark_dpa4_reverse.py --model "$checkpoint" --structure "$fixture" --routes stock transport local --repeats 6 --steps 6 --check-vjp local --output comparison.json
python scripts/benchmark_dpa4_reverse.py --model "$checkpoint" --structure "$fixture" --routes local stock --repeats 8 --steps 8 --check-vjp local --output confirmation.json
python scripts/benchmark_dpa4_reverse.py --model "$checkpoint" --structure "$fixture" --routes stock reverse transport --repeats 6 --steps 6 --check-vjp reverse --output backward-control.json
```

For the two-rotation route, the count of individual pairs saving at least 10% (as distinct from just running faster) was as follows.

- Au/water (192), local: 6/6, paired range 10.3% to 13.5%.
- Au/water (192), confirm-local: 7/8, paired range 9.7% to 12.4%.
- Cu (108), local: 6/6, paired range 10.3% to 13.1%.
- Cu (108), confirm-local: 7/8, paired range 9.4% to 13.6%.
