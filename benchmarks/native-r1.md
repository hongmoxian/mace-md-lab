# An edge-parallel reverse for native XL R1

This experiment specializes the second interaction of the MACE-MP-0 Small compact export for the CUDA 12 SM70 build of Symmetrix-XL 0.2.0. The scalar source has 128 channels. Four radial coefficients per channel couple it to the sixteen spherical harmonics through degree three. The builder checks the scalar-source contract fingerprints before changing the generated kernels. All spline coefficients, harmonic derivatives, angular normalization constants, learnable weights, energy, arithmetic precision and host-visible observables are retained. Scope is the direct FP64 fixed-composition, nonalchemical evaluator.

For an edge from source $j$ to receiver $i$, write $h_{jc}$ for the source channel and $u_{i\ell mc}$ for the adjoint of the receiver's output. Define $B_{\ell c}=\sum_m u_{i\ell mc}Y_{\ell m}$ and $\mathbf D_{\ell c}=\sum_m u_{i\ell mc}\nabla Y_{\ell m}$. The forward contribution is $h_{jc}R_{\ell c}(r)Y_{\ell m}$. Its scalar-source adjoint is $\sum_\ell R_{\ell c}B_{\ell c}$. Its contribution to the edge derivative is

$$
\sum_c h_{jc}\left[
  \hat{\mathbf r}\sum_\ell R'_{\ell c}B_{\ell c}+
  \sum_\ell R_{\ell c}\mathbf D_{\ell c}
\right].
$$

One warp handles all four groups of 32 channels for a single directed edge. The lanes contract over the angular components, evaluate the same cubic and its derivative, and reduce all channels' force contributions inside that warp. Different warps process disjoint edges incident on the same source. They reduce their accumulated source-feature adjoints once after the source's entire neighbor list. Upstream processes these neighbors serially with a block-wide reduction and barriers for each edge. The new schedule retains one native reverse kernel and one writer per directed-force slot. The two additional forward experiments factor $h_{jc}R_{\ell c}$ once for the $2\ell+1$ components and optionally split the receiver's neighbor list into two parts.

## Development result

In the 48-step, six-repeat Au/water development comparison the original prepared evaluator took 3.337 ms/step, the unchanged upstream R1 compiled with nvcc took 3.409 ms, and the eight-warp reverse took 3.006 ms with unchanged R1 forward. Four separately traced moving steps measured R1 reverse at 0.665, 0.725 and 0.317 ms respectively. Those three forward times were 0.275, 0.274 and 0.274 ms. The new reverse cut 52.3% from the shipped R1 reverse time. Full-step time fell by 9.9% with the R0 evaluator held at its upstream setting. Hoisting the source factor into the radial term increased forward time to 0.347 ms; splitting the receiver’s neighbor interval in two increased it to 0.338 ms. Both are retained as ablations. This development build was followed by an explicit coordinate-storage-width guard for the fallback Cartesian-gradient branch. [Development timings and kernel dispatch](raw/xl-r1-forward-development.json)

The final guarded implementation repeated the Au/water development ablation with 32 timed steps in six paired rounds. The full native trajectory matched the shipped XL observables and state at every step. Each forward/reverse figure below comes from four CUPTI-confirmed moving steps outside timing. [Raw trials and errors](raw/xl-r1-schedules-qualified.json)

| Native R1 route | R1 forward (ms) | R1 reverse (ms) | Complete MD step (ms) |
|---|---:|---:|---:|
| Shipped upstream | 0.273 | 0.666 | 3.312 |
| Upstream with nvcc | 0.275 | 0.717 | 3.365 |
| 2 warps | 0.273 | 0.422 | 3.064 |
| 4 warps | 0.273 | 0.367 | 3.005 |
| 8 warps | 0.274 | 0.323 | 2.964 |
| 8 warps plus factored forward | 0.350 | 0.321 | 3.037 |
| 8 warps plus split forward | 0.328 | 0.324 | 3.041 |

## On-demand angular derivatives

The optional XL Y-only policy retains spherical-harmonic values and computes their Cartesian derivatives inside the force graph. The original eight-warp implementation invokes XL's direct polynomial gradient per edge when that pointer is absent. The additional `edgewarp8lower` candidate assigns a harmonic component to each of 16 warp lanes. It forms the gradient of the degree-$l$ solid harmonic from the stored degree-$(l-1)$ components, then projects to the angular gradient on the tangent plane of the unit sphere. The reverse still consumes derivatives of the same real harmonic functions. This specialization and the upstream/unchanged eight-warp comparators all run with the same Y-only storage option. [Sparse coefficient rows and CUDA helper](../src/mace_md_lab/xl_harmonic_lowering.py)

On seed 17, six 48-step Y-only trials measured 3.283 ms/step for the shipped upstream R1, 2.929 ms for the retained eight-warp R1 using XL’s on-demand gradient, and 2.941 ms for the new cooperative lowering. The R1 reverse kernel increased from 0.335 to 0.373 ms. The lowered derivatives passed all trajectory checks. The selected algorithm retains the upstream gradient helper with the eight-warp neighbor schedule. The native loader returned the expected absolute cubin path for every installed module, in addition to its ABI artifact ID and four traced forward/reverse dispatches. [Y-only control, numerical checks and module identities](raw/xl-r1-harmonic-lowering-au17.json)

## Reproduce the development comparison

Build the upstream R0/M1 modules as in the full-step acceptance protocol. The command below compares the shipped R1 against a compiler-matched R1 with the unchanged upstream source and five conservative alternatives. Each route uses the same prepared graph and integration code. [Periodic setup](acceptance.md#reproduction) · [Native R1 builder and installer](../src/mace_md_lab/xl_r1_algorithm.py)

```bash
compact=runs/compact-small.json
mods=runs/r1-qualified/artifacts
python -m mace_md_lab.xl_r1_algorithm --compact "$compact" --directory "$mods" \
  --variants upstream edgewarp2 edgewarp4 edgewarp8 edgewarp8forward edgewarp8pairs
python scripts/benchmark_xl_r1_warps.py --compact "$compact" \
  --structure fixtures-interface/au-water.extxyz --r0-directory runs/xl-compiled \
  --r1-directory "$mods" --variants stock upstream edgewarp2 edgewarp4 edgewarp8 edgewarp8forward edgewarp8pairs \
  --steps 32 --repeats 6 --output runs/r1-schedules-qualified.json
```

To compare the optional on-demand derivative, add the `edgewarp8lower` variant to the module build, then pass `--y-only --variants stock upstream edgewarp8 edgewarp8lower --steps 48 --repeats 6` to the same benchmark.
