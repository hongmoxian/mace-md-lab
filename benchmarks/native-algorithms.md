# Conservative-force algorithms inside the prepared XL backend

This experiment holds the Python driver, neighbor-list policy and integrator fixed. The comparator is the prepared Symmetrix-XL 0.2.0 evaluator with upstream native kernels, static launch policy, FP64, direct speed plan and the 8,192-point compact export of MACE-MP-0 Small. R0 and M1 use their tuned generated kernels. Cu108 additionally uses generated M0, following the preceding stock-only comparison. Every implementation propagates the same unconstrained fixed-cell velocity-Verlet trajectory and returns energy, Cartesian forces and global stress to the host every step. [Previous full-step acceptance](acceptance.md)

## Locked results inside the strong backend

Every cell used the faster official on-demand angular derivative policy on all three routes. Each entry is the median wall time in ms per complete step. The previous column retains the older factored R0. Both comparison columns use the same paired timing invocation as the candidate.

| System | Upstream prepared XL | Previous factored R0 | Locked R0 + R1 | Time saved vs upstream | Time saved vs previous | Paired wins vs upstream |
|---|---:|---:|---:|---:|---:|---:|
| [Au/water, new seed 109](raw/native-yonly-au109.json) | 3.166 | 3.036 | 2.577 | 18.6% | 15.1% | 8/8 |
| [Cu, 108 atoms](raw/native-yonly-cu108.json) | 2.552 | 2.414 | 2.294 | 10.1% | 5.0% | 7/8 |
| [Au/water, development seed 17](raw/native-yonly-au17.json) | 3.138 | 3.033 | 2.584 | 17.7% | 14.8% | 8/8 |
| [Au/water, previous fixture seed 73](raw/native-yonly-au73.json) | 3.150 | 3.058 | 2.587 | 17.9% | 15.4% | 8/8 |

The first Cu test passed the 10% threshold by 0.11 percentage points. With the same frozen code, two additional allocations gave the following results. These are separate timing comparisons and do not replace the first result.

| Confirmation | Upstream | Previous | Locked | Time saved vs upstream | Time saved vs previous | Paired wins vs upstream |
|---|---:|---:|---:|---:|---:|---:|
| [Cu108, 24 steps, separate allocation](raw/native-yonly-cu108-repeat.json) | 2.642 | 2.522 | 2.270 | 14.1% | 10.0% | 8/8 |
| [Cu108, 48 steps, separate allocation](raw/native-yonly-cu108-48step.json) | 2.629 | 2.584 | 2.295 | 12.7% | 11.2% | 8/8 |

For the four primary experiments, the largest force-component difference from the same XL energy evaluator was 3.11 × 10⁻¹⁵ eV/Å over all common frames. The maximum difference in the change of NVE energy was 1.19 × 10⁻¹⁵ eV/atom. The largest force difference from the source MACE checkpoint on the three stationary comparisons per case was below 8.5 × 10⁻⁸ eV/Å. All 22 repository tests passed. The short timings characterize single-GPU fixed-cell MD on the tested checkpoint. The R1 schedule uses its scalar source and qualified coupling structure.

## Qualification

- The practical threshold is **at least 10% less wall time per complete step** than the same prepared driver with the upstream force evaluator. Score the ratio of the two within-run medians. At least seven of eight paired runs must favor the candidate. Record all paired times and the previous factored R0 control in the same experiment. This is a predefined practical performance rule.
- Select the native algorithm using development runs on the Au/water fixture with seed 17. Keep the algorithm, precision, export and numerical gates fixed when applying it to the Au/water fixture with previously unused seed 109 and to 108-atom Cu. The interface receives five warmup steps and eight rounds of 48 timed steps, Cu five warmup steps and eight rounds of 24. The integrator step is 0.5 fs.
- Check energy, force and stress against the original MACE checkpoint and the matched XL evaluator. Compare the change of NVE total energy, positions and velocities over the moving trajectory. A new algebraically equivalent kernel is also checked against its corresponding native XL outputs at FP64 roundoff. Collect CUDA activity independently of wall timing to verify kernel dispatch.
- Runtime initialization, compilation and offline preprocessing precede timed integration. Any per-step or per-rebuild work is charged to the candidate. A changed radial interpolation or precision is a separate sensitivity experiment.

On the previous held-out Au/water run the matched prepared upstream control took 3.409 ms/step. Replacing its R0 reverse alone took 3.330 ms/step and won four of six pairs. The new 10% criterion would have required 3.069 ms/step on that reference. The final comparison uses the reference measured in the same run. [Previous paired timings](raw/locked-r0-au73.json)

On the development interface, XL's official option to retain spherical harmonics and evaluate their Cartesian derivatives on demand reduced stock step time from 3.351 to 3.128 ms. The same official control reduced Cu108 time from 2.510 to 2.369 ms. All primary comparisons therefore enable this policy on every engine. The denominator includes this native XL gain. [Official harmonic storage control](native-affine-geometry.md)

## Frozen implementation

The R0 forward assigns a contiguous channel tile to each warp. The reverse contracts the incoming channel adjoints with the radial spline values and their derivatives before reducing the angular contributions. For R1, one warp handles each directed edge, and eight warps in a block share a source atom. Each warp evaluates radial and angular contributions to the edge derivative and accumulates its partial source-feature gradient, followed by one reduction at the source. R1 forward uses the original implementation. Both replacements keep the trained parameters and the original export's FP64 spline coefficients. [Frozen route manifest](../examples/native-xl-locked.json) · [R0 methods and component tests](native-r0.md) · [R1 contraction and compiler control](native-r1.md)

A six-round development comparison using the original retained-gradient policy gave 3.316 ms/step for the prepared upstream evaluator, 3.223 ms for the original factored R0 and 2.827 ms after also replacing the R1 reverse. That comparison isolated R1 before choosing the final R0 channel schedule. [Development report](raw/native-dev-factored-r1.json)

## Reproduce the same-driver comparison

Prepare the official MACE checkpoint, 8,192-point compact export, Cu108 and seed-17 Au/water fixtures using the periodic setup. Compile the three upstream modules for the model and V100, and make the previous factored R0 available in that same directory. The frozen JSON specifies the two additional candidate artifact directories relative to the repository root. [Inputs and upstream modules](acceptance.md#reproduction) · [Locked manifest](../examples/native-xl-locked.json)

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export SYMMETRIX_JIT_CACHE="$(mktemp -d)"
compact=runs/compact-small.json
model=checkpoints/mace-mp-0-small.model
mods=runs/xl-compiled
python -m mace_md_lab.xl_r0_algorithm --symmetrix-json "$compact" \
  --artifact-dir runs/r0-algorithm --variants tilegrad
python -m mace_md_lab.xl_r1_algorithm --compact "$compact" \
  --directory runs/r1-qualified/artifacts --variants edgewarp8
python examples/make_interface.py --seed 109 --output fixtures-interface/au-water-seed109.extxyz
compare() {
    python scripts/benchmark_xl_algorithms.py --model "$model" --symmetrix-json "$compact" \
      --product-dir "$mods" --routes-json examples/native-xl-locked.json \
      --harmonic-storage y-only-direct-v1 --trace --require-speedup locked \
      --warmup 5 --repeats 8 --structure "$1" --steps "$2" --output "$3" "${@:4}"
}
compare fixtures-interface/au-water-seed109.extxyz 48 runs/native-yonly-au109.json
compare fixtures108/cu.extxyz 24 runs/native-yonly-cu108.json --m0-dir "$mods"
```

The runner records the isolated native modules, their binary fingerprints, the full trial order, per-trial rebuild counts, same-frame errors against XL and MACE, and finite differences of the candidate's own energy in coordinate and strain directions. The numerical result and the practical speed rule are separate fields in the JSON. The requested locked-route mode returns a failure exit status if either check fails.

Independent development controls tested re-indexing byte-identical spline rows, coarser interpolation and a species-only affine reparameterization. Their wall-time and accuracy measurements are in the component reports. [Radial representation](native-radial-storage.md) · [Affine and geometry controls](native-affine-geometry.md) · [Full native profile](prepared-kernel-profile.md)
