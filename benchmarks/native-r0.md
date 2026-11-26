# Channel-first conservative derivatives for XL's first interaction

The native R0 message for an edge uses a radial function $R_{\ell c}(r)$ and a spherical harmonic `Y[l,m]`. Let `b[i,l,m,c]` be the receiver's energy derivative with respect to this message. The new reverse contracts the 128 channels into two partial sums for each angular component:

```math
u_{\ell m}=\sum_c b_{i\ell mc}R_{\ell c}(r),\qquad
v_{\ell m}=\sum_c b_{i\ell mc}R'_{\ell c}(r).
```

Each subgroup lane forms these sums over its own channel subset, projects them onto the angular gradients and radial direction, and reduces three Cartesian components across lanes. This moves the Cartesian projection outside the channel loop. The implementation visits the angular orders separately so each lane holds at most seven pairs of partial sums. The physical spline coefficients and their distances remain in FP64, with the native spline derivative and cutoff.

At 128 channels and four angular orders, a logical R0 spline evaluation requests 2,048 FP64 coefficients per active edge. GPU cache hits determine the actual device-memory traffic. Channel-first contraction reduces the work after fetching those coefficients.

The stateless `tilegrad` variant also assigns each 32-channel tile of a receiver/order to a warp and holds its angular partial sums in registers while traversing neighbors. The evaluator launches the same R0 entry points with the same validated geometry. All outputs are written by their owners; no persistent geometry cache is needed by this variant. [Native builder and installer](../src/mace_md_lab/xl_r0_algorithm.py)

## Au/water development results

The tests use the frozen 192-atom seed-17 structure, a single V100, 8,192 spline knots, five warmup steps and four interleaved 24-step trials with the same prepared ArrayVerlet driver. Every step returns energy, forces and stress on the host. R0 and M1 are explicitly loaded and their native entry points verified by an untimed trace. This is an R0-only ablation: R1 and M1 keep their upstream algorithms. [Raw channel-contraction results](raw/r0-channel-development.json) · [Raw scheduling controls](raw/r0-tile-development.json)

| R0 method | Median complete ms/step | R0 forward in trace (ms) | R0 reverse in trace (ms) |
|---|---:|---:|---:|
| Upstream generated | 3.225 | 0.414 | 0.444 |
| Earlier factored reverse | 3.088 | 0.405 | 0.306 |
| Channel-first reverse | 3.056 | 0.403 | 0.271 |
| Register tiles and channel-first reverse | 3.031 | 0.380 | 0.271 |

The new schedule saves 6.0% of complete step time against the upstream generated operator and 1.8% against the earlier factored operator. Its maximum component difference from the reference XL force on the common frames was 2.20 × 10⁻¹³ eV/Å. A separate device-launched edge-sharded forward increased end-to-end time. The short trace reports its parent kernel only, so its recorded parent duration excludes the child grid.

## Reproduce

Use the FP64 SM70 setup and compiled upstream R0 and M1 from the [prepared benchmark](acceptance.md#reproduction). Compile the new operators and select them through the common prepared-driver manifest. Its field `r0` specifies the initial module, which the named installer replaces.

```bash
python -m mace_md_lab.xl_r0_algorithm --symmetrix-json runs/compact-small.json \
    --artifact-dir runs/r0-algorithm --variants tilegrad
cat > runs/channel-first.json <<'JSON'
{"tilegrad":{"r0":"upstream","install":[{"module":"mace_md_lab.xl_r0_algorithm","config":{"directory":"runs/r0-algorithm","variant":"tilegrad"}}]}}
JSON
python scripts/benchmark_xl_algorithms.py --model checkpoints/mace-mp-0-small.model \
    --symmetrix-json runs/compact-small.json --product-dir runs/xl-compiled \
    --structure fixtures-interface/au-water.extxyz --routes-json runs/channel-first.json \
    --steps 24 --warmup 5 --repeats 4 --trace --output runs/channel-first-au.json
```

The [native-algorithm acceptance protocol](native-algorithms.md) specifies paired timing and the input paths for the holdout and Cu controls.

### Storage controls

An isolated single-evaluator probe staged radial values and slopes in a private, bounded device workspace during forward and read them back during reverse. For each cached edge it checks the radial table pointer, pair type and exact radius, and uses the original spline when the key or capacity check fails. This experiment was confined to serial forward/reverse calls in one dedicated process. With the channel-first adjoint, staging reduced reverse time but raised the forward time to 0.537 ms. Its complete step was 3.166 ms, compared with 3.094 ms for the previously factored reverse in the same run. Evaluating the same-grid cubic from consecutive knots' values and slopes also passed the numerical gates but took 3.465 ms per step. The final stateless R0 operator keeps the native coefficient representation. [Raw storage comparison](raw/r0-staged-development.json) · [Isolated sensitivity driver](../scripts/benchmark_xl_r0_sensitivity.py)

To reproduce the serial development reports above, compile the desired routes, each using the same stock compact checkpoint, then pass the route names to the sensitivity driver. Use `arrays_tile1 arrays_warp arrays_child4 arrays_child8` for scheduling, `arrays_channelgrad arrays_tilegrad arrays_childgrad` for the contraction study, and the routes below for the storage study.

```bash
python -m mace_md_lab.xl_r0_algorithm --symmetrix-json runs/compact-small.json \
    --artifact-dir runs/r0-algorithm --variants staged stagedgrad endpoints
python scripts/benchmark_xl_r0_sensitivity.py --model checkpoints/mace-mp-0-small.model \
    --symmetrix-json runs/compact-small.json --structure fixtures-interface/au-water.extxyz \
    --r0-algorithm-dir runs/r0-algorithm --product-dir runs/xl-compiled --steps 24 --warmup 5 \
    --repeats 4 --kernel-launch-policy static --routes arrays_upstream_r0 arrays_factored_r0 \
    arrays_staged arrays_stagedgrad arrays_endpoints --output runs/r0-staged-development.json
```

### On-the-fly angular gradients

With XL's shared Y-only policy, native R0 reconstructs the Cartesian derivatives of the 16 spherical harmonics on each edge. For a unit direction `n`, the Cartesian angular gradient is the projection of a solid-harmonic gradient with `I − nnᵀ`, scaled by the inverse radius. Applying this linear projection once to the summed edge adjoint gives the same derivative. The experimental `tangentproj` implements this rearrangement, while `ambientwarp` also assigns separate harmonic components to different subgroup lanes. The `lowerproj` variant constructs ambient derivatives directly from lower-degree stored harmonics before the common projection. These routes use the same generated R0 tables as `tilegrad`.

All four passed the own-energy finite-difference checks and common-frame tests. The tracked R0 reverse took 0.265 ms for the locked tile, 0.265 ms for a single accumulated projection, 0.328 ms for the parallel raw harmonics and 0.271 ms for the lower-degree formula. For the last variant CUPTI recorded two reverse launches and one of the two required forward launches. Its common-frame force differed from stock by 6.00 × 10⁻¹⁵ eV/Å. The joint report records the failed dispatch check and suppresses timings of all variants. No full-step speedup is attributed to these angular-only modifications. [Raw Y-only numerical checks, traces and incomplete dispatch witness](raw/r0-projection-development.json)

```bash
python -m mace_md_lab.xl_r0_algorithm --symmetrix-json runs/compact-small.json \
    --artifact-dir runs/r0-algorithm --variants tilegrad tangentproj ambientwarp lowerproj
python - <<'PY'
import json
from pathlib import Path

routes = {}
for name in ("tilegrad", "tangentproj", "ambientwarp", "lowerproj"):
    config = {"directory": "runs/r0-algorithm", "variant": name}
    routes[name] = {"r0": "upstream", "install": [{"module": "mace_md_lab.xl_r0_algorithm",
                                                   "config": config}]}
Path("runs/r0-proj-dev.json").write_text(json.dumps(routes, indent=2))
PY
python scripts/benchmark_xl_algorithms.py --model checkpoints/mace-mp-0-small.model \
    --symmetrix-json runs/compact-small.json --product-dir runs/xl-compiled \
    --structure fixtures-interface/au-water.extxyz --routes-json runs/r0-proj-dev.json \
    --steps 24 --warmup 5 --repeats 6 --trace --harmonic-storage y-only-direct-v1 \
    --output runs/r0-projection-development.json
```
