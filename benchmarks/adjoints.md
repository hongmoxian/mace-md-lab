# Conservative MACE force and MD benchmarks

Specializing both scalar product layers and the spherical harmonics gave a **1.23–1.29× speedup per MD step** on four periodic MACE-MP-0 Small inputs, measured against the unmodified model using the same padded CUDA Graph. The product specialization supplied most of that gain. On MACE-MP-0 Medium, one scalar product layer qualified and the speedup was **1.05–1.06×**. The external Symmetrix-XL backend was the fastest implementation in all three direct comparisons below.

## Which operators save time?

Times are **milliseconds per complete velocity-Verlet step**, on one Tesla V100-SXM2 16 GB in FP64 with fixed weights, the default head, CUDA 12.6, PyTorch 2.13.0+cu126, MACE 0.3.16 and ASE 3.29.0. Runs start from the same coordinates and momenta, with a 0.5 fs timestep and a 0.12 Å neighbor skin. Each value is the median of five restarted trials with rotated method order and three warmup steps. The timer includes topology checks, graph refreshes, recaptures and synchronization. Model setup, transformation, first force and warmup are recorded separately from the timed steps.

| Checkpoint and system | Timed steps | ASE E/F | Stock resident | Angular only | Product only | Both | Stock / both |
|---|---:|---:|---:|---:|---:|---:|---:|
| Small · Cu32 | 20 | 17.47 | 6.61 | 6.42 | 5.47 | 5.27 | 1.254× |
| Small · Cu108 | 20 | 25.80 | 15.33 | 15.09 | 12.12 | 11.86 | 1.293× |
| Small · Cu32, compressed | 12 | 17.56 | 7.12 | 6.94 | 5.95 | 5.79 | 1.230× |
| Small · Au48(H₂O)48 | 16 | 31.63 | 26.75 | 26.53 | 21.24 | 21.02 | 1.273× |
| Medium · Cu32 | 12 | 22.88 | 14.85 | 14.64 | 14.28 | 14.08 | 1.055× |
| Medium · Au48(H₂O)48 | 12 | 71.07 | 67.90 | 67.66 | 65.16 | 64.94 | 1.046× |

The ASE E/F control uses upstream batching and returns energy and forces to the host with stress disabled. All four resident columns use the same GPU integrator, candidate neighbor list, physical cutoff and padded replay. Angular specializes the value and first derivative of the component-normalized spherical harmonics. Product evaluates eligible scalar contractions as a fixed polynomial with species-specific coefficients. Medium retains the stock implementation in its vector-valued block. Across these six runs, both transforms matched the official calculator to within 3.11 × 10⁻¹⁵ eV/Å for force components at the initial and final reference geometries. Every timed padded step replayed without an eager fallback. [Scalar contraction](../docs/frozen-product.md) · [Small Cu32](raw/ab-cu32.json) · [Small Cu108](raw/ab-cu108.json) · [Compressed Cu32](raw/ab-cross.json) · [Small interface](raw/ab-interface.json) · [Medium Cu32](raw/medium-cu32.json) · [Medium interface](raw/medium-interface.json)

Two additional same-policy controls tested the effect of graph replay. On Small Cu32 with exact-size replay, stock and both took 6.32 and 4.98 ms/step (1.269×). At the Small interface with eager execution, they took 25.14 and 19.13 ms/step (1.314×). [Exact Cu32](raw/exact-cu32.json) · [Eager interface](raw/eager-interface.json)

## How fast is the compiled external backend?

Each row is a separate invocation with **all six implementations**, the same Small source checkpoint, initial frame and momenta. These are medians of three trials of 12 steps following three warmups. Official MACE ASE and Symmetrix-XL ASE both returned stress alongside the requested energy and forces. The resident modes keep positions and velocities on the GPU and compute energy and forces. All entries include integration, the backend's neighbor handling and synchronized endpoints. Conversion, compilation and initialization precede timing.

| System | Official ASE | Stock exact | Stock padded | Both exact | Both padded | Symmetrix-XL direct/speed | Max Symmetrix ΔF (eV/Å) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cu32 | 19.08 | 6.26 | 6.53 | 4.90 | 5.19 | **3.58** | 3.43 × 10⁻⁹ |
| Cu108 | 28.12 | 13.87 | 14.96 | 10.42 | 11.52 | **4.11** | 3.34 × 10⁻⁹ |
| Au48(H₂O)48 | 34.36 | 116.66 | 26.45 | 90.81 | 20.74 | **8.14** | 6.67 × 10⁻⁸ |

At the interface the active edge count changed often enough that the exact-size modes recaptured on all 12 timed steps of each trial. Their padded counterparts reused their captures throughout. Symmetrix-XL 0.2.0 used its CUDA 12 SM70 wheel, the native `mh0-direct-speed` plan and a four-element export with 8,192 radial spline samples. All six implementations passed the original FP64 thresholds. The external timing is for this converted representation and hardware. [Cu32](raw/strong-cu32.json) · [Cu108](raw/strong-cu108.json) · [Interface](raw/strong-interface.json)

The follow-up benchmarks tune the native neighbor builder and skin, then compare the same native engine with and without a fixed-cell reuse check. They also test two changes to the standalone MACE interaction. [Tuned Au/water experiments](xl-parity.md)

## Does the computed force follow the energy?

The 192-atom fixture contains 48 gold atoms and 48 waters in a periodic, unequilibrated slab geometry. With Small and both transforms, force components at its initial geometry matched the official calculator within 2.45 × 10⁻¹⁵ eV/Å. At a centered step of 0.0001 Å, the absolute discrepancies between the force projection and the energy finite difference were 3.52 × 10⁻⁹ eV/Å along a random normalized direction and 1.70 × 10⁻⁶ eV/Å along an O–H stretch. The raw report gives four displacements per direction. [Force checks](raw/fd-interface.json)

Over the same 24 fs from the same initial state, the maximum absolute per-atom total-energy excursion was **0.14257 meV/atom at 0.5 fs** (48 steps) and **0.03459 meV/atom at 0.25 fs** (96 steps). The half-step/full-step ratio was 0.243. Official MACE at 0.5 fs also gave 0.14257 meV/atom. Over all 48 steps, the two independently integrated 0.5 fs trajectories agreed to 1.07 × 10⁻¹⁴ Å in position and 3.68 × 10⁻¹³ eV/Å in force. This test measures short-time integration error on the specified potential. [Half-timestep NVE data](raw/fd-interface.json)

For a local chemical observable, we rigidly translated one interfacial water along the surface normal while holding its orientation and the surrounding atoms fixed. At displacements −0.2, 0 and +0.2 Å, the normal force on that water was +0.46517, +0.11027 and −0.06125 eV/Å. Across the five displacements, the largest optimized-versus-stock differences were 1.14 × 10⁻¹³ eV in relative energy and 4.45 × 10⁻¹⁶ eV/Å in projected force. Centered differences with a 0.001 Å displacement agreed with the force to 1.26 × 10⁻⁶ eV/Å. The curve is an instantaneous potential-energy slice. [All five energies and projected forces](raw/interface-scan.json)

## Where does the reverse pass spend time?

On the 13,306-edge interface structure, the median eager reverse pass for the full energy-to-position derivative took 14.30 ms with stock MACE and 11.10 ms with both transforms. With stock activations and actual incoming gradients supplied to isolated modules, the two product-layer reverse passes went from 1.81/1.81 ms to 0.21/0.22 ms. Spherical harmonics went from 0.74 to 0.14 ms. Each isolated backward uses its own fresh forward outside the timer. Their times cannot be summed to predict an MD step. Per-frame whole-reverse results for Cu108 varied, including a slower first frame with the transforms. [Interface reverse profile](raw/reverse-interface.json) · [Cu108 reverse profile](raw/reverse-cu108.json)

Keeping the polynomial and angular operator fixed separates the effect of forming the product derivative during the forward pass from evaluating it during reverse. On Cu108 the precomputed and deferred schedules took 11.91 and 12.06 ms/step, compared with 15.36 ms/step for stock. At the interface the corresponding times were 20.87, 21.04 and 26.62 ms/step. Changing only the schedule altered whole-step time by approximately 1% here. The cross-family study separately tests position-dependent edge geometry and graph replay on molecular checkpoints. [Cu108 schedules](raw/schedule-cu108.json) · [Interface schedules](raw/schedule-interface.json) · [Portable geometry benchmarks](portable.md)

## Reproduce the tables

Use a CUDA 12 environment with a compatible PyTorch build. The shell blocks below run from the repository root in Bash. They generate test structures and perform short inference runs using two published checkpoints.

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m pip install -e '.[cuda12]' ase==3.29.0 matscipy==1.2.0 e3nn==0.4.4
mkdir -p runs checkpoints
release=https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0
small=checkpoints/mace-mp-0-small.model
medium=checkpoints/mace-mp-0-medium.model
curl -fL "$release/2023-12-10-mace-128-L0_energy_epoch-249.model" -o "$small"
curl -fL "$release/2023-12-03-mace-128-L1_epoch-199.model" -o "$medium"
sha256sum -c <<'CHECKSUMS'
2ddb079cee0e131eaaf6912ba581b394551ead283e95c99cfe78c605d10b5736  checkpoints/mace-mp-0-small.model
01bfe22100139f424713cf921144e5509cbe353d67aa9fa1be9c6e1e0ed35845  checkpoints/mace-mp-0-medium.model
CHECKSUMS

python examples/make_fixture.py --out fixtures --repeat 2 --frames-only
python examples/make_fixture.py --out fixtures108 --repeat 3 --frames-only
python examples/make_fixture.py --out fixtures-crossing --repeat 2 --lattice-a 3.464101615 --frames-only
python examples/make_interface.py --output fixtures-interface/au-water.extxyz

cu32=fixtures/cu.extxyz
cu108=fixtures108/cu.extxyz
cross=fixtures-crossing/cu.extxyz
interface=fixtures-interface/au-water.extxyz
sha256sum -c <<'CHECKSUMS'
288df73f35c10f79db54d8f8e9ab072f1c1b059c9759d6a2e941695ce67f1f86  fixtures/cu.extxyz
caf53b8fdc45601d1d781b1f91467a58bae78db87be195d4de6bae55124cd7f9  fixtures108/cu.extxyz
4176326e59a1b43f2747e266f69165c147206d7f80fe5e575196e41923abc4c0  fixtures-crossing/cu.extxyz
674186cd63ac40385917aa952cf204c88d263fac13d2af3f5257c01e1d3a24a5  fixtures-interface/au-water.extxyz
CHECKSUMS
```

The Cu structures start from a perturbed FCC crystal. The compressed cell produces frequent crossings of the 6 Å model cutoff. The interface generator constructs a three-layer Au(111) slab and assigns velocities at 300 K. The numerical gates are 10⁻⁸ eV/atom for energy and changes in NVE total energy, 10⁻⁶ eV/Å for force components, 10⁻⁶ Å for positions and 10⁻⁷ Å/fs for velocities. Validation runs before timing. [Fixture generation](../examples/make_fixture.py) · [Interface generation](../examples/make_interface.py)

```bash
run_md() {
  local name="$1" checkpoint="$2" input="$3" steps="$4" replay="$5"
  shift 5
  mace-md-adjoint --model "$checkpoint" --structure "$input" --device cuda \
    --dtype float64 --dt-fs 0.5 --skin 0.12 --certify-edges --warmup 3 \
    --repeats 5 --steps "$steps" --replay "$replay" --output "runs/$name.json" "$@"
}
run_md ab-cu32 "$small" "$cu32" 20 padded
run_md ab-cu108 "$small" "$cu108" 20 padded
run_md ab-cross "$small" "$cross" 12 padded
run_md ab-interface "$small" "$interface" 16 padded
run_md medium-cu32 "$medium" "$cu32" 12 padded
run_md medium-interface "$medium" "$interface" 12 padded
run_md exact-cu32 "$small" "$cu32" 20 exact
run_md eager-interface "$small" "$interface" 16 eager
run_md schedule-cu108 "$small" "$cu108" 20 padded --modes none both both_deferred
run_md schedule-interface "$small" "$interface" 16 padded --modes none both both_deferred

mace-md-force-checks --model "$small" --structure "$interface" --device cuda \
  --dtype float64 --backend padded --adjoint both --certify-edges --skin 0.12 \
  --steps 48 --dt-fs 0.5 --fd-h 0.003 0.001 0.0003 0.0001 --output runs/fd-interface.json
python -m mace_md_lab.interface_scan --model "$small" --structure "$interface" \
  --device cuda --dtype float64 --backend padded --adjoint both --skin 0.12 \
  --fd-h 0.001 --q -0.2 -0.1 0 0.1 0.2 --output runs/interface-scan.json
mace-md-reverse-profile --model "$small" --structure "$cu108" --device cuda \
  --dtype float64 --adjoint both --frames 3 --repeats 3 --output runs/reverse-cu108.json
mace-md-reverse-profile --model "$small" --structure "$interface" --device cuda \
  --dtype float64 --adjoint both --frames 1 --repeats 5 --output runs/reverse-interface.json
```

For the strong baseline, the following setup selects the published V100 wheel and the original four-element export. The three benchmark inputs contain only the listed elements. The recorded compact JSON has SHA-256 `80c2ee186373a08e538018a44127782d3f960a3d3520841a391abe4fc89c3a42`. The comparison records both the source and converted hashes, backend selection, actual spline count, stress output, trajectory checks and raw trial times.

```bash
python -m pip install symmetrix-xl==0.2.0 symmetrix-xl-cuda12-sm70==0.2.0
symmetrix_extract_mace --model "$small" --chemical-symbols Cu Au O H \
  --num-spline-points 8192 --output runs/compact-small.json
export SYMMETRIX_JIT_CACHE="$(mktemp -d)"
strong() {
  python scripts/benchmark_backend.py --model "$small" --symmetrix-json runs/compact-small.json \
    --symmetrix-selector cuda12-sm70 --structure "$2" --dtype float64 --dt-fs 0.5 \
    --skin 0.12 --warmup 3 --steps 12 --repeats 3 --output "runs/strong-$1.json"
}
strong cu32 "$cu32"
strong cu108 "$cu108"
strong interface "$interface"
```
