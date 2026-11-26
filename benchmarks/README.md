# V100 benchmarks

[SevenNet message-passing reverse](sevennet-message-reverse.md) reports convolution reverse shares and paired full-step savings with a fixed driver.

[Periodic DPA4 and SevenNet transfer plus GPU memory scaling](periodic-transfer-memory.md) compares each checkpoint with its own reference and measures fresh-process peak memory across lateral Au/water replications.

[Direct comparisons with original MACE and CHGNet](direct-comparison.md) report wall time per full MD step and per evaluation on identical moving frames for Au/water and Cu108.

[Native force algorithm comparisons](native-algorithms.md) hold the prepared driver, integration, observables and graph maintenance fixed.

[Current acceptance results and reproduction](acceptance.md) cover the independent analytic tangent and its comparison with the combined native XL route.

MACE-MP-0 Small ran at **6.34 ms/step for 32 Cu atoms** and **14.27 ms/step for 108 Cu atoms** with exact-size CUDA Graph replay. Both trajectories kept a constant number of active edges during timing. A separate 32-atom trajectory deliberately crossed the 6 Å cutoff. On that trajectory, reserving edge capacity reduced time from **16.82 to 7.13 ms/step**, with identical physical edges supplied to the model.

## Matched MD measurements

Entries are median wall-clock milliseconds per velocity-Verlet step over three trials. Every entry in a column was measured in the same benchmark invocation, with the same weights, initial coordinates, velocities and FP64 precision. The official ASE path requests stress in MACE 0.3.16. The ASE E/F control turns off stress while retaining upstream batching and host outputs. The resident paths return only energy and forces and keep the integrator state on the GPU.

| Backend | Cu 32 | Cu 108 | Cu 32, cutoff crossings |
|---|---:|---:|---:|
| Official ASE | 19.10 | 28.16 | 19.00 |
| ASE E/F control | 17.69 | 25.98 | 17.63 |
| Resident, cached neighbor candidates | 13.52 | 15.78 | 13.37 |
| Resident, with cutoff-gap certificate | 13.40 | 15.62 | 13.61 |
| CUDA Graph, exact active edge count | **6.34** | **14.27** | 16.82 |
| CUDA Graph, isolated padding | 6.64 | 15.37 | **7.13** |

The two CUDA Graph paths use the same cutoff-gap certificate. During each crossing trial, both replayed 12 times. Exact-size replay captured twice inside the timed interval, and padded replay reused one graph throughout. Certificate hits were zero in the crossing validation, so reducing graph recapture explains the improvement. With stable edge counts, padding added approximately 5% per step at 32 atoms and 8% at 108 atoms. All three padded runs recorded zero eager fallbacks. The median setup plus first-force costs for exact/padded replay were 72.3/73.6 ms, 103.3/105.2 ms and 73.9/73.7 ms in table order. Setup and three warmup steps are excluded from the displayed ms/step. [32-atom raw data](raw/padded-32.json) · [108-atom raw data](raw/padded-108.json) · [Crossing raw data](raw/padded-crossing.json)

### Numerical checks

The benchmark compares both initial and final reference configurations at identical coordinates. It also checks position and velocity on independently integrated trajectories and compares the per-atom change in NVE total energy every step. Values below describe **padded replay** relative to official ASE. Maximum excursion is measured from the initial NVE total energy of the padded trajectory.

| Case | Maximum ΔE on identical frames (eV/atom) | Maximum force-component error (eV/Å) | Maximum position-component error (Å) | Maximum NVE excursion (meV/atom) |
|---|---:|---:|---:|---:|
| Cu 32 | 0 | 5.55 × 10⁻¹⁶ | 2.22 × 10⁻¹⁶ | 0.000336 |
| Cu 108 | 5.26 × 10⁻¹⁶ | 6.66 × 10⁻¹⁶ | 8.88 × 10⁻¹⁶ | 0.000169 |
| Cu 32, cutoff crossings | 0 | 6.66 × 10⁻¹⁶ | 8.67 × 10⁻¹⁸ | 0.000526 |

The largest sampled velocity-component discrepancy was 6.94 × 10⁻¹⁸ Å/fs. The per-step NVE change agreed within 8.89 × 10⁻¹⁶ eV/atom across the three cases. The validation windows cover 6–10 fs.

## Reproduce the comparison

The measurements used one Tesla V100-SXM2 16 GB, CUDA 12.6, PyTorch 2.13.0+cu126, MACE 0.3.16, ASE 3.29.0, matscipy 1.2.0 and e3nn 0.4.4 with one CPU thread. The checkpoint was `mace-mp-0-small.model`, SHA-256 `2ddb079cee0e131eaaf6912ba581b394551ead283e95c99cfe78c605d10b5736`. Supply this trusted checkpoint locally. The runs use the native e3nn backend. Each input trajectory contains eight frames from a perturbed FCC crystal propagated with ASE/EMT. The NVE benchmarks start from frame 0 and its stored momenta, at 0.5 fs and with a 0.12 Å neighbor skin.

Run from the repository root in a fresh working directory for the fixtures. The script creates three untrained fixture networks solely for the committee experiment. All single-model benchmarks use the pretrained checkpoint supplied through `MODEL`.

```bash
export MODEL="$PWD/mace-mp-0-small.model"
curl -fL -o "$MODEL" https://github.com/ACEsuit/mace-foundations/releases/download/mace_mp_0/2023-12-10-mace-128-L0_energy_epoch-249.model
echo "2ddb079cee0e131eaaf6912ba581b394551ead283e95c99cfe78c605d10b5736  $MODEL" | sha256sum -c -
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m pip install -e .
python examples/make_fixture.py --out fixtures --repeat 2
python examples/make_fixture.py --out fixtures108 --repeat 3 --frames-only
python examples/make_fixture.py --out fixtures-crossing --repeat 2 --lattice-a 3.464101615 --frames-only
mkdir -p runs

bench() {
  mace-md-bench --model "$MODEL" --structure "$1" --output "$2" --steps "$3" \
    --device cuda --dtype float64 --dt-fs 0.5 --skin 0.12 --warmup 3 --repeats 3 \
    --certify-edges --capture-forces --pad-capture
}
bench fixtures/cu.extxyz runs/padded-32.json 20
bench fixtures108/cu.extxyz runs/padded-108.json 20
bench fixtures-crossing/cu.extxyz runs/padded-crossing.json 12
```

## Locate time within a force call

On three distinct frames per size, the mean **unhooked** forward and position-gradient times were 5.82 and 7.37 ms at 32 atoms, and 6.67 and 8.84 ms at 108 atoms. At 108 atoms the instrumented forward spent 3.39 ms across its two interaction blocks, 2.03 ms across the product blocks and 0.84 ms in spherical harmonics plus radial embedding. The backward pass accounted for 57% of unhooked force-evaluation time. CUDA events include launch gaps on the active stream. These measurements use prepared graphs and exclude the neighbor-list, transfers and integrator. The profile measures eager evaluation. [32-atom operator timings](raw/profile-mp0.json) · [108-atom operator timings](raw/profile-mp0-108.json)

```bash
mace-md-profile --model "$MODEL" --structure fixtures/cu.extxyz --device cuda \
  --dtype float64 --frames 3 --output runs/profile-32.json
mace-md-profile --model "$MODEL" --structure fixtures108/cu.extxyz --device cuda \
  --dtype float64 --frames 3 --output runs/profile-108.json
```

## Independent-checkpoint committee

The three untrained 32-channel MACE fixtures use different initialization seeds 17, 29 and 43. On the same eight 32-atom frames, the official calculator took **59.91 ms/observation**, the cached-topology E/F path took **51.42 ms**, and sharing compatible geometry features took **49.72 ms**. The last comparison isolates the added benefit of geometry sharing, **1.034×**. Each path returned all member forces and computed the same disagreement statistic on the host. The largest shared-feature force-component error was 1.57 × 10⁻¹⁶ eV/Å and the largest discrepancy in atomic force disagreement was 5.56 × 10⁻¹⁷ eV/Å on the replayed frames. A four-step online driver check also passed. The fixture weights test numerical equivalence and execution cost. Physical force accuracy requires trained members. [Committee raw data](raw/committee-cu.json)

```bash
mace-md-committee --models fixtures/member-17.model fixtures/member-29.model fixtures/member-43.model \
  --trajectory fixtures/cu.extxyz --geometry require --limit 8 --device cuda --dtype float64 \
  --certify-edges --warmup 1 --repeats 3 --check-driver-steps 4 --observe-every 2 \
  --output runs/committee.json
```

The longest timed MD segment was 20 steps. The committee replayed eight frames per pass, and each operator profile used three frames. The experiments required no model training or DFT.
