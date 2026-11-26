# MACE MD Lab

Compare conservative MD implementations in three separately trained architectures using MACE, PyG SchNet, and SchNetPack PaiNN. The same Cartesian edge derivative and CUDA force replay can be enabled independently in all three. The frozen scalar product optimization below is specific to compatible MACE blocks. [Cross-framework results and reproduction](benchmarks/portable.md)

Run a fixed-cell molecular dynamics benchmark with an existing MACE checkpoint, or measure force disagreement across several checkpoints on the same frames. The package compares every optimized path with the original member energies and forces. It writes timings, numerical errors, and neighbor-list rebuild counts to JSON.

**Inputs** are trusted local MACE or ScaleShiftMACE checkpoints and ASE-readable periodic structures. **Outputs** include wall time per MD step, milliseconds per committee observation, per-member force errors, and the maximum atomic force disagreement on each sampled frame. The initial release targets scalar-energy MACE 0.3.16 on CPU or one GPU, with optional CUDA Graph replay.

## Measure periodic transfer and GPU capacity

[DPA4 SO(2) message reverse](benchmarks/dpa4-message-reverse.md) gives the hotspot profile, same-driver MD timings and per-block gradient checks. [DPA4 forward replay](benchmarks/dpa4-forward-replay.md) compares forward-only and full energy/force replay, including a matched stock-kernel replay control.

[SevenNet message-passing reverse](benchmarks/sevennet-message-reverse.md) gives the five-layer profile, the edge adjoint and matched full-step measurements.

Run comparable short trajectories with the published SevenNet-0 or DPA4-Mini weights. Inspect per-trial speed and same-frame errors, or measure complete-process GPU peaks while increasing the periodic Au/water cell. [Results and benchmark commands](benchmarks/periodic-transfer-memory.md)

## Benchmark a native force algorithm

Compare two conservative-force implementations inside the same prepared Symmetrix-XL evaluator and fixed-cell integrator. The runner reports per-trial full-step timings, numerical checks against XL and MACE, and finite differences of the candidate energy. [Algorithm-only results and reproduction](benchmarks/native-algorithms.md)

## Reproduce the full-step acceptance tests

Run the frozen-checkpoint MACE implementation and the prepared Symmetrix-XL driver on the same moving structures. Both return the energy, conservative forces and stress to the host at every step. The report compares against XL with its upstream generated kernels enabled and includes separate tests of real SchNet and PaiNN models. [Protocol, results and commands](benchmarks/acceptance.md)

## Benchmark conservative force derivatives

Use an existing frozen MACE checkpoint with the same weights and precision as the native reference. The GPU implementations use Triton for the angular derivative and CuPy/NVRTC for scalar product layers. For CUDA 12 on Linux, install PyTorch for the device and run `python -m pip install -e ".[cuda12]"`.

```bash
python examples/make_interface.py --output fixtures-interface/au-water.extxyz
mace-md-adjoint --model mace-mp-0-small.model --structure fixtures-interface/au-water.extxyz \
  --device cuda --dtype float64 --certify-edges --replay padded --skin 0.12 \
  --steps 16 --warmup 3 --repeats 5 --output runs/adjoint.json
```

The report compares `none`, `angular`, `product`, and `both` with an identical replay policy and reports the separate ASE energy/force control. Use the transforms for first energy derivatives in fixed-weight inference. A local Python trajectory can select the same operators with `with optimized_model(calc.models[0], "both", atoms.numbers):` around creation and use of `ResidentMACE`. Import `optimized_model` from `mace_md_lab.transforms`. Operators support component-normalized spherical harmonics through degree three and scalar-output product layers through correlation order three. Compatible scalar blocks can be optimized in a model whose other blocks retain vector outputs. [Validation, data and commands](benchmarks/adjoints.md) · [Scalar product derivation](docs/frozen-product.md)

## Run a short benchmark

Install PyTorch for your hardware and install this repository from its root.

```bash
python -m pip install -e .
python examples/make_fixture.py --out fixtures

mace-md-bench --model fixtures/member-17.model --structure fixtures/cu.extxyz \
  --device cpu --steps 5 --warmup 1 --repeats 1 --certify-edges --output runs/md.json
```

The example generates eight frames of a moving 32-atom Cu crystal using ASE/EMT and three independently initialized, untrained MACE networks. It requires no downloads or training. Use a trained checkpoint for material simulations. Use a trusted local checkpoint because MACE 0.3.16 loads serialized Python models.

## Compare CUDA replay on moving atoms

Use the same checkpoint and geometry as the CPU example. Both replay modes keep the model weights, precision and autograd forces fixed.

```bash
mace-md-bench --model fixtures/member-17.model --structure fixtures/cu.extxyz \
  --device cuda --dtype float64 --skin 0.12 --dt-fs 0.5 \
  --steps 20 --warmup 3 --repeats 3 --certify-edges --capture-forces --pad-capture \
  --output runs/replay.json
```

**Choose the path from its measured edge-count behavior.** On a V100 with the trained MACE-MP-0-small checkpoint in FP64, a moving 32-atom Cu cell with stable edge count took 6.34 ms/step with exact-size replay and 6.64 ms/step with padded replay. With a compressed Cu cell, the physical edge count changed during the timing window. Exact-size replay recaptured twice in 12 steps and took 16.82 ms/step. Padded replay used its existing capture for all 12 steps and took 7.13 ms/step. These are medians of three short runs. The ordinary and compressed inputs are separate examples. [Stable count](benchmarks/raw/padded-32.json) · [Changing count](benchmarks/raw/padded-crossing.json)

Each run includes native ASE, an ASE energy-and-force control, and a resident eager control. The flags add comparison paths. `--certify-edges` enables exact topology reuse under a displacement bound. `--capture-forces` records the MACE forward and force gradient for the current number of physical edges. `--pad-capture` captures the same calculation with a second, disconnected padding graph to accommodate changes in edge count. `--rebuild-ablation` adds a resident control that rebuilds neighbors every step. Read `median_ms_step` together with `setup_and_first_force_ms` and the timed recapture counts. [Runtime and timing definitions](docs/runtime.md)

## Advance one trajectory

After comparing the backends on your structure, use the Python runner directly. This example advances the input velocities for 20 steps of 0.5 fs and writes the final coordinates and momenta as a restart frame.

```python
from ase.io import read, write
from mace_md_lab.runtime import ResidentMACE
from mace.calculators import MACECalculator

atoms = read("fixtures/cu.extxyz", index=0)
calc = MACECalculator(model_paths="fixtures/member-17.model", device="cuda",
                      default_dtype="float64")
md = ResidentMACE(atoms, calc, skin=0.12, certify_edges=True,
                  capture_forces=True, pad_capture=True)
for _ in range(20):
    energy, forces = md.step(dt_fs=0.5)
atoms.set_positions(md.positions.cpu().numpy())
atoms.set_velocities(md.velocities.cpu().numpy())
write("final.extxyz", atoms)
```

## Measure a committee

```bash
mace-md-committee --models fixtures/member-17.model fixtures/member-29.model fixtures/member-43.model \
  --trajectory fixtures/cu.extxyz --geometry require --limit 8 \
  --device cpu --warmup 1 --repeats 1 --check-driver-steps 4 --observe-every 2 \
  --output runs/committee.json
```

Each model has its own weights and energy gradient. The committee benchmark replays the same trajectory through the upstream multi-model calculator, an energy-and-forces loop with shared topology, and a version that additionally shares compatible spherical harmonics and fixed radial features. The latter two have identical requested observables. Use `--geometry auto` for an explicit topology-only fallback when feature sharing is incompatible. Input frames retain their element order, cell and static model inputs.

For atom `i`, the reported disagreement is the population RMS length of the Cartesian force deviation across members. `native_max_force_disagreement_eV_A` is the maximum over atoms. The optional short online check propagates member 0 and evaluates the remaining weights at observation steps. Changing the observation interval changes when disagreement is sampled.

## Locate the next bottleneck

```bash
mace-md-profile --model fixtures/member-17.model --structure fixtures/cu.extxyz \
  --device cpu --frames 3 --output runs/profile.json
python -m unittest discover -s tests -v
```

The profiler separates the energy forward pass from the position gradient. It times radial features, spherical harmonics, interaction blocks and product blocks inside the forward pass. The report includes hooked and unhooked totals. Use the MD benchmark to measure how much an operator change actually saves per step.

[Measured results and reproduction commands](benchmarks/README.md) · [Papers, existing software and next experiments](docs/literature.md)

## Scope

The trajectory runner implements unconstrained, fully periodic, fixed-cell velocity Verlet with positions and velocities resident on the integration device. It supports a single scalar head selection per run, one fixed elemental encoding, and energies and conservative forces. Model weights and requested precision are the same in each comparison. The official calculator remains the reference for observables. The periodic acceptance driver additionally supports the fixed-cell global stress for its qualified checkpoint. Barostats, enhanced sampling, long-range polar models and custom constant-potential or magnetic-MACE subclasses require separate validation. Existing biased trajectories can be replayed for committee diagnostics using a compatible energy model.

Neighbor candidates include periodic images out to the physical cutoff plus a skin. Atoms can move across cell boundaries. The optional certificate tracks both the neighbor-list reference and the last exactly filtered edge graph. The host rebuilds a candidate list when necessary. A displacement check transfers a scalar on GPU. Regression tests compare the physical image tuples against fresh upstream MACE graphs as atoms cross the cutoff in skew and small periodic cells.
