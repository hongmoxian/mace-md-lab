# Run the same MD experiment across three neural potentials

The experiment changes two operations independently in pretrained MACE, SchNet and PaiNN. One switch changes how the model differentiates edge geometry. The other records the force call in a CUDA Graph and replays it. All four combinations advance the atoms and return conservative forces from the same checkpoint within each model family.

## Models and inputs

| Framework and weights | Geometry consumed by the model | Energy readout and reference |
|---|---|---|
| MACE 0.3.16, MACE-MP-0 Small | Distances and directions at 6 Å cutoff. Stock e3nn spherical harmonics consume the unit vectors. | Scalar energy in eV. The reference uses original model code and the calculator’s atomic-data builder. |
| PyTorch Geometric 2.6.1, SchNet QM9 U0 | Distances at 10 Å cutoff. Six 128-channel continuous-filter interaction blocks. | Molecular U0 in eV with additive atomic references. The reference invokes the upstream SchNet forward. These runs supplied an exact all-pairs cutoff graph through its public graph callback. |
| SchNetPack 2.3.0, PaiNN rMD17 ethanol, member 1 | Distances and directions at 5 Å cutoff. Three interaction blocks with 128 features. | The native energy head and offsets return kcal/mol. The adapter multiplies by 0.04336410390059322 for eV. The reference uses the original representation, energy head, postprocessing and SchNetPack’s matscipy neighbor list. |

The same starting coordinates are used across frameworks. Ethanol has nine atoms and its geometry comes from SchNetPack. Benzene has twelve atoms and comes from ASE. The size control contains eight copies of the ethanol coordinates in a single 72-atom graph. Each fragment starts at least 20 Å from its neighbor along the replication axis. It tests increased edge workload with the local molecular environments held fixed. PaiNN was trained on ethanol. Benzene tests execution outside that training composition. SchNet’s QM9 energies were fitted to equilibrium molecules. Short force and integration checks measure reproduction of each model. [MACE weights](https://github.com/ACEsuit/mace-mp/releases/tag/mace_mp_0) · [PyG SchNet and official parameter mapping](https://pytorch-geometric.readthedocs.io/en/2.6.1/_modules/torch_geometric/nn/models/schnet.html) · [PaiNN weights and training configuration](https://github.com/atomistic-machine-learning/schnetpack/tree/69d26f3842a012f1f1e461ad7f1984d353843c8d/examples/trained_models/rmd17_ethanol/painn_1) · [Ethanol coordinates](../examples/structures/ethanol.xyz)

## Compare complete MD steps

All four paths use the same FP64 checkpoint, edge order, exact cutoff test and device-resident velocity-Verlet integrator within each comparison. **Native** uses PyTorch indexing, norms and ordinary autograd. **Analytic** gathers the directed edge vectors and propagates their first Cartesian derivatives to the two atoms with shared Triton kernels. The SchNet kernel requests distance alone. **Replay** captures energy and its position gradient together. The force graph recaptures when its physical edge count changes and copies new indices when the shape stays fixed. All paths check topology outside the captured graph.

The table reports median milliseconds per complete MD step on one Tesla V100-SXM2 16 GB, PyTorch 2.13.0+cu126 and CUDA 12.6. Five trials restart from the same coordinates and seeded 300 K momenta. Each trial takes three warmup steps followed by 12 timed steps of 0.25 fs. The harness rotates method order. Timing includes position and velocity updates, topology checks, any recapture, force evaluation and synchronization at the boundaries. Setup and the first force are recorded separately. The last column isolates replacing native geometry with its analytic derivative under an identical replay policy.

| Framework | Input | Native eager | Analytic eager | Native replay | Analytic replay | Step-time reduction under replay |
|---|---|---:|---:|---:|---:|---:|
| MACE | [Ethanol, 9](raw/portable-mace-ethanol.json) | 13.347 | 13.247 | 2.834 | 2.774 | 2.1% |
| MACE | [Benzene, 12](raw/portable-mace-benzene.json) | 12.808 | 12.712 | 3.047 | 2.978 | 2.3% |
| MACE | [8 × ethanol, 72](raw/portable-mace-ethanol8.json) | 12.895 | 12.793 | 5.531 | 5.461 | 1.3% |
| SchNet | [Ethanol, 9](raw/portable-schnet-ethanol.json) | 5.558 | 5.604 | 1.275 | 1.244 | 2.4% |
| SchNet | [Benzene, 12](raw/portable-schnet-benzene.json) | 5.627 | 5.684 | 1.330 | 1.291 | 2.9% |
| SchNet | [8 × ethanol, 72](raw/portable-schnet-ethanol8.json) | 5.700 | 5.777 | 1.748 | 1.708 | 2.3% |
| PaiNN | [Ethanol, 9](raw/portable-painn-ethanol.json) | 5.589 | 5.447 | 1.368 | 1.286 | 5.9% |
| PaiNN | [Benzene, 12](raw/portable-painn-benzene.json) | 5.874 | 5.730 | 1.542 | 1.467 | 4.9% |
| PaiNN | [8 × ethanol, 72](raw/portable-painn-ethanol8.json) | 5.808 | 5.680 | 1.781 | 1.697 | 4.7% |

Under replay, changing the geometry derivative reduced median step time by 1.3–5.9% across the nine cases. In eager SchNet it increased median step time on all three inputs. The largest change in each framework came from enabling replay. Every captured trial replayed all 12 measured steps with zero timed recaptures. Across all nine trajectories and four modes, the largest force-component difference from the upstream reference was 2.22e-14 eV/Å.

The molecular experiment leaves MACE’s symmetric product blocks and spherical-harmonics polynomials in their stock form. Frozen scalar product specialization, the direct spherical-harmonics derivative and disconnected padding for cutoff crossings have separate periodic MACE benchmarks. [MACE-specific ablation and external backend comparison](adjoints.md) · [Runner](../src/mace_md_lab/portable/runner.py) · [Shared derivative](../src/mace_md_lab/portable/geometry.py)

## Reproduce a row

Use Linux, a CUDA-enabled PyTorch build for your GPU and the dependencies of the model being tested. Run from the checkout root. These are the package versions used in the reports. The PyG adapter can use its exact all-pairs reference without torch-cluster.

```bash
python -m pip install -e '.[cuda12]' ase==3.29.0 matscipy==1.2.0 e3nn==0.4.4
python -m pip install torch-geometric==2.6.1    # SchNet
python -m pip install schnetpack==2.3.0         # PaiNN
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p checkpoints fixtures-molecules runs
python examples/make_molecule.py --output fixtures-molecules/benzene.extxyz
python examples/make_molecule.py --from-geometry examples/structures/ethanol.xyz \
    --copies 8 --output fixtures-molecules/ethanol8.extxyz
```

Download the checkpoints for the framework rows you want. SchNet’s conversion extracts the U0 member from the original archive and verifies its SHA-256 before reading it. We compared all 86 tensors of the converted PyG state, both normalization constants and the energy scale against PyG’s own `SchNet.from_qm9_pretrained(target=7)` mapping. PaiNN loads the published file directly.

```bash
mace_release=https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0
curl -fL "$mace_release/2023-12-10-mace-128-L0_energy_epoch-249.model" \
    -o checkpoints/mace-mp-0-small.model
python -m mace_md_lab.portable.schnet checkpoints/pyg-u0.pt
spk=https://raw.githubusercontent.com/atomistic-machine-learning/schnetpack/69d26f3842a012f1f1e461ad7f1984d353843c8d
curl -fL "$spk/examples/trained_models/rmd17_ethanol/painn_1/best_model" \
    -o checkpoints/painn-ethanol.model
sha256sum -c <<'CHECKSUMS'
2ddb079cee0e131eaaf6912ba581b394551ead283e95c99cfe78c605d10b5736  checkpoints/mace-mp-0-small.model
260f262f4021215dc65edec62c2c3128b910b413e3070024db3b21510ef30b29  checkpoints/painn-ethanol.model
CHECKSUMS
```

For example, time all four SchNet paths on benzene. Choose `--framework mace --model checkpoints/mace-mp-0-small.model` or `--framework painn --model checkpoints/painn-ethanol.model` to change the potential. Use `examples/structures/ethanol.xyz` or `fixtures-molecules/ethanol8.extxyz` for the other structures.

```bash
mace-md-portable --framework schnet --model checkpoints/pyg-u0.pt \
  --structure fixtures-molecules/benzene.extxyz --device cuda --dtype float64 \
  --dt-fs 0.25 --warmup 3 --steps 12 --repeats 5 --output runs/portable-schnet-benzene.json
```

Read `passed`, `validation`, `trials` and `median_ms_step` in the report. Validation compares energy, all Cartesian force components, positions, velocities and changes in total NVE energy with the independently propagated upstream trajectory. A centered difference of upstream energies at two displacement sizes tests the force projection at the shared starting geometry. Each trial also records setup time, physical edge count, timed refreshes, recaptures and replays. [Benchmark and tolerances](../src/mace_md_lab/portable/benchmark.py)
