# Fixed-cell NVE and CUDA Graph replay

The runner advances one fully periodic structure using velocity Verlet. It keeps unwrapped positions, velocities and masses on the integration device in FP64. Model inputs and the autograd calculation use the requested checkpoint precision. Native ASE and the resident paths use the same weights, model head, initial positions and velocities.

## Run the comparisons

From the repository root, install the package and generate the fixtures as described in the README. This command measures every single-model path. A local trained checkpoint can replace the illustrative fixture checkpoint.

```bash
mace-md-bench --model fixtures/member-17.model --structure fixtures/cu.extxyz \
  --device cuda --dtype float64 --dt-fs 0.5 --skin 0.12 \
  --steps 20 --warmup 3 --repeats 3 --rebuild-ablation --certify-edges \
  --capture-forces --pad-capture --output runs/all-paths.json
```

| JSON path | CLI selection | Operation |
|---|---|---|
| `ase_native` | Default | Native MACECalculator, including its usual stress evaluation. |
| `ase_ef` | Default | The same native calculator with stress disabled. |
| `resident_cached` | Default | Velocity Verlet on device, a Verlet candidate list, and exact-cutoff filtering at every force evaluation. |
| `resident_rebuild` | `--rebuild-ablation` | Same resident E/F driver with fresh upstream neighbor construction at every force evaluation. |
| `resident_certified` | `--certify-edges` | Reuses physical indices whenever both displacement bounds pass. |
| `resident_capture` | `--capture-forces` | Replays one CUDA Graph for the current exact edge count. |
| `resident_padded` | `--pad-capture` | Replays one fixed-capacity CUDA Graph for the real structure and its disconnected padding graph. |

The two capture flags select their comparisons independently. Supplying both runs a direct raw-versus-padded comparison. Both use the same topology policy as `resident_certified` when `--certify-edges` is set, otherwise `resident_cached`. CUDA Graphs record the unchanged MACE energy forward and autograd position derivative. The integration updates and topology checks run outside the recording.

## Exact topology under motion

At a candidate-list rebuild, the upstream neighbor builder enumerates all directed periodic images inside the model cutoff plus the skin. If every atom subsequently travels less than half the skin, an omitted pair cannot enter the physical cutoff. The runner recomputes edge distances and applies the true physical cutoff when the exact list needs refreshing.

For optional exact-edge reuse, the runner also stores the smallest absolute difference between a candidate distance and the physical cutoff at the **last exact refresh**. Twice the maximum displacement from that refresh bounds the change in any pair distance. If this bound is strictly smaller than the gap, and the candidate-list skin condition still holds, all previous edge memberships remain valid. A floating-point guard narrows the gap. A pair exactly at cutoff forces a fresh selection. The validity check on GPU synchronizes one scalar. A candidate rebuild moves the coordinates to the host.

## What changes at replay

For exact-size replay, each change in the number of physical edges triggers a new recording. Changing an edge's endpoints or lattice image at fixed count updates the index and shift buffers and keeps the recording. This is inexpensive for a crystal with steady coordination.

For padded replay, the runner reserves roughly 6.25% additional edge slots, rounded up to a multiple of eight, with a minimum capacity of eight. Unused slots belong to a separate graph containing one artificial atom. Each spare edge connects this atom to its own shifted image. The runner uses upstream MACE padding and returns graph 0 energy and the real atoms' forces. It overwrites the spare tail whenever the physical topology changes. Only exceeding capacity triggers recapture. Padded replay supports standard MACE or ScaleShiftMACE inputs without extended joint embeddings.

Extra edges increase the cost of each warm step. Recaptures can cost more than those extra edges when coordination changes frequently. Use measurements from the same checkpoint, initial structure, timestep and trajectory length to choose the faster path. The shipped V100 data include both behaviors. [Moving 32-atom Cu](../benchmarks/raw/padded-32.json) · [Compressed Cu](../benchmarks/raw/padded-crossing.json) · [108-atom Cu](../benchmarks/raw/padded-108.json)

## Read the JSON

`validation` uses native ASE as the reference. It checks E/F on the identical initial and final reference geometries, samples the integrated positions and velocities, and compares the per-step total-energy excursions. Acceptance thresholds for the requested dtype appear in the report.

Each timing repeat starts from the same initial state. `setup_and_first_force_ms` measures path construction and its first force calculation with the model already loaded. For CUDA Graph paths this includes initial capture. Each path then advances through the requested warm-up. The timed block advances the requested number of moving-atom steps and synchronizes CUDA at both endpoints. It includes neighbor construction, integration, topology refreshes and recaptures that occur within that block. Validation, trajectory snapshots and output writes are separate. Initialization and per-step logging affect short production trajectories and can be measured separately for the intended workload.

| Field | Interpretation |
|---|---|
| `median_ms_step` | Median of repeat wall times divided by timed steps, in milliseconds per full MD step. |
| `counts_before_timed_steps` | Cumulative counters after startup and warm-up. |
| `neighbor_rebuilds`, `edge_refreshes`, `certificate_hits` | Counts within the timed block for candidate builds, exact physical graph filtering and certified reuse. |
| `captures`, `replays`, `fallback_evaluations` | Counts during timed steps. A nonzero capture count is a **timed recapture**. Successful replay comparisons require actual replay and zero fallback evaluations. |
| `final_capacity_edges` | Available edge slots in the final recording, including any padding capacity. |
| `speedup_*_vs` | Ratio of baseline median step time to candidate median step time. For matched requested observables, read the comparison to `ase_ef`. To isolate capture, read the comparison to the matched resident eager path. |

Committee observations and operator profiling use separate entry points and timing units. [Committee and profiling examples](../README.md) · [Benchmark conditions and scripts](../benchmarks/README.md)
