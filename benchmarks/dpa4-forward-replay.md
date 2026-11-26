# Forward and conservative force replay for DPA4-Mini

These experiments keep the official DeepPot neighbor builder, the same frozen DPA4-Mini-OMat24 weights and the same fixed-cell 0.5 fs Verlet driver. Every measured step updates positions and returns energy, all forces and stress. One Tesla V100 16 GB runs FP32 model operations with the upstream energy accumulation precision. The stock baseline uses DeePMD-kit 3.2.0 and PyTorch 2.12 eager. Its optional compilation and inference backends were not benchmarked.

## What still costs time after the edge-adjoint optimization?

CUDA stream events bracket the built-in DeePMD stages with the previous [local SO(2) implementation](dpa4-message-reverse.md) enabled. The two interaction blocks are nested inside the descriptor. The force stage covers differentiation of the full energy and virial assembly. The four columns below are medians over four trials of four moving steps, in ms/step.

| System | Build neighbors | Descriptor forward (of which two blocks) | Force and virial | Full step, uninstrumented |
|---|---:|---:|---:|---:|
| [Au/water (192)](raw/dpa-front-profile-au.json) | 1.82 | 13.10 (8.55) | 13.68 | 31.80 |
| [Cu (108)](raw/dpa-front-profile-cu.json) | 1.90 | 13.01 (8.43) | 13.73 | 31.92 |

## Same-driver timing

Stock is the original eager model. Local is our earlier eager SO(2) forward/reverse kernel. Front replays only the energy forward of Local and performs the force VJP eagerly. Stock graph and Joint both replay the energy/force/virial core, using stock and optimized operators respectively. Each route uses its own identically loaded model. The comparison rotates route order across five paired trials. Primary times eight moving steps/trial; the independent confirmation reverses the starting order and times 18 steps/trial. Three untimed warmups precede each timed window. Graph initializations, dispatcher counters, per-trial captures and synchronized raw times are recorded separately.

| System | Run | Stock | Local | Front | Stock graph | Joint | Stock/Joint |
|---|---|---:|---:|---:|---:|---:|---:|
| Au/water (192) | [primary](raw/dpa-front-primary-au.json) | 35.50 | 31.21 | 17.60 | 21.25 | 15.56 | 2.28 |
| Au/water (192) | [confirm](raw/dpa-front-confirm-au.json) | 35.76 | 31.69 | 17.56 | 21.31 | 15.60 | 2.29 |
| Cu (108) | [primary](raw/dpa-front-primary-cu.json) | 34.88 | 31.14 | 17.82 | 17.00 | 13.01 | 2.68 |
| Cu (108) | [confirm](raw/dpa-front-confirm-cu.json) | 35.02 | 31.27 | 17.99 | 16.98 | 13.00 | 2.69 |

Au/water (192): forward-only replay saved 43.6–44.6% versus Local. With equivalent full-core replay on both routes, Joint saved 26.8% in both runs versus Stock graph and won 10/10 paired windows. Joint recaptures during those windows: 0.

Cu (108): forward-only replay saved 42.5–42.8% versus Local. With equivalent full-core replay on both routes, Joint saved 23.4–23.5% versus Stock graph and won 10/10 paired windows. Joint recaptures during those windows: 0.

A further independently launched eight-step [Cu repeat](raw/dpa-front-repeat-cu.json) is retained. Cold timings in the JSON files include runner construction and 12 full steps (13 evaluations), after model load and with process-level library and Triton caches shared among routes. Route order affects first-use compilation. When Joint was the first route to use the custom kernels, first capture/warmup took 1.94 s on the interface and 2.05 s on Cu. In that order, initialization plus 12 steps took 2.14 and 2.22 s for Joint compared with 0.46 and 0.45 s for warmed-framework stock. After Local had already used those kernels, Joint capture took about 0.07 s. These startup measurements are order-dependent.

## Dynamic topology and checks

The GPU graph records an execution schedule while the physical edge membership can change. Each call first runs the original neighbor builder. It copies current coordinates, types, edge indices, displacements, scatter indices and validity masks into a bounded static buffer. Inactive slots use a false mask and a finite dummy vector. The initial edge capacity rounds up to a multiple of 64 after adding the greater of 64 edges and 2% headroom (floored). On overflow it replaces the one live graph. Reuse assumes a frozen single-frame energy model and serial execution. The forward-only control reuses the captured energy tape and calls DeePMD's own eager edge derivative on newly computed activations. Full replay captures both phases. Returned tensors have independent storage.

A shape-management ablation on the moving interface rounded each step's edge count independently. In [five paired eight-step windows](raw/dpa-front-aligned-au.json), each graph route recaptured four times per window. Joint took 52.65 ms/step versus 30.56 for Local. This is the reason the growing-capacity implementation is used for the primary comparisons.

All graph routes were compared to the official ASE calculator on five moving frames, two seeded position perturbations, one periodic translation of an atom, and a return to the first frame. Across both primary and confirmation experiments, maximum absolute component differences over graph routes were 6.3e-08 eV/atom, 1.8e-05 eV/Å and 9.8e-09 eV/Å³. Acceptance limits were 2e-5, 2e-4 and 2e-5 in those units.

## Reproduce

Use the environment, public checkpoint and the two fixtures in the [SO(2) baseline](dpa4-message-reverse.md#reproduce). Use one CPU thread and a single CUDA device. Choose a fresh process for each command and fixture.

```bash
python scripts/profile_dpa4_stages.py --model "$checkpoint" --structure "$fixture" --output stages.json
python scripts/benchmark_dpa4_forward.py --model "$checkpoint" --structure "$fixture" --routes stock local front joint stockgraph --repeats 5 --steps 8 --output primary.json
python scripts/benchmark_dpa4_forward.py --model "$checkpoint" --structure "$fixture" --routes stockgraph joint front local stock --repeats 5 --steps 18 --output confirm.json
python scripts/benchmark_dpa4_forward.py --model "$checkpoint" --structure "$fixture" --bucket-policy aligned --routes stock local front joint stockgraph --repeats 5 --steps 8 --output aligned-control.json
```
