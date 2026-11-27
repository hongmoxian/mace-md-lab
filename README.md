# Liquid copper trajectories from accelerated MACE

These archives contain the unwrapped trajectories for self-diffusion in liquid Cu with MACE-MP-0 Small. The optimized MACE driver and conservative force kernels run on the Symmetrix-XL compact export. All trajectories use one V100 GPU each, a 2 fs integration step and a coordinate sampling interval of 0.05 ps.

| Archive | Production runs |
| --- | --- |
| liquid-cu-continuous-1ns.tar.gz | One continuous 1 ns run at 1520 K with 108 atoms |
| liquid-cu-independent-replicas.tar.gz | Three independent 100 ps runs at each of 1420, 1520 and 1620 K with 108 atoms, plus three 50 ps runs at 1520 K with 256 atoms |

Check the downloaded archive with `sha256sum -c SHA256SUMS` (both archives in the same directory). Extract each archive into its own directory. Each run contains `start.extxyz`, `final.extxyz`, `result.json` and `samples.npz`. The NPZ contains unwrapped positions in Å (`xyz`), time in ps (`t_ps`), temperature in K, total energy per atom in eV, the configurational stress in ASE Voigt order (eV/Å³), the cell in Å and an FCC order parameter. The analysis folder also contains checksums for individual NPZ files.

```python
import numpy as np
with np.load("T1520-N108-r3/samples.npz") as trajectory:
    print(trajectory["xyz"].shape, trajectory["t_ps"][-1])
```

[Reproduce the simulations and analysis](https://github.com/hongmoxian/mace-md-lab/blob/main/benchmarks/liquid-copper.md)
