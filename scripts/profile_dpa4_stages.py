"""Time sparse DPA4 stages on the current CUDA stream in complete moving MD steps."""
import argparse
from collections import defaultdict
from contextlib import contextmanager
from importlib import import_module
import json
from pathlib import Path
from statistics import median
from unittest.mock import patch

from ase.io import read
from deepmd.calculator import DP
from deepmd.pt.model.descriptor.sezm_nn.so2 import SO2Convolution
import torch

from benchmark_periodic_transfer import deep_eval
from profile_dpa4_reverse import event, timed
from mace_md_lab.portable.dpa4_reverse import use_adjoint
from mace_md_lab.xl_fixed_step import ArrayVerlet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "structure", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    a = parser.parse_args()
    atoms = read(a.structure)
    calc = DP(model=str(a.model))
    blocks = [(n, m) for n, m in calc.dp.deep_eval.dp.named_modules() if isinstance(m, SO2Convolution)]
    model_source = import_module("deepmd.pt.model.model.sezm_model")
    descriptor_source = import_module("deepmd.pt.model.descriptor.sezm")
    samples = defaultdict(list)

    @contextmanager
    def marker(name):
        start = event()
        try:
            yield
        finally:
            samples[name].append((start, event()))

    original = model_source.edge_energy_deriv
    def derivative(*args, **kwargs):
        with marker("force_and_virial"):
            return original(*args, **kwargs)

    trials = []
    with use_adjoint(blocks, "local"):
        for i in range(4):
            row = {}
            for trace in ((False, True) if i % 2 == 0 else (True, False)):
                runner = ArrayVerlet(atoms, deep_eval(atoms, calc), .5)
                runner.advance(3)
                if not trace:
                    row["bare_ms"] = timed(runner, 4)
                    continue
                samples.clear()
                with patch.object(model_source, "nvtx_range", marker), patch.object(descriptor_source, "nvtx_range", marker), patch.object(model_source, "edge_energy_deriv", derivative):
                    row["traced_ms"] = timed(runner, 4)
                assert len(samples["force_and_virial"]) == 4
                row["ms"] = {k: sum(s.elapsed_time(e) for s, e in spans)/4 for k, spans in samples.items()}
            trials.append(row)
    med = {k: median(row["ms"][k] for row in trials) for k in trials[0]["ms"]}
    data = dict(fixture=a.structure.name, gpu=torch.cuda.get_device_name(), trials=trials, stage_medians_ms=med)
    a.output.write_text(json.dumps(data, indent=2) + "\n")
    print("STAGES", med)


if __name__ == "__main__":
    main()
