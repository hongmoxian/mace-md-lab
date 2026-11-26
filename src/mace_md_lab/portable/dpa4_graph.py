"""Bounded CUDA graph replay for serial, frozen DPA4 energy/force/virial evaluation."""
from collections import Counter
from contextlib import contextmanager
from time import perf_counter
from unittest.mock import patch

import torch
from deepmd.pt.model.model.transform_output import edge_energy_deriv
from deepmd.pt.utils import env


class _CoreGraph:
    def __init__(self, original, values, options, forward_only):
        self.forward_only = forward_only
        self.inputs = tuple(torch.empty_like(x) for x in values)
        self.optional = {k: torch.empty_like(x) if isinstance(x, torch.Tensor) else x for k, x in options.items()}
        self.copy_inputs(values, options)
        self.graph = torch.cuda.CUDAGraph()

        def compute():
            # Only staging inputs change between calls. The captured copies feeding
            # the eager derivative keep stable addresses and autograd versions.
            w = tuple(x.clone() for x in self.inputs)
            kwargs = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in self.optional.items()}
            if not forward_only:
                return original(*w, **kwargs)
            kwargs["atomic_output_only"] = True
            leaf = w[3].requires_grad_()
            out = original(*w, **kwargs)
            energy = out["energy"].sum(dim=1, dtype=env.GLOBAL_PT_ENER_FLOAT_PRECISION)
            return out, energy, leaf, w[4], w[5]

        stream = torch.cuda.Stream(device=values[0].device)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.enable_grad():
            compute()
        torch.cuda.current_stream().wait_stream(stream)
        with torch.enable_grad(), torch.cuda.graph(self.graph, stream=stream):
            self.out = compute()

    def copy_inputs(self, values, options):
        for dest, source in zip(self.inputs, values, strict=True):
            dest.copy_(source.detach())
        for k, source in options.items():
            if isinstance(source, torch.Tensor):
                self.optional[k].copy_(source.detach())

    def run(self, values, options):
        self.copy_inputs(values, options)
        self.graph.replay()
        if self.forward_only:
            out, energy, leaf, scat, mask = self.out
            nf, nall = values[0].shape[:2]
            with torch.enable_grad():
                f, av, vir, _ = edge_energy_deriv(energy, leaf, scat[0], scat[1], mask, nf, nall, False)
            result = dict(energy=out["energy"], energy_redu=energy, energy_derv_r=f,
                          energy_derv_c=av, energy_derv_c_redu=vir, mask=out["mask"])
        else:
            result = self.out
        # Return independent storage to callers that retain outputs across steps.
        return {k: v.detach().clone() for k, v in result.items()}


def _pad_edges(values, capacity):
    coord, atype, index, vec, scat, mask = values
    n = len(vec)
    pad = capacity - n
    if not pad:
        return values
    # Inactive slots use valid indices and a finite dummy displacement. The physical
    # masks, indices, and displacements are refreshed by the official builder.
    from torch.nn.functional import pad as append
    dummy = vec.new_zeros((pad, 3))
    dummy[:, 2] = 1.
    return (coord, atype, append(index, (0, pad)), torch.cat((vec, dummy)),
            append(scat, (0, pad)), append(mask, (0, pad), value=False))


@contextmanager
def replay_energy_core(model, *, forward_only, edge_multiple=64, headroom=True):
    """Serial single-frame, first-derivative inference. Holds one live shape bucket."""
    if model.training or model.get_active_mode() != "ener" or edge_multiple < 1:
        raise ValueError("requires frozen energy-mode inference and a positive bucket width")
    original = model.core_compute
    active = None
    active_key = None
    capacity = 0
    metrics = Counter()
    def call(*values, **options):
        nonlocal active, active_key, capacity
        if (len(values) != 6 or values[0].shape[0] != 1 or not values[0].is_cuda
                or any(options.get(k) is not None for k in ("comm_dict","spin","extended_atype","extended_coord_corr"))
                or options.get("embedding_only") or options.get("atomic_output_only")
                or not options.get("conservative", True)):
            return original(*values, **options)
        metrics["calls"] += 1
        metrics["min_edges"] = min(metrics.get("min_edges", len(values[3])), len(values[3]))
        metrics["max_edges"] = max(metrics["max_edges"], len(values[3]))
        n = len(values[3])
        if not headroom or n > capacity or capacity == 0:
            target = n + max(edge_multiple, n // 50) if headroom else n
            capacity = max(edge_multiple, ((target + edge_multiple - 1) // edge_multiple) * edge_multiple)
        metrics["capacity"] = capacity
        padded = _pad_edges(values, capacity)
        def signature(x):
            return (tuple(x.shape), x.dtype, x.device) if isinstance(x, torch.Tensor) else x
        key = (tuple(map(signature, padded)), tuple(sorted((k, signature(v)) for k,v in options.items())))
        if key != active_key:
            start = perf_counter()
            active = None
            active = _CoreGraph(original, padded, options, forward_only)
            active_key = key
            metrics["captures"] += 1
            metrics["capture_seconds"] += perf_counter() - start
        return active.run(padded, options)
    with patch.object(model, "core_compute", call):
        yield metrics
