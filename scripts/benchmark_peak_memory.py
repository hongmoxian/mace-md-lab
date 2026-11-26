"""Measure complete-process GPU memory in fresh processes on replicated Au/water."""
import argparse
import json
import os
from pathlib import Path
from statistics import median
import subprocess
import sys
import tempfile
import time


def resident_mib():
    """Take an untimed process-specific snapshot after the CUDA stream has drained."""
    import pynvml as nv
    nv.nvmlInit()
    try:
        for index in range(nv.nvmlDeviceGetCount()):
            handle = nv.nvmlDeviceGetHandleByIndex(index)
            try:
                ps = nv.nvmlDeviceGetComputeRunningProcesses(handle)
            except nv.NVMLError_NoPermission:
                continue
            for proc in ps:
                if proc.pid == os.getpid():
                    return proc.usedGpuMemory / 2**20
        raise RuntimeError("worker missing from NVML process list")
    finally:
        nv.nvmlShutdown()


def worker(a):
    import numpy as np
    import torch
    from mace_md_lab.bench import ASEPath, file_hash, starting_structure
    from benchmark_xl_algorithms import new_path, read_routes, setup_engine

    a.seed, a.frame, a.temperature, a.skin, a.dt_fs = 17, 0, 300., 1., .5
    a.harmonic_storage, a.m0_dir = "y-only-direct-v1", None
    base, _ = starting_structure(a)
    initial = base.repeat((a.tiles, a.tiles, 1))
    n = len(initial)
    torch.cuda.init()
    if a.worker == "mace":
        from mace.calculators import MACECalculator
        engine = MACECalculator(model_paths=[str(a.model)], device="cuda", default_dtype="float64")
        path = ASEPath(initial, engine, a.dt_fs)
    else:
        os.environ["SYMMETRIX_NEIGHBOR_BACKEND"] = "kokkos"
        from symmetrix import load_backend
        load_backend("cuda12-sm70")
        routes = read_routes(a.routes_json)
        spec = routes["locked" if a.worker == "locked" else "upstream"]
        engine, _ = setup_engine(a, spec, file_hash(a.symmetrix_json), sorted(set(map(int, initial.numbers))))
        path = ASEPath(initial, engine, a.dt_fs) if a.worker == "xl_ase" else new_path(initial, engine, a)[0]
    s = path.snapshot()
    stress = s[5] if len(s) == 6 else path.atoms.get_stress()
    np.savez(Path(a.work) / (a.worker + ".npz"), e=s[2]/n, f=s[3], s=stress)
    if not all(np.isfinite(x).all() for x in (*s, stress)):
        raise ValueError("nonfinite initial state")
    path.advance(a.warmup)
    torch.cuda.synchronize()
    post_warmup = resident_mib()
    times = []
    for _ in range(a.repeats):
        torch.cuda.synchronize()
        t = time.perf_counter()
        path.advance(a.steps)
        torch.cuda.synchronize()
        times.append(1000*(time.perf_counter()-t)/a.steps)
    if not all(np.isfinite(x).all() for x in path.snapshot()):
        raise ValueError("nonfinite final state")
    print("MEMRESULT " + json.dumps(dict(atoms=n, gpu=torch.cuda.get_device_name(), times=times,
          ms_step=median(times), post_warmup_process_MiB=post_warmup,
          post_md_process_MiB=resident_mib(), torch_reserved_peak_MiB=torch.cuda.max_memory_reserved()/2**20,
          torch_allocated_peak_MiB=torch.cuda.max_memory_allocated()/2**20,
          model_sha256=file_hash(a.model), compact_sha256=file_hash(a.symmetrix_json),
          fixture_sha256=file_hash(a.structure), torch=str(torch.__version__))), flush=True)


def monitor(argv):
    """Sample all allocations belonging to a fresh worker, including native libraries."""
    import pynvml as nv
    from collections import Counter
    nv.nvmlInit()
    handles = [nv.nvmlDeviceGetHandleByIndex(i) for i in range(nv.nvmlDeviceGetCount())]
    peaks = Counter()
    found = None
    samples = 0
    foreign = False
    with tempfile.TemporaryFile(mode="w+") as log:
        python = [sys.executable, "-S"] if sys.flags.no_site else [sys.executable]
        proc = subprocess.Popen([*python, *argv],
                                stdout=log, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            for handle in ([found] if found else handles):
                try:
                    ps = nv.nvmlDeviceGetComputeRunningProcesses(handle)
                except nv.NVMLError_NoPermission:
                    continue
                match = next((p for p in ps if p.pid == proc.pid), None)
                if match is None:
                    continue
                info = nv.nvmlDeviceGetMemoryInfo(handle)
                peaks["process_peak_MiB"] = max(peaks["process_peak_MiB"], match.usedGpuMemory/2**20)
                peaks["board_peak_MiB"] = max(peaks["board_peak_MiB"], info.used/2**20)
                peaks["board_total_MiB"] = info.total/2**20
                found = handle
                foreign |= any(p.pid != proc.pid for p in ps)
                samples += 1
                break
            time.sleep(.01)
        log.seek(0)
        output = log.read()
    nv.nvmlShutdown()
    lines = [s[10:] for s in output.splitlines() if s.startswith("MEMRESULT ")]
    result = dict(peaks, samples=samples, foreign_compute_pid_seen=foreign)
    if proc.returncode == 0 and lines and samples:
        result.update(json.loads(lines[-1]), status="ok")
    else:
        reason = output[-2300:]
        oom = any(msg in reason.lower() for msg in ("out of memory", "cudaerrormemoryallocation"))
        if oom:
            reason = next((s for s in reversed(output.splitlines()) if "out of memory" in s.lower()), "CUDA memory allocation failed")
        result.update(status="oom" if oom else "failed", exit_code=proc.returncode, diagnostic=reason)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "symmetrix-json", "product-dir", "structure", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--routes-json", type=Path, default=Path("examples/native-xl-locked.json"))
    p.add_argument("--tiles", type=int, required=True)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repeats", type=int, default=3)
    choices = ("mace", "xl_ase", "prepared", "locked")
    p.add_argument("--routes", nargs="+", choices=choices, default=list(choices))
    p.add_argument("--worker", choices=choices)
    p.add_argument("--work")
    p.add_argument("--initial-reference", type=Path, help="compare to an independently saved initial state")
    p.add_argument("--save-initial", type=Path, help="save the first successful route initial state")
    a = p.parse_args()
    if min(a.steps, a.tiles, a.repeats, a.warmup) < 1:
        p.error("positive steps and tiling required")
    if a.worker:
        worker(a)
        return
    import numpy as np
    data = dict(tiles=[a.tiles, a.tiles, 1], steps=a.steps, warmup=a.warmup, repeats=a.repeats,
                sample_sleep_ms=10, dtype="float64", routes={})
    with tempfile.TemporaryDirectory() as work:
        for route in a.routes:
            data["routes"][route] = monitor([__file__, *sys.argv[1:], "--worker", route, "--work", work])
            a.output.write_text(json.dumps(data, indent=2) + "\n")
        results = data["routes"]
        ref = next((name for name in ("mace","prepared","xl_ase","locked")
                    if results.get(name, {}).get("status") == "ok"), None)
        if ref is not None and a.save_initial:
            import shutil
            shutil.copyfile(Path(work)/(ref+".npz"),a.save_initial)
        baseline_file = a.initial_reference or (Path(work)/(str(ref)+".npz"))
        for route, row in results.items():
            if row["status"] != "ok":
                continue
            with np.load(baseline_file) as baseline, np.load(Path(work)/(route+".npz")) as other:
                errors = {k: float(np.max(np.abs(other[k]-baseline[k]))) for k in ("e","f","s")}
            row["initial_efs_reference"] = a.initial_reference.name if a.initial_reference else ref
            row["max_error_eV_atom_eV_A_eV_A3"] = errors
        a.output.write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
