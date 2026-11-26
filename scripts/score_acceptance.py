"""Score full-step MACE and XL routes against stock XL in the same benchmark run."""

import argparse
import json
import math
from pathlib import Path
from statistics import median


def assess(report, name, limit):
    baseline = "xl_efs"
    if name == baseline:
        raise ValueError("the reference cannot also be the candidate")
    times = report["raw_ms_step"]
    candidate, stock = times[name], times[baseline]
    if len(stock) != report["repeats"] or len(candidate) != len(stock) or not stock:
        raise ValueError(f"incomplete timing for {name}")
    for path in (baseline, name):
        checks = report["checks"][path]
        if not checks["passed"] or "stress_eV_A3" not in checks["errors"]:
            raise ValueError(f"{path} did not pass the matched energy, force and stress checks")
        if not all(math.isfinite(t) and t > 0 for t in times[path]):
            raise ValueError(f"{path} has invalid trial times")
    b, t = median(stock), median(candidate)
    ratio = t / b
    return {
        "route": name, "stock_ms_step": b, "candidate_ms_step": t,
        "candidate_over_stock": ratio, "limit_ratio": limit, "limit_ms_step": b * limit,
        "paired_ratios": [c / s for c, s in zip(candidate, stock)],
        "paired_trials_within_limit": sum(c <= limit * s for c, s in zip(candidate, stock)),
        "passed": ratio <= limit,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("report", type=Path)
    p.add_argument("--exact", required=True, help="independent route using the original energy expression")
    p.add_argument("--combined", required=True, help="prepared route using the native XL evaluator")
    p.add_argument("--interpolated", help="optional qualified spline route, scored separately")
    args = p.parse_args()
    d = json.loads(args.report.read_text())
    if not d["passed"] or not all(d.get(k) for k in ("xl_stock_product", "xl_stock_r0", "host_efs_each_step")):
        p.error("use a passing same-run benchmark with upstream-generated R0/M1 and host E/F/stress")
    modules = [d["xl_stock_product"], d["xl_stock_r0"]]
    if d.get("xl_stock_m0"):
        modules.append(d["xl_stock_m0"])
    if any(meta["variant"] != "upstream" or meta["compact_sha256"] != d["xl_json_sha256"] for meta in modules):
        p.error("stock modules must be the upstream builds for the measured compact model")
    if d["dtype"] != "float64" or d["xl_launch_policy"] != "static" or not args.combined.startswith("xl_prepared"):
        p.error("expected a static-policy FP64 benchmark with a native combined route")
    exact = args.exact.split("_", 1)[0]
    if exact not in {"base", "first", "folded", "second", "both", "bothfolded", "candidate", "replay",
                     "sched", "packed", "tangent", "schedpacked", "schedtangent"}:
        p.error("the exact route must evaluate the original radial network independently of XL")
    rows = {"independent_exact": assess(d, args.exact, 1.10), "combined_native": assess(d, args.combined, 0.90)}
    if args.interpolated:
        if "table" not in args.interpolated:
            p.error("the interpolated comparison requires a table route")
        rows["independent_spline"] = assess(d, args.interpolated, 1.10)
    print(json.dumps({"source": args.report.name, "stock_generated_m0": bool(d.get("xl_stock_m0")),
                      "results": rows}, indent=2))
    if not all(rows[k]["passed"] for k in ("independent_exact", "combined_native")):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
