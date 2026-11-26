"""Check the R0 spline factorization supplied by Symmetrix-XL 0.2.0.

Run on one CPU thread with the compact export used for the GPU benchmarks.
The XL diagnostic compares projected and reconstructed nodal values and slopes
over the entire spline grid. It does not launch message passing or MD.
"""

import argparse
import hashlib
import json
import math
from importlib.metadata import version
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symmetrix-json", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--elements", nargs=2, type=int, default=[79, 8],
                        help="atomic numbers of the receiver and sender")
    args = parser.parse_args()
    if args.symmetrix_json.resolve() == args.output.resolve():
        parser.error("input and output paths must differ")
    if version("symmetrix-xl") != "0.2.0":
        parser.error("this diagnostic uses the Symmetrix-XL 0.2.0 native interface")

    from symmetrix import load_backend

    native = load_backend("cpu")
    payload = args.symmetrix_json.read_bytes()
    model = json.loads(payload)
    compact = model["compact_radial"]
    elements = model["atomic_numbers"]
    if any(z not in elements for z in args.elements):
        parser.error(f"available atomic numbers are {elements}")
    owns_kokkos = not native._kokkos_is_initialized()
    if owns_kokkos:
        native._init_kokkos()
    try:
        evaluator = native.MACEKokkos(str(args.symmetrix_json))
        try:
            report = evaluator._compact_radial_factorization_report(
                "R0", *(elements.index(z) for z in args.elements)
            )
        finally:
            del evaluator
    finally:
        if owns_kokkos:
            native._finalize_kokkos()
    errors = [float(v) for key, v in report.items() if "error" in key]
    if len(errors) != 2:
        raise RuntimeError(f"unexpected native report fields {report.keys()}")
    result = {
        "scope": "Native CPU R0 radial values and slopes at every spline node",
        "version": version("symmetrix-xl"),
        "export_sha256": hashlib.sha256(payload).hexdigest(),
        "receiver_sender_Z": args.elements,
        "radial_shape": compact["networks"]["R0"]["shape"],
        "spline_nodes": compact["num_spline_points"],
        "distance_transform": compact["distance_transform"],
        "native_report": report,
        "tolerance": 1e-8,
        "passed": all(math.isfinite(v) and v <= 1e-8 for v in errors),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if not result["passed"]:
        raise SystemExit("R0 factorization exceeded tolerance")


if __name__ == "__main__":
    main()
