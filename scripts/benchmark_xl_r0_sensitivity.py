"""Reproduce first-layer algebra and storage ablations in one serial process.

The two staged variants own a fixed device workspace, with one forward and one
reverse per call. Only this standalone executable admits these variants. All
evaluation and capture operations in benchmark_xl_fixed_step are sequential.
"""
import argparse
from pathlib import Path
import sys
from unittest.mock import patch

import benchmark_xl_fixed_step as bench
from mace_md_lab import xl_product, xl_r0_algorithm


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--r0-algorithm-dir", type=Path, required=True)
    known, remaining = parser.parse_known_args()
    upstream = xl_product.install_xl_r0

    def select_module(calc, artifacts, *, variant):
        if variant in xl_r0_algorithm.VARIANTS:
            return xl_r0_algorithm._load_compiled(calc, known.r0_algorithm_dir, variant=variant)
        return upstream(calc, artifacts, variant=variant)

    extra = {"arrays_"+v:v for v in xl_r0_algorithm.VARIANTS}
    with patch.dict(bench.R0_VARIANTS, extra), patch.object(xl_product, "install_xl_r0", select_module):
        with patch.object(sys, "argv", [sys.argv[0], *remaining]):
            bench.main()


if __name__ == "__main__":
    main()
