#!/usr/bin/env python3
"""
Diagnostic script for linumpy 3D reconstruction pipeline performance.

This script checks the server configuration to identify bottlenecks:
- CPU core detection and thread configuration
- GPU availability and CUDA setup (CuPy for linumpy GPU ops; PyTorch for BaSiCPy)
- Memory availability
- Nextflow parameter recommendations
- Performance baseline tests

Usage:
    linum_diagnose_pipeline.py                    # Full diagnostic
    linum_diagnose_pipeline.py --quick            # Quick system check only
    linum_diagnose_pipeline.py --benchmark        # Include performance benchmarks
    linum_diagnose_pipeline.py --debug-cuda       # Detailed CUDA library debugging
    linum_diagnose_pipeline.py --verbose          # Show full error tracebacks
    linum_diagnose_pipeline.py --output report.txt  # Save results to file
"""

# Configure thread limits before numpy/scipy imports
import linumpy.config.threads  # noqa: F401

import argparse
import json
from datetime import datetime

from linumpy.diagnostics.pipeline import SystemDiagnostics, get_terminal_width

__all__ = ["SystemDiagnostics", "get_terminal_width"]


def main() -> None:
    """Run function."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--quick", action="store_true", help="Quick system check only (no benchmarks)")
    p.add_argument("--benchmark", action="store_true", help="Include performance benchmarks")
    p.add_argument("--debug-cuda", action="store_true", help="Show detailed CUDA library debugging info")
    p.add_argument("--verbose", "-v", action="store_true", help="Show verbose error output for debugging")
    p.add_argument("--output", "-o", type=str, help="Save results to JSON file")
    args = p.parse_args()

    print("=" * get_terminal_width())
    print(" LINUMPY 3D RECONSTRUCTION PIPELINE DIAGNOSTICS")
    print(f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * get_terminal_width())

    diag = SystemDiagnostics(verbose=args.verbose)

    if args.debug_cuda:
        diag.debug_cuda_libraries()
        return

    diag.check_cpu()
    diag.check_memory()
    diag.check_gpu()
    diag.check_python_packages()
    diag.check_nextflow_config()

    if args.benchmark or not args.quick:
        diag.run_benchmarks()

    results = diag.generate_report()

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Results saved to: {args.output}")

    print("\n" + "=" * get_terminal_width())
    print(" DIAGNOSTICS COMPLETE")
    print("=" * get_terminal_width() + "\n")


if __name__ == "__main__":
    main()
