#!/usr/bin/env python3
"""Estimate a single 2x2 tile-placement affine pooled across many 3D mosaic grids.

For each input ``mosaic_grid_*.ome.zarr`` volume, load only the central Z
plane and call
:func:`linumpy.mosaic.motor.compute_registration_refinements` to
measure per-pair absolute tile displacements via phase correlation.
Pairs from every input are concatenated into one pool and a single 2×2
affine transform is fitted via
:func:`~linumpy.mosaic.motor.estimate_affine_from_pairs`.

The resulting transform captures instrument-level geometry (scan-to-stage
rotation θ, motor non-perpendicularity φ, effective per-axis step in
pixels) which is constant across an acquisition session. Use the
resulting ``.npy`` as ``--input_transform`` for
``linum_stitch_3d_refined.py`` to remove per-slice affine jitter while
keeping the blend-shift sub-pixel refinement.

The script is read-only with respect to its inputs and does not touch
any pipeline outputs.

GPU acceleration (CuPy-backed phase correlation) is used when available
(--use_gpu, default on). Falls back to CPU automatically if no GPU is
detected.
"""

# Configure thread limits before numpy/scipy imports
import linumpy.config.threads  # noqa: F401

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np

from linumpy.gpu import GPU_AVAILABLE, print_gpu_info
from linumpy.io import slice_config as slice_config_io
from linumpy.mosaic.motor import pool_pairs_and_fit_global_affine

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_SLICE_RE = re.compile(r"z(\d+)")


def _extract_slice_id(path: Path) -> str:
    match = _SLICE_RE.search(path.name)
    return match.group(1) if match else path.stem


def _discover_volumes(
    input_dir: Path,
    pattern: str,
    slice_config_path: Path | None,
    explicit_ids: list[str] | None,
) -> list[tuple[str, Path]]:
    zarr_paths = sorted(input_dir.glob(pattern))
    allowed: set[str] | None = None
    if slice_config_path is not None:
        allowed = slice_config_io.filter_slices_to_use(slice_config_path)
        logger.info("slice_config: %d slices marked use=true", len(allowed))
    if explicit_ids is not None:
        explicit_set = {sid.strip().zfill(2) for sid in explicit_ids}
        allowed = explicit_set if allowed is None else allowed & explicit_set
        logger.info("--include_slice: restricting to %d slice ids", len(explicit_set))

    volumes: list[tuple[str, Path]] = []
    for path in zarr_paths:
        slice_id = _extract_slice_id(path)
        if allowed is not None and slice_id not in allowed:
            continue
        volumes.append((slice_id, path))
    return volumes


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input_dir", help="Directory containing mosaic_grid_*z??.ome.zarr files.")
    p.add_argument("output_transform", help="Output path for the fitted 2x2 affine transform (.npy).")
    p.add_argument(
        "--overlap_fraction",
        type=float,
        default=0.2,
        help="Expected tile overlap fraction (must match acquisition). [%(default)s]",
    )
    p.add_argument(
        "--pattern",
        type=str,
        default="mosaic_grid*_z*.ome.zarr",
        help="Glob pattern used to discover input mosaic grids. [%(default)s]",
    )
    p.add_argument(
        "--slice_config",
        type=str,
        default=None,
        help="Optional slice_config.csv — rows with use=false are skipped.",
    )
    p.add_argument(
        "--include_slice",
        type=str,
        nargs="+",
        default=None,
        help="Optional explicit list of slice ids (zero-padded, e.g. '10 11 12')\n"
        "to include. Combined with --slice_config via intersection when both\n"
        "are provided.",
    )
    p.add_argument(
        "--histogram_match",
        action="store_true",
        help="Match overlap histograms before phase correlation (more robust\n"
        "to uneven tile-edge illumination; matches the old\n"
        "linum_estimate_transform.py behaviour).",
    )
    p.add_argument(
        "--max_empty_fraction",
        type=float,
        default=None,
        help="If set, use an Otsu threshold to detect empty overlaps and skip\n"
        "any pair with more than this fraction of background pixels.\n"
        "When unset, the default per-volume 'mean(overlap > 0) < 0.1'\n"
        "heuristic is used.",
    )
    p.add_argument(
        "--n_samples",
        type=int,
        default=None,
        help="Maximum number of pooled pairs to feed into the LS fit.\n"
        "If set and the pool exceeds this size, a reproducible random\n"
        "sub-sample is drawn. Unset means use every pair.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for pair sub-sampling (used only when --n_samples is set). [%(default)s]",
    )
    p.add_argument(
        "--diagnostics_json",
        type=str,
        default=None,
        help="Optional JSON sidecar for fit diagnostics and per-volume stats.",
    )
    p.add_argument("--overwrite", "-f", action="store_true", help="Overwrite the output transform if it already exists.")
    p.add_argument(
        "--use_gpu",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use GPU-accelerated phase correlation via CuPy if available. [%(default)s]",
    )
    p.add_argument("--verbose", "-v", action="store_true", help="Print GPU information on startup.")
    return p


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()

    use_gpu = args.use_gpu and GPU_AVAILABLE
    if args.verbose:
        print_gpu_info()
    if args.use_gpu and not GPU_AVAILABLE:
        logger.info("No CUDA device detected; falling back to CPU phase correlation")

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        parser.error(f"Input directory does not exist: {input_dir}")

    output_transform = Path(args.output_transform)
    if output_transform.exists() and not args.overwrite:
        parser.error(f"Output exists: {output_transform}. Use -f to overwrite.")
    if output_transform.suffix != ".npy":
        parser.error("output_transform must end in .npy")

    slice_config_path = Path(args.slice_config) if args.slice_config else None
    if slice_config_path is not None and not slice_config_path.exists():
        parser.error(f"slice_config.csv not found: {slice_config_path}")

    volumes = _discover_volumes(input_dir, args.pattern, slice_config_path, args.include_slice)
    if not volumes:
        parser.error(f"No mosaic grids selected (pattern={args.pattern!r}, dir={input_dir})")
    logger.info("pooling pairs from %d mosaic grids", len(volumes))

    transform, diagnostics = pool_pairs_and_fit_global_affine(
        [(sid, p) for sid, p in volumes],
        overlap_fraction=args.overlap_fraction,
        histogram_match=args.histogram_match,
        max_empty_fraction=args.max_empty_fraction,
        n_samples=args.n_samples,
        seed=args.seed,
        use_gpu=use_gpu,
    )

    model = diagnostics["displacement_model"]
    logger.info("Global displacement model (backend=%s):", diagnostics["backend"])
    logger.info("  Transform: %s", np.array2string(transform, precision=3))
    logger.info("  theta_deg = %+.3f  (scan-to-stage rotation; 0 = aligned)", model["theta_deg"])
    logger.info("  phi_deg   = %+.3f  (motor-axes angle; 90 = perpendicular)", model["phi_deg"])
    logger.info("  Ox_frac   = %.4f (expected %.4f)", model["Ox_fraction"], args.overlap_fraction)
    logger.info("  Oy_frac   = %.4f (expected %.4f)", model["Oy_fraction"], args.overlap_fraction)
    logger.info("  lstsq_residual = %s", diagnostics["lstsq_residual"])

    output_transform.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_transform), transform)
    logger.info("wrote transform to %s", output_transform)

    if args.diagnostics_json is not None:
        diagnostics_path = Path(args.diagnostics_json)
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_path.write_text(json.dumps(diagnostics, indent=2))
        logger.info("wrote diagnostics to %s", diagnostics_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
