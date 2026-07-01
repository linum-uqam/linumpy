#!/usr/bin/env python3
"""
Assess slice quality for 3D mosaic grids and optionally update slice configuration.

This script analyzes mosaic grid slices to detect quality issues that might affect
reconstruction. It uses multiple metrics to identify problematic slices:

- SSIM (Structural Similarity): compares each slice to its neighbors.
- Edge Preservation: detects if edge structures are preserved compared to neighbors.
- Variance Consistency: checks for unusual signal variance (data loss/corruption).
- First Slice Detection: automatically identifies calibration slices (thicker/different).

GPU acceleration is used when available (--use_gpu, default on) for SSIM and
edge-detection computations. Falls back to CPU automatically if no GPU is detected.

The output can be:

- A new slice_config.csv with quality scores and recommendations.
- An update to an existing slice_config.csv with quality assessments.
- A quality report for review.

Example usage::

    # Assess quality and create/update slice config
    linum_assess_slice_quality.py /path/to/mosaics slice_config.csv

    # Assess and exclude low-quality slices automatically
    linum_assess_slice_quality.py /path/to/mosaics slice_config.csv --min_quality 0.3

    # Exclude first N calibration slices
    linum_assess_slice_quality.py /path/to/mosaics slice_config.csv --exclude_first 1

    # Update existing config with quality info
    linum_assess_slice_quality.py /path/to/mosaics slice_config.csv --update_existing

    # Force CPU fallback
    linum_assess_slice_quality.py /path/to/mosaics slice_config.csv --no-use_gpu
"""

from __future__ import annotations

# Configure thread limits before numpy/scipy imports
import linumpy.config.threads  # noqa: F401

import argparse
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tqdm.auto import tqdm

if TYPE_CHECKING:
    import numpy as np

from linumpy.cli.args import add_overwrite_arg, assert_output_exists
from linumpy.gpu import GPU_AVAILABLE
from linumpy.gpu.image_quality import (
    assess_slice_quality_gpu,
    clear_gpu_memory,
)
from linumpy.io import slice_config as slice_config_io
from linumpy.io.zarr import read_omezarr
from linumpy.metrics.image_quality import (
    assess_slice_quality,
    detect_calibration_slice,
)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input", help="Input directory containing mosaic grids (`*.ome.zarr`)")
    p.add_argument("output_file", help="Output slice configuration CSV file")

    gpu_group = p.add_argument_group("GPU Options")
    gpu_group.add_argument(
        "--use_gpu",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use GPU acceleration if available. [%(default)s]",
    )
    gpu_group.add_argument("--gpu_id", type=int, default=0, help="GPU device ID to use. [%(default)s]")

    quality_group = p.add_argument_group("Quality Assessment")
    quality_group.add_argument(
        "--min_quality",
        type=float,
        default=0.0,
        help="Minimum quality score to include slice (0-1). Default: 0.0 (include all, just report)",
    )
    quality_group.add_argument(
        "--sample_depth",
        type=int,
        default=5,
        help="Number of z-planes to sample per slice for faster assessment. Default: 5 (0=all)",
    )
    quality_group.add_argument(
        "--pyramid_level",
        type=int,
        default=0,
        help="Pyramid level to use for assessment (0=full res). Higher levels are faster but less accurate. Default: 0",
    )
    quality_group.add_argument(
        "--roi_size",
        type=int,
        default=0,
        help="Side length of center crop in XY (pixels) used for "
        "all quality metrics. 0 = full plane (slow for large "
        "single-resolution mosaics). Recommended: 1024.",
    )
    quality_group.add_argument(
        "--processes",
        type=int,
        default=1,
        help="Number of parallel workers for slice assessment (CPU mode only).\n"
        "Each worker reads its own zarr planes concurrently.\n"
        "Default: 1 (sequential). Set to params.processes.",
    )

    calib_group = p.add_argument_group("Calibration Slice Handling")
    calib_group.add_argument(
        "--exclude_first",
        type=int,
        default=1,
        help="Exclude first N slices as calibration slices. Default: 1 (first slice is usually calibration)",
    )
    calib_group.add_argument(
        "--detect_calibration",
        action="store_true",
        help="Automatically detect calibration slices by their different thickness/structure",
    )
    calib_group.add_argument(
        "--calibration_thickness_ratio",
        type=float,
        default=1.5,
        help="Slices with thickness ratio > this are flagged as calibration. Default: 1.5",
    )

    update_group = p.add_argument_group("Update Existing Config")
    update_group.add_argument(
        "--update_existing", action="store_true", help="Update an existing slice_config.csv with quality info"
    )
    update_group.add_argument("--existing_config", type=str, default=None, help="Path to existing slice config to update")

    output_group = p.add_argument_group("Output Options")
    output_group.add_argument("--report_only", action="store_true", help="Only print report, don't write config file")
    output_group.add_argument("-v", "--verbose", action="store_true", help="Print detailed quality metrics per slice")

    add_overwrite_arg(p)
    return p


def get_mosaic_files(directory: Path) -> dict[int, Path]:
    """Find all mosaic grid files and extract slice IDs."""
    pattern = r".*z(\d+).*\.ome\.zarr$"
    mosaics = {}

    for f in directory.iterdir():
        if f.is_dir() and f.suffix == ".zarr":
            match = re.match(pattern, f.name)
            if match:
                slice_id = int(match.group(1))
                mosaics[slice_id] = f

    return dict(sorted(mosaics.items()))


def read_existing_config(config_path: Path) -> dict[int, dict[str, Any]]:
    """Read an existing slice configuration file keyed by integer ``slice_id``."""
    rows = slice_config_io.read(config_path)
    return {int(sid): dict(row) for sid, row in rows.items()}


def write_slice_config_with_quality(
    output_file: Path,
    slice_ids: list[int],
    quality_results: dict[int, dict[str, Any]],
    exclude_ids: list[int],
    existing_config: dict[int, dict[str, Any]] | None = None,
) -> None:
    """Write ``slice_config.csv`` with the decision columns set from the quality.

    assessment. Raw per-metric scores (ssim_mean / edge_score / variance_score /
    depth) intentionally stay out of the CSV -- they live in the pipeline report
    and per-stage diagnostics JSON, not in the per-slice decision trace.
    """
    out_rows: list[dict[str, object]] = []
    for slice_id in slice_ids:
        quality = quality_results.get(slice_id, {})
        use = "true"
        reason = ""
        if slice_id in exclude_ids:
            use = "false"
            if quality.get("is_calibration", False):
                reason = "calibration_slice"
            elif quality.get("overall", 1.0) < quality.get("min_threshold", 0):
                reason = "low_quality"
            elif quality.get("exclude_first", False):
                reason = "first_slice_excluded"
            else:
                reason = "manually_excluded"

        existing = existing_config.get(slice_id, {}) if existing_config else {}
        if existing.get("use", "true").lower() in ["false", "0", "no"]:
            use = "false"
            if not reason:
                reason = existing.get("exclude_reason") or existing.get("notes") or "previously_excluded"

        row: dict[str, object] = {
            "slice_id": f"{slice_id:02d}",
            "use": use,
            "quality_score": f"{float(quality.get('overall', 0.0)):.3f}",
            "exclude_reason": reason,
        }
        if existing.get("galvo_confidence", ""):
            row["galvo_confidence"] = existing["galvo_confidence"]
        if existing.get("galvo_fix", ""):
            row["galvo_fix"] = existing["galvo_fix"]
        for carry in ("notes",):
            val = existing.get(carry)
            if val:
                row[carry] = val
        out_rows.append(row)

    slice_config_io.write(output_file, out_rows)


def main() -> None:
    """Run function operation."""
    p = _build_arg_parser()
    args = p.parse_args()

    input_path = Path(args.input)
    output_file = Path(args.output_file)

    if not args.report_only:
        assert_output_exists(output_file, p, args)

    if not input_path.is_dir():
        p.error(f"Input directory not found: {input_path}")

    use_gpu = args.use_gpu and GPU_AVAILABLE
    if args.use_gpu and not GPU_AVAILABLE:
        print("Warning: GPU requested but not available. Using CPU.")
    elif use_gpu:
        try:
            import cupy as cp

            cp.cuda.Device(args.gpu_id).use()
            print(f"Using GPU device {args.gpu_id}")
        except Exception as e:
            print(f"Warning: Could not select GPU {args.gpu_id}: {e}. Using default.")

    print(f"Scanning for mosaic grids in: {input_path}")
    mosaic_files = get_mosaic_files(input_path)

    if not mosaic_files:
        p.error(f"No mosaic grid files found in {input_path}")

    slice_ids = sorted(mosaic_files.keys())
    print(f"Found {len(slice_ids)} slices: {[f'{s:02d}' for s in slice_ids]}")

    existing_config = None
    if args.update_existing:
        config_to_load = args.existing_config if args.existing_config else output_file
        if Path(config_to_load).exists():
            existing_config = read_existing_config(Path(config_to_load))
            print(f"Loaded existing config with {len(existing_config)} entries")

    exclude_ids = set()

    if args.exclude_first > 0:
        first_slices = slice_ids[: args.exclude_first]
        exclude_ids.update(first_slices)
        print(f"Excluding first {args.exclude_first} slice(s) as calibration: {first_slices}")

    print(f"\nLoading slices (pyramid_level={args.pyramid_level})...")
    volumes: dict[int, np.ndarray | None] = {}
    for slice_id in tqdm(slice_ids, desc="Loading slices"):
        try:
            vol, _ = read_omezarr(mosaic_files[slice_id], level=args.pyramid_level)
            volumes[slice_id] = vol
        except Exception as e:
            print(f"  Warning: Could not load slice {slice_id:02d}: {e}")
            volumes[slice_id] = None

    calibration_slices = []
    if args.detect_calibration:
        print(f"Detecting calibration slices (thickness ratio > {args.calibration_thickness_ratio})...")
        valid_volumes = {sid: vol for sid, vol in volumes.items() if vol is not None}
        calibration_slices = detect_calibration_slice(valid_volumes, args.calibration_thickness_ratio)
        if calibration_slices:
            exclude_ids.update(calibration_slices)
            print(f"Detected calibration slices: {calibration_slices}")

    print(f"\nAssessing slice quality (GPU={'enabled' if use_gpu else 'disabled'}, sample_depth={args.sample_depth})...")
    quality_results: dict[int, dict[str, Any]] = {}

    if use_gpu:
        for i, slice_id in enumerate(tqdm(slice_ids, desc="Assessing quality")):
            vol = volumes.get(slice_id)
            if vol is None:
                quality_results[slice_id] = {
                    "overall": 0.0,
                    "ssim_mean": 0.0,
                    "edge_score": 0.0,
                    "variance_score": 0.0,
                    "depth": 0,
                    "has_data": False,
                    "error": "load_failed",
                }
                continue
            vol_before = volumes.get(slice_ids[i - 1]) if i > 0 else None
            vol_after = volumes.get(slice_ids[i + 1]) if i < len(slice_ids) - 1 else None
            overall, metrics = assess_slice_quality_gpu(vol, vol_before, vol_after, args.sample_depth)
            metrics["is_calibration"] = slice_id in calibration_slices
            metrics["exclude_first"] = slice_id in slice_ids[: args.exclude_first]
            metrics["min_threshold"] = args.min_quality
            quality_results[slice_id] = metrics
            if args.min_quality > 0 and overall < args.min_quality:
                exclude_ids.add(slice_id)
        clear_gpu_memory()
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _assess_one(idx_and_id: tuple) -> Any:
            i, slice_id = idx_and_id
            vol = volumes.get(slice_id)
            if vol is None:
                return slice_id, {
                    "overall": 0.0,
                    "ssim_mean": 0.0,
                    "edge_score": 0.0,
                    "variance_score": 0.0,
                    "depth": 0,
                    "has_data": False,
                    "error": "load_failed",
                }
            vol_before = volumes.get(slice_ids[i - 1]) if i > 0 else None
            vol_after = volumes.get(slice_ids[i + 1]) if i < len(slice_ids) - 1 else None
            _overall, metrics = assess_slice_quality(vol, vol_before, vol_after, args.sample_depth, xy_roi=args.roi_size)
            metrics["is_calibration"] = slice_id in calibration_slices
            metrics["exclude_first"] = slice_id in slice_ids[: args.exclude_first]
            metrics["min_threshold"] = args.min_quality
            return slice_id, metrics

        tasks = list(enumerate(slice_ids))
        with ThreadPoolExecutor(max_workers=args.processes) as executor:
            futures = {executor.submit(_assess_one, t): t for t in tasks}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Assessing quality"):
                slice_id, metrics = future.result()
                quality_results[slice_id] = metrics
                if args.min_quality > 0 and metrics.get("overall", 0.0) < args.min_quality:
                    exclude_ids.add(slice_id)

    print("\n" + "=" * 70)
    print(f"SLICE QUALITY REPORT{' (GPU-accelerated)' if use_gpu else ' (CPU)'}")
    print("=" * 70)
    print(f"{'Slice':<8} {'Quality':<10} {'SSIM':<10} {'Edge':<10} {'Var':<10} {'Depth':<8} {'Status':<15}")
    print("-" * 70)

    for slice_id in slice_ids:
        q = quality_results.get(slice_id, {})
        status = []
        if slice_id in exclude_ids:
            if q.get("is_calibration"):
                status.append("CALIBRATION")
            elif q.get("exclude_first"):
                status.append("FIRST_SLICE")
            elif q.get("overall", 1.0) < args.min_quality:
                status.append("LOW_QUALITY")
            else:
                status.append("EXCLUDED")
        else:
            status.append("OK")

        status_str = ",".join(status)
        print(
            f"{slice_id:02d}      {q.get('overall', 0):.3f}      "
            f"{q.get('ssim_mean', 0):.3f}      {q.get('edge_score', 0):.3f}      "
            f"{q.get('variance_score', 0):.3f}      {q.get('depth', 0):<8} {status_str}"
        )

    print("-" * 70)
    print(f"Total slices: {len(slice_ids)}")
    print(f"Excluded: {len(exclude_ids)}")
    print(f"Included: {len(slice_ids) - len(exclude_ids)}")

    if args.min_quality > 0:
        low_quality = [s for s in slice_ids if quality_results.get(s, {}).get("overall", 1.0) < args.min_quality]
        if low_quality:
            print(f"Low quality slices (< {args.min_quality}): {low_quality}")

    if not args.report_only:
        write_slice_config_with_quality(output_file, slice_ids, quality_results, list(exclude_ids), existing_config)
        print(f"\nSlice configuration written to: {output_file}")

    if exclude_ids:
        print(f"\nExcluded slice IDs: {sorted(exclude_ids)}")


if __name__ == "__main__":
    main()
