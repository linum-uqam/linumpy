#!/usr/bin/env python3
"""Export lightweight data package for the manual alignment tool.

Reads common-space slices (OME-Zarr) and pairwise registration outputs,
then produces a self-contained directory with the following layout::

    manual_align_package/
      aips/           XY AIPs: per-slice fallback (mean over Z) + per-pair edge
                      projections (pair_z{fid}_z{mid}_{role}.npz) restricted to
                      the overlap-edge depth slab of each volume    -- XY alignment
      aips_xz/        XZ cross-sections                  -- Z-overlap review
      aips_yz/        YZ cross-sections                  -- Z-overlap review
      transforms/     .tfm + offsets.txt + metrics JSON
      manual_align_metadata.json

XZ/YZ cross-sections are generated in two complementary ways:

  Per-pair files (preferred): ``pair_z{fid:02d}_z{mid:02d}_fixed.npz`` and
  ``pair_z{fid:02d}_z{mid:02d}_moving.npz``.  Both slices in the pair share
  the same Y/X column, chosen by maximising the *combined* intensity at the
  overlap depth -- so the two cross-sections always show the same anatomical
  plane and can be compared directly.

  Per-slice fallback: ``slice_z{sid:02d}.npz``, one per slice, using the
  globally brightest column.  Kept for backward-compatibility with older
  packages.

The package can be downloaded locally and opened directly by the
``linumpy-manual-align`` Napari plugin without needing the full 3-D volumes.
"""

import linumpy.config.threads  # noqa: F401

import argparse
import json
import logging
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm

from linumpy.registration.manual import (
    _discover_slices,
    _discover_transforms,
    _is_interpolated,
    _pair_task,
    _read_overlap_z_offsets,
    _slice_task,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument(
        "slices_dir",
        help="Directory containing common-space slices (slice_z##.ome.zarr).",
    )
    p.add_argument(
        "transforms_dir",
        help="Directory containing pairwise registration outputs (slice_z##*/transform.tfm).",
    )
    p.add_argument(
        "output_dir",
        help="Output directory for the manual alignment data package.",
    )
    p.add_argument(
        "--level",
        type=int,
        default=1,
        help="Pyramid level for AIP computation (0=full, 1=2x downsample, ...). [%(default)s]",
    )
    p.add_argument(
        "--slices",
        type=int,
        nargs="*",
        default=None,
        help="Only export specific slice IDs. Default: all.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=0,
        help=("Number of parallel worker processes. 0 = cpu_count - 2 (leaving 2 cores free). [%(default)s]"),
    )
    p.add_argument(
        "--slices_remote_dir",
        default=None,
        help=(
            "Absolute server path to the published common-space slice directory "
            "(e.g. /scratch/workspace/sub-22/output/bring_to_common_space). "
            "Stored in metadata.json so the manual-align plugin can open "
            "persistent SSH readers for interactive XZ/YZ cross-sections. "
            "Defaults to slices_dir when not provided."
        ),
    )
    p.add_argument(
        "--interpolated_slices_remote_dir",
        default=None,
        help=(
            "Absolute server path to the published interpolated-slice directory "
            "(e.g. /scratch/workspace/sub-22/output/interpolate_missing_slice). "
            "When set, per-slice remote paths for interpolated slices "
            "(detected by '_interpolated' in the filename) point to this "
            "directory instead of --slices_remote_dir. Stored in metadata.json "
            "so the manual-align plugin can find interpolated slices on the server."
        ),
    )
    p.add_argument(
        "--xy_overlap_px",
        type=int,
        default=20,
        metavar="PX",
        help=(
            "Number of Z voxels (at the working pyramid level) to project at the"
            " boundary of each slice for the XY overlap AIPs."
            " Fixed: last PX voxels; Moving: first PX voxels. [%(default)s]"
        ),
    )
    return p


def main(argv: Any = None) -> None:
    """Run function."""
    p = _build_arg_parser()
    args = p.parse_args(argv)

    slices_dir = Path(args.slices_dir)
    transforms_dir = Path(args.transforms_dir)
    output_dir = Path(args.output_dir)
    level = args.level
    # Use the explicitly provided server path when available; fall back to slices_dir.
    # Normalize to remove any double-slashes produced by a trailing slash in params.output.
    slices_remote_dir = str(Path(args.slices_remote_dir)) if args.slices_remote_dir else str(slices_dir)
    # Separate remote dir for interpolated slices (e.g. interpolate_missing_slice/).
    # Falls back to slices_remote_dir when not provided (backward-compatible).
    interp_remote_dir = (
        str(Path(args.interpolated_slices_remote_dir)) if args.interpolated_slices_remote_dir else slices_remote_dir
    )
    workers = args.workers or max(1, (os.process_cpu_count() or os.cpu_count() or 4) - 2)
    overlap_px = args.xy_overlap_px
    logger.info("XY overlap slab: %s voxels at pyramid level %s", overlap_px, args.level)

    if not slices_dir.exists():
        logger.error("Slices directory not found: %s", slices_dir)
        return

    if not transforms_dir.exists():
        logger.error("Transforms directory not found: %s", transforms_dir)
        return

    slice_paths = _discover_slices(slices_dir)
    transform_paths = _discover_transforms(transforms_dir)

    if not slice_paths:
        logger.error("No slice_z##.ome.zarr files found in %s", slices_dir)
        return

    logger.info("Found %s slices, %s transform dirs", len(slice_paths), len(transform_paths))

    # Filter slices if requested
    if args.slices:
        requested = set(args.slices)
        slice_paths = {k: v for k, v in slice_paths.items() if k in requested}
        logger.info("Filtered to %s requested slices", len(slice_paths))

    aips_dir = output_dir / "aips"
    aips_xz_dir = output_dir / "aips_xz"
    aips_yz_dir = output_dir / "aips_yz"
    tfm_dir = output_dir / "transforms"
    for d in (aips_dir, aips_xz_dir, aips_yz_dir, tfm_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Pass 1: XY AIPs (per slice) + per-slice XZ/YZ fallback files.
    # Each slice is independent -- process in parallel.
    # ------------------------------------------------------------------
    logger.info("Computing XY AIPs and per-slice XZ/YZ fallbacks at pyramid level %s using %s workers...", level, workers)
    slice_tasks = [
        (sid, str(spath), level, str(aips_dir), str(aips_xz_dir), str(aips_yz_dir)) for sid, spath in slice_paths.items()
    ]
    with ProcessPoolExecutor(max_workers=min(workers, len(slice_tasks))) as pool:
        futures = {pool.submit(_slice_task, t): t[0] for t in slice_tasks}
        with tqdm(total=len(futures), desc="AIPs") as bar:
            for fut in as_completed(futures):
                sid = futures[fut]
                try:
                    fut.result()
                except Exception as exc:
                    logger.error("z%d failed: %s", sid, exc)
                bar.update(1)

    # ------------------------------------------------------------------
    # Pass 2: Paired XZ/YZ files -- both slices share the same column,
    # chosen from the combined signal at their mutual overlap depth.
    # Each pair is independent -- process in parallel.
    # ------------------------------------------------------------------
    sorted_ids = sorted(slice_paths.keys())
    pairs = [(sorted_ids[i - 1], sorted_ids[i]) for i in range(1, len(sorted_ids)) if sorted_ids[i] in transform_paths]

    if pairs:
        logger.info("Generating paired XZ/YZ cross-sections for %s pairs using %s workers...", len(pairs), workers)
        pair_tasks = []
        for fid, mid in pairs:
            tpath = transform_paths[mid]
            fixed_z, moving_z = _read_overlap_z_offsets(tpath / "offsets.txt")
            pair_tasks.append(
                (
                    fid,
                    mid,
                    str(slice_paths[fid]),
                    str(slice_paths[mid]),
                    fixed_z,
                    moving_z,
                    level,
                    overlap_px,
                    str(aips_dir),
                    str(aips_xz_dir),
                    str(aips_yz_dir),
                )
            )

        with ProcessPoolExecutor(max_workers=min(workers, len(pair_tasks))) as pool:
            futures = {pool.submit(_pair_task, t): (t[0], t[1]) for t in pair_tasks}
            with tqdm(total=len(futures), desc="paired XZ/YZ") as bar:
                for fut in as_completed(futures):
                    fid, mid = futures[fut]
                    try:
                        fut.result()
                    except Exception as exc:
                        logger.error("pair z%d/z%d failed: %s", fid, mid, exc)
                    bar.update(1)

    # Export transforms
    logger.info("Copying pairwise transforms...")
    for tpath in transform_paths.values():
        out_tdir = tfm_dir / tpath.name
        out_tdir.mkdir(parents=True, exist_ok=True)
        # Copy .tfm files
        for tfm_file in tpath.glob("*.tfm"):
            shutil.copy2(tfm_file, out_tdir / tfm_file.name)
        # Copy offsets.txt
        offsets_file = tpath / "offsets.txt"
        if offsets_file.exists():
            shutil.copy2(offsets_file, out_tdir / "offsets.txt")
        # Copy metrics JSON
        metrics_file = tpath / "pairwise_registration_metrics.json"
        if metrics_file.exists():
            shutil.copy2(metrics_file, out_tdir / "pairwise_registration_metrics.json")

    # Write metadata
    interpolated_ids = sorted(sid for sid, p in slice_paths.items() if _is_interpolated(p))
    # Per-slice remote paths: interpolated slices come from a separate publish
    # directory (interpolate_missing_slice/) while real slices live in
    # bring_to_common_space/.  Storing an explicit path per slice lets the
    # plugin's SSH reader always find the file regardless of its origin.
    slice_remote_paths = {
        str(sid): (
            f"{interp_remote_dir}/{p.name}"
            if _is_interpolated(p) and interp_remote_dir != slices_remote_dir
            else f"{slices_remote_dir}/{p.name}"
        )
        for sid, p in slice_paths.items()
    }
    if interpolated_ids:
        logger.info(
            "Detected %s interpolated slice(s): %s",
            len(interpolated_ids),
            interpolated_ids,
        )
    metadata = {
        "pyramid_level": level,
        "n_slices": len(slice_paths),
        "slice_ids": sorted(slice_paths.keys()),
        # Exact filename for each slice (e.g. "slice_z02_normalize.ome.zarr").
        # The suffix varies by pipeline step, so the widget uses this mapping
        # rather than constructing a fixed pattern like "slice_z02.ome.zarr".
        "slice_filenames": {str(sid): p.name for sid, p in slice_paths.items()},
        "axis_views": {"xz_dir": "aips_xz", "yz_dir": "aips_yz", "paired": bool(pairs)},
        "n_transforms": sum(1 for tpath in transform_paths.values() if list(tpath.glob("*.tfm"))),
        # Absolute server path to the published common-space OME-Zarr files.
        # Passed via --slices_remote_dir from the Nextflow process so it points to
        # the publishDir path rather than the work-directory staging path.
        # Used by the plugin to open persistent SSH+Python readers for interactive
        # cross-section navigation (slider to select Y or X position at full resolution).
        "slices_remote_dir": slices_remote_dir,
        # Per-slice remote paths: accounts for interpolated slices that live in
        # a different publish directory (interpolate_missing_slice/) than the
        # common-space slices (bring_to_common_space/).  Takes precedence over
        # slices_remote_dir when the plugin resolves a slice path.
        "slice_remote_paths": slice_remote_paths,
        # IDs of slices that were synthesised by the interpolation step rather
        # than acquired directly.  The plugin can use this list to label them
        # as "[interpolated]" and to warn the user that the content is synthetic.
        "interpolated_slice_ids": interpolated_ids,
        "cross_section_level": level,
    }
    metadata_path = output_dir / "manual_align_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    logger.info(
        "Exported %s AIPs, %s paired XZ/YZ sets, and %s transforms to %s",
        len(slice_paths),
        len(pairs),
        len(transform_paths),
        output_dir,
    )
    logger.info("Metadata: %s", metadata_path)


if __name__ == "__main__":
    main()
