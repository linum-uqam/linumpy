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
import re
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from linumpy.io.zarr import read_omezarr

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _save_aip_npz(
    aip: np.ndarray,
    scale: np.ndarray,
    out_path: Path,
    center_pos: int | None = None,
) -> None:
    """Save one AIP projection to NPZ using the standard schema.

    *center_pos* is the Y index (for XZ cross-sections) or X index (for YZ
    cross-sections) at which the cross-section was taken.  Stored so the
    plugin can initialise its interactive slider at the tissue centroid.
    """
    kwargs: dict[str, Any] = {"aip": aip.astype(np.float32), "scale": np.array(scale, dtype=float)}
    if center_pos is not None:
        kwargs["center_pos"] = np.array(center_pos, dtype=np.int32)
    np.savez_compressed(str(out_path), **kwargs)


def _brightest_index(volume: np.ndarray, axis: int) -> int:
    """Return the index along *axis* whose summed intensity is highest."""
    return int(np.argmax(volume.sum(axis=tuple(i for i in range(volume.ndim) if i != axis))))


def _save_axis_views(
    volume: np.ndarray,
    scale: np.ndarray,
    sid: int,
    aips_xz_dir: Path,
    aips_yz_dir: Path,
) -> None:
    """Save XZ and YZ cross-sections as NPZ files.

    Unlike mean projections, single-slice cross-sections preserve structural
    detail (e.g. tissue boundaries) needed to judge Z-overlap alignment.
    The slice is chosen at the Y/X position with the highest integrated
    intensity, so the image is guaranteed to contain tissue even when the
    tissue does not occupy the geometric center of the field.

    Volume axis order is (Z, Y, X). The cross-sections are:
      XZ: brightest Y row  → shape (Z, X), scale (Z, X)
      YZ: brightest X col  → shape (Z, Y), scale (Z, Y)
    Both are flipped along Z so depth increases downward in the viewer.
    """
    if volume.ndim != 3 or min(volume.shape) == 0:
        return

    scale_arr = np.array(scale, dtype=float)
    cy = _brightest_index(volume, axis=1)  # best Y row for XZ view
    cx = _brightest_index(volume, axis=2)  # best X col for YZ view

    views = [
        # XZ: brightest row (fix Y = cy) → (Z, X), flip Z; center_pos = cy
        (aips_xz_dir, volume[:, cy, :][::-1, :], scale_arr[[0, 2]] if scale_arr.size >= 3 else scale_arr, cy),
        # YZ: brightest column (fix X = cx) → (Z, Y), flip Z; center_pos = cx
        (aips_yz_dir, volume[:, :, cx][::-1, :], scale_arr[[0, 1]] if scale_arr.size >= 3 else scale_arr, cx),
    ]

    for out_dir, img, img_scale, cp in views:
        _save_aip_npz(img, img_scale, out_dir / f"slice_z{sid:02d}.npz", center_pos=cp)


def _tissue_centroid(profile: np.ndarray) -> float:
    """Return the intensity-weighted centroid of a 1-D column/row profile.

    Weights are squared so that bright tissue dominates over low-level
    background noise.  Falls back to the mid-point if the profile is flat.
    """
    w = profile.astype(float) ** 2
    total = w.sum()
    if total == 0:
        return float(profile.size) / 2.0
    return float(np.dot(np.arange(profile.size, dtype=float), w) / total)


def _save_xy_aips_for_pair(
    fixed_arr: np.ndarray,
    moving_arr: np.ndarray,
    fixed_scale: np.ndarray,
    moving_scale: np.ndarray,
    overlap_px: int,
    fid: int,
    mid: int,
    aips_dir: Path,
) -> None:
    """Save paired XY AIPs covering the overlap zone at the edges of each volume.

    ``overlap_px`` is the number of Z voxels (at the working pyramid level) to
    average at each boundary:

    - **Fixed slice**: last *overlap_px* voxels of Z -- the bottom of the fixed
      volume, which physically overlaps with the top of the moving volume.
    - **Moving slice**: first *overlap_px* voxels of Z -- the top of the moving
      volume, which physically overlaps with the bottom of the fixed volume.

    Both projections cover the same tissue depth, giving matching structure in
    the XY overlay without relying on registration-derived Z offsets.

    Output filenames follow the same convention as paired XZ/YZ files:
    ``pair_z{fid:02d}_z{mid:02d}_fixed.npz`` and
    ``pair_z{fid:02d}_z{mid:02d}_moving.npz``.
    """
    if fixed_arr.ndim != 3 or moving_arr.ndim != 3:
        return
    if min(fixed_arr.shape) == 0 or min(moving_arr.shape) == 0:
        return

    nz_f = fixed_arr.shape[0]
    nz_m = moving_arr.shape[0]
    slab_f = min(overlap_px, nz_f)
    slab_m = min(overlap_px, nz_m)

    fixed_slab = fixed_arr[nz_f - slab_f :]
    moving_slab = moving_arr[:slab_m]

    fixed_aip = fixed_slab.mean(axis=0).astype(np.float32)
    moving_aip = moving_slab.mean(axis=0).astype(np.float32)

    pair_stem = f"pair_z{fid:02d}_z{mid:02d}"
    _save_aip_npz(fixed_aip, np.array(fixed_scale, dtype=float), aips_dir / f"{pair_stem}_fixed.npz")
    _save_aip_npz(moving_aip, np.array(moving_scale, dtype=float), aips_dir / f"{pair_stem}_moving.npz")


def _save_axis_views_for_pair(
    fixed_arr: np.ndarray,
    moving_arr: np.ndarray,
    fixed_scale: np.ndarray,
    moving_scale: np.ndarray,
    fixed_z: int,
    moving_z: int,
    fid: int,
    mid: int,
    aips_xz_dir: Path,
    aips_yz_dir: Path,
) -> None:
    """Save paired XZ/YZ cross-sections that share the same column position.

    Column selection strategy
    -------------------------
    Rather than picking the global intensity peak (which is biased toward
    whichever slice is brighter), we:

    1. Average a ±5 % Z-slab around each volume's overlap depth to suppress
       noisy single-slice artefacts at the section boundary.
    2. Compute the intensity-weighted centroid of the column profile for each
       slice independently and take their average.  The centroid is robust to
       lateral tissue displacement between consecutive slices, which is exactly
       the misalignment the plugin is designed to correct.

    Both slices are then cut at this shared Y (XZ) and X (YZ) column,
    guaranteeing that consecutive slices always show the same anatomical
    cross-section plane.

    Output filenames: ``pair_z{fid:02d}_z{mid:02d}_fixed.npz`` and
    ``pair_z{fid:02d}_z{mid:02d}_moving.npz``.
    """
    if fixed_arr.ndim != 3 or moving_arr.ndim != 3:
        return
    if min(fixed_arr.shape) == 0 or min(moving_arr.shape) == 0:
        return

    # Clamp overlap indices to valid range
    fz = max(0, min(fixed_z, fixed_arr.shape[0] - 1))
    mz = max(0, min(moving_z, moving_arr.shape[0] - 1))

    # Average a ±5 % Z-slab so a single noisy boundary slice does not dominate
    slab = max(1, int(0.05 * fixed_arr.shape[0]))
    fo_slab = fixed_arr[max(0, fz - slab) : min(fixed_arr.shape[0], fz + slab + 1)]
    mo_slab = moving_arr[max(0, mz - slab) : min(moving_arr.shape[0], mz + slab + 1)]

    def _mean2d(vol_slab: np.ndarray) -> np.ndarray:
        """Mean over Z slab, normalised to [0, 1]."""
        img = vol_slab.mean(axis=0).astype(float)
        mx = img.max()
        return img / mx if mx > 0 else img

    fo = _mean2d(fo_slab)  # (Y, X)
    mo = _mean2d(mo_slab)  # (Y, X)

    ny = min(fo.shape[0], mo.shape[0])
    nx = min(fo.shape[1], mo.shape[1])
    fo, mo = fo[:ny, :nx], mo[:ny, :nx]

    # Centroid of each slice's column profile, averaged to find the shared column.
    # Using the average of two centroids rather than argmax of the combined sum
    # handles the common case where the two slices have laterally shifted tissue.
    cy_f = _tissue_centroid(fo.sum(axis=1))
    cy_m = _tissue_centroid(mo.sum(axis=1))
    cy = round((cy_f + cy_m) / 2.0)

    cx_f = _tissue_centroid(fo.sum(axis=0))
    cx_m = _tissue_centroid(mo.sum(axis=0))
    cx = round((cx_f + cx_m) / 2.0)

    pair_stem = f"pair_z{fid:02d}_z{mid:02d}"

    for role, arr, scale_arr in [
        ("fixed", fixed_arr, fixed_scale),
        ("moving", moving_arr, moving_scale),
    ]:
        # Clamp to this volume's actual dimensions
        cy_i = min(cy, arr.shape[1] - 1)
        cx_i = min(cx, arr.shape[2] - 1)
        sc = np.array(scale_arr, dtype=float)
        sc_xz = sc[[0, 2]] if sc.size >= 3 else sc
        sc_yz = sc[[0, 1]] if sc.size >= 3 else sc

        # XZ: fix Y = cy_i → (Z, X), flip Z so depth increases downward
        _save_aip_npz(arr[:, cy_i, :][::-1, :], sc_xz, aips_xz_dir / f"{pair_stem}_{role}.npz", center_pos=cy_i)
        # YZ: fix X = cx_i → (Z, Y), flip Z
        _save_aip_npz(arr[:, :, cx_i][::-1, :], sc_yz, aips_yz_dir / f"{pair_stem}_{role}.npz", center_pos=cx_i)


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


def _is_interpolated(path: Path) -> bool:
    """Return True if this slice was produced by the interpolation step.

    Interpolated slices are named ``slice_z{N}_interpolated.ome.zarr``
    (the ``_interpolated`` suffix is set by ``linum_interpolate_missing_slice.py``).
    """
    return "_interpolated" in path.name


def _discover_slices(slices_dir: Path) -> dict[int, Path]:
    """Discover common-space slice files."""
    pattern = re.compile(r"slice_z(\d+)")
    slices = {}
    for p in sorted(slices_dir.iterdir()):
        m = pattern.search(p.name)
        if m and p.name.endswith(".ome.zarr"):
            slices[int(m.group(1))] = p
    return dict(sorted(slices.items()))


def _discover_transforms(transforms_dir: Path) -> dict[int, Path]:
    """Discover pairwise transform directories."""
    pattern = re.compile(r"slice_z(\d+)")
    transforms = {}
    for p in sorted(transforms_dir.iterdir()):
        if p.is_dir():
            m = pattern.search(p.name)
            if m:
                transforms[int(m.group(1))] = p
    return dict(sorted(transforms.items()))


def _read_overlap_z_offsets(offsets_file: Path) -> tuple[int, int]:
    """Load (fixed_z, moving_z) from pairwise ``offsets.txt``, or (0, 0) if missing/invalid."""
    if not offsets_file.exists():
        return 0, 0
    try:
        arr_off = np.loadtxt(str(offsets_file), dtype=int)
        if arr_off.size >= 2:
            return int(arr_off[0]), int(arr_off[1])
    except OSError, ValueError:
        pass
    return 0, 0


def _slice_task(args: tuple) -> int:
    """Worker for Pass 1: load one zarr slice, write XY AIP + per-slice XZ/YZ NPZ files."""
    sid, spath_str, level, aips_dir, aips_xz_dir, aips_yz_dir = args
    vol, scale = read_omezarr(spath_str, level=level)
    arr = np.asarray(vol)
    scale_arr = np.array(scale, dtype=float)
    _save_aip_npz(arr.mean(axis=0), scale_arr, Path(aips_dir) / f"slice_z{sid:02d}.npz")
    _save_axis_views(arr, scale_arr, sid, Path(aips_xz_dir), Path(aips_yz_dir))
    return sid


def _pair_task(args: tuple) -> tuple[int, int]:
    """Worker for Pass 2: load two zarr slices, write paired XY, XZ, and YZ NPZ files."""
    (
        fid,
        mid,
        fpath_str,
        mpath_str,
        fixed_z,
        moving_z,
        level,
        overlap_px,
        aips_dir,
        aips_xz_dir,
        aips_yz_dir,
    ) = args
    fixed_vol, fixed_scale = read_omezarr(fpath_str, level=level)
    moving_vol, moving_scale = read_omezarr(mpath_str, level=level)
    fixed_arr = np.asarray(fixed_vol)
    moving_arr = np.asarray(moving_vol)
    fixed_scale_arr = np.array(fixed_scale, dtype=float)
    moving_scale_arr = np.array(moving_scale, dtype=float)
    _save_axis_views_for_pair(
        fixed_arr,
        moving_arr,
        fixed_scale_arr,
        moving_scale_arr,
        fixed_z,
        moving_z,
        fid,
        mid,
        Path(aips_xz_dir),
        Path(aips_yz_dir),
    )
    _save_xy_aips_for_pair(
        fixed_arr,
        moving_arr,
        fixed_scale_arr,
        moving_scale_arr,
        overlap_px,
        fid,
        mid,
        Path(aips_dir),
    )
    return fid, mid


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
