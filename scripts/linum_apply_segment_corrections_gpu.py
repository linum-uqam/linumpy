#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apply segment break corrections to common-space slices.

GPU-accelerated version using CuPy for 3D volume resampling.
Falls back to CPU (SimpleITK) if no GPU is available.

Uses the ``segment_corrections.json`` produced by
``linum_detect_segment_breaks.py`` to rigidly correct the orientation of slices
that belong to acquisition segments acquired after a sample-remounting event.

For each affected slice (i.e., slices whose ID appears in
``per_slice_corrections``), the script:

1. Loads the 3D OME-Zarr volume.
2. Applies a rigid transform (in-plane rotation around the Z-axis + XY
   translation) to the entire volume.
3. Saves the corrected volume as a new OME-Zarr file in ``out_directory``.

The GPU path uses ``cupyx.scipy.ndimage.affine_transform`` to resample the 3D
volume, which is significantly faster than CPU for large volumes.

Unaffected slices (no correction needed) are symlinked into ``out_directory``
so that the output directory always contains the complete, correctly ordered
set of slices regardless of how many breaks were detected.

Usage
-----
    linum_apply_segment_corrections_gpu.py \\
        <in_slices_dir> <segment_corrections.json> <out_directory>
"""

import linumpy._thread_config  # noqa: F401

import argparse
import json
import logging
import os
import re
import shutil
from pathlib import Path

import dask.array as da
import numpy as np
import SimpleITK as sitk

from linumpy.gpu import GPU_AVAILABLE
from linumpy.io.zarr import read_omezarr, save_omezarr
from linumpy.utils.io import add_overwrite_arg

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_slices_dir',
                   help='Directory containing common-space slice .ome.zarr files '
                        '(output of bring_to_common_space).')
    p.add_argument('in_corrections',
                   help='segment_corrections.json produced by '
                        'linum_detect_segment_breaks.py.')
    p.add_argument('out_directory',
                   help='Output directory for corrected slices.')

    p.add_argument('--n_levels', type=int, default=3,
                   help='Number of pyramid levels to write for corrected slices. '
                        '[%(default)s]')
    p.add_argument('--use_gpu', default=True,
                   action=argparse.BooleanOptionalAction,
                   help='Use GPU for 3D volume resampling when available. '
                        '[%(default)s]')

    add_overwrite_arg(p)
    return p


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_slice_id(path: Path) -> int:
    m = re.search(r'z(\d+)', path.name)
    return int(m.group(1)) if m else -1


def _discover_slices(slices_dir: Path) -> list[Path]:
    slices = sorted(
        [p for p in slices_dir.iterdir()
         if p.is_dir() and p.name.endswith('.ome.zarr') and re.search(r'z\d+', p.name)],
        key=_extract_slice_id,
    )
    if not slices:
        raise FileNotFoundError(
            f"No .ome.zarr slice directories found in {slices_dir}")
    return slices


# ---------------------------------------------------------------------------
# Transform application
# ---------------------------------------------------------------------------

def apply_rigid_correction(
    vol: np.ndarray,
    res: tuple,
    rotation_deg: float,
    tx_px: float,
    ty_px: float,
    use_gpu: bool = False,
) -> np.ndarray:
    """
    Apply an in-plane rigid correction (rotation around Z + XY translation) to
    a 3D volume with shape (Z, Y, X).

    Parameters
    ----------
    vol : np.ndarray
        Input volume, shape (Z, Y, X).
    res : tuple
        Voxel size in mm, (z_mm, y_mm, x_mm).
    rotation_deg : float
        Rotation angle around the Z-axis in degrees.
        Positive = counter-clockwise when viewed from above.
    tx_px : float
        Translation in the X direction (columns) in pixels.
    ty_px : float
        Translation in the Y direction (rows) in pixels.
    use_gpu : bool
        Use GPU (CuPy) for the resampling step when available.

    Returns
    -------
    np.ndarray
        Corrected volume, same shape and dtype as input.
    """
    if abs(rotation_deg) < 1e-6 and abs(tx_px) < 1e-3 and abs(ty_px) < 1e-3:
        return vol  # nothing to do

    nz, ny, nx = vol.shape

    if use_gpu and GPU_AVAILABLE:
        import cupy as cp
        from cupyx.scipy.ndimage import affine_transform as cp_affine

        a = np.radians(rotation_deg)
        cos_a, sin_a = np.cos(a), np.sin(a)

        # Array-index center (assumes isotropic in-plane spacing at level-0).
        cx_arr = nx / 2.0
        cy_arr = ny / 2.0

        # Inverse (output→input) mapping for affine_transform in (Z, Y, X) space.
        # Replicates the Euler3DTransform inverse used in the CPU path:
        #   X_in = cx + cos(a)*(X_out - cx - tx) + sin(a)*(Y_out - cy - ty)
        #   Y_in = cy - sin(a)*(X_out - cx - tx) + cos(a)*(Y_out - cy - ty)
        matrix = np.array([[1.0,    0.0,     0.0],
                           [0.0,  cos_a, -sin_a],
                           [0.0,  sin_a,   cos_a]])
        offset = np.array([
            0.0,
            cy_arr + sin_a * (cx_arr + tx_px) - cos_a * (cy_arr + ty_px),
            cx_arr - sin_a * (cy_arr + ty_px) - cos_a * (cx_arr + tx_px),
        ])

        vol_gpu = cp.asarray(vol.astype(np.float32))
        corrected_gpu = cp_affine(vol_gpu, cp.asarray(matrix), offset=cp.asarray(offset),
                                  order=1, mode='constant', cval=0.0)
        corrected = cp.asnumpy(corrected_gpu).astype(vol.dtype)
        del vol_gpu, corrected_gpu
        cp.get_default_memory_pool().free_all_blocks()
        return corrected

    # -----------------------------------------------------------------------
    # CPU path via SimpleITK Euler3DTransform
    # -----------------------------------------------------------------------

    # Build SimpleITK image: GetImageFromArray expects (Z, Y, X) = (z, y, x)
    # and converts to physical (x, y, z) internally.
    vol_f32 = vol.astype(np.float32)
    sitk_vol = sitk.GetImageFromArray(vol_f32)

    # Physical voxel spacing: SimpleITK uses (x, y, z) order for spacing.
    # res = (z_mm, y_mm, x_mm) from OME metadata; convert µm convention if needed.
    # read_omezarr returns resolution in mm for OME-NGFF.
    if len(res) >= 3:
        sp_x, sp_y, sp_z = float(res[2]), float(res[1]), float(res[0])
    elif len(res) == 2:
        sp_x, sp_y, sp_z = float(res[1]), float(res[0]), 1.0
    else:
        sp_x = sp_y = sp_z = 1.0
    sitk_vol.SetSpacing((sp_x, sp_y, sp_z))

    # Euler 3D transform: rotate only around Z-axis.
    transform = sitk.Euler3DTransform()

    # Centre of rotation: physical coordinates at (cx, cy, 0)
    cx = nx * sp_x / 2.0
    cy = ny * sp_y / 2.0
    transform.SetCenter([cx, cy, 0.0])

    # Rotation around Z (third Euler angle), others zero.
    transform.SetRotation(0.0, 0.0, float(np.radians(rotation_deg)))

    # Translation in physical units (mm).
    transform.SetTranslation([tx_px * sp_x, ty_px * sp_y, 0.0])

    # Resample
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(sitk_vol)
    resampler.SetTransform(transform)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0.0)

    corrected_sitk = resampler.Execute(sitk_vol)
    corrected = sitk.GetArrayFromImage(corrected_sitk)  # back to (Z, Y, X)

    return corrected.astype(vol.dtype)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = _build_arg_parser()
    args = p.parse_args()

    slices_dir = Path(args.in_slices_dir)
    corrections_path = Path(args.in_corrections)
    out_dir = Path(args.out_directory)

    if out_dir.exists() and not args.overwrite:
        p.error(f"Output directory already exists: {out_dir}. Use -f to overwrite.")
    out_dir.mkdir(parents=True, exist_ok=True)

    use_gpu = args.use_gpu and GPU_AVAILABLE
    if args.use_gpu and not GPU_AVAILABLE:
        logger.warning("--use_gpu requested but no GPU available; falling back to CPU.")
    logger.info(f"GPU resampling: {'enabled' if use_gpu else 'disabled'}")

    # -----------------------------------------------------------------------
    # Load corrections
    # -----------------------------------------------------------------------
    with open(corrections_path) as fh:
        corrections_data = json.load(fh)

    n_breaks = corrections_data.get('n_breaks', 0)
    per_slice = corrections_data.get('per_slice_corrections', {})
    # Keys are stored as strings in JSON; convert to int
    per_slice = {int(k): v for k, v in per_slice.items()}

    logger.info(f"Corrections file: {corrections_path}")
    logger.info(f"Breaks in dataset: {n_breaks}")
    logger.info(f"Slices requiring correction: {len(per_slice)}")

    # -----------------------------------------------------------------------
    # Discover slices
    # -----------------------------------------------------------------------
    slices = _discover_slices(slices_dir)
    logger.info(f"Found {len(slices)} slices in input directory.")

    # -----------------------------------------------------------------------
    # Process each slice
    # -----------------------------------------------------------------------
    n_corrected = 0
    n_linked = 0

    for slice_path in slices:
        sid = _extract_slice_id(slice_path)
        out_path = out_dir / slice_path.name

        if sid in per_slice:
            correction = per_slice[sid]
            rot = correction['rotation_deg']
            tx = correction['tx_px']
            ty = correction['ty_px']

            logger.info(
                f"  z{sid:02d}: applying correction "
                f"rot={rot:+.2f}°  tx={tx:+.1f}px  ty={ty:+.1f}px"
            )

            vol, res = read_omezarr(str(slice_path), level=0)
            arr = np.array(vol)

            corrected = apply_rigid_correction(arr, res, rot, tx, ty, use_gpu=use_gpu)

            save_omezarr(
                da.from_array(corrected),
                str(out_path),
                voxel_size=res,
                n_levels=args.n_levels,
                overwrite=True,
            )
            n_corrected += 1

        else:
            # No correction needed — symlink to save space and time.
            if out_path.exists() or out_path.is_symlink():
                out_path.unlink() if out_path.is_file() else shutil.rmtree(out_path)
            os.symlink(slice_path.resolve(), out_path)
            logger.debug(f"  z{sid:02d}: symlinked (no correction)")
            n_linked += 1

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SEGMENT CORRECTION SUMMARY")
    print("=" * 60)
    print(f"Total slices      : {len(slices)}")
    print(f"Corrected (written): {n_corrected}")
    print(f"Unchanged (linked) : {n_linked}")
    if n_breaks == 0:
        print("\nNo segment breaks detected — all slices passed through unchanged.")
    else:
        print(f"\n{n_breaks} break(s) corrected.")
    print("=" * 60)


if __name__ == '__main__':
    main()
