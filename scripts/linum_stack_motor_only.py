#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stack slices into a 3D volume using only motor positions (no pairwise registration).

This diagnostic tool creates a 3D stack using ONLY the XY shifts from the shifts_xy.csv
file (motor/stage positions recorded during acquisition). By comparing this "motor-only"
stack against the fully-registered stack, you can:

1. **Validate motor positions**: Check if motor positions alone provide good alignment
2. **Identify registration artifacts**: See if pairwise registration introduces errors
3. **Debug dilation issues**: Verify if XY drift is the main cause of misalignment

The script reads the shifts file and positions each slice according to its cumulative
XY shift, without any image-based registration refinement.
"""
import linumpy._thread_config  # noqa: F401

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from linumpy.io.zarr import read_omezarr, AnalysisOmeZarrWriter
from linumpy.utils.io import add_overwrite_arg, assert_output_exists

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_slices_dir',
                   help='Directory containing slice volumes (.ome.zarr)')
    p.add_argument('in_shifts',
                   help='CSV file with XY shifts (shifts_xy.csv)')
    p.add_argument('out_volume',
                   help='Output stacked volume (.ome.zarr)')

    p.add_argument('--blending', type=str, default='none',
                   choices=['none', 'average', 'max', 'feather'],
                   help='Blending method for overlapping regions [%(default)s]\n'
                        '  none: No blending, later slices overwrite earlier ones\n'
                        '  average: Average overlapping voxels\n'
                        '  max: Take maximum of overlapping voxels\n'
                        '  feather: Feathered blending based on distance from edge')
    p.add_argument('--overlap_slices', type=int, default=0,
                   help='Number of z-slices to overlap between consecutive slices [%(default)s]\n'
                        'Set to 0 to stack without z-overlap (just XY alignment)')
    p.add_argument('--z_spacing_um', type=float, default=None,
                   help='Z spacing between slices in microns.\n'
                        'If not provided, uses spacing from first slice metadata.')
    p.add_argument('--center_drift', action='store_true', default=True,
                   help='Center cumulative drift around middle slice [%(default)s]')
    p.add_argument('--no_center_drift', action='store_false', dest='center_drift',
                   help='Do not center drift')
    p.add_argument('--slice_pattern', type=str, default=r'slice_z(\d+)',
                   help='Regex pattern to extract slice ID from filename [%(default)s]')
    p.add_argument('--preview', type=str, default=None,
                   help='Output path for preview image (.png)')
    p.add_argument('--max_slices', type=int, default=None,
                   help='Maximum number of slices to stack (for testing)')

    add_overwrite_arg(p)
    return p


def load_shifts(shifts_path):
    """Load shifts CSV and build cumulative shift lookup."""
    df = pd.read_csv(shifts_path)

    # Build cumulative shifts
    # The shifts file contains pairwise shifts: fixed_id -> moving_id
    # We need to accumulate these to get absolute positions

    # Get all unique slice IDs
    all_ids = sorted(set(df['fixed_id'].tolist() + df['moving_id'].tolist()))

    # Build shift lookup
    shift_lookup = {}
    for _, row in df.iterrows():
        fixed_id = int(row['fixed_id'])
        moving_id = int(row['moving_id'])
        shift_lookup[(fixed_id, moving_id)] = (row['x_shift_mm'], row['y_shift_mm'])

    # Compute cumulative shifts (first slice is at origin)
    cumsum = {all_ids[0]: (0.0, 0.0)}
    for i in range(len(all_ids) - 1):
        fixed_id = all_ids[i]
        moving_id = all_ids[i + 1]

        if (fixed_id, moving_id) in shift_lookup:
            dx_mm, dy_mm = shift_lookup[(fixed_id, moving_id)]
        else:
            logger.warning(f"No shift found for {fixed_id} -> {moving_id}, using 0")
            dx_mm, dy_mm = 0.0, 0.0

        prev_dx, prev_dy = cumsum[fixed_id]
        cumsum[moving_id] = (prev_dx + dx_mm, prev_dy + dy_mm)

    return cumsum, all_ids


def convert_shifts_to_pixels(cumsum, resolution_um):
    """Convert mm shifts to pixel shifts."""
    # resolution_um is in microns per pixel
    mm_to_px = 1000.0 / resolution_um

    return {
        slice_id: (dx_mm * mm_to_px, dy_mm * mm_to_px)
        for slice_id, (dx_mm, dy_mm) in cumsum.items()
    }


def center_shifts(cumsum_px, slice_ids):
    """Center shifts around the middle slice."""
    if not slice_ids:
        return cumsum_px

    middle_idx = len(slice_ids) // 2
    middle_id = slice_ids[middle_idx]
    center_dx, center_dy = cumsum_px.get(middle_id, (0, 0))

    return {
        slice_id: (dx - center_dx, dy - center_dy)
        for slice_id, (dx, dy) in cumsum_px.items()
    }


def compute_output_shape(slice_files, cumsum_px, overlap_slices=0):
    """Compute the output volume shape to fit all slices."""
    xmin, xmax, ymin, ymax = [], [], [], []
    total_z = 0

    for slice_id, slice_file in slice_files.items():
        vol, _ = read_omezarr(str(slice_file), level=0)
        dx, dy = cumsum_px.get(slice_id, (0, 0))

        z_depth = vol.shape[0]
        height = vol.shape[1]  # Y
        width = vol.shape[2]   # X

        xmin.append(dx)
        xmax.append(dx + width)
        ymin.append(dy)
        ymax.append(dy + height)
        total_z += z_depth

    # Account for overlap
    n_slices = len(slice_files)
    if overlap_slices > 0 and n_slices > 1:
        total_z -= overlap_slices * (n_slices - 1)

    x0 = min(xmin)
    y0 = min(ymin)
    nx = int(np.ceil(max(xmax) - x0))
    ny = int(np.ceil(max(ymax) - y0))

    return (total_z, ny, nx), (x0, y0)


def apply_shift_to_slice(slice_data, dx, dy, output_shape_yx):
    """
    Compute the output region and source region for placing a shifted slice.

    Returns the slice data and the destination coordinates, without allocating
    a full-size output array.

    Returns
    -------
    slice_data : np.ndarray
        The (possibly cropped) slice data to write
    dst_coords : tuple
        (y_start, y_end, x_start, x_end) in output coordinates
    """
    out_ny, out_nx = output_shape_yx

    # Compute destination coordinates
    dst_x_start = int(round(dx))
    dst_y_start = int(round(dy))
    dst_x_end = dst_x_start + slice_data.shape[2]
    dst_y_end = dst_y_start + slice_data.shape[1]

    # Compute source crop if destination is out of bounds
    src_x_start = max(0, -dst_x_start)
    src_y_start = max(0, -dst_y_start)
    src_x_end = slice_data.shape[2] - max(0, dst_x_end - out_nx)
    src_y_end = slice_data.shape[1] - max(0, dst_y_end - out_ny)

    # Clip destination to output bounds
    dst_x_start = max(0, dst_x_start)
    dst_y_start = max(0, dst_y_start)
    dst_x_end = min(out_nx, dst_x_end)
    dst_y_end = min(out_ny, dst_y_end)

    # Crop the source data
    if src_x_end > src_x_start and src_y_end > src_y_start:
        cropped = slice_data[:, src_y_start:src_y_end, src_x_start:src_x_end]
        return cropped, (dst_y_start, dst_y_end, dst_x_start, dst_x_end)
    else:
        return None, None


def blend_overlap(existing, new_data, method='none'):
    """Blend overlapping regions."""
    if method == 'none':
        # New data overwrites existing (where new data is non-zero)
        mask = new_data != 0
        existing[mask] = new_data[mask]
        return existing

    elif method == 'average':
        # Average where both have data
        both_valid = (existing != 0) & (new_data != 0)
        only_new = (existing == 0) & (new_data != 0)
        existing[both_valid] = (existing[both_valid] + new_data[both_valid]) / 2
        existing[only_new] = new_data[only_new]
        return existing

    elif method == 'max':
        # Take maximum
        return np.maximum(existing, new_data)

    elif method == 'feather':
        # Feathered blending (simple version based on distance from edge)
        # TODO: Implement proper distance-based feathering
        return blend_overlap(existing, new_data, 'average')

    return existing


def generate_preview(volume, output_path):
    """Generate a preview image of the stacked volume."""
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Middle slices in each dimension
        z_mid = volume.shape[0] // 2
        y_mid = volume.shape[1] // 2
        x_mid = volume.shape[2] // 2

        # XY slice (axial)
        axes[0].imshow(volume[z_mid, :, :], cmap='gray')
        axes[0].set_title(f'XY (z={z_mid})')
        axes[0].axis('off')

        # XZ slice (coronal)
        axes[1].imshow(volume[:, y_mid, :], cmap='gray', aspect='auto')
        axes[1].set_title(f'XZ (y={y_mid})')
        axes[1].axis('off')

        # YZ slice (sagittal)
        axes[2].imshow(volume[:, :, x_mid], cmap='gray', aspect='auto')
        axes[2].set_title(f'YZ (x={x_mid})')
        axes[2].axis('off')

        plt.suptitle('Motor-Only Stack Preview')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

        logger.info(f"Preview saved to {output_path}")
    except Exception as e:
        logger.warning(f"Could not generate preview: {e}")


def generate_preview_from_slice(slice_2d, output_path):
    """Generate a preview image from a single 2D slice."""
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        # Normalize for display
        vmin = np.percentile(slice_2d[slice_2d > 0], 1) if np.any(slice_2d > 0) else 0
        vmax = np.percentile(slice_2d, 99)

        ax.imshow(slice_2d, cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_title('Motor-Only Stack (middle Z slice)')
        ax.axis('off')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Preview saved to {output_path}")
    except Exception as e:
        logger.warning(f"Could not generate preview: {e}")


def generate_preview_from_zarr(zarr_output, output_path):
    """Generate a 3-panel preview (XY, XZ, YZ) from a zarr output without loading full volume."""
    try:
        import matplotlib.pyplot as plt

        # Get shape from zarr
        shape = zarr_output.shape  # (Z, Y, X)
        z_mid = shape[0] // 2
        y_mid = shape[1] // 2
        x_mid = shape[2] // 2

        # Read only the slices we need
        xy_slice = np.array(zarr_output[z_mid, :, :])  # XY at middle Z
        xz_slice = np.array(zarr_output[:, y_mid, :])  # XZ at middle Y
        yz_slice = np.array(zarr_output[:, :, x_mid])  # YZ at middle X

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Normalize each slice for display
        def normalize_slice(s):
            valid = s > 0
            if np.any(valid):
                vmin = np.percentile(s[valid], 1)
                vmax = np.percentile(s, 99)
            else:
                vmin, vmax = 0, 1
            return vmin, vmax

        # XY slice (axial)
        vmin, vmax = normalize_slice(xy_slice)
        axes[0].imshow(xy_slice, cmap='gray', vmin=vmin, vmax=vmax)
        axes[0].set_title(f'XY (z={z_mid})')
        axes[0].axis('off')

        # XZ slice (coronal)
        vmin, vmax = normalize_slice(xz_slice)
        axes[1].imshow(xz_slice, cmap='gray', vmin=vmin, vmax=vmax, aspect='auto')
        axes[1].set_title(f'XZ (y={y_mid})')
        axes[1].axis('off')

        # YZ slice (sagittal)
        vmin, vmax = normalize_slice(yz_slice)
        axes[2].imshow(yz_slice, cmap='gray', vmin=vmin, vmax=vmax, aspect='auto')
        axes[2].set_title(f'YZ (x={x_mid})')
        axes[2].axis('off')

        plt.suptitle(f'Motor-Only Stack Preview ({shape[0]} x {shape[1]} x {shape[2]})')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"3-panel preview saved to {output_path}")
    except Exception as e:
        logger.warning(f"Could not generate preview: {e}")


def main():
    p = _build_arg_parser()
    args = p.parse_args()

    slices_dir = Path(args.in_slices_dir)
    shifts_path = Path(args.in_shifts)
    output_path = Path(args.out_volume)

    assert_output_exists(output_path, p, args)

    # Find slice files
    slice_files_list = sorted(slices_dir.glob('*.ome.zarr'))
    if not slice_files_list:
        p.error(f"No .ome.zarr files found in {slices_dir}")

    # Extract slice IDs and build mapping
    pattern = re.compile(args.slice_pattern)
    slice_files = {}
    for f in slice_files_list:
        match = pattern.search(f.name)
        if match:
            slice_id = int(match.group(1))
            slice_files[slice_id] = f

    if not slice_files:
        p.error(f"No files matched pattern '{args.slice_pattern}' in {slices_dir}")

    logger.info(f"Found {len(slice_files)} slice files")

    # Limit slices for testing
    if args.max_slices:
        slice_ids = sorted(slice_files.keys())[:args.max_slices]
        slice_files = {k: slice_files[k] for k in slice_ids}
        logger.info(f"Limited to {len(slice_files)} slices for testing")

    # Load shifts
    logger.info(f"Loading shifts from {shifts_path}")
    cumsum_mm, all_shift_ids = load_shifts(shifts_path)

    # Get resolution from first slice
    # NOTE: read_omezarr returns resolution in MILLIMETERS (OME-NGFF standard)
    first_slice_id = sorted(slice_files.keys())[0]
    first_vol, first_res = read_omezarr(str(slice_files[first_slice_id]), level=0)

    # Resolution: res is [z, y, x] in mm from OME-NGFF, convert to µm
    res_x_mm = first_res[-1] if len(first_res) >= 3 else first_res[0]
    res_y_mm = first_res[-2] if len(first_res) >= 3 else first_res[0]
    res_z_mm = first_res[0] if len(first_res) >= 3 else 0.200  # default 200 µm

    # Convert to µm for display and calculations
    res_x_um = res_x_mm * 1000
    res_y_um = res_y_mm * 1000
    res_z_um = res_z_mm * 1000

    if args.z_spacing_um:
        res_z_um = args.z_spacing_um

    logger.info(f"Resolution: Z={res_z_um:.2f} µm, Y={res_y_um:.2f} µm, X={res_x_um:.2f} µm")

    # Convert shifts to pixels (use X resolution for both X and Y for simplicity)
    cumsum_px = convert_shifts_to_pixels(cumsum_mm, res_x_um)

    # Filter to only slices we have files for
    available_ids = sorted(slice_files.keys())
    cumsum_px = {k: v for k, v in cumsum_px.items() if k in available_ids}

    # Fill in missing shifts with zero
    for slice_id in available_ids:
        if slice_id not in cumsum_px:
            logger.warning(f"No shift for slice {slice_id}, using (0, 0)")
            cumsum_px[slice_id] = (0.0, 0.0)

    # Center drift if requested
    if args.center_drift:
        cumsum_px = center_shifts(cumsum_px, available_ids)
        logger.info("Centered drift around middle slice")

    # Compute output shape
    logger.info("Computing output shape...")
    output_shape, (x0, y0) = compute_output_shape(slice_files, cumsum_px, args.overlap_slices)
    logger.info(f"Output shape: {output_shape} (Z, Y, X)")
    logger.info(f"Origin offset: ({x0:.1f}, {y0:.1f}) px")

    # Adjust shifts by origin
    cumsum_px = {
        slice_id: (dx - x0, dy - y0)
        for slice_id, (dx, dy) in cumsum_px.items()
    }

    # Stack slices
    logger.info(f"Stacking {len(slice_files)} slices (blending: {args.blending})...")

    # Use chunked writer to avoid memory issues
    output = AnalysisOmeZarrWriter(
        str(output_path), output_shape,
        chunk_shape=(min(100, output_shape[0]), min(512, output_shape[1]), min(512, output_shape[2])),
        dtype=np.float32
    )

    z_cursor = 0
    for i, slice_id in enumerate(available_ids):
        slice_file = slice_files[slice_id]
        vol, _ = read_omezarr(str(slice_file), level=0)
        vol_data = np.array(vol[:]).astype(np.float32)

        dx, dy = cumsum_px[slice_id]

        # Get the cropped slice and destination coordinates
        shifted, dst_coords = apply_shift_to_slice(vol_data, dx, dy, (output_shape[1], output_shape[2]))

        if shifted is None:
            logger.warning(f"Slice {slice_id} is entirely outside output bounds, skipping")
            continue

        dst_y_start, dst_y_end, dst_x_start, dst_x_end = dst_coords

        # Determine Z range for this slice
        z_start = z_cursor
        z_end = z_start + shifted.shape[0]

        # Handle overlap
        if args.overlap_slices > 0 and i > 0:
            z_start -= args.overlap_slices

        # Ensure we don't exceed output bounds
        if z_end > output_shape[0]:
            z_end = output_shape[0]
            shifted = shifted[:z_end - z_start]

        # Place/blend into output at the correct XY position
        if z_start < z_end:
            if args.blending == 'none' or i == 0:
                # No blending - just write to the specific region
                output[z_start:z_end, dst_y_start:dst_y_end, dst_x_start:dst_x_end] = shifted
            else:
                # Read existing region, blend, write back
                existing = np.array(output[z_start:z_end, dst_y_start:dst_y_end, dst_x_start:dst_x_end])
                blended = blend_overlap(existing, shifted, args.blending)
                output[z_start:z_end, dst_y_start:dst_y_end, dst_x_start:dst_x_end] = blended

        z_cursor = z_end

        logger.info(f"  Slice {slice_id:02d}: shift=({dx:.1f}, {dy:.1f}) px, z=[{z_start}:{z_end}], "
                    f"xy=[{dst_y_start}:{dst_y_end}, {dst_x_start}:{dst_x_end}]")

    # Finalize with pyramid
    logger.info("Finalizing and generating pyramid levels...")
    resolution = [res_z_um, res_y_um, res_x_um]
    output.finalize(resolution, n_levels=3)

    # Generate preview if requested
    if args.preview:
        # Generate 3-panel preview from zarr output
        generate_preview_from_zarr(output, args.preview)

    logger.info("Done!")


if __name__ == '__main__':
    main()
