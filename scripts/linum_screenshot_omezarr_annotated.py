#!/usr/bin/env python3
"""
Generate orthogonal view screenshots from an OME-Zarr volume with Z-slice index annotations.

Creates a figure showing coronal and sagittal views with Z-slice index numbers
marked on the side, making it easy to identify which input slice corresponds
to which horizontal band in the reconstruction.
"""
# Configure thread limits before numpy/scipy imports
import linumpy._thread_config  # noqa: F401

import argparse
from pathlib import Path
from linumpy.io.zarr import read_omezarr
import numpy as np
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("in_zarr",
                   help="Full path to a zarr file.")
    p.add_argument("out_figure",
                   help="Full path to the output figure")
    p.add_argument('--x_slice', type=int,
                   help='Slice index along the second axis (X/rows) for ZY view.')
    p.add_argument('--y_slice', type=int,
                   help='Slice index along the last axis (Y/columns) for ZX view.')
    p.add_argument('--n_slices', type=int,
                   help='Number of input slices (auto-detected from OME-Zarr metadata if not specified).')
    p.add_argument('--slice_ids', type=str,
                   help='Comma-separated list of actual slice IDs (e.g., "05,12,18"). '
                        'If provided, these will be shown instead of sequential numbers.')
    p.add_argument('--font_size', type=int, default=7,
                   help='Font size for slice labels (default: 7)')
    p.add_argument('--label_every', type=int, default=1,
                   help='Label every Nth slice (default: 1, label all)')
    p.add_argument('--show_lines', action='store_true',
                   help='Draw horizontal lines at slice boundaries')
    return p


def estimate_n_slices_from_zarr(zarr_path):
    """Try to estimate number of input slices from OME-Zarr metadata."""
    import zarr

    try:
        store = zarr.open(zarr_path, mode='r')

        # Check for custom metadata that might store slice info
        if '.zattrs' in store:
            attrs = dict(store.attrs)
            if 'n_input_slices' in attrs:
                return attrs['n_input_slices']
            if 'slice_boundaries' in attrs:
                return len(attrs['slice_boundaries'])

        # Check multiscales metadata for any hints
        if 'multiscales' in store.attrs:
            multiscales = store.attrs['multiscales']
            if isinstance(multiscales, list) and len(multiscales) > 0:
                ms = multiscales[0]
                if 'metadata' in ms and 'n_input_slices' in ms['metadata']:
                    return ms['metadata']['n_input_slices']
    except Exception:
        pass

    return None


def add_z_slice_labels(ax, n_input_slices, img_height, font_size=7, label_every=1,
                       show_lines=False, side='left', slice_ids=None):
    """Add Z-slice index labels on the side of a coronal/sagittal view.

    Parameters
    ----------
    ax : matplotlib axis
        The axis to annotate
    n_input_slices : int
        Number of input slices that were stacked (e.g., 64 physical slices)
    img_height : int
        Height of the displayed image in pixels (Z dimension of volume)
    font_size : int
        Font size for labels
    label_every : int
        Label every Nth slice (1 = all, 2 = every other, etc.)
    show_lines : bool
        Whether to draw horizontal lines at slice boundaries
    side : str
        Which side to put labels ('left' or 'right')
    slice_ids : list of str, optional
        Actual slice IDs to display (e.g., ['05', '12', '18']). If None, uses sequential numbers.
    """
    # Calculate voxels per input slice (each physical slice spans multiple Z voxels)
    voxels_per_slice = img_height / n_input_slices

    # Position for labels (slightly outside the image)
    x_pos = -0.02 if side == 'left' else 1.02
    ha = 'right' if side == 'left' else 'left'

    for slice_idx in range(n_input_slices):
        # Calculate y position in image coordinates (in pixels)
        # With origin='lower', y=0 is at bottom, y=img_height at top
        # slice_idx=0 should be at the bottom
        y_center_pixels = (slice_idx + 0.5) * voxels_per_slice

        # Only label every Nth slice
        if slice_idx % label_every == 0:
            # Use actual slice ID if provided, otherwise use sequential number
            if slice_ids is not None and slice_idx < len(slice_ids):
                label = f'z{slice_ids[slice_idx]}'
            else:
                label = f'z{slice_idx:02d}'
            # Use data coordinates for positioning (more accurate)
            ax.text(x_pos, y_center_pixels / img_height, label,
                    transform=ax.transAxes,
                    fontsize=font_size, color='white',
                    ha=ha, va='center',
                    fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.1', facecolor='black',
                              alpha=0.7, edgecolor='none'))

        # Draw slice boundary lines at the top of each slice region
        if show_lines and slice_idx > 0:
            y_line = slice_idx * voxels_per_slice
            ax.axhline(y=y_line, color='cyan', alpha=0.3, linewidth=0.5, linestyle='--')


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Validate input path
    in_path = Path(args.in_zarr)
    if not in_path.exists():
        parser.error(f"Input file not found: {args.in_zarr}")

    # Resolve symlinks (common in Nextflow work directories)
    in_path = in_path.resolve()

    image, _ = read_omezarr(str(in_path))

    # Get volume dimensions
    n_z_voxels, n_rows, n_cols = image.shape

    # Determine number of input slices
    n_input_slices = args.n_slices

    if n_input_slices is not None and n_input_slices > 0:
        print(f"Using provided n_slices: {n_input_slices}")
    else:
        n_input_slices = None  # Reset if invalid
        # Try to get from metadata
        n_input_slices = estimate_n_slices_from_zarr(str(in_path))
        if n_input_slices is not None:
            print(f"Got n_slices from metadata: {n_input_slices}")

    if n_input_slices is None:
        # Estimate from directory name or use heuristic
        # Look for pattern like "30_slices" or count from nearby slice files
        parent_dir = in_path.parent
        slice_files = list(parent_dir.glob('slice_z*.ome.zarr'))
        if slice_files:
            # Extract slice numbers and count
            slice_nums = []
            for f in slice_files:
                match = re.search(r'slice_z(\d+)', f.name)
                if match:
                    slice_nums.append(int(match.group(1)))
            if slice_nums:
                n_input_slices = max(slice_nums) - min(slice_nums) + 1
                print(f"Estimated n_slices from slice files: {n_input_slices}")

    if n_input_slices is None:
        # Last resort: estimate based on typical slice thickness
        # Assuming ~60 voxels per input slice at 10µm resolution (600µm slices)
        n_input_slices = max(1, n_z_voxels // 60)
        print(f"Warning: Could not determine n_input_slices, estimating {n_input_slices}")

    # Parse slice_ids if provided
    slice_ids = None
    if args.slice_ids:
        slice_ids = [s.strip() for s in args.slice_ids.split(',')]
        print(f"Using provided slice IDs: {slice_ids}")
        # Update n_input_slices to match slice_ids if not explicitly set
        if args.n_slices is None:
            n_input_slices = len(slice_ids)

    x_slice = args.x_slice if args.x_slice is not None else n_rows // 2
    y_slice = args.y_slice if args.y_slice is not None else n_cols // 2

    # Extract coronal and sagittal views (these show Z slices as horizontal bands)
    # ZY view (coronal) - shows Z on vertical axis
    image_zy = image[:, x_slice, :]
    # ZX view (sagittal) - shows Z on vertical axis
    image_zx = image[:, :, y_slice]

    # Calculate display range
    allvals = np.concatenate([image_zy.flatten(), image_zx.flatten()])
    vmin = np.min(allvals)
    vmax = np.percentile(allvals, 99.9)

    # Create figure with dark background - two panels side by side
    # Use aspect='equal' to avoid stretching the image
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12), facecolor='black')

    for ax in [ax1, ax2]:
        ax.set_facecolor('black')

    # Plot ZY view (coronal) - Z increases upward with origin='lower'
    # Use aspect='equal' to preserve true proportions (no stretching)
    ax1.imshow(image_zy, cmap='magma', origin='lower', vmin=vmin, vmax=vmax, aspect='equal')
    ax1.set_title(f'Coronal (ZY) view at X={x_slice}', color='white', fontsize=12, pad=10)
    ax1.set_xlabel('Y', color='white', fontsize=10)
    ax1.set_ylabel('Z', color='white', fontsize=10)
    ax1.tick_params(colors='white', labelsize=8)
    for spine in ax1.spines.values():
        spine.set_color('white')

    # Add Z-slice labels on the left side
    add_z_slice_labels(ax1, n_input_slices, image_zy.shape[0],
                       font_size=args.font_size,
                       label_every=args.label_every,
                       show_lines=args.show_lines,
                       side='left',
                       slice_ids=slice_ids)

    # Plot ZX view (sagittal) - Z increases upward with origin='lower'
    # Use aspect='equal' to preserve true proportions (no stretching)
    ax2.imshow(image_zx, cmap='magma', origin='lower', vmin=vmin, vmax=vmax, aspect='equal')
    ax2.set_title(f'Sagittal (ZX) view at Y={y_slice}', color='white', fontsize=12, pad=10)
    ax2.set_xlabel('X', color='white', fontsize=10)
    ax2.set_ylabel('Z', color='white', fontsize=10)
    ax2.tick_params(colors='white', labelsize=8)
    for spine in ax2.spines.values():
        spine.set_color('white')

    # Add Z-slice labels on the right side
    add_z_slice_labels(ax2, n_input_slices, image_zx.shape[0],
                       font_size=args.font_size,
                       label_every=args.label_every,
                       show_lines=args.show_lines,
                       side='right',
                       slice_ids=slice_ids)

    # Generate slice range string for title
    if slice_ids is not None:
        slice_range_str = f"slices: {slice_ids[0]}-{slice_ids[-1]}" if len(slice_ids) > 1 else f"slice: {slice_ids[0]}"
    else:
        slice_range_str = f"z00-z{n_input_slices-1:02d}"

    # Add overall title with volume info
    fig.suptitle(f'Z-Slice Index Reference: {n_input_slices} input slices ({slice_range_str})\n'
                 f'Volume: {n_z_voxels} Z × {n_rows} X × {n_cols} Y voxels',
                 color='white', fontsize=14, y=0.98)

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(args.out_figure, facecolor='black', edgecolor='none', dpi=150)
    plt.close(fig)

    print(f"Z-slice annotated screenshot saved to {args.out_figure}")
    if slice_ids is not None:
        print(f"Input slices: {n_input_slices} ({slice_range_str})")
    else:
        print(f"Input slices: {n_input_slices} (z00-z{n_input_slices-1:02d})")
    print(f"Volume: {n_z_voxels} Z × {n_rows} X × {n_cols} Y voxels")


if __name__ == '__main__':
    main()
