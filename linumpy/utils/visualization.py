# -*- coding: utf-8 -*-
"""
Volume visualization utilities.

Consolidated from linum_screenshot_omezarr.py and linum_screenshot_omezarr_annotated.py.
"""
import re
from pathlib import Path
from typing import Optional, List

import numpy as np


def save_orthogonal_views(image, out_path: str,
                          z_slice: int = None,
                          x_slice: int = None,
                          y_slice: int = None,
                          cmap: str = 'magma',
                          percentile_max: float = 99.9) -> None:
    """Save orthogonal (XY, XZ, YZ) views of a volume as a figure.

    Parameters
    ----------
    image : array-like
        3D volume (Z, X, Y) - as returned by read_omezarr.
    out_path : str
        Output figure path (e.g. 'view.png').
    z_slice, x_slice, y_slice : int or None
        Slice indices. Default: center of each axis.
    cmap : str
        Colormap (default 'magma').
    percentile_max : float
        Values above this percentile are clipped for display.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    z_slice = z_slice if z_slice is not None else image.shape[0] // 2
    x_slice = x_slice if x_slice is not None else image.shape[1] // 2
    y_slice = y_slice if y_slice is not None else image.shape[2] // 2

    image_z = np.array(image[z_slice, :, :]).T
    image_x = np.array(image[:, x_slice, :])
    image_x = image_x[::-1, ::-1]
    image_y = np.array(image[:, :, y_slice])
    image_y = image_y[::-1]

    width_ratio = [i.shape[1] for i in (image_z, image_x, image_y)]

    allvals = np.concatenate([image_x.flatten(), image_y.flatten(), image_z.flatten()])
    vmin = float(np.min(allvals))
    vmax = float(np.percentile(allvals, percentile_max))

    fig, ax = plt.subplots(1, 3, width_ratios=width_ratio)
    fig.set_size_inches(24, 10)
    fig.set_dpi(512)

    ax[0].imshow(image_z, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
    ax[1].imshow(image_x, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
    ax[2].imshow(image_y, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)

    for a in ax:
        a.set_axis_off()

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def estimate_n_slices_from_zarr(zarr_path: str) -> Optional[int]:
    """Try to estimate number of input slices from OME-Zarr metadata.

    Checks custom metadata fields, multiscales metadata, sibling slice files
    in the directory, and falls back to a heuristic estimate.

    Parameters
    ----------
    zarr_path : str or Path
        Path to the OME-Zarr file.

    Returns
    -------
    int or None
        Estimated number of input slices, or None if undeterminable.
    """
    import zarr

    try:
        store = zarr.open(str(zarr_path), mode='r')

        if hasattr(store, 'attrs'):
            attrs = dict(store.attrs)
            if 'n_input_slices' in attrs:
                return attrs['n_input_slices']
            if 'slice_boundaries' in attrs:
                return len(attrs['slice_boundaries'])

        if 'multiscales' in store.attrs:
            multiscales = store.attrs['multiscales']
            if isinstance(multiscales, list) and len(multiscales) > 0:
                ms = multiscales[0]
                if 'metadata' in ms and 'n_input_slices' in ms['metadata']:
                    return ms['metadata']['n_input_slices']
    except Exception:
        pass

    # Try sibling slice files
    parent_dir = Path(zarr_path).parent
    slice_files = list(parent_dir.glob('slice_z*.ome.zarr'))
    if slice_files:
        slice_nums = []
        for f in slice_files:
            match = re.search(r'slice_z(\d+)', f.name)
            if match:
                slice_nums.append(int(match.group(1)))
        if slice_nums:
            return max(slice_nums) - min(slice_nums) + 1

    return None


def add_z_slice_labels(ax, n_input_slices: int, img_height: int,
                       font_size: int = 7, label_every: int = 1,
                       show_lines: bool = False, side: str = 'left',
                       slice_ids: Optional[List[str]] = None) -> None:
    """Add Z-slice index labels on the side of a coronal/sagittal view.

    Parameters
    ----------
    ax : matplotlib axis
        The axis to annotate.
    n_input_slices : int
        Number of input slices stacked (e.g. 64 physical slices).
    img_height : int
        Height of the displayed image in pixels (Z dimension).
    font_size : int
        Font size for labels.
    label_every : int
        Label every Nth slice.
    show_lines : bool
        Draw horizontal lines at slice boundaries.
    side : str
        'left' or 'right' for label placement.
    slice_ids : list of str or None
        Actual slice IDs (e.g. ['05', '12']). If None, uses sequential numbers.
    """
    voxels_per_slice = img_height / n_input_slices
    x_pos = -0.02 if side == 'left' else 1.02
    ha = 'right' if side == 'left' else 'left'

    for slice_idx in range(n_input_slices):
        y_center_pixels = (slice_idx + 0.5) * voxels_per_slice

        if slice_idx % label_every == 0:
            if slice_ids is not None and slice_idx < len(slice_ids):
                label = f'z{slice_ids[slice_idx]}'
            else:
                label = f'z{slice_idx:02d}'

            ax.text(x_pos, y_center_pixels / img_height, label,
                    transform=ax.transAxes,
                    fontsize=font_size, color='white',
                    ha=ha, va='center',
                    fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.1', facecolor='black',
                              alpha=0.7, edgecolor='none'))

        if show_lines and slice_idx > 0:
            y_line = slice_idx * voxels_per_slice
            ax.axhline(y=y_line, color='cyan', alpha=0.3, linewidth=0.5, linestyle='--')


def save_annotated_views(image, out_path: str,
                         n_input_slices: int = None,
                         x_slice: int = None,
                         y_slice: int = None,
                         font_size: int = 7,
                         label_every: int = 1,
                         show_lines: bool = False,
                         slice_ids: Optional[List[str]] = None,
                         zarr_path: str = None) -> None:
    """Save coronal and sagittal views with Z-slice index annotations.

    Parameters
    ----------
    image : array-like
        3D volume (Z, X, Y).
    out_path : str
        Output figure path.
    n_input_slices : int or None
        Number of input slices. Auto-detected if zarr_path provided.
    x_slice, y_slice : int or None
        Slice indices. Default: center.
    font_size : int
        Font size for slice labels.
    label_every : int
        Label every Nth slice.
    show_lines : bool
        Draw horizontal lines at slice boundaries.
    slice_ids : list of str or None
        Actual slice IDs to display.
    zarr_path : str or None
        If provided, try to auto-detect n_input_slices from metadata.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_z_voxels, n_rows, n_cols = image.shape[0], image.shape[1], image.shape[2]

    if n_input_slices is None and zarr_path is not None:
        n_input_slices = estimate_n_slices_from_zarr(zarr_path)

    if n_input_slices is None:
        n_input_slices = max(1, n_z_voxels // 60)

    if slice_ids is not None and n_input_slices is None:
        n_input_slices = len(slice_ids)

    x_slice = x_slice if x_slice is not None else n_rows // 2
    y_slice = y_slice if y_slice is not None else n_cols // 2

    image_zy = np.array(image[:, x_slice, :])
    image_zx = np.array(image[:, :, y_slice])

    allvals = np.concatenate([image_zy.flatten(), image_zx.flatten()])
    vmin = float(np.min(allvals))
    vmax = float(np.percentile(allvals, 99.9))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12), facecolor='black')
    for ax in [ax1, ax2]:
        ax.set_facecolor('black')

    ax1.imshow(image_zy, cmap='magma', origin='lower', vmin=vmin, vmax=vmax, aspect='equal')
    ax1.set_title(f'Coronal (ZY) view at X={x_slice}', color='white', fontsize=12, pad=10)
    ax1.set_xlabel('Y', color='white', fontsize=10)
    ax1.set_ylabel('Z', color='white', fontsize=10)
    ax1.tick_params(colors='white', labelsize=8)
    for spine in ax1.spines.values():
        spine.set_color('white')
    add_z_slice_labels(ax1, n_input_slices, image_zy.shape[0],
                       font_size=font_size, label_every=label_every,
                       show_lines=show_lines, side='left', slice_ids=slice_ids)

    ax2.imshow(image_zx, cmap='magma', origin='lower', vmin=vmin, vmax=vmax, aspect='equal')
    ax2.set_title(f'Sagittal (ZX) view at Y={y_slice}', color='white', fontsize=12, pad=10)
    ax2.set_xlabel('X', color='white', fontsize=10)
    ax2.set_ylabel('Z', color='white', fontsize=10)
    ax2.tick_params(colors='white', labelsize=8)
    for spine in ax2.spines.values():
        spine.set_color('white')
    add_z_slice_labels(ax2, n_input_slices, image_zx.shape[0],
                       font_size=font_size, label_every=label_every,
                       show_lines=show_lines, side='right', slice_ids=slice_ids)

    if slice_ids is not None:
        slice_range_str = (f"slices: {slice_ids[0]}-{slice_ids[-1]}"
                           if len(slice_ids) > 1 else f"slice: {slice_ids[0]}")
    else:
        slice_range_str = f"z00-z{n_input_slices-1:02d}"

    fig.suptitle(
        f'Z-Slice Index Reference: {n_input_slices} input slices ({slice_range_str})\n'
        f'Volume: {n_z_voxels} Z × {n_rows} X × {n_cols} Y voxels',
        color='white', fontsize=14, y=0.98
    )

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, facecolor='black', edgecolor='none', dpi=150)
    plt.close(fig)
