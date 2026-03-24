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


# ---------------------------------------------------------------------------
# Orientation helpers
# ---------------------------------------------------------------------------

# Map from anatomical letter to target-axis group index (0=S/I, 1=R/L, 2=A/P)
_LETTER_GROUP = {'S': 0, 'I': 0, 'R': 1, 'L': 1, 'A': 2, 'P': 2}

# Map from pair of axis-group indices to anatomical plane name
_GROUP_PLANE = {
    frozenset({1, 2}): 'Axial',
    frozenset({0, 1}): 'Coronal',
    frozenset({0, 2}): 'Sagittal',
}


def _panel_labels_from_orientation(orientation: str):
    """Derive anatomical panel labels from a 3-letter orientation code.

    Validates the code using :func:`linumpy.utils.orientation.parse_orientation_code`
    then computes panel names and axis labels from the source-dimension letters.

    The volume has shape (Z=dim0, X=dim1, Y=dim2).
    Panel 1 is ``image[:, x_slice, :]`` — shows (dim0, dim2), fixes dim1.
    Panel 2 is ``image[:, :, y_slice]``  — shows (dim0, dim1), fixes dim2.

    Parameters
    ----------
    orientation : str
        3-letter RAS-style code, e.g. ``'RIA'`` means dim0→R, dim1→I, dim2→A.
        Surrounding quotes are stripped automatically.

    Returns
    -------
    tuple or None
        ``(p1_name, p1_xlabel, p1_ylabel, p1_fixed_label,
           p2_name, p2_xlabel, p2_ylabel, p2_fixed_label)``
        where *name* is the anatomical plane ('Axial'/'Coronal'/'Sagittal'),
        *xlabel*/*ylabel* are the axis letters for the plot,
        and *fixed_label* is the axis letter that is held constant.
        Returns ``None`` for an invalid code.
    """
    from linumpy.utils.orientation import parse_orientation_code

    code = orientation.strip("'\" ").upper()
    try:
        parse_orientation_code(code)  # validation only
    except (ValueError, KeyError):
        return None

    a0, a1, a2 = code  # anatomical letter for source dim0, dim1, dim2
    g0, g1, g2 = _LETTER_GROUP[a0], _LETTER_GROUP[a1], _LETTER_GROUP[a2]

    # Panel 1: shows (dim0=Z, dim2=Y), fixes dim1 at x_slice
    p1_name = _GROUP_PLANE.get(frozenset({g0, g2}), 'ZY')
    # Panel 2: shows (dim0=Z, dim1=X), fixes dim2 at y_slice
    p2_name = _GROUP_PLANE.get(frozenset({g0, g1}), 'ZX')

    return (
        p1_name, a2, a0, a1,   # panel1: xlabel=dim2, ylabel=dim0, fixed=dim1
        p2_name, a1, a0, a2,   # panel2: xlabel=dim1, ylabel=dim0, fixed=dim2
    )


def save_annotated_views(image, out_path: str,
                         n_input_slices: int = None,
                         x_slice: int = None,
                         y_slice: int = None,
                         font_size: int = 7,
                         label_every: int = 1,
                         show_lines: bool = False,
                         slice_ids: Optional[List[str]] = None,
                         zarr_path: str = None,
                         orientation: str = None) -> None:
    """Save anatomically-labelled orthogonal views with Z-slice index annotations.

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
    orientation : str or None
        3-letter RAS orientation code (e.g. ``'RIA'``).
        When provided, panel titles and axis labels use anatomical names
        (Axial/Coronal/Sagittal) derived from this code instead of the
        generic ``'Coronal (ZY)'`` / ``'Sagittal (ZX)'`` defaults.
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

    # Derive panel titles and axis labels from orientation when available.
    _orient = _panel_labels_from_orientation(orientation) if orientation else None
    if _orient:
        p1_name, p1_xlabel, p1_ylabel, p1_fixed, p2_name, p2_xlabel, p2_ylabel, p2_fixed = _orient
        title1 = f'{p1_name} ({p1_ylabel}\u00d7{p1_xlabel}) view at {p1_fixed}={x_slice}'
        title2 = f'{p2_name} ({p2_ylabel}\u00d7{p2_xlabel}) view at {p2_fixed}={y_slice}'
        xlabel1, ylabel1 = p1_xlabel, p1_ylabel
        xlabel2, ylabel2 = p2_xlabel, p2_ylabel
    else:
        title1 = f'Coronal (ZY) view at X={x_slice}'
        title2 = f'Sagittal (ZX) view at Y={y_slice}'
        xlabel1, ylabel1 = 'Y', 'Z'
        xlabel2, ylabel2 = 'X', 'Z'

    image_zy = np.array(image[:, x_slice, :])
    image_zx = np.array(image[:, :, y_slice])

    allvals = np.concatenate([image_zy.flatten(), image_zx.flatten()])
    vmin = float(np.min(allvals))
    vmax = float(np.percentile(allvals, 99.9))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12), facecolor='black')
    for ax in [ax1, ax2]:
        ax.set_facecolor('black')

    ax1.imshow(image_zy, cmap='magma', origin='lower', vmin=vmin, vmax=vmax, aspect='equal')
    ax1.set_title(title1, color='white', fontsize=12, pad=10)
    ax1.set_xlabel(xlabel1, color='white', fontsize=10)
    ax1.set_ylabel(ylabel1, color='white', fontsize=10)
    ax1.tick_params(colors='white', labelsize=8)
    for spine in ax1.spines.values():
        spine.set_color('white')
    add_z_slice_labels(ax1, n_input_slices, image_zy.shape[0],
                       font_size=font_size, label_every=label_every,
                       show_lines=show_lines, side='left', slice_ids=slice_ids)

    ax2.imshow(image_zx, cmap='magma', origin='lower', vmin=vmin, vmax=vmax, aspect='equal')
    ax2.set_title(title2, color='white', fontsize=12, pad=10)
    ax2.set_xlabel(xlabel2, color='white', fontsize=10)
    ax2.set_ylabel(ylabel2, color='white', fontsize=10)
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

    orient_note = (
        f"  ·  orientation: {orientation.strip(chr(39)).upper()} (acquisition space, pre-atlas-alignment)"
        if orientation else ""
    )
    fig.suptitle(
        f'Z-Slice Alignment View — {n_input_slices} input slices ({slice_range_str}){orient_note}\n'
        f'Volume: {n_z_voxels} Z × {n_rows} X × {n_cols} Y voxels'
        f'  ·  NOTE: axes reflect raw acquisition geometry, NOT final neuroimaging orientation',
        color='yellow', fontsize=11, y=0.98
    )

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, facecolor='black', edgecolor='none', dpi=150)
    plt.close(fig)
