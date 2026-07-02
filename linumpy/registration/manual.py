"""Manual image registration and correction GUI for z-slice stacks."""

import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RadioButtons, RangeSlider, Slider
from scipy.interpolate import RegularGridInterpolator

from linumpy.io.zarr import read_omezarr

PREV_REF_LABEL = "Previous slice as reference"
NEXT_REF_LABEL = "Next slice as reference"
NO_REF_LABEL = "No reference slice"


class ManualImageCorrection:
    """Manual image correction using a graphical user interface.

    Corrections
    include independent translation and rotation of each z-slice as well
    as image intensities rescaling per z-slice.

    Parameters
    ----------
    data: ndarray of shape (nz, ny, nx)
        Stack of images, where images are stacked along the first axis (z).
    resolution: 3-tuple
        Resolution of the dataset (rz, ry, rx).
    downsample_factor: int
        Factor by which the full resolution images are downscaled
        prior to rendering. Tradeoff between image quality and interactivity
        of the GUI. Does not influence the resolution of the corrected image.
    transforms: ndarray (nz, 3), optional
        Transform for each slice where each array (3,) contains a translation
        (ty, tx) and a rotation (theta).
    custom_ranges: ndarray (nz, 2), optional
        Intensities for rescaling each slice. One (vmin, vmax) per slice.
    """

    def __init__(
        self,
        data: np.ndarray,
        resolution: tuple,
        downsample_factor: int,
        transforms: np.ndarray | None = None,
        custom_ranges: np.ndarray | None = None,
    ) -> None:
        # We will work on a dataset rescaled between [0, 1]
        data = data - data.min()
        data = data / data.max()

        self.downsample = downsample_factor
        z = np.arange(data.shape[0])
        self.max_z = np.max(z)

        y = np.arange(data.shape[1])
        x = np.arange(data.shape[2])
        self.image_interpolator = RegularGridInterpolator((z, y, x), data, bounds_error=False, fill_value=0)
        self.grid_coordinates = np.stack(np.meshgrid(z, y, x, indexing="ij"), axis=-1)

        # Transforms array contains translation and rotation
        # for each slice in the order (ty, tx, theta)
        if transforms is None:
            self.transforms: np.ndarray = np.zeros((len(z), 3))
        else:
            self.transforms = transforms
        if self.transforms.shape != (len(z), 3):
            raise ValueError(f"Invalid shape for transforms file: expected ({len(z)}, 3), got {self.transforms.shape}.")

        # Base intensity normalization will rescale each slice
        # between its min and max values to the range [0, 1]
        if custom_ranges is None:
            self.custom_ranges: np.ndarray = np.array([np.min(data, axis=(1, 2)), np.max(data, axis=(1, 2))]).T
        else:
            self.custom_ranges = custom_ranges
        if self.custom_ranges.shape != (len(z), 2):
            raise ValueError(f"Invalid shape for custom ranges file: expected ({len(z)}, 3), got {self.custom_ranges.shape}.")

        self.ref_z_mode = NO_REF_LABEL
        self.current_x = len(x) // 2
        self.current_y = len(y) // 2
        self.current_z = 0

        self.fig, axs = plt.subplots(1, 3, figsize=(16, 8))
        self.fig.subplots_adjust(bottom=0.38, top=0.95, left=0.05, right=0.9)

        # intensities will always be displayed between (0, 1)
        aspect_a = resolution[0] / resolution[2]
        self.axim_a = axs[0].imshow(
            self.get_view_a(), aspect=aspect_a, vmin=0.0, vmax=1.0, interpolation="nearest", cmap="magma"
        )
        aspect_b = resolution[0] / resolution[1]
        self.axim_b = axs[1].imshow(
            self.get_view_b(), aspect=1.0 / aspect_b, vmin=0.0, vmax=1.0, interpolation="nearest", cmap="magma"
        )
        aspect_c = resolution[1] / resolution[2]
        self.axim_c = axs[2].imshow(
            self.get_view_c(), aspect=aspect_c, vmin=0.0, vmax=1.0, interpolation="nearest", cmap="magma"
        )
        axs[0].set_axis_off()
        axs[1].set_axis_off()
        axs[2].set_axis_off()

        ax_current_z = self.fig.add_axes((0.15, 0.30, 0.45, 0.03))
        ax_ref_z = self.fig.add_axes((0.65, 0.30, 0.25, 0.05))
        ax_offset_a = self.fig.add_axes((0.15, 0.25, 0.75, 0.03))
        ax_offset_b = self.fig.add_axes((0.15, 0.20, 0.75, 0.03))
        ax_theta = self.fig.add_axes((0.15, 0.15, 0.75, 0.03))
        ax_current_y = self.fig.add_axes((0.15, 0.10, 0.75, 0.03))
        ax_current_x = self.fig.add_axes((0.15, 0.05, 0.75, 0.03))
        ax_scalebar = self.fig.add_axes((0.91, 0.40, 0.01, 0.55))

        self.scalebar = RangeSlider(
            ax_scalebar,
            "Scalebar",
            valmin=0.0,
            valmax=1.0,
            valinit=(self.custom_ranges[self.current_z, 0], self.custom_ranges[self.current_z, 1]),
            orientation="vertical",
        )

        self.s_offset_a = Slider(
            ax_offset_a,
            "Offset left image",
            valmin=-data.shape[2] / 2,
            valmax=data.shape[2] / 2,
            valinit=self.transforms[self.current_z, 0],
        )
        self.s_offset_b = Slider(
            ax_offset_b,
            "Offset right image",
            valmin=-data.shape[1] / 2,
            valmax=data.shape[1] / 2,
            valinit=self.transforms[self.current_z, 1],
        )
        self.s_current_z = Slider(
            ax_current_z, "Current slice z", valmin=0, valmax=data.shape[0], valinit=0, valstep=np.arange(data.shape[0])
        )
        self.s_current_y = Slider(
            ax_current_y,
            "Current slice y",
            valmin=0,
            valmax=data.shape[1],
            valinit=self.current_y,
            valstep=np.arange(data.shape[1]),
        )
        self.s_current_x = Slider(
            ax_current_x,
            "Current slice x",
            valmin=0,
            valmax=data.shape[2],
            valinit=self.current_x,
            valstep=np.arange(data.shape[2]),
        )
        self.s_theta = Slider(
            ax_theta, "Rotation", valmin=-np.pi / 6.0, valmax=np.pi / 6.0, valinit=self.transforms[self.current_z, 2]
        )
        self.radio_buttons = RadioButtons(ax_ref_z, [NO_REF_LABEL, PREV_REF_LABEL, NEXT_REF_LABEL], 0)

        # register callbacks
        self.s_current_z.on_changed(self.on_change_z)
        self.s_current_y.on_changed(self.on_change_y)
        self.s_current_x.on_changed(self.on_change_x)
        self.s_offset_a.on_changed(self.on_change_offset_a)
        self.s_offset_b.on_changed(self.on_change_offset_b)
        self.s_theta.on_changed(self.on_change_theta)
        self.radio_buttons.on_clicked(self.on_change_ref_z)
        self.scalebar.on_changed(self.on_change_scaling)

    def start(self) -> bool:
        """
        Start GUI.

        Returns
        -------
        return: bool
            True when the window closes.
        """
        plt.show(block=True)
        return True

    def on_change_scaling(self, scaling_range: tuple) -> None:
        """Update intensity rescaling for the current z-slice."""
        self.custom_ranges[self.current_z] = scaling_range
        self.axim_a.set(data=self.get_view_a())
        self.axim_b.set(data=self.get_view_b())
        self.axim_c.set(data=self.get_view_c())
        self.fig.canvas.draw_idle()

    def on_change_z(self, val: float) -> None:
        """Update current z-slice index."""
        self.current_z = int(val)
        self.s_offset_a.set_val(self.transforms[self.current_z, 0])
        self.s_offset_b.set_val(self.transforms[self.current_z, 1])
        self.s_theta.set_val(self.transforms[self.current_z, 2])
        self.scalebar.set_val(self.custom_ranges[self.current_z, :])
        self.axim_c.set(data=self.get_view_c())
        self.fig.canvas.draw_idle()

    def on_change_y(self, val: float) -> None:
        """Update current y-plane index."""
        self.current_y = int(val)
        self.axim_b.set(data=self.get_view_b())
        self.fig.canvas.draw_idle()

    def on_change_x(self, val: float) -> None:
        """Update current x-plane index."""
        self.current_x = int(val)
        self.axim_a.set(data=self.get_view_a())
        self.fig.canvas.draw_idle()

    def on_change_offset_a(self, val: float) -> None:
        """Update y-translation for the current z-slice."""
        self.transforms[self.current_z, 0] = val
        self.axim_a.set(data=self.get_view_a())
        self.axim_b.set(data=self.get_view_b())
        self.axim_c.set(data=self.get_view_c())
        self.fig.canvas.draw_idle()

    def on_change_offset_b(self, val: float) -> None:
        """Update x-translation for the current z-slice."""
        self.transforms[self.current_z, 1] = val
        self.axim_a.set(data=self.get_view_a())
        self.axim_b.set(data=self.get_view_b())
        self.axim_c.set(data=self.get_view_c())
        self.fig.canvas.draw_idle()

    def on_change_theta(self, val: float) -> None:
        """Update rotation angle for the current z-slice."""
        self.transforms[self.current_z, 2] = val
        self.axim_a.set(data=self.get_view_a())
        self.axim_b.set(data=self.get_view_b())
        self.axim_c.set(data=self.get_view_c())
        self.fig.canvas.draw_idle()

    def on_change_ref_z(self, label: str | None) -> None:
        """Update reference z-slice mode."""
        self.ref_z_mode = label
        self.axim_c.set(data=self.get_view_c())
        self.fig.canvas.draw_idle()

    def transform_coordinates(self, coordinates: np.ndarray, z: int | None = None) -> np.ndarray:
        """Apply the stored transform to a set of grid coordinates."""
        # will consider either all z or a single one
        if z is None:
            ty = self.transforms[:, 0]
            tx = self.transforms[:, 1]
            theta = self.transforms[:, 2]
        else:
            ty = self.transforms[z, 0]
            tx = self.transforms[z, 1]
            theta = self.transforms[z, 2]

        coordinates = apply_transform(ty, tx, theta, coordinates)

        return coordinates

    def apply_scaling(self, data: np.ndarray, z: int | None = None) -> np.ndarray:
        """Rescale slice intensities using the stored per-slice ranges."""
        if z is not None:
            clip_min = self.custom_ranges[z, 0]
            clip_max = self.custom_ranges[z, 1]
        else:
            clip_min = self.custom_ranges[:, 0, None]
            clip_max = self.custom_ranges[:, 1, None]

        data = apply_scaling(data, clip_min, clip_max)

        # at this point the data is between [0, 1]
        return data

    def draw_cursor(self, data: np.ndarray) -> np.ndarray:
        """Draw a cursor line at the current z position on a view."""
        # keeping in mind that axis=0 is the z axis
        cursor_len = int(0.02 * data.shape[-1])
        data[self.current_z, :cursor_len] = 1.0
        data[self.current_z, -cursor_len:] = 1.0
        return data

    def get_view_a(self) -> np.ndarray:
        """Return the YZ view (x-plane) as a transformed, scaled image."""
        view_coords = self.grid_coordinates[:, :, self.current_x, :]
        transformed_coords = self.transform_coordinates(view_coords)
        data = self.apply_scaling(self.image_interpolator(transformed_coords))
        data = self.draw_cursor(data)
        return data

    def get_view_b(self) -> np.ndarray:
        """Return the XZ view (y-plane) as a transformed, scaled image."""
        view_coords = self.grid_coordinates[:, self.current_y, :, :]
        transformed_coords = self.transform_coordinates(view_coords)
        data = self.apply_scaling(self.image_interpolator(transformed_coords))
        data = self.draw_cursor(data)
        return data.T

    def get_view_c(self) -> np.ndarray:
        """Return the XY view (z-slice) as a transformed, scaled RGB image."""
        # subsample coordinates for better interactivity
        view_coords = self.grid_coordinates[self.current_z, :: self.downsample, :: self.downsample, :]
        transformed_coords = self.transform_coordinates(view_coords, self.current_z)
        data_view = self.apply_scaling(self.image_interpolator(transformed_coords), self.current_z)

        data_rgb = np.zeros((*data_view.shape, 3))
        data_rgb[..., :] = data_view[..., None]

        if self.ref_z_mode != NO_REF_LABEL:
            ref_z = self.current_z - 1 if self.ref_z_mode == PREV_REF_LABEL else self.current_z + 1
            if ref_z >= 0 and ref_z <= self.max_z:
                ref_coords = self.grid_coordinates[ref_z, :: self.downsample, :: self.downsample, :]
                transformed_ref_coords = self.transform_coordinates(ref_coords, ref_z)
                data_ref = self.apply_scaling(self.image_interpolator(transformed_ref_coords), self.current_z)
                data_rgb[..., 0] = data_ref
        return np.clip(data_rgb, 0.0, 1.0)

    def save_results(self, filename: Path) -> None:
        """
        Save resulting corrections to npz file.

        Parameters
        ----------
        filename: string or Path
            Output filename.
        """
        np.savez_compressed(filename, custom_ranges=self.custom_ranges, transforms=self.transforms)


def apply_transform(
    ty: float | np.ndarray, tx: float | np.ndarray, theta: float | np.ndarray, coordinates: np.ndarray
) -> np.ndarray:
    """Apply transformation to coordinates.

    Coordinates are expected to be of shape (nz, ny, nx, 3), with each
    coordinate given in the order (z, y, x).

    Parameters
    ----------
    ty: float or ndarray of shape (nz,)
        Translation along y axis.
    tx: float or ndarray of shape (nz,)
        Translation along x axis.
    theta: float or ndarray of shape (nz,)
        Rotation around z axis in radians. The center of rotation
        is the center of the image.
    coordinates : ndarray of shape (nz, ny, nx, 3)
        Input grid coordinates in (z, y, x) order.

    Returns
    -------
    coordinates: ndarray (nz, ny, nx, 3)
        Transformed coordinates.
    """
    # Step 1. Rotate coordinates
    center_y = np.max(coordinates[:, :, 1]) / 2.0
    center_x = np.max(coordinates[:, :, 2]) / 2.0
    coordinates = coordinates - np.reshape([0, center_y, center_x], (1, 1, 3))
    rotated_y = np.atleast_2d(np.cos(theta)).T * coordinates[..., 1] - np.atleast_2d(np.sin(theta)).T * coordinates[..., 2]
    rotated_x = np.atleast_2d(np.sin(theta)).T * coordinates[..., 1] + np.atleast_2d(np.cos(theta)).T * coordinates[..., 2]
    coordinates[:, :, 1] = rotated_y + center_y
    coordinates[:, :, 2] = rotated_x + center_x

    # Step 2. Translate coordinates
    coordinates[:, :, 1] += np.atleast_2d(ty).T
    coordinates[:, :, 2] += np.atleast_2d(tx).T

    return coordinates


def apply_scaling(data: np.ndarray, vmin: float | np.ndarray, vmax: float | np.ndarray) -> np.ndarray:
    """Rescale image intensities from (vmin, vmax) to (0.0, 1.0).

    Values
    outside the range (vmin, vmax) are clipped.

    Rescaling can be performed with a single range for the whole image
    or with a different range for each ROW. In the case, the first dimension
    of data should correspond to the number of elements in vmin, vmax.

    Parameters
    ----------
    data: ndarray
        The intensities to rescale.
    vmin: float or ndarray of shape (data.shape[0],)
        Minimum value. Will be worth 0 after rescaling.
    vmax: float or ndarray of shape (data.shape[0],)
        Maximum value. Will be worth 1 after rescaling.

    Returns
    -------
    data: ndarray
        Recaled intensities.
    """
    data = np.clip(data, vmin, vmax)
    data -= vmin
    clip_range = vmax - vmin
    if isinstance(clip_range, np.ndarray):
        safe_range = np.where(clip_range > 0, clip_range, 1.0)
        data /= safe_range
    elif clip_range > 0.0:
        data /= clip_range
    return data


def transform_and_rescale_slice(slice: np.ndarray, ty: float, tx: float, theta: float, vmin: float, vmax: float) -> np.ndarray:
    """Transform and rescale a 2D slice.

    Transform consists of a translation
    (ty, tx) and a rotation theta. Rescaling clips intensities to (vmin, vmax)
    and rescales the resulting values to the range (0, 1).

    Parameters
    ----------
    slice: ndarray of shape (ny, nx)
        Slice to process.
    ty: float
        Translation along y axis (first axis).
    tx: float
        Translation along x axis (second axis).
    theta: float
        Rotation in radians.
    vmin: float
        Minimum value for rescaling.
    vmax: float
        Maximum value for rescaling.

    Returns
    -------
    slice: ndarray of shape (ny, nx)
        Processed slice.
    """
    y = np.arange(slice.shape[0])
    x = np.arange(slice.shape[1])
    image_interpolator = RegularGridInterpolator((y, x), slice, bounds_error=False, fill_value=0)
    grid_coordinates = np.stack(np.meshgrid(0, y, x, indexing="ij"), axis=-1)

    # transform coordinates
    transformed_coordinates = apply_transform(ty, tx, theta, grid_coordinates[0])
    transformed_image = image_interpolator(transformed_coordinates[..., 1:])
    # rescale intensities
    transformed_image = apply_scaling(transformed_image, vmin, vmax)

    return transformed_image


# ---------------------------------------------------------------------------
# Manual alignment data package export (linum_export_manual_align.py)
#
# Reads common-space slices (OME-Zarr) and pairwise registration outputs and
# produces the AIPs / cross-sections / transforms consumed by the
# ``linumpy-manual-align`` Napari plugin. Extracted from the script per D-84
# (#8) / D-86 -- the script remains a thin CLI wrapper around these helpers.
# ---------------------------------------------------------------------------


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
