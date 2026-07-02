#!/usr/bin/env python3

"""
Align a 3D brain volume to RAS orientation using rigid registration to the Allen atlas.

This script computes a rigid transform from the input brain volume to a RAS-aligned
version by registering it to the Allen Brain Atlas. The transform can be applied
directly to the zarr file (resampling) or stored in OME-Zarr metadata.
"""

# Configure thread limits before numpy/scipy imports
import linumpy.config.threads  # noqa: F401
from linumpy.config.threads import configure_all_libraries

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from tqdm.auto import tqdm

from linumpy.imaging import orientation as orientation_lib
from linumpy.imaging.orientation import (
    apply_orientation_transform,
    compute_centered_reference_and_transform,
    parse_orientation_code,
    reorder_resolution,
    store_transform_in_metadata,
)
from linumpy.io.zarr import AnalysisOmeZarrWriter, read_omezarr
from linumpy.reference import allen

matplotlib.use("Agg")  # Non-interactive backend
configure_all_libraries()

# Preserve direct helper imports for characterization tests and downstream reuse.
sitk_transform_to_affine_matrix = orientation_lib.sitk_transform_to_affine_matrix

# Constants
DEFAULT_ALLEN_RESOLUTION = 100
DEFAULT_MAX_ITERATIONS = 1000
DEFAULT_METRIC = "MI"


def _debug_log(message: str, **fields: Any) -> None:
    """Append an NDJSON line describing a slicing/labelling decision.

    Active only when ``LINUMPY_DEBUG_LOG`` is set, so production runs pay
    nothing. Used to capture runtime evidence of which volume conventions
    each preview function actually receives.
    """
    import os

    path = os.environ.get("LINUMPY_DEBUG_LOG")
    if not path:
        return
    try:
        import time

        entry = {
            "id": f"log_{int(time.time() * 1000)}_panels",
            "timestamp": int(time.time() * 1000),
            "sessionId": "6fa1b3",
            "runId": "panels-fix",
            "hypothesisId": "H1",
            "location": "linum_align_to_ras.py",
            "message": message,
            "data": fields,
        }
        with Path(path).open("a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input_zarr", help="Input OME-Zarr file from 3D reconstruction pipeline")
    p.add_argument("output_zarr", help="Output OME-Zarr file (RAS-aligned)")
    p.add_argument(
        "--allen-resolution",
        type=int,
        default=DEFAULT_ALLEN_RESOLUTION,
        choices=allen.AVAILABLE_RESOLUTIONS,
        help="Allen atlas resolution in micron [%(default)s]",
    )
    p.add_argument(
        "--metric",
        type=str,
        default=DEFAULT_METRIC,
        choices=["MI", "MSE", "CC", "AntsCC"],
        help="Registration metric [%(default)s]",
    )
    p.add_argument(
        "--max-iterations",
        type=int,
        default=DEFAULT_MAX_ITERATIONS,
        help="Maximum registration iterations [%(default)s]",
    )
    p.add_argument(
        "--store-transform-only", action="store_true", help="Store transform in metadata only (don't resample volume)"
    )
    p.add_argument("--level", type=int, default=0, help="Pyramid level for registration (0 = full resolution) [%(default)s]")
    p.add_argument(
        "--chunks", type=int, nargs=3, default=None, help="Chunk size for output zarr. Uses input chunks when None."
    )
    p.add_argument(
        "--n-levels", type=int, default=None, help="Number of pyramid levels for output. Uses Allen atlas levels when None."
    )
    p.add_argument(
        "--pyramid_resolutions",
        type=float,
        nargs="+",
        default=None,
        help="Target pyramid resolution levels in µm (e.g. 10 25 50 100).\n"
        "If omitted, inherits levels from input zarr metadata or uses Allen resolutions.",
    )
    p.add_argument(
        "--make_isotropic", action="store_true", default=True, help="Resample to isotropic voxels at each pyramid level."
    )
    p.add_argument("--no_isotropic", dest="make_isotropic", action="store_false")
    p.add_argument("--verbose", action="store_true", help="Print registration progress")
    p.add_argument("--preview", type=str, default=None, help="Generate preview image showing alignment comparison")
    p.add_argument(
        "--input-orientation",
        type=str,
        default=None,
        help="Input volume orientation code (3 letters: R/L, A/P, S/I)\nExamples: 'RAS' (Allen), 'LPI', 'PIR'",
    )
    p.add_argument(
        "--initial-rotation",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        metavar=("RX", "RY", "RZ"),
        help="Initial rotation angles in degrees (Rx, Ry, Rz).\nUse to provide initial orientation hint for registration.",
    )
    p.add_argument("--preview-only", action="store_true", help="Only generate preview of input volume (no registration)")
    p.add_argument(
        "--orientation-preview",
        type=str,
        default=None,
        metavar="PATH",
        help="Save a 3-panel preview of the volume after --input-orientation and\n"
        "--initial-rotation are applied. Use to verify these parameters\n"
        "before committing to a full registration run.",
    )
    p.add_argument(
        "--orientation-preview-only",
        action="store_true",
        help="Generate --orientation-preview and exit without running registration.",
    )
    return p


# =============================================================================
# Orientation utilities -- imported from linumpy.imaging.orientation
# =============================================================================


def create_registration_progress_callback(
    max_iterations: int,
    n_resolution_levels: int = 3,
    pbar: tqdm | None = None,
    registration_start_step: int = 0,
    registration_steps: int = 0,
) -> Callable:
    """
    Create a progress callback for registration.

    Parameters
    ----------
    max_iterations : int
        Maximum iterations per level
    n_resolution_levels : int
        Number of resolution levels in the registration pyramid
    pbar : tqdm, optional
        Progress bar to update
    registration_start_step : int
        Step number where registration starts in progress bar
    registration_steps : int
        Number of steps allocated for registration

    Returns
    -------
    callable
        Progress callback function compatible with SimpleITK registration
    """
    total_iterations = [0]
    level_counter = [0]
    last_iteration = [-1]
    # Worst-case budget (used only as the denominator for the progress bar).
    estimated_total = float(max_iterations * n_resolution_levels)

    def callback(method: Any) -> None:
        """Update progress during registration iterations."""
        iteration = method.GetOptimizerIteration()
        metric = method.GetMetricValue()

        # Detect resolution-level transitions (iteration counter resets to 0
        # when SimpleITK starts the next pyramid level).
        if iteration < last_iteration[0]:
            level_counter[0] += 1
        last_iteration[0] = iteration

        total_iterations[0] += 1

        if pbar is not None:
            # Blend "within-level" progress with completed levels so the bar
            # advances smoothly across resolutions and does not stall when a
            # level converges early or hits max_iterations.
            within_level = min(1.0, (iteration + 1) / max_iterations)
            level_progress = (level_counter[0] + within_level) / n_resolution_levels
            progress_ratio = min(1.0, max(level_progress, total_iterations[0] / estimated_total))
            target_step = registration_start_step + int(registration_steps * progress_ratio)
            if target_step > pbar.n:
                pbar.n = target_step
                pbar.set_postfix_str(f"metric={metric:.6f} level={level_counter[0] + 1}/{n_resolution_levels}")
                pbar.refresh()

    return callback


# =============================================================================
# Resolution utilities
# =============================================================================


def get_pyramid_resolutions_from_zarr(zarr_path: Path) -> list[float] | None:
    """
    Extract pyramid resolution levels from OME-Zarr metadata.

    Parameters
    ----------
    zarr_path : Path
        Path to OME-Zarr file

    Returns
    -------
    list of float or None
        Target resolutions in microns, or None if not found
    """
    for metadata_file in ["zarr.json", ".zattrs"]:
        metadata_path = zarr_path / metadata_file
        if not metadata_path.exists():
            continue

        try:
            with Path(metadata_path).open(encoding="utf-8") as f:
                metadata = json.load(f)
        except OSError, json.JSONDecodeError:
            continue

        multiscales = metadata.get("multiscales", [])
        if not multiscales:
            continue

        resolutions = []
        for dataset in multiscales[0].get("datasets", []):
            transforms = dataset.get("coordinateTransformations", [])
            for tr in transforms:
                if tr.get("type") == "scale" and "scale" in tr:
                    # Get finest spatial dimension, convert mm to µm
                    scale = tr["scale"][-3:]
                    res_um = min(float(s) for s in scale) * 1000
                    resolutions.append(res_um)
                    break

        if resolutions:
            return resolutions

    return None


def apply_transform_to_zarr(
    input_path: Path,
    output_path: Path,
    transform: sitk.Transform,
    chunks: tuple | None = None,
    n_levels: int | None = None,
    pyramid_resolutions: list | None = None,
    make_isotropic: bool = True,
    orientation_permutation: tuple | None = None,
    orientation_flips: tuple | None = None,
    pbar: tqdm | None = None,
) -> None:
    """
    Apply transform to zarr file by resampling into RAS-aligned space.

    The output is centered on the transformed brain volume, preserving the
    original resolution. This corrects any rotation/off-axis alignment without
    placing the brain in the Allen atlas coordinate system.

    Parameters
    ----------
    input_path: Path
        Path to input OME-Zarr
    output_path: Path
        Path to output OME-Zarr
    transform : sitk.Transform
        Transform to apply
    chunks : tuple, optional
        Chunk size for output
    n_levels : int, optional
        Number of pyramid levels (if None, use source pyramid or Allen resolutions)
    orientation_permutation : tuple, optional
        Axis permutation for orientation correction
    orientation_flips : tuple, optional
        Axis flips for orientation correction
    pbar : tqdm, optional
        Progress bar
    pyramid_resolutions : list, optional
        Explicit list of resolutions for the output pyramid
    make_isotropic : bool
        If True, resample output to isotropic resolution
    """

    def update_pbar() -> None:
        if pbar:
            pbar.update(1)

    # Load volume at full resolution (level 0) and capture its actual spacing.
    # base_resolution comes from the downsampled registration level, so we must
    # read the level-0 spacing from the file to get the correct physical extent.
    vol_zarr, level0_resolution = read_omezarr(input_path, level=0)
    if chunks is None:
        chunks = getattr(vol_zarr, "chunks", None)
    if chunks is None:
        chunks = (128,) * len(vol_zarr.shape)

    vol = np.asarray(vol_zarr[:])
    original_dtype = vol.dtype
    update_pbar()

    # Apply orientation correction
    resolution = level0_resolution
    if orientation_permutation is not None:
        vol = apply_orientation_transform(vol, orientation_permutation, orientation_flips)
        resolution = reorder_resolution(resolution, orientation_permutation)
        # ``apply_orientation_transform`` returns a non-contiguous view.
        # ``sitk.GetImageFromArray`` would internally call
        # ``np.ascontiguousarray`` AND make its own SITK copy — that is
        # two ~36 GB allocations on top of the still-live source.
        # Materialise the contiguous buffer here and immediately drop the
        # original so the SITK conversion only adds one full-volume copy.
        vol = np.ascontiguousarray(vol)

    # Compute a tissue-representative background value on the numpy array
    # BEFORE allocating the (potentially large) SimpleITK float32 copy.  Using
    # this as the default pixel value avoids black borders that would skew
    # downstream normalization and visualization.  Subsample to keep the
    # boolean-mask gather from allocating a multi-GB temp on a level-0 brain.
    flat = vol.reshape(-1)
    stride = max(1, flat.size // 10_000_000)  # cap sample at ~10M voxels
    sample = flat[::stride]
    nonzero_sample = sample[sample > 0]
    bg_value = float(np.percentile(nonzero_sample, 1)) if nonzero_sample.size else 0.0
    del flat, sample, nonzero_sample

    # Convert to SimpleITK
    vol_sitk = allen.numpy_to_sitk_image(vol, resolution, cast_dtype=np.float32)
    del vol  # free original volume before resampling
    update_pbar()

    # Compute reference image and modified transform that centers the output
    reference, centered_transform = compute_centered_reference_and_transform(vol_sitk, transform)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(bg_value)
    resampler.SetTransform(centered_transform)

    transformed_sitk = resampler.Execute(vol_sitk)
    del vol_sitk  # free input before allocating output array
    transformed = sitk.GetArrayFromImage(transformed_sitk)
    del transformed_sitk  # free SimpleITK image after extracting numpy array
    update_pbar()

    # GetArrayFromImage already yields numpy (Z, Y, X) matching our convention.
    update_pbar()

    # Convert back to original dtype.  Use ``copy=False`` so the no-op
    # float32→float32 path does not allocate another full-volume buffer
    # (~36 GB on a level-0 brain).  For integer targets, do rint/clip
    # in place on the float32 buffer before the narrowing astype to
    # avoid two extra full-volume temporaries.
    if np.issubdtype(original_dtype, np.integer):
        info = np.iinfo(original_dtype)
        np.rint(transformed, out=transformed)
        np.clip(transformed, info.min, info.max, out=transformed)
        transformed = transformed.astype(original_dtype, copy=False)
    else:
        transformed = transformed.astype(original_dtype, copy=False)

    # Write output
    writer = AnalysisOmeZarrWriter(
        output_path,
        shape=transformed.shape,
        chunk_shape=chunks,
        dtype=transformed.dtype,
        overwrite=True,
    )
    writer[:] = transformed

    if n_levels is not None:
        writer.finalize(list(resolution), n_levels=n_levels)
    else:
        if pyramid_resolutions is not None:
            target_resolutions = pyramid_resolutions
        else:
            # Fallback: inherit levels from input zarr metadata, or use Allen resolutions
            target_resolutions = get_pyramid_resolutions_from_zarr(Path(input_path))
            if target_resolutions is None:
                target_resolutions = list(allen.AVAILABLE_RESOLUTIONS)
        writer.finalize(list(resolution), target_resolutions_um=target_resolutions, make_isotropic=make_isotropic)

    update_pbar()


# =============================================================================
# Preview generation
# =============================================================================


def create_input_preview(input_path: Path, output_path: Path, level: int = 0) -> None:
    """Create preview of input volume to help determine orientation."""
    vol_zarr, resolution = read_omezarr(input_path, level=level)
    vol = np.asarray(vol_zarr[:])

    z_mid = vol.shape[0] // 2
    x_mid = vol.shape[1] // 2
    y_mid = vol.shape[2] // 2

    vmin, vmax = np.percentile(vol, [1, 99])

    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    fig.suptitle(f"Input Volume Preview\nShape: {vol.shape} (Z, Y, X), Resolution: {resolution} mm", fontsize=14, y=0.98)

    # Axial slice (dim0 midpoint)
    axes[0, 0].imshow(vol[z_mid, :, :].T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
    axes[0, 0].set_title("Slice at dim0 midpoint\nShows: dim1 x dim2")
    axes[0, 0].set_xlabel("dim1 →")
    axes[0, 0].set_ylabel("dim2 →")

    # Sagittal slice (dim1 midpoint)
    axes[0, 1].imshow(vol[::-1, x_mid, :], cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
    axes[0, 1].set_title("Slice at dim1 midpoint\nShows: dim2 x dim0")
    axes[0, 1].set_xlabel("dim2 →")
    axes[0, 1].set_ylabel("dim0 →")

    # Coronal slice (dim2 midpoint)
    axes[1, 0].imshow(vol[::-1, :, y_mid], cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
    axes[1, 0].set_title("Slice at dim2 midpoint\nShows: dim1 x dim0")
    axes[1, 0].set_xlabel("dim1 →")
    axes[1, 0].set_ylabel("dim0 →")

    # Help text
    axes[1, 1].axis("off")
    help_text = """
ORIENTATION GUIDE (Allen Atlas = RAS+)

Allen RAS+ convention:
  • R (Right):    +X direction
  • A (Anterior): +Y direction (nose)
  • S (Superior): +Z direction (top)

For each dimension, identify the anatomical direction:
  R/L for right/left
  A/P for anterior/posterior
  S/I for superior/inferior

Example:
  dim0→Superior, dim1→Anterior, dim2→Right
  → orientation code = 'SAR'
"""
    axes[1, 1].text(
        0.02,
        0.98,
        help_text,
        transform=axes[1, 1].transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Input preview saved to: {output_path}")


def create_alignment_preview(
    input_path: Path,
    output_path: Path | None,
    transform: sitk.Transform,
    resolution: tuple,
    preview_path: str,
    allen_resolution: int = DEFAULT_ALLEN_RESOLUTION,
    level: int = 0,
    orientation_permutation: tuple | None = None,
    orientation_flips: tuple | None = None,
    pbar: tqdm | None = None,
) -> None:
    """Create preview comparing original, aligned, and Allen template.

    Shows center slices from each volume in their own coordinate frames.
    The Allen template is shown for reference but may not spatially align
    with the brain volume since we're not placing it in Allen coordinate space.
    """

    def update_pbar() -> None:
        if pbar:
            pbar.update(1)

    # Load original
    vol_original, orig_res = read_omezarr(input_path, level=level)
    vol_original = np.asarray(vol_original[:])

    if orientation_permutation is not None:
        vol_original = apply_orientation_transform(vol_original, orientation_permutation, orientation_flips)
        orig_res = reorder_resolution(tuple(orig_res), orientation_permutation)

    # apply_orientation_transform yields the linumpy convention (S, A, R)
    # matching the Allen RAS+ template: dim0=S, dim1=A, dim2=R.  Both the
    # aligned and Allen-template volumes below already use the same numpy
    # ordering, so all three columns share one convention and a single set
    # of "Axial / Coronal / Sagittal" labels applies uniformly.
    update_pbar()

    # Load aligned volume from output file, or compute it
    if output_path and Path(output_path).exists():
        vol_aligned, _aligned_res = read_omezarr(output_path, level=level)
        vol_aligned = np.asarray(vol_aligned[:])
    else:
        # Compute aligned volume using the transform
        vol_sitk = allen.numpy_to_sitk_image(vol_original, resolution)
        # Create reference and centered transform
        reference, centered_transform = compute_centered_reference_and_transform(vol_sitk, transform)

        vol_arr = sitk.GetArrayViewFromImage(vol_sitk)
        nonzero = vol_arr[vol_arr > 0]
        bg_value = float(np.percentile(nonzero, 1)) if len(nonzero) > 0 else 0.0
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(reference)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(bg_value)
        resampler.SetTransform(centered_transform)
        transformed_sitk = resampler.Execute(vol_sitk)
        vol_aligned = sitk.GetArrayFromImage(transformed_sitk)
    update_pbar()

    # Load Allen template at native resolution for reference
    # We'll just show it as a reference, not spatially aligned
    allen_sitk = allen.download_template_ras_aligned(allen_resolution, cache=True)
    allen_template = sitk.GetArrayFromImage(allen_sitk)
    # GetArrayFromImage already yields numpy (Z, Y, X) matching our convention.
    update_pbar()

    # Helper functions
    def get_center_slices(vol: Any) -> Any:
        """Get center slices in each plane."""
        z, y, x = vol.shape[0] // 2, vol.shape[1] // 2, vol.shape[2] // 2
        return vol[z, :, :], vol[:, y, :], vol[:, :, x]

    def get_display_range(vol: Any) -> Any:
        """Get display range from tissue voxels.

        The aligned volume is resampled with a non-zero default pixel value
        (sitk's `SetDefaultPixelValue(bg_value)`), so plain `vol > 0` includes
        the entire background floor and the resulting [1, 99] percentile is
        dominated by background — clipping all real tissue to white. Estimate
        the background floor from the 1st percentile of all voxels and
        compute the display range from voxels strictly above that floor.
        Zero-padded volumes (original / Allen template) are unchanged because
        their floor is 0.
        """
        if vol.size == 0 or float(vol.max()) == 0.0:
            return 0, 1
        flat = vol.ravel()
        floor = float(np.percentile(flat, 1))
        tissue = flat[flat > floor + 1e-6]
        if tissue.size == 0:
            return 0, 1
        return np.percentile(tissue, [1, 99])

    def find_content_center_slices(vol: Any) -> Any:
        """Find the slice with maximum content independently for each axis.

        Using a shared 3D centroid for all three views fails when the brain is
        asymmetric (e.g. cut at 45°): the centroid lands near the cut boundary,
        so one or more of the orthogonal slice views passes through the cut plane
        and shows a black stripe.  Instead, pick each index independently as the
        slice with the highest total signal along that axis.
        """
        if vol.max() == 0:
            return get_center_slices(vol)
        z = int(np.argmax(vol.sum(axis=(1, 2))))
        x = int(np.argmax(vol.sum(axis=(0, 2))))
        y = int(np.argmax(vol.sum(axis=(0, 1))))
        return vol[z, :, :], vol[:, x, :], vol[:, :, y]

    # Get slices - use content-centered slices for aligned volume
    orig_slices = get_center_slices(vol_original)
    aligned_slices = find_content_center_slices(vol_aligned)
    allen_slices = get_center_slices(allen_template)

    orig_vmin, orig_vmax = get_display_range(vol_original)
    align_vmin, align_vmax = get_display_range(vol_aligned)
    allen_vmin, allen_vmax = get_display_range(allen_template)

    # Create figure
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    fig.suptitle("Alignment Preview: Original vs Aligned vs Allen Template (Reference)", fontsize=16)

    # All three volumes are in standard RAS, numpy (S, A, R):
    #   dim0=S (Superior), dim1=A (Anterior), dim2=R (Right).
    # Slicing → anatomical plane:
    #   vol[z, :, :]  fixes S → AXIAL    (rows=A, cols=R)
    #   vol[:, y, :]  fixes A → CORONAL  (rows=S, cols=R)
    #   vol[:, :, x]  fixes R → SAGITTAL (rows=S, cols=A)
    plane_names = ["Axial (AR)", "Coronal (SR)", "Sagittal (SA)"]

    _debug_log(
        "create_alignment_preview: shapes & labels",
        original_shape=list(vol_original.shape),
        aligned_shape=list(vol_aligned.shape),
        allen_shape=list(allen_template.shape),
        plane_names=plane_names,
    )

    for row, plane_name in enumerate(plane_names):
        # All three columns share the same convention (S, A, R).  With
        # origin="lower", row 0 is rendered at the bottom of the figure, so:
        #   axial    rows=A → Posterior at bottom, Anterior up
        #   coronal  rows=S → Inferior at bottom,  Superior up
        #   sagittal rows=S → Inferior at bottom,  Superior up
        # No transpose or row reversal is needed.
        axes[row, 0].imshow(orig_slices[row], cmap="gray", origin="lower", vmin=orig_vmin, vmax=orig_vmax)
        axes[row, 0].set_title(f"Original - {plane_name}")
        axes[row, 0].axis("off")

        # Aligned
        axes[row, 1].imshow(aligned_slices[row], cmap="gray", origin="lower", vmin=align_vmin, vmax=align_vmax)
        axes[row, 1].set_title(f"Aligned - {plane_name}")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(allen_slices[row], cmap="gray", origin="lower", vmin=allen_vmin, vmax=allen_vmax)
        axes[row, 2].set_title(f"Allen {allen_resolution}µm - {plane_name}")
        axes[row, 2].axis("off")

    # Add info text
    info_text = (
        f"Original shape: {vol_original.shape}\nAligned shape: {vol_aligned.shape}\nAllen shape: {allen_template.shape}"
    )
    bbox_props = {"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5}
    fig.text(0.02, 0.02, info_text, fontsize=10, family="monospace", bbox=bbox_props)

    plt.tight_layout()
    Path(preview_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(preview_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    update_pbar()

    print(f"Alignment preview saved to: {preview_path}")


# =============================================================================
# Main entry point
# =============================================================================


def create_orientation_preview(
    input_path: Path,
    preview_path: str,
    level: int = 0,
    orientation_permutation: tuple | None = None,
    orientation_flips: tuple | None = None,
    initial_rotation_deg: tuple = (0.0, 0.0, 0.0),
) -> None:
    """
    Save a 3-panel orthogonal preview of the volume after orientation correction and initial rotation are applied.

    Axes are labelled in RAS space (Z=S, X=R, Y=A) so the result can be
    inspected directly against the Allen atlas orientation.

    Parameters
    ----------
    input_path: Path
        Path to input OME-Zarr.
    preview_path : str
        Output PNG path.
    level : int
        Pyramid level to load (lower = higher resolution but slower).
    orientation_permutation : tuple, optional
        Axis permutation from ``parse_orientation_code``.
    orientation_flips : tuple, optional
        Axis flips from ``parse_orientation_code``.
    initial_rotation_deg : tuple of float
        (Rx, Ry, Rz) initial rotation angles in degrees applied after orientation.
    """
    vol_zarr, resolution = read_omezarr(input_path, level=level)
    vol = np.asarray(vol_zarr[:]).astype(np.float32)

    # Apply orientation permutation + flips
    if orientation_permutation is not None:
        vol = apply_orientation_transform(vol, orientation_permutation, orientation_flips)
        resolution = list(reorder_resolution(tuple(resolution), orientation_permutation))

    # Apply initial rotation via SimpleITK (same path as the registration uses)
    if any(r != 0.0 for r in initial_rotation_deg):
        vol_sitk = allen.numpy_to_sitk_image(vol, resolution, cast_dtype=np.float32)
        center = vol_sitk.TransformContinuousIndexToPhysicalPoint([s / 2.0 for s in vol_sitk.GetSize()])
        rx, ry, rz = [np.deg2rad(a) for a in initial_rotation_deg]
        t = sitk.Euler3DTransform()
        t.SetCenter(center)
        t.SetRotation(rx, ry, rz)
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(vol_sitk)
        resampler.SetTransform(t.GetInverse())
        resampler.SetInterpolator(sitk.sitkLinear)
        vol = sitk.GetArrayFromImage(resampler.Execute(vol_sitk))

    # Display range from non-zero voxels
    nonzero = vol[vol > 0]
    vmin, vmax = np.percentile(nonzero if len(nonzero) else vol.ravel(), [1, 99])

    # Build title
    applied = []
    if orientation_permutation is not None:
        applied.append("orientation")
    if any(r != 0.0 for r in initial_rotation_deg):
        applied.append(f"rotation {list(initial_rotation_deg)}°")
    subtitle = f"({', '.join(applied)} applied)" if applied else "(no corrections applied)"

    z_mid = vol.shape[0] // 2
    y_mid = vol.shape[1] // 2
    x_mid = vol.shape[2] // 2

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"Orientation Preview -- {subtitle}\n"
        f"Shape: {vol.shape}  |  After corrections: dim0=S (Superior), dim1=A (Anterior), dim2=R (Right)",
        fontsize=11,
    )

    # After apply_orientation_transform the volume is in linumpy convention
    # (S, A, R): dim0=S (Superior), dim1=A (Anterior), dim2=R (Right).
    # Slicing → anatomical plane:
    #   vol[z, :, :]   fixes S → AXIAL    (rows=A, cols=R)
    #   vol[:, y, :]   fixes A → CORONAL  (rows=S, cols=R)
    #   vol[:, :, x]   fixes R → SAGITTAL (rows=S, cols=A)
    # We use origin="lower" so that row 0 sits at the bottom of the figure;
    # for the axial panel this places Posterior at bottom (Anterior up), and
    # for coronal/sagittal it places Inferior at bottom (Superior up).  No
    # row reversal is needed -- it would invert this and flip S↔I.
    axes[0].imshow(vol[z_mid, :, :], cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
    axes[0].set_title(f"Axial  (dim0=S={z_mid})")
    axes[0].set_xlabel("dim2=R  (← L    R →)")
    axes[0].set_ylabel("dim1=A  (← P    A →)")

    axes[1].imshow(vol[:, y_mid, :], cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Coronal  (dim1=A={y_mid})")
    axes[1].set_xlabel("dim2=R  (← L    R →)")
    axes[1].set_ylabel("dim0=S  (← I    S →)")

    axes[2].imshow(vol[:, :, x_mid], cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
    axes[2].set_title(f"Sagittal  (dim2=R={x_mid})")
    axes[2].set_xlabel("dim1=A  (← P    A →)")
    axes[2].set_ylabel("dim0=S  (← I    S →)")

    _debug_log(
        "create_orientation_preview: slicing decisions",
        vol_shape=list(vol.shape),
        panels=[
            {"axes": 0, "slice": f"vol[{z_mid}, :, :]", "fixed_axis": "dim0=S", "plane": "Axial"},
            {"axes": 1, "slice": f"vol[:, {y_mid}, :]", "fixed_axis": "dim1=A", "plane": "Coronal"},
            {"axes": 2, "slice": f"vol[:, :, {x_mid}]", "fixed_axis": "dim2=R", "plane": "Sagittal"},
        ],
    )

    plt.tight_layout()
    Path(preview_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(preview_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Orientation preview saved to: {preview_path}")


# =============================================================================
# Main entry point
# =============================================================================


def main() -> None:
    """Run the script. parse arguments and run alignment workflow."""
    parser = _build_arg_parser()
    args = parser.parse_args()

    input_path = Path(args.input_zarr)
    output_path = Path(args.output_zarr)

    if not input_path.exists():
        raise FileNotFoundError(f"Input zarr not found: {input_path}")

    # Preview-only mode
    if args.preview_only:
        preview_path = Path(args.preview) if args.preview else Path("input_preview.png")
        create_input_preview(input_path, preview_path, level=args.level)
        return

    # Parse orientation
    orientation_permutation = None
    orientation_flips = None
    if args.input_orientation:
        try:
            orientation_permutation, orientation_flips = parse_orientation_code(args.input_orientation)
            print(f"Input orientation '{args.input_orientation}':")
            print(f"  Axis permutation: {orientation_permutation}")
            print(f"  Axis flips: {orientation_flips}")
        except ValueError as e:
            parser.error(str(e))

    # Orientation + initial-rotation preview (can exit before registration)
    if args.orientation_preview or args.orientation_preview_only:
        preview_out = args.orientation_preview or "orientation_preview.png"
        create_orientation_preview(
            input_path,
            preview_out,
            level=args.level,
            orientation_permutation=orientation_permutation,
            orientation_flips=orientation_flips,
            initial_rotation_deg=tuple(args.initial_rotation),
        )
        if args.orientation_preview_only:
            return

    # Load input volume
    vol_zarr, zarr_resolution = read_omezarr(Path(input_path), level=args.level)
    resolution = tuple(zarr_resolution)

    # Progress bar - allocate steps for each phase
    registration_steps = 3  # Steps allocated for registration progress
    base_steps = 2 if args.store_transform_only else 5  # Load + save steps
    total_steps = base_steps + registration_steps
    if args.preview:
        total_steps += 4
    pbar = tqdm(total=total_steps, desc="Aligning to RAS")

    vol = np.asarray(vol_zarr[:])
    pbar.update(1)

    if args.verbose:
        print(f"Volume shape: {vol.shape}, Resolution: {resolution} mm")

    # Apply orientation correction for registration
    if orientation_permutation is not None:
        vol = apply_orientation_transform(vol, orientation_permutation, orientation_flips)
        resolution = reorder_resolution(resolution, orientation_permutation)

    # Create progress callback for registration
    registration_start_step = pbar.n
    progress_callback = create_registration_progress_callback(
        max_iterations=args.max_iterations,
        n_resolution_levels=3,
        pbar=pbar,
        registration_start_step=registration_start_step,
        registration_steps=registration_steps,
    )

    # Register to Allen atlas
    pbar.set_postfix_str("registering...")
    transform, stop_condition, error = allen.register_3d_rigid_to_allen(
        moving_image=vol,
        moving_spacing=resolution,
        allen_resolution=args.allen_resolution,
        metric=args.metric,
        max_iterations=args.max_iterations,
        verbose=args.verbose,
        progress_callback=progress_callback,
        initial_rotation_deg=tuple(args.initial_rotation),
    )
    # Ensure progress bar reaches end of registration steps
    pbar.n = registration_start_step + registration_steps
    pbar.refresh()

    print(f"Registration complete: {stop_condition}")
    print(f"Final metric value: {error:.6f}")
    del vol  # free registration-level volume before loading full-resolution data

    # Apply or store transform
    if args.store_transform_only:
        store_transform_in_metadata(input_path, transform)
        pbar.update(1)
    else:
        apply_transform_to_zarr(
            input_path,
            output_path,
            transform,
            chunks=tuple(args.chunks) if args.chunks else None,
            n_levels=args.n_levels,
            pyramid_resolutions=args.pyramid_resolutions,
            make_isotropic=args.make_isotropic,
            orientation_permutation=orientation_permutation,
            orientation_flips=orientation_flips,
            pbar=pbar,
        )
        print(f"Aligned volume saved to: {output_path}")

        # Save transform file
        # Strip the compound .ome.zarr extension (Path.stem only removes the last suffix)
        stem = output_path.with_suffix("").with_suffix("").name
        transform_path = output_path.parent / f"{stem}_transform.tfm"
        sitk.WriteTransform(transform, str(transform_path))
        print(f"Transform saved to: {transform_path}")
        pbar.update(1)

    # Generate preview
    if args.preview:
        pbar.set_postfix_str("generating preview...")
        create_alignment_preview(
            input_path,
            output_path if not args.store_transform_only else None,
            transform,
            resolution,
            args.preview,
            allen_resolution=args.allen_resolution,
            level=args.level,
            orientation_permutation=orientation_permutation,
            orientation_flips=orientation_flips,
            pbar=pbar,
        )

    pbar.set_postfix_str("complete")
    pbar.close()


if __name__ == "__main__":
    main()
