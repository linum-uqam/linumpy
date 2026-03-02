#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interpolate a missing or degraded slice using information from adjacent slices.

This script implements registration-based morphing interpolation to reconstruct
a missing slice in a serial sectioning dataset. The method:

1. Registers the slice before the gap to the slice after
2. Computes a half-transform (midpoint transformation)
3. Warps both adjacent slices toward the midpoint
4. Blends the warped slices to create the interpolated result
5. Optionally blends with a degraded slice if available

If a degraded slice is provided, its quality is automatically assessed using
structural similarity (SSIM) and edge preservation metrics. The degraded slice
is blended with the interpolated result based on its quality score.

This approach is based on motion-compensated interpolation techniques commonly
used in video frame interpolation, adapted for 3D microscopy volumes.

References:
- Lee et al. (1991) "Shape-based interpolation of multidimensional grey-level 
  images", IEEE Trans. Medical Imaging
- Bao et al. (2019) "Depth-Aware Video Frame Interpolation", CVPR
- Penney et al. (2004) "A comparison of similarity measures for use in 2-D-3-D 
  medical image registration", IEEE Trans. Medical Imaging

Note: This method is only suitable for interpolating a SINGLE missing slice.
When two or more consecutive slices are missing, there is insufficient 
information for accurate reconstruction.

Example usage:
    # Without degraded slice (pure interpolation)
    linum_interpolate_missing_slice.py slice_z00.ome.zarr slice_z02.ome.zarr \\
        slice_z01_interpolated.ome.zarr --method registration
    
    # With degraded slice (quality-weighted blend)
    linum_interpolate_missing_slice.py slice_z00.ome.zarr slice_z02.ome.zarr \\
        slice_z01_interpolated.ome.zarr --degraded_slice slice_z01_bad.ome.zarr
"""
# Configure thread limits before numpy/scipy imports
import linumpy._thread_config  # noqa: F401

import argparse
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from pathlib import Path

from linumpy.io.zarr import read_omezarr, save_omezarr
from linumpy.stitching.interpolation import (
    compute_half_affine_transform,
    interpolate_average,
    interpolate_weighted,
    interpolate_registration_based,
    assess_degraded_slice_quality,
    blend_with_degraded,
)
from linumpy.stitching.registration import register_2d_images_sitk, apply_transform
from linumpy.utils.io import add_overwrite_arg, assert_output_exists
from linumpy.utils.image_quality import (
    compute_ssim_3d,
    compute_edge_score,
    compute_variance_score,
)
import dask.array as da

# Configure all libraries (especially SimpleITK) to respect thread limits
from linumpy._thread_config import configure_all_libraries
configure_all_libraries()


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("slice_before",
                   help="Path to the slice BEFORE the missing slice (*.ome.zarr)")
    p.add_argument("slice_after",
                   help="Path to the slice AFTER the missing slice (*.ome.zarr)")
    p.add_argument("output",
                   help="Output path for the interpolated slice (*.ome.zarr)")
    p.add_argument("--method", choices=['registration', 'average', 'weighted'],
                   default='registration',
                   help="Interpolation method:\n"
                        "  registration - Registration-based morphing (recommended)\n"
                        "  average - Simple average of adjacent slices\n"
                        "  weighted - Weighted average with distance falloff\n"
                        "[default: %(default)s]")
    p.add_argument("--blend_method", choices=['linear', 'gaussian'],
                   default='gaussian',
                   help="Blending method for combining warped slices:\n"
                        "  linear - Equal 50/50 blend (may show edges)\n"
                        "  gaussian - Feathered blend using distance transform (recommended)\n"
                        "[default: %(default)s]")
    p.add_argument("--registration_metric", choices=['MSE', 'CC', 'MI'],
                   default='MSE',
                   help="Metric for registration [default: %(default)s]")
    p.add_argument("--max_iterations", type=int, default=1000,
                   help="Maximum iterations for registration [default: %(default)s]")
    p.add_argument("--reference_slice", type=int, default=None,
                   help="Z-index in slice_before to use as registration reference.\n"
                        "If not specified, uses the middle slice.")
    
    # Degraded slice options
    degraded_group = p.add_argument_group('Degraded Slice Options',
                                          'Use a degraded/damaged slice to improve interpolation')
    degraded_group.add_argument("--degraded_slice", type=str, default=None,
                                help="Path to a degraded slice that has usable data (*.ome.zarr).\n"
                                     "Quality is automatically assessed and blended accordingly.")
    degraded_group.add_argument("--degraded_weight", type=float, default=None,
                                help="Manual override for degraded slice weight (0.0-1.0).\n"
                                     "If not specified, weight is automatically computed from quality.")
    degraded_group.add_argument("--min_quality_threshold", type=float, default=0.2,
                                help="Minimum quality score to use degraded slice.\n"
                                     "Below this threshold, degraded slice is ignored.\n"
                                     "[default: %(default)s]")
    
    # Preview/debug options
    preview_group = p.add_argument_group('Preview Options',
                                         'Generate visual previews for quality checking')
    preview_group.add_argument("--preview", type=str, default=None,
                               help="Path to save a preview image (PNG) showing:\n"
                                    "- Slice before, slice after\n"
                                    "- Interpolated result\n"
                                    "- Degraded slice (if provided)\n"
                                    "Useful for verifying interpolation quality.")
    preview_group.add_argument("--preview_slice", type=int, default=None,
                               help="Z-index to use for preview. Default: middle slice.")
    preview_group.add_argument("--preview_dpi", type=int, default=150,
                               help="DPI for preview image [default: %(default)s]")

    add_overwrite_arg(p)
    return p


def invert_transform(transform):
    """
    Compute the inverse of a transform.
    
    Parameters
    ----------
    transform : sitk.Transform
        Transform to invert.
        
    Returns
    -------
    sitk.Transform
        Inverted transform.
    """
    return transform.GetInverse()


def register_slices_2d(fixed_slice, moving_slice, metric='MSE', max_iterations=1000):
    """
    Register a 2D slice from moving volume to fixed volume.
    
    Parameters
    ----------
    fixed_slice : np.ndarray
        2D fixed image.
    moving_slice : np.ndarray
        2D moving image.
    metric : str
        Registration metric.
    max_iterations : int
        Maximum iterations.
        
    Returns
    -------
    sitk.Transform
        The computed transform.
    float
        Registration error metric.
    """
    # Normalize images for registration
    fixed_norm = fixed_slice.astype(np.float32)
    moving_norm = moving_slice.astype(np.float32)
    
    # Normalize to [0, 1]
    if fixed_norm.max() > fixed_norm.min():
        fixed_norm = (fixed_norm - fixed_norm.min()) / (fixed_norm.max() - fixed_norm.min())
    if moving_norm.max() > moving_norm.min():
        moving_norm = (moving_norm - moving_norm.min()) / (moving_norm.max() - moving_norm.min())
    
    transform, _, error = register_2d_images_sitk(
        fixed_norm, moving_norm,
        method='affine',
        metric=metric,
        max_iterations=max_iterations,
        return_3d_transform=False,
        verbose=False
    )
    
    return transform, error


def generate_preview(vol_before, vol_after, interpolated, output_path,
                     vol_degraded=None, final_result=None,
                     preview_slice=None, dpi=150,
                     degraded_weight=None, quality_threshold=None):
    """
    Generate a preview image showing the interpolation results.

    Parameters
    ----------
    vol_before : np.ndarray
        Volume before the missing slice.
    vol_after : np.ndarray
        Volume after the missing slice.
    interpolated : np.ndarray
        Pure interpolated result.
    output_path : str or Path
        Path to save the preview image.
    vol_degraded : np.ndarray, optional
        Degraded slice volume if provided.
    final_result : np.ndarray, optional
        Final result after blending with degraded (if different from interpolated).
    preview_slice : int, optional
        Z-index to use for preview. Default: middle slice.
    dpi : int
        DPI for the output image.
    degraded_weight : float, optional
        Weight used for degraded slice blending.
    quality_threshold : float, optional
        Quality threshold used.
    """
    # Determine slice index
    if preview_slice is None:
        preview_slice = vol_before.shape[0] // 2
    preview_slice = max(0, min(preview_slice, vol_before.shape[0] - 1))

    # Normalize function for display
    def normalize_for_display(img):
        img = img.astype(np.float32)
        p1, p99 = np.percentile(img[img > 0], [1, 99]) if np.any(img > 0) else (0, 1)
        if p99 > p1:
            img = (img - p1) / (p99 - p1)
        return np.clip(img, 0, 1)

    # Extract slices
    before_slice = normalize_for_display(vol_before[preview_slice])
    after_slice = normalize_for_display(vol_after[preview_slice])
    interp_slice = normalize_for_display(interpolated[preview_slice])

    # Determine layout based on whether we have degraded slice
    has_degraded = vol_degraded is not None
    has_final = final_result is not None and not np.allclose(final_result, interpolated)

    if has_degraded and has_final:
        # 2x3 layout: before, after, degraded, interpolated, final, difference
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        degraded_slice = normalize_for_display(vol_degraded[preview_slice])
        final_slice = normalize_for_display(final_result[preview_slice])

        # Row 1: inputs
        axes[0].imshow(before_slice, cmap='gray')
        axes[0].set_title('Slice Before (input)')
        axes[0].axis('off')

        axes[1].imshow(after_slice, cmap='gray')
        axes[1].set_title('Slice After (input)')
        axes[1].axis('off')

        axes[2].imshow(degraded_slice, cmap='gray')
        title = f'Degraded Slice'
        if degraded_weight is not None:
            title += f'\n(quality={degraded_weight:.2f})'
        axes[2].set_title(title)
        axes[2].axis('off')

        # Row 2: outputs
        axes[3].imshow(interp_slice, cmap='gray')
        axes[3].set_title('Pure Interpolation')
        axes[3].axis('off')

        axes[4].imshow(final_slice, cmap='gray')
        title = 'Final Result (blended)'
        if degraded_weight is not None:
            title += f'\n(w={degraded_weight:.2f})'
        axes[4].set_title(title)
        axes[4].axis('off')

        # Difference image
        diff = np.abs(interp_slice - degraded_slice)
        axes[5].imshow(diff, cmap='hot')
        axes[5].set_title('|Interpolated - Degraded|')
        axes[5].axis('off')

    elif has_degraded:
        # 2x2 layout: before, after, interpolated, degraded
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        degraded_slice = normalize_for_display(vol_degraded[preview_slice])

        axes[0].imshow(before_slice, cmap='gray')
        axes[0].set_title('Slice Before (input)')
        axes[0].axis('off')

        axes[1].imshow(after_slice, cmap='gray')
        axes[1].set_title('Slice After (input)')
        axes[1].axis('off')

        axes[2].imshow(interp_slice, cmap='gray')
        axes[2].set_title('Interpolated (output)')
        axes[2].axis('off')

        axes[3].imshow(degraded_slice, cmap='gray')
        title = f'Degraded (not used)'
        if degraded_weight is not None:
            title = f'Degraded (q={degraded_weight:.2f} < {quality_threshold})'
        axes[3].set_title(title)
        axes[3].axis('off')

    else:
        # 2x2 layout: before, after, interpolated, XZ view
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        axes[0].imshow(before_slice, cmap='gray')
        axes[0].set_title('Slice Before (input)')
        axes[0].axis('off')

        axes[1].imshow(after_slice, cmap='gray')
        axes[1].set_title('Slice After (input)')
        axes[1].axis('off')

        axes[2].imshow(interp_slice, cmap='gray')
        axes[2].set_title('Interpolated (output)')
        axes[2].axis('off')

        # Show XZ cross-section to visualize z-continuity
        y_mid = vol_before.shape[1] // 2
        xz_before = normalize_for_display(vol_before[:, y_mid, :])
        xz_interp = normalize_for_display(interpolated[:, y_mid, :])
        xz_after = normalize_for_display(vol_after[:, y_mid, :])

        # Stack them for comparison
        xz_combined = np.vstack([xz_before, xz_interp, xz_after])
        axes[3].imshow(xz_combined, cmap='gray', aspect='auto')
        axes[3].set_title('XZ View: Before | Interp | After')
        axes[3].axhline(y=xz_before.shape[0], color='cyan', linestyle='--', linewidth=0.5)
        axes[3].axhline(y=xz_before.shape[0] + xz_interp.shape[0], color='cyan', linestyle='--', linewidth=0.5)
        axes[3].axis('off')

    fig.suptitle(f'Slice Interpolation Preview (z={preview_slice})', fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"Preview saved to: {output_path}")


def main():
    p = _build_arg_parser()
    args = p.parse_args()
    
    # Validate inputs
    slice_before_path = Path(args.slice_before)
    slice_after_path = Path(args.slice_after)
    output_path = Path(args.output)
    
    if not slice_before_path.exists():
        p.error(f"Slice before not found: {slice_before_path}")
    if not slice_after_path.exists():
        p.error(f"Slice after not found: {slice_after_path}")
    
    assert_output_exists(output_path, p, args)
    
    print(f"Loading slice before: {slice_before_path}")
    vol_before, res_before = read_omezarr(slice_before_path)
    vol_before = np.array(vol_before)
    
    print(f"Loading slice after: {slice_after_path}")
    vol_after, res_after = read_omezarr(slice_after_path)
    vol_after = np.array(vol_after)
    
    # Handle shape mismatches
    if vol_before.shape != vol_after.shape:
        print(f"Shape mismatch detected: {vol_before.shape} vs {vol_after.shape}")

        # Handle z-dimension mismatch by truncating to minimum
        min_z = min(vol_before.shape[0], vol_after.shape[0])
        if vol_before.shape[0] != vol_after.shape[0]:
            print(f"  Truncating z-dimension to minimum: {min_z}")
            vol_before = vol_before[:min_z]
            vol_after = vol_after[:min_z]

        # Handle X/Y dimension mismatch by using maximum and zero-padding
        if vol_before.shape[1:] != vol_after.shape[1:]:
            max_x = max(vol_before.shape[1], vol_after.shape[1])
            max_y = max(vol_before.shape[2], vol_after.shape[2])
            print(f"  Adjusting X/Y dimensions to: ({max_x}, {max_y})")

            # Pad vol_before if needed
            if vol_before.shape[1] < max_x or vol_before.shape[2] < max_y:
                padded = np.zeros((min_z, max_x, max_y), dtype=vol_before.dtype)
                padded[:, :vol_before.shape[1], :vol_before.shape[2]] = vol_before
                vol_before = padded

            # Pad vol_after if needed
            if vol_after.shape[1] < max_x or vol_after.shape[2] < max_y:
                padded = np.zeros((min_z, max_x, max_y), dtype=vol_after.dtype)
                padded[:, :vol_after.shape[1], :vol_after.shape[2]] = vol_after
                vol_after = padded

        print(f"  Adjusted shapes: {vol_before.shape}")

    # Store original z-depth for output (use the target z-depth, which is average of neighbors)
    output_z_depth = vol_before.shape[0]

    # Validate resolutions match
    if res_before != res_after:
        print(f"Warning: Resolution mismatch: {res_before} vs {res_after}")
    
    # Load degraded slice if provided
    vol_degraded = None
    if args.degraded_slice is not None:
        degraded_path = Path(args.degraded_slice)
        if degraded_path.exists():
            print(f"Loading degraded slice: {degraded_path}")
            vol_degraded, res_degraded = read_omezarr(degraded_path)
            vol_degraded = np.array(vol_degraded)
            
            if vol_degraded.shape != vol_before.shape:
                print(f"Degraded slice shape mismatch: {vol_degraded.shape} vs {vol_before.shape}")
                # Try to adjust degraded slice to match
                target_shape = vol_before.shape
                try:
                    # Truncate z if needed
                    if vol_degraded.shape[0] > target_shape[0]:
                        vol_degraded = vol_degraded[:target_shape[0]]
                    elif vol_degraded.shape[0] < target_shape[0]:
                        # Pad z with zeros
                        padded = np.zeros(target_shape, dtype=vol_degraded.dtype)
                        padded[:vol_degraded.shape[0]] = vol_degraded
                        vol_degraded = padded

                    # Handle X/Y mismatch
                    if vol_degraded.shape[1:] != target_shape[1:]:
                        padded = np.zeros(target_shape, dtype=vol_degraded.dtype)
                        min_x = min(vol_degraded.shape[1], target_shape[1])
                        min_y = min(vol_degraded.shape[2], target_shape[2])
                        padded[:, :min_x, :min_y] = vol_degraded[:, :min_x, :min_y]
                        vol_degraded = padded

                    print(f"  Adjusted degraded slice shape to: {vol_degraded.shape}")
                except Exception as e:
                    print(f"  Could not adjust degraded slice shape, ignoring: {e}")
                    vol_degraded = None
        else:
            print(f"Warning: Degraded slice not found, proceeding without it: {degraded_path}")
    
    print(f"Volume shape: {vol_before.shape}")
    print(f"Resolution: {res_before}")
    print(f"Method: {args.method}")
    
    # Perform interpolation
    if args.method == 'registration':
        print("Performing registration-based interpolation...")
        interpolated = interpolate_registration_based(
            vol_before, vol_after,
            metric=args.registration_metric,
            max_iterations=args.max_iterations,
            reference_slice=args.reference_slice,
            blend_method=args.blend_method
        )
    elif args.method == 'average':
        print("Performing simple average interpolation...")
        interpolated = interpolate_average(vol_before, vol_after)
    elif args.method == 'weighted':
        print("Performing weighted average interpolation...")
        interpolated = interpolate_weighted(vol_before, vol_after)
    else:
        p.error(f"Unknown method: {args.method}")
    
    # Blend with degraded slice if available
    final_result = interpolated
    quality_weight = None
    used_degraded = False
    if vol_degraded is not None:
        # Determine quality weight
        if args.degraded_weight is not None:
            # Manual override
            quality_weight = args.degraded_weight
            print(f"Using manual degraded weight: {quality_weight:.3f}")
        else:
            # Automatic quality assessment
            quality_weight, metrics = assess_degraded_slice_quality(
                vol_degraded, vol_before, vol_after
            )
        
        # Check if quality is above threshold
        if quality_weight >= args.min_quality_threshold:
            print(f"Blending with degraded slice (weight={quality_weight:.3f})")
            final_result = blend_with_degraded(interpolated, vol_degraded, quality_weight)
            used_degraded = True
        else:
            print(f"Degraded slice quality ({quality_weight:.3f}) below threshold "
                  f"({args.min_quality_threshold}), using pure interpolation")
    
    # Generate preview if requested
    if args.preview is not None:
        print(f"Generating preview...")
        generate_preview(
            vol_before, vol_after, interpolated,
            output_path=args.preview,
            vol_degraded=vol_degraded,
            final_result=final_result if used_degraded else None,
            preview_slice=args.preview_slice,
            dpi=args.preview_dpi,
            degraded_weight=quality_weight,
            quality_threshold=args.min_quality_threshold
        )

    # Convert to original dtype if needed
    original_dtype = vol_before.dtype
    if np.issubdtype(original_dtype, np.integer):
        final_result = np.clip(final_result, 0, np.iinfo(original_dtype).max)
        final_result = final_result.astype(original_dtype)
    
    # Save result
    print(f"Saving interpolated slice to: {output_path}")
    save_omezarr(da.from_array(final_result), str(output_path), res_before)
    
    print("Done!")


if __name__ == "__main__":
    main()
