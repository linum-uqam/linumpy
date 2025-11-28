#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interpolate a missing slice using information from adjacent slices.

This script implements registration-based morphing interpolation to reconstruct
a missing slice in a serial sectioning dataset. The method:

1. Registers the slice before the gap to the slice after
2. Computes a half-transform (midpoint transformation)
3. Warps both adjacent slices toward the midpoint
4. Blends the warped slices to create the interpolated result

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
    linum_interpolate_missing_slice.py slice_z00.ome.zarr slice_z02.ome.zarr \\
        slice_z01_interpolated.ome.zarr --method registration
"""
import argparse
import numpy as np
import SimpleITK as sitk
from pathlib import Path

from linumpy.io.zarr import read_omezarr, save_omezarr
from linumpy.stitching.registration import register_2d_images_sitk, apply_transform
from linumpy.utils.io import add_overwrite_arg, assert_output_exists
import dask.array as da


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
                   default='linear',
                   help="Blending method for combining warped slices:\n"
                        "  linear - Equal 50/50 blend\n"
                        "  gaussian - Distance-weighted gaussian blend\n"
                        "[default: %(default)s]")
    p.add_argument("--registration_metric", choices=['MSE', 'CC', 'MI'],
                   default='MSE',
                   help="Metric for registration [default: %(default)s]")
    p.add_argument("--max_iterations", type=int, default=1000,
                   help="Maximum iterations for registration [default: %(default)s]")
    p.add_argument("--reference_slice", type=int, default=None,
                   help="Z-index in slice_before to use as registration reference.\n"
                        "If not specified, uses the middle slice.")
    add_overwrite_arg(p)
    return p


def compute_half_affine_transform(transform):
    """
    Compute a transform that is 'halfway' to the given transform.
    
    For affine transforms, this decomposes the transform and applies
    half of the rotation, translation, and scaling.
    
    Parameters
    ----------
    transform : sitk.Transform
        The full transform from image A to image B.
        
    Returns
    -------
    sitk.AffineTransform
        A transform representing half the transformation.
    """
    # Get transform parameters
    if isinstance(transform, sitk.CompositeTransform):
        # Flatten composite transform to affine
        transform = sitk.AffineTransform(transform.GetNthTransform(0))
    
    # Create identity transform for starting point
    dim = transform.GetDimension()
    
    if dim == 2:
        half_transform = sitk.AffineTransform(2)
        
        # Get the matrix and translation
        matrix = np.array(transform.GetMatrix()).reshape(2, 2)
        translation = np.array(transform.GetTranslation())
        center = np.array(transform.GetCenter())
        
        # Compute matrix square root using eigendecomposition
        # For rotation/scale matrices: sqrt(M) gives half the transformation
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        sqrt_eigenvalues = np.sqrt(eigenvalues.astype(complex))
        half_matrix = (eigenvectors @ np.diag(sqrt_eigenvalues) @ 
                       np.linalg.inv(eigenvectors)).real
        
        # Half the translation
        half_translation = translation / 2.0
        
        half_transform.SetMatrix(half_matrix.flatten().tolist())
        half_transform.SetTranslation(half_translation.tolist())
        half_transform.SetCenter(center.tolist())
        
    elif dim == 3:
        half_transform = sitk.AffineTransform(3)
        
        matrix = np.array(transform.GetMatrix()).reshape(3, 3)
        translation = np.array(transform.GetTranslation())
        center = np.array(transform.GetCenter())
        
        # Matrix square root
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        sqrt_eigenvalues = np.sqrt(eigenvalues.astype(complex))
        half_matrix = (eigenvectors @ np.diag(sqrt_eigenvalues) @ 
                       np.linalg.inv(eigenvectors)).real
        
        half_translation = translation / 2.0
        
        half_transform.SetMatrix(half_matrix.flatten().tolist())
        half_transform.SetTranslation(half_translation.tolist())
        half_transform.SetCenter(center.tolist())
    else:
        raise ValueError(f"Unsupported transform dimension: {dim}")
    
    return half_transform


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


def interpolate_registration_based(vol_before, vol_after, metric='MSE', 
                                   max_iterations=1000, reference_slice=None,
                                   blend_method='linear'):
    """
    Interpolate missing slice using registration-based morphing.
    
    This method:
    1. Registers 2D slices from vol_before to vol_after
    2. Computes half-transforms for each z-level
    3. Warps both volumes toward the midpoint
    4. Blends the results
    
    Parameters
    ----------
    vol_before : np.ndarray
        3D volume (Z, X, Y) before the missing slice.
    vol_after : np.ndarray
        3D volume (Z, X, Y) after the missing slice.
    metric : str
        Registration metric.
    max_iterations : int
        Maximum registration iterations.
    reference_slice : int
        Z-index to use for registration reference. If None, uses middle.
    blend_method : str
        'linear' or 'gaussian' blending.
        
    Returns
    -------
    np.ndarray
        Interpolated 3D volume.
    """
    nz, nx, ny = vol_before.shape
    
    if reference_slice is None:
        reference_slice = nz // 2
    
    print(f"  Using z-slice {reference_slice} as registration reference")
    
    # Register using the reference slice
    fixed_2d = vol_after[reference_slice]
    moving_2d = vol_before[reference_slice]
    
    transform_2d, error = register_slices_2d(
        fixed_2d, moving_2d, 
        metric=metric, 
        max_iterations=max_iterations
    )
    
    print(f"  Registration error: {error:.6f}")
    
    # Compute half transform
    half_transform = compute_half_affine_transform(transform_2d)
    inv_half_transform = invert_transform(half_transform)
    
    # Apply transforms to each z-slice
    warped_before = np.zeros_like(vol_before, dtype=np.float32)
    warped_after = np.zeros_like(vol_after, dtype=np.float32)
    
    for z in range(nz):
        # Warp vol_before forward (toward midpoint)
        warped_before[z] = apply_transform(vol_before[z].astype(np.float32), half_transform)
        # Warp vol_after backward (toward midpoint)
        warped_after[z] = apply_transform(vol_after[z].astype(np.float32), inv_half_transform)
    
    # Blend the two warped volumes
    if blend_method == 'linear':
        # Simple 50/50 blend
        interpolated = 0.5 * warped_before + 0.5 * warped_after
    elif blend_method == 'gaussian':
        # Gaussian-weighted blend (slightly favor the center)
        # Create spatial weights based on content (non-zero regions)
        weight_before = (warped_before > 0).astype(np.float32)
        weight_after = (warped_after > 0).astype(np.float32)
        
        # Normalize weights
        total_weight = weight_before + weight_after + 1e-10
        w_before = weight_before / total_weight
        w_after = weight_after / total_weight
        
        # Where both have content, use 50/50
        both_valid = (weight_before > 0) & (weight_after > 0)
        w_before[both_valid] = 0.5
        w_after[both_valid] = 0.5
        
        interpolated = w_before * warped_before + w_after * warped_after
    else:
        raise ValueError(f"Unknown blend method: {blend_method}")
    
    return interpolated


def interpolate_average(vol_before, vol_after):
    """
    Simple average of two volumes.
    
    Parameters
    ----------
    vol_before : np.ndarray
        Volume before missing slice.
    vol_after : np.ndarray
        Volume after missing slice.
        
    Returns
    -------
    np.ndarray
        Average of the two volumes.
    """
    return 0.5 * vol_before.astype(np.float32) + 0.5 * vol_after.astype(np.float32)


def interpolate_weighted(vol_before, vol_after, sigma=2.0):
    """
    Weighted average with gaussian smoothing.
    
    Parameters
    ----------
    vol_before : np.ndarray
        Volume before missing slice.
    vol_after : np.ndarray
        Volume after missing slice.
    sigma : float
        Gaussian smoothing sigma.
        
    Returns
    -------
    np.ndarray
        Weighted average.
    """
    from scipy.ndimage import gaussian_filter
    
    avg = 0.5 * vol_before.astype(np.float32) + 0.5 * vol_after.astype(np.float32)
    
    # Light smoothing along z-axis to reduce discontinuities
    smoothed = gaussian_filter(avg, sigma=(sigma, 0, 0))
    
    return smoothed


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
    
    # Validate shapes match
    if vol_before.shape != vol_after.shape:
        p.error(f"Shape mismatch: {vol_before.shape} vs {vol_after.shape}")
    
    # Validate resolutions match
    if res_before != res_after:
        print(f"Warning: Resolution mismatch: {res_before} vs {res_after}")
    
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
    
    # Convert to original dtype if needed
    original_dtype = vol_before.dtype
    if np.issubdtype(original_dtype, np.integer):
        interpolated = np.clip(interpolated, 0, np.iinfo(original_dtype).max)
        interpolated = interpolated.astype(original_dtype)
    
    # Save result
    print(f"Saving interpolated slice to: {output_path}")
    save_omezarr(da.from_array(interpolated), str(output_path), res_before)
    
    print("Done!")


if __name__ == "__main__":
    main()

