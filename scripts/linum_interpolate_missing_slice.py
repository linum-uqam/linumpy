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


def compute_ssim_3d(vol1, vol2, win_size=7):
    """
    Compute mean Structural Similarity Index (SSIM) between two 3D volumes.
    
    Computes SSIM for each z-slice and returns the mean.
    
    Parameters
    ----------
    vol1 : np.ndarray
        First volume (Z, X, Y).
    vol2 : np.ndarray
        Second volume (Z, X, Y).
    win_size : int
        Window size for SSIM computation.
        
    Returns
    -------
    float
        Mean SSIM score (0 to 1, higher is better).
    """
    from skimage.metrics import structural_similarity as ssim
    
    # Normalize volumes to [0, 1]
    v1 = vol1.astype(np.float32)
    v2 = vol2.astype(np.float32)
    
    if v1.max() > v1.min():
        v1 = (v1 - v1.min()) / (v1.max() - v1.min())
    if v2.max() > v2.min():
        v2 = (v2 - v2.min()) / (v2.max() - v2.min())
    
    ssim_scores = []
    for z in range(vol1.shape[0]):
        # Handle small images
        actual_win_size = min(win_size, min(v1.shape[1], v1.shape[2]) - 1)
        if actual_win_size % 2 == 0:
            actual_win_size -= 1
        if actual_win_size < 3:
            actual_win_size = 3
            
        try:
            score = ssim(v1[z], v2[z], win_size=actual_win_size, data_range=1.0)
            ssim_scores.append(score)
        except Exception:
            # If SSIM fails, use normalized cross-correlation as fallback
            ncc = np.corrcoef(v1[z].flatten(), v2[z].flatten())[0, 1]
            ssim_scores.append(max(0, ncc))
    
    return np.mean(ssim_scores)


def compute_edge_preservation_score(vol_degraded, vol_reference):
    """
    Compute edge preservation score between degraded and reference volume.
    
    Uses Sobel edge detection to compare edge structures.
    
    Parameters
    ----------
    vol_degraded : np.ndarray
        Degraded volume.
    vol_reference : np.ndarray
        Reference volume (e.g., average of neighbors).
        
    Returns
    -------
    float
        Edge preservation score (0 to 1, higher is better).
    """
    from scipy.ndimage import sobel
    
    # Normalize
    v_deg = vol_degraded.astype(np.float32)
    v_ref = vol_reference.astype(np.float32)
    
    if v_deg.max() > v_deg.min():
        v_deg = (v_deg - v_deg.min()) / (v_deg.max() - v_deg.min())
    if v_ref.max() > v_ref.min():
        v_ref = (v_ref - v_ref.min()) / (v_ref.max() - v_ref.min())
    
    # Compute edges using Sobel
    edges_deg = np.sqrt(sobel(v_deg, axis=1)**2 + sobel(v_deg, axis=2)**2)
    edges_ref = np.sqrt(sobel(v_ref, axis=1)**2 + sobel(v_ref, axis=2)**2)
    
    # Normalize edges
    if edges_deg.max() > 0:
        edges_deg = edges_deg / edges_deg.max()
    if edges_ref.max() > 0:
        edges_ref = edges_ref / edges_ref.max()
    
    # Compute correlation of edge maps
    correlation = np.corrcoef(edges_deg.flatten(), edges_ref.flatten())[0, 1]
    
    # Handle NaN (can occur if one image has no edges)
    if np.isnan(correlation):
        return 0.0
    
    # Convert to 0-1 range (correlation is -1 to 1)
    return max(0, correlation)


def compute_variance_ratio(vol_degraded, vol_reference):
    """
    Compute variance ratio between degraded and reference volumes.
    
    Low variance in degraded slice may indicate data loss or corruption.
    
    Parameters
    ----------
    vol_degraded : np.ndarray
        Degraded volume.
    vol_reference : np.ndarray
        Reference volume.
        
    Returns
    -------
    float
        Variance ratio score (0 to 1, higher means more similar variance).
    """
    var_deg = np.var(vol_degraded)
    var_ref = np.var(vol_reference)
    
    if var_ref == 0:
        return 0.0
    
    ratio = var_deg / var_ref
    
    # Score is 1 when variances are equal, decreases as they diverge
    # Using a logistic-like function
    score = 2.0 / (1.0 + np.abs(np.log(ratio + 1e-10)))
    
    return min(1.0, max(0.0, score))


def assess_degraded_slice_quality(vol_degraded, vol_before, vol_after):
    """
    Automatically assess the quality of a degraded slice.
    
    Uses multiple metrics to determine how usable the degraded slice is:
    1. SSIM with neighboring slices
    2. Edge preservation compared to expected structure
    3. Variance consistency
    
    Parameters
    ----------
    vol_degraded : np.ndarray
        The degraded slice volume.
    vol_before : np.ndarray
        Volume before the degraded slice.
    vol_after : np.ndarray
        Volume after the degraded slice.
        
    Returns
    -------
    float
        Quality score from 0 (unusable) to 1 (perfect).
    dict
        Individual metric scores for debugging.
    """
    print("  Assessing degraded slice quality...")
    
    # Create expected reference (simple average of neighbors)
    reference = 0.5 * vol_before.astype(np.float32) + 0.5 * vol_after.astype(np.float32)
    
    # Metric 1: SSIM with neighbors
    ssim_before = compute_ssim_3d(vol_degraded, vol_before)
    ssim_after = compute_ssim_3d(vol_degraded, vol_after)
    ssim_score = (ssim_before + ssim_after) / 2
    print(f"    SSIM with neighbors: {ssim_score:.3f} (before={ssim_before:.3f}, after={ssim_after:.3f})")
    
    # Metric 2: Edge preservation
    edge_score = compute_edge_preservation_score(vol_degraded, reference)
    print(f"    Edge preservation: {edge_score:.3f}")
    
    # Metric 3: Variance consistency
    variance_score = compute_variance_ratio(vol_degraded, reference)
    print(f"    Variance ratio: {variance_score:.3f}")
    
    # Combine metrics (weighted average)
    # SSIM is most important, then edges, then variance
    weights = {'ssim': 0.5, 'edge': 0.3, 'variance': 0.2}
    
    quality_score = (
        weights['ssim'] * ssim_score +
        weights['edge'] * edge_score +
        weights['variance'] * variance_score
    )
    
    metrics = {
        'ssim_before': ssim_before,
        'ssim_after': ssim_after,
        'ssim_mean': ssim_score,
        'edge_preservation': edge_score,
        'variance_ratio': variance_score,
        'overall': quality_score
    }
    
    print(f"    Overall quality score: {quality_score:.3f}")
    
    return quality_score, metrics


def blend_with_degraded(interpolated, degraded, quality_weight):
    """
    Blend interpolated result with degraded slice based on quality weight.
    
    Parameters
    ----------
    interpolated : np.ndarray
        Pure interpolated volume.
    degraded : np.ndarray
        Degraded slice volume.
    quality_weight : float
        Weight for degraded slice (0 to 1).
        
    Returns
    -------
    np.ndarray
        Blended result.
    """
    w = quality_weight
    return w * degraded.astype(np.float32) + (1 - w) * interpolated.astype(np.float32)


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
    
    # Load degraded slice if provided
    vol_degraded = None
    if args.degraded_slice is not None:
        degraded_path = Path(args.degraded_slice)
        if degraded_path.exists():
            print(f"Loading degraded slice: {degraded_path}")
            vol_degraded, res_degraded = read_omezarr(degraded_path)
            vol_degraded = np.array(vol_degraded)
            
            if vol_degraded.shape != vol_before.shape:
                print(f"Warning: Degraded slice shape mismatch, ignoring: {vol_degraded.shape} vs {vol_before.shape}")
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
        else:
            print(f"Degraded slice quality ({quality_weight:.3f}) below threshold "
                  f"({args.min_quality_threshold}), using pure interpolation")
    
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

