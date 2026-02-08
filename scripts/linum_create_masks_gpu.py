#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create tissue masks from OME-Zarr images.

GPU-accelerated version using linumpy.gpu module for filtering and morphology.
Falls back to CPU if GPU is not available.
Masks are saved with matching pyramid levels to the input image.
"""
# Configure thread limits before numpy/scipy imports
import linumpy._thread_config  # noqa: F401

import argparse

import numpy as np
import dask.array as da
from scipy.ndimage import distance_transform_edt
from skimage.morphology import remove_small_objects, local_maxima
from skimage.measure import label
from skimage.segmentation import watershed
from skimage.filters import median
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader, Multiscales

from linumpy.io import read_omezarr, save_omezarr
from linumpy.gpu import GPU_AVAILABLE, print_gpu_info
from linumpy.gpu.array_ops import (
    threshold_otsu,
    compute_percentiles_memory_efficient,
    compute_nonzero_percentile_memory_efficient
)
from linumpy.gpu.morphology import (
    gaussian_filter, binary_opening, binary_closing, binary_fill_holes
)


def get_num_pyramid_levels(zarr_path):
    """Get the number of pyramid levels in an OME-Zarr file."""
    reader = Reader(parse_url(zarr_path))
    nodes = list(reader())
    image_node = nodes[0]
    
    for spec in image_node.specs:
        if isinstance(spec, Multiscales):
            return len(spec.datasets)
    return 1


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('image',
                   help='Input image in .ome.zarr format to mask.')
    p.add_argument('out_file',
                   help='Output path for the masks in .ome.zarr format.')
    p.add_argument('--sigma', type=float, default=5.0,
                   help='Standard deviation for Gaussian smoothing. [%(default)s]')
    p.add_argument('--selem_radius', type=int, default=1,
                   help='Radius of the structuring element for morphological operations. [%(default)s]')
    p.add_argument('--min_size', type=int, default=100,
                   help='Minimum size of objects to keep in the final mask. [%(default)s]')
    p.add_argument('--normalize', action='store_true',
                   help='Normalize the image before processing.')
    p.add_argument('--fill_holes', type=str, default='3d', choices=['none', '3d', 'slicewise'],
                   help='Hole filling method: none (no extra filling), 3d (standard scipy fill), '
                        'slicewise (fill in all 3 axes - best for masks with through-holes). [%(default)s]')
    p.add_argument('--use_gpu', default=True,
                   action=argparse.BooleanOptionalAction,
                   help='Use GPU acceleration if available. [%(default)s]')
    p.add_argument('--verbose', '-v', action='store_true',
                   help='Print GPU information')
    p.add_argument('--n_levels', type=int, default=None,
                   help='Number of pyramid levels. If not specified, matches input image pyramid. [%(default)s]')
    p.add_argument('--preview', type=str, default=None,
                   help='Path to save a preview image (PNG) for visual verification.')
    return p


def fill_holes_slicewise(mask, use_gpu=False):
    """
    Fill holes in a 3D mask by applying 2D hole filling in all three axes.

    Parameters
    ----------
    mask : np.ndarray
        3D binary mask
    use_gpu : bool
        Whether to use GPU acceleration

    Returns
    -------
    np.ndarray
        Mask with holes filled
    """
    if mask.ndim != 3:
        return binary_fill_holes(mask, use_gpu=use_gpu)

    # First do 3D fill
    mask = binary_fill_holes(mask, use_gpu=use_gpu)

    # Then fill slice-by-slice in all three axes
    # For GPU efficiency, we process on CPU for the slicewise operations
    # as the overhead of many small GPU transfers outweighs the benefit
    from scipy.ndimage import binary_fill_holes as scipy_fill_holes

    nz, ny, nx = mask.shape

    # Fill in XY planes (along Z axis)
    for z in range(nz):
        mask[z, :, :] = scipy_fill_holes(mask[z, :, :])

    # Fill in XZ planes (along Y axis)
    for y in range(ny):
        mask[:, y, :] = scipy_fill_holes(mask[:, y, :])

    # Fill in YZ planes (along X axis)
    for x in range(nx):
        mask[:, :, x] = scipy_fill_holes(mask[:, :, x])

    # Final 3D fill
    mask = binary_fill_holes(mask, use_gpu=use_gpu)

    return mask



def create_mask_gpu(image: np.ndarray, sigma: float = 5.0, selem_radius: int = 1,
                    min_size: int = 100, normalize: bool = True, use_gpu: bool = True) -> np.ndarray:
    """
    Create a mask using GPU-accelerated operations where possible.
    
    This function is optimized for memory efficiency by:
    - Using subsampling for percentile calculations
    - Explicitly freeing GPU memory after operations
    - Processing on CPU when GPU memory is insufficient

    Parameters
    ----------
    image : np.ndarray
        Input image
    sigma : float
        Gaussian smoothing sigma
    selem_radius : int
        Structuring element radius
    min_size : int
        Minimum object size
    normalize : bool
        Whether to normalize
    use_gpu : bool
        Whether to use GPU
        
    Returns
    -------
    np.ndarray
        Binary mask
    """
    # Import cupy here to check memory and handle cleanup
    if use_gpu and GPU_AVAILABLE:
        import cupy as cp
        # Clear any existing GPU memory before starting
        cp.get_default_memory_pool().free_all_blocks()

    if normalize:
        # Memory-efficient normalization using subsampling for percentile calculation
        image = image.copy().astype(np.float32)
        
        # Get percentiles using memory-efficient subsampling
        p_low = compute_nonzero_percentile_memory_efficient(image, 0.5, use_gpu=use_gpu)
        p_high = compute_percentiles_memory_efficient(image, [99.5], use_gpu=use_gpu)[0]

        # Normalize in-place to save memory
        image -= p_low
        if (p_high - p_low) > 1e-10:
            image /= (p_high - p_low)
        np.clip(image, 0, 1, out=image)

    # GPU-accelerated Gaussian smoothing
    image = gaussian_filter(image, sigma=sigma, use_gpu=use_gpu)
    
    # Free GPU memory after gaussian filter
    if use_gpu and GPU_AVAILABLE:
        cp.get_default_memory_pool().free_all_blocks()

    # Median filter (CPU - skimage version handles edge cases better)
    image = median(image)

    # GPU-accelerated threshold
    threshold = threshold_otsu(image, use_gpu=use_gpu)
    
    # Create binary mask - do this on CPU to save GPU memory
    mask = image > threshold
    del image  # Free the image memory as we only need the mask now

    if use_gpu and GPU_AVAILABLE:
        cp.get_default_memory_pool().free_all_blocks()

    # GPU-accelerated morphology
    mask = binary_opening(mask, iterations=1, use_gpu=use_gpu)
    if use_gpu and GPU_AVAILABLE:
        cp.get_default_memory_pool().free_all_blocks()

    mask = binary_closing(mask, iterations=1, use_gpu=use_gpu)
    if use_gpu and GPU_AVAILABLE:
        cp.get_default_memory_pool().free_all_blocks()

    mask = binary_fill_holes(mask, use_gpu=use_gpu)
    if use_gpu and GPU_AVAILABLE:
        cp.get_default_memory_pool().free_all_blocks()

    # Remove small objects (CPU - complex connected component analysis)
    mask = remove_small_objects(mask, min_size=min_size)

    # Distance transform and watershed (CPU - complex algorithms)
    dist = distance_transform_edt(mask)
    peaks = local_maxima(dist)
    markers = label(peaks)
    labels = watershed(-dist, markers=markers, mask=mask)
    del dist, peaks, markers  # Free intermediate arrays

    mask = labels > 0
    del labels

    mask = binary_fill_holes(mask, use_gpu=use_gpu)
    if use_gpu and GPU_AVAILABLE:
        cp.get_default_memory_pool().free_all_blocks()

    mask = remove_small_objects(mask, min_size=min_size)

    return mask


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    
    use_gpu = args.use_gpu and GPU_AVAILABLE
    
    if args.verbose:
        print_gpu_info()
        print(f"Using GPU: {use_gpu}")

    # Load image
    vol, res = read_omezarr(args.image, level=0)
    vol = vol[:]

    # Create mask with GPU acceleration
    mask = create_mask_gpu(
        vol, 
        sigma=args.sigma, 
        selem_radius=args.selem_radius, 
        min_size=args.min_size,
        normalize=args.normalize,
        use_gpu=use_gpu
    )

    # Apply additional hole filling if requested
    if args.fill_holes == 'slicewise':
        print("Applying slicewise hole filling...")
        mask = fill_holes_slicewise(mask, use_gpu=use_gpu)
    elif args.fill_holes == '3d':
        mask = binary_fill_holes(mask, use_gpu=use_gpu)
    # 'none' - no additional filling

    # Determine number of pyramid levels
    if args.n_levels is not None:
        n_levels = args.n_levels
    else:
        # Match the input image pyramid levels
        n_levels = get_num_pyramid_levels(args.image) - 1  # -1 because n_levels is additional levels beyond base
    
    # Save mask with pyramid levels
    save_omezarr(da.from_array(mask), args.out_file, res, n_levels=n_levels)

    # Generate preview if requested
    if args.preview:
        from scripts.linum_create_masks import generate_mask_preview
        generate_mask_preview(vol, mask, args.preview)
        print(f"Preview saved: {args.preview}")

    # Collect metrics using helper function
    from linumpy.utils.metrics import collect_mask_metrics
    collect_mask_metrics(
        mask=mask,
        input_vol=vol,
        output_path=args.out_file,
        input_path=args.image,
        params={'sigma': args.sigma, 'selem_radius': args.selem_radius, 
                'min_size': args.min_size, 'normalize': args.normalize,
                'use_gpu': use_gpu, 'fill_holes': args.fill_holes}
    )


if __name__ == '__main__':
    main()
