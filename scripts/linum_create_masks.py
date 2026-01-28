#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create binary masks from input images for use in registration.
Masks are saved with matching pyramid levels to the input image.
Optionally generates preview images for visual verification.
"""
# Configure thread limits before numpy/scipy imports
import linumpy._thread_config  # noqa: F401

import argparse

import dask.array as da
import numpy as np
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader, Multiscales

from linumpy.io import read_omezarr, save_omezarr
from linumpy.segmentation import create_mask
from linumpy.utils.metrics import collect_mask_metrics


def get_num_pyramid_levels(zarr_path):
    """Get the number of pyramid levels in an OME-Zarr file."""
    reader = Reader(parse_url(zarr_path))
    nodes = list(reader())
    image_node = nodes[0]
    
    for spec in image_node.specs:
        if isinstance(spec, Multiscales):
            return len(spec.datasets)
    return 1


def generate_mask_preview(image, mask, output_path, slice_idx=None):
    """
    Generate a preview image showing the mask overlaid on the original image.

    Parameters
    ----------
    image : np.ndarray
        Original 2D or 3D image
    mask : np.ndarray
        Binary mask (same shape as image)
    output_path : str
        Path to save the preview PNG
    slice_idx : int, optional
        For 3D images, which slice to use. If None, uses middle slice.
    """
    import matplotlib.pyplot as plt

    # Handle 3D images
    if image.ndim == 3:
        if slice_idx is None:
            slice_idx = image.shape[0] // 2
        img_2d = image[slice_idx]
        mask_2d = mask[slice_idx]
    else:
        img_2d = image
        mask_2d = mask

    # Normalize image for display
    img_norm = img_2d.astype(np.float32)
    p1, p99 = np.percentile(img_norm[img_norm > 0], [1, 99]) if np.any(img_norm > 0) else (0, 1)
    if p99 > p1:
        img_norm = np.clip((img_norm - p1) / (p99 - p1), 0, 1)
    else:
        img_norm = np.zeros_like(img_norm)

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(img_norm, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Mask only
    axes[1].imshow(mask_2d, cmap='gray')
    axes[1].set_title('Generated Mask')
    axes[1].axis('off')

    # Overlay (mask edges on image)
    from scipy import ndimage
    edges = ndimage.binary_dilation(mask_2d) ^ mask_2d

    overlay = np.stack([img_norm, img_norm, img_norm], axis=-1)
    overlay[edges, 0] = 1.0  # Red edges
    overlay[edges, 1] = 0.0
    overlay[edges, 2] = 0.0

    axes[2].imshow(overlay)
    axes[2].set_title('Mask Overlay')
    axes[2].axis('off')

    # Add statistics
    mask_coverage = np.sum(mask_2d) / mask_2d.size * 100
    fig.suptitle(f'Mask Preview (coverage: {mask_coverage:.1f}%)', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


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
    p.add_argument('--n_levels', type=int, default=None,
                   help='Number of pyramid levels. If not specified, matches input image pyramid. [%(default)s]')
    p.add_argument('--preview', type=str, default=None,
                   help='Path to save a preview image (PNG) for visual verification.')
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Load image
    vol, res = read_omezarr(args.image, level=0)
    vol = vol[:]

    # Create mask
    mask = create_mask(vol, sigma=args.sigma, selem_radius=args.selem_radius, min_size=args.min_size,
                       normalize=args.normalize)

    # Determine number of pyramid levels
    if args.n_levels is not None:
        n_levels = args.n_levels
    else:
        n_levels = get_num_pyramid_levels(args.image) - 1
    
    # Save mask with pyramid levels
    save_omezarr(da.from_array(mask), args.out_file, res, n_levels=n_levels)

    # Generate preview if requested
    if args.preview:
        generate_mask_preview(vol, mask, args.preview)
        print(f"Preview saved: {args.preview}")

    # Collect metrics using helper function
    collect_mask_metrics(
        mask=mask,
        input_vol=vol,
        output_path=args.out_file,
        input_path=args.image,
        params={'sigma': args.sigma, 'selem_radius': args.selem_radius, 
                'min_size': args.min_size, 'normalize': args.normalize}
    )


if __name__ == '__main__':
    main()
