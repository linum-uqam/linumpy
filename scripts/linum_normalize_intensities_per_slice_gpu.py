#!/usr/bin/env python3
# Configure thread limits before numpy/scipy imports
import linumpy._thread_config  # noqa: F401

#-*- coding:utf-8 -*-

"""
GPU-accelerated intensity normalization of ome.zarr volume along z axis.

Uses linumpy.gpu module for:
- Gaussian filtering (7-20x speedup) on 2D mean projection
- Otsu thresholding (4-10x speedup) for mask creation

Falls back to CPU if GPU is not available.
"""

import argparse
from linumpy.io.zarr import read_omezarr, save_omezarr
from linumpy.gpu import GPU_AVAILABLE, print_gpu_info
from linumpy.gpu.morphology import gaussian_filter
from linumpy.gpu.array_ops import threshold_otsu
from linumpy.preproc.normalization import normalize_volume
from linumpy.utils.metrics import collect_normalization_metrics

import dask.array as da
import numpy as np


def _build_arg_parser():
    p = argparse.ArgumentParser(description='__doc__',
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_image',
                   help='Input image.')
    p.add_argument('out_image',
                   help='Output image.')
    p.add_argument('--percentile_max', type=float, default=99.9,
                   help='Values above the ith percentile will be clipped. [%(default)s]')
    p.add_argument('--sigma', type=float, default=1.0,
                   help='Smoothing sigma for estimating the agarose mask. [%(default)s]')
    p.add_argument("--use_gpu", default=True,
                   action=argparse.BooleanOptionalAction,
                   help="Use GPU acceleration if available")
    p.add_argument("--verbose", action="store_true",
                   help="Print GPU information")
    return p


def get_agarose_mask(vol, smoothing_sigma, use_gpu=True):
    """Compute agarose mask using GPU-accelerated operations."""
    reference = np.mean(vol, axis=0)
    reference_smooth = gaussian_filter(reference, sigma=smoothing_sigma, use_gpu=use_gpu)
    threshold = threshold_otsu(reference_smooth[reference > 0], use_gpu=use_gpu)
    agarose_mask = np.logical_and(reference_smooth < threshold, reference > 0)
    return agarose_mask, float(threshold)


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    use_gpu = args.use_gpu and GPU_AVAILABLE

    if args.verbose:
        print_gpu_info()
        print(f"Using GPU: {use_gpu}")
        if args.use_gpu and not GPU_AVAILABLE:
            print("GPU requested but not available, falling back to CPU")

    vol, res = read_omezarr(args.in_image, level=0)
    vol_data = vol[:]

    # Get agarose mask (GPU-accelerated)
    agarose_mask, otsu_threshold = get_agarose_mask(vol_data, args.sigma, use_gpu=use_gpu)

    # Normalize using shared function
    vol_normalized, background_thresholds = normalize_volume(vol_data, agarose_mask, args.percentile_max)

    # Save
    save_omezarr(da.from_array(vol_normalized), args.out_image, res, n_levels=3)

    # Collect metrics using helper function
    collect_normalization_metrics(
        vol_normalized=vol_normalized,
        agarose_mask=agarose_mask,
        otsu_threshold=otsu_threshold,
        background_thresholds=background_thresholds,
        output_path=args.out_image,
        input_path=args.in_image,
        params={'percentile_max': args.percentile_max, 'sigma': args.sigma, 'use_gpu': use_gpu}
    )


if __name__ == '__main__':
    main()
