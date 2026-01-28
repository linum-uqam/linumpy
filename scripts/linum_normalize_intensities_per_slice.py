#!/usr/bin/env python3
# Configure thread limits before numpy/scipy imports
import linumpy._thread_config  # noqa: F401

#-*- coding:utf-8 -*-
"""
Normalize intensities of ome.zarr volume along z axis. Intensities for
each z are rescaled between the minimum value inside agarose and the value
defined by the `percentile_max` argument.
"""
import argparse
from linumpy.io.zarr import read_omezarr, save_omezarr
from linumpy.preproc.normalization import normalize_volume
from linumpy.utils.metrics import collect_normalization_metrics
from skimage.filters import threshold_otsu
from scipy.ndimage import gaussian_filter

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
    return p


def get_agarose_mask(vol, smoothing_sigma):
    """Compute agarose mask using Otsu thresholding."""
    reference = np.mean(vol, axis=0)
    reference_smooth = gaussian_filter(reference, sigma=smoothing_sigma)
    threshold = threshold_otsu(reference_smooth[reference > 0])
    agarose_mask = np.logical_and(reference_smooth < threshold, reference > 0)
    return agarose_mask, threshold


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    vol, res = read_omezarr(args.in_image, level=0)
    vol_data = vol[:]

    # Get agarose mask
    agarose_mask, otsu_threshold = get_agarose_mask(vol_data, args.sigma)

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
        params={'percentile_max': args.percentile_max, 'sigma': args.sigma}
    )


if __name__ == '__main__':
    main()
