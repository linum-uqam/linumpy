#!/usr/bin/env python3
#-*- coding:utf-8 -*-
"""
Normalize intensities of ome.zarr volume along z axis. Intensities for
each z are rescaled between the minimum value inside agarose and the value
defined by the `percentile_max` argument.
"""
import argparse
from linumpy.io.zarr import read_omezarr, save_omezarr
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
    reference = np.mean(vol, axis=0)
    reference_smooth = gaussian_filter(reference, sigma=smoothing_sigma)
    threshold = threshold_otsu(reference_smooth[reference > 0])

    # voxels in mask are expected to be agarose voxels
    agarose_mask = np.logical_and(reference_smooth < threshold, reference > 0)
    return agarose_mask


def normalize(vol, percentile_max, smoothing_sigma):
    # voxels in mask are expected to be agarose voxels
    agarose_mask = get_agarose_mask(vol, smoothing_sigma)

    pmax = np.percentile(vol, percentile_max, axis=(1, 2))
    vol = np.clip(vol, None, pmax[:, None, None])

    background_thresholds = []
    for curr_slice in vol:
        agarose = curr_slice[agarose_mask]
        bg_median = np.median(agarose)
        background_thresholds.append(bg_median)

    background_thresholds = np.array(background_thresholds)
    vol = np.clip(vol, background_thresholds[:, None, None], None)

    # rescale
    vol = vol - np.min(vol, axis=(1, 2), keepdims=True)
    vmax = np.max(vol, axis=(1, 2))
    vol[vmax > 0] = vol[vmax > 0] / vmax[:, None, None]
    return vol


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    vol, res = read_omezarr(args.in_image, level=0)
    vol = vol[:]

    vol = normalize(vol, args.percentile_max, args.sigma)

    save_omezarr(da.from_array(vol), args.out_image, res, n_levels=3)


if __name__ == '__main__':
    main()
