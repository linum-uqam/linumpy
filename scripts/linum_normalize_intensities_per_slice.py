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
from linumpy.preproc.normalization import normalize_volume, get_agarose_mask
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
    p.add_argument('--min_contrast_fraction', type=float, default=0.1,
                   help='Minimum contrast as fraction of global max to prevent\n'
                        'over-amplification of weak/bad slices. [%(default)s]')
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    vol, res = read_omezarr(args.in_image, level=0)
    vol_data = vol[:]

    # Get agarose mask
    agarose_mask, otsu_threshold = get_agarose_mask(vol_data, args.sigma)

    # Normalize using shared function
    vol_normalized, background_thresholds = normalize_volume(
        vol_data, agarose_mask, args.percentile_max, args.min_contrast_fraction
    )

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
        params={
            'percentile_max': args.percentile_max,
            'sigma': args.sigma,
            'min_contrast_fraction': args.min_contrast_fraction
        }
    )


if __name__ == '__main__':
    main()
