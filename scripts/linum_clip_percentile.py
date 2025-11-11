#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clip .ome.zarr volume intensities between lower and upper percentile.
"""
import argparse
import numpy as np

from linumpy.io.zarr import read_omezarr, save_omezarr
import dask.array as da


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_volume',
                   help='Input volume .ome.zarr.')
    p.add_argument('out_volume',
                   help='Output volume .ome.zarr.')
    p.add_argument('--percentile_lower', default=0, type=float,
                   help='Percentile below which values will be clipped [%(default)s].')
    p.add_argument('--percentile_upper', default=99.9, type=float,
                   help='Percentile above which values will be clipped [%(default)s].')
    p.add_argument('--rescale', action='store_true',
                   help='Rescale volume intensities after clipping.')
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    vol, res = read_omezarr(args.in_volume)
    darr = da.from_zarr(vol)
    p_lower = np.percentile(vol[:], args.percentile_lower)
    p_upper = np.percentile(vol[:], args.percentile_upper)
    darr = da.clip(darr, p_lower, p_upper)

    if args.rescale:
        darr = darr - darr.flatten().min()
        darr = darr / darr.flatten().max()

    save_omezarr(darr, args.out_volume, res, vol.chunks)


if __name__ == '__main__':
    main()
