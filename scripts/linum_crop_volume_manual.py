#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crop volume manually between (xmin, xmax), (ymin, ymin) and (zmin, zmax),
corresponding to the first, second and third axes in the dataset, respectively.
"""
import argparse
from linumpy.io import save_omezarr, read_omezarr

import dask.array as da


def _build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument('in_volume',
                   help='Input volume in .ome.zarr format.')
    p.add_argument('out_volume',
                   help='Output volume in .ome.zarr format.')
    p.add_argument('--xmin', default=0, type=int,
                   help='Minimum index for first axis [%(default)s].')
    p.add_argument('--xmax', default=-1, type=int,
                   help='Maximum index for first axis [%(default)s].')
    p.add_argument('--ymin', default=0, type=int,
                   help='Minimum index for second axis [%(default)s].')
    p.add_argument('--ymax', default=-1, type=int,
                   help='Maximum index for second axis [%(default)s].')
    p.add_argument('--zmin', default=0, type=int,
                   help='Minimum index for third axis [%(default)s].')
    p.add_argument('--zmax', default=-1, type=int,
                   help='Maximum index for third axis [%(default)s].')
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    vol, res = read_omezarr(args.in_volume)
    darr = da.from_zarr(vol)
    darr = darr[args.xmin:args.xmax,
                args.ymin:args.ymax,
                args.zmin:args.zmax]
    save_omezarr(darr, args.out_volume, res, vol.chunks)


if __name__ == '__main__':
    main()
