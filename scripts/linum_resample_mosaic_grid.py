#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Configure thread limits before numpy/scipy imports
import linumpy._thread_config  # noqa: F401

import argparse

from linumpy.io import read_omezarr
from linumpy.preproc.resampling import resample_mosaic_grid


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_mosaic',
                   help='Input mosaic grid in .ome.zarr.')
    p.add_argument('out_mosaic',
                   help='Output resampled mosaic .ome.zarr.')
    p.add_argument('--resolution', '-r', type=float, default=10.0,
                   help='Isotropic resolution for resampling in microns.')
    p.add_argument('--n_levels', type=int, default=5,
                   help='Number of levels in pyramid decomposition [%(default)s].')
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    vol, source_res = read_omezarr(args.in_mosaic)
    resample_mosaic_grid(vol, source_res, args.resolution, n_levels=args.n_levels, out_path=args.out_mosaic)


if __name__ == '__main__':
    main()
