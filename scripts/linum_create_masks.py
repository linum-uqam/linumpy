#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estimate the transform aligning `in_moving` volume over `in_fixed` volume.
Finds the best shift along the stacking direction and the best 2D transform.
The output is a directory containing a transform (.mat) and a z offset (.txt)
for aligning the moving slice over the fixed slice.
"""
import argparse

import dask.array as da

from linumpy.io import read_omezarr, save_omezarr
from linumpy.segmentation import create_mask


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
    p.add_argument()
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Load image
    vol, res = read_omezarr(args.in_image, level=0)
    vol = vol[:]

    # Create mask
    mask = create_mask(vol, sigma=args.sigma, selem_radius=args.selem_radius, min_size=args.min_size,
                       normalize=args.normalize)

    # Save mask
    save_omezarr(da.from_array(mask), args.out_file, res, chunks=vol.chunks)


if __name__ == '__main__':
    main()
