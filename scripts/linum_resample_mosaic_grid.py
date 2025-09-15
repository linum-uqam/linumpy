#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import numpy as np
import zarr
import dask.array as da
import itertools

from skimage.transform import rescale
from linumpy.io import read_omezarr, save_omezarr, create_tempstore


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
    target_res = args.resolution / 1000.0  # conversion um to mm

    tile_shape = vol.chunks
    scaling_factor = np.asarray(source_res) / target_res
    tile_00 = vol[:tile_shape[0], :tile_shape[1], :tile_shape[2]]

    # process first tile to get output shape
    out_tile00 = rescale(tile_00, scaling_factor, order=1,
                         preserve_range=True, anti_aliasing=True)
    out_tile_shape = out_tile00.shape

    nx = vol.shape[1] // tile_shape[1]
    ny = vol.shape[2] // tile_shape[2]

    out_shape = (out_tile_shape[0], nx*out_tile_shape[1], ny*out_tile_shape[2])
    out_zarr = zarr.open(create_tempstore(), mode='w', shape=out_shape,
                         chunks=out_tile_shape, dtype=vol.dtype)
    for i, j in itertools.product(range(nx), range(ny)):
        out_zarr[:, i*out_tile_shape[1]:(i + 1)*out_tile_shape[1],
                 j*out_tile_shape[2]:(j + 1)*out_tile_shape[2]] =\
            rescale(vol[:, i*tile_shape[1]:(i + 1)*tile_shape[1], j*tile_shape[2]:(j + 1)*tile_shape[2]],
                    scaling_factor, order=1, preserve_range=True, anti_aliasing=True)

    darr = da.from_zarr(out_zarr)
    save_omezarr(darr, args.out_mosaic, [target_res]*3,
                 chunks=out_tile_shape, n_levels=args.n_levels)


if __name__ == '__main__':
    main()
