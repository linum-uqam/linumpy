#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert folder of tiff files to omezarr. Expected file structure is:
    in_folder/
    ├── z0.tif
    ├── z1.tif
    └── ...

If there are more than one channel, file structure should be
    in_folder/
    ├── channel_00/
    │   ├── z0.tif
    │   ├── z1.tif
    │   └── ...
    ├── channel_01/
    │   └── ...
    └── ...
"""
import argparse
from glob import glob
import logging
import os

import numpy as np
import dask.array as da
import zarr
from tifffile import imread
from skimage.transform import resize

from linumpy.io.zarr import save_omezarr, create_tempstore
from linumpy.utils.io import add_overwrite_arg, add_verbose_arg


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_folder',
                   help="Folder with tiff files."
                        "If you have multiple channels, images have to "
                        "be split into different subfolders within in_folder.")
    p.add_argument('in_dimensions', nargs=3, type=float,
                   help='Dimensions of the input data (X,Y,Z).')
    p.add_argument("--resolution", type=float, default=None,
                   help="Output isotropic resolution "
                        "in micron per pixel. (default=%(default)s)")
    p.add_argument('--chunks', nargs=3, type=int,
                   help="Chunks of the output zarr file.")
    p.add_argument('--n_levels', type=int, default=5,
                   help="Number of levels in the pyramid."
                        " (default=%(default)s)")
    p.add_argument('out_zarr',
                   help='Output zarr file.')
    p.add_argument('--zarr_root', default='/tmp/',
                   help='Path to parent directory under which the zarr'
                        ' temporary directory will be created [/tmp/].')
    add_overwrite_arg(p)
    add_verbose_arg(p)
    return p


def check_folders(parser, folder):
    """
    Check if the folder contains tiff files or subfolders with tiff files.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Argument parser.
    folder : str
        Folder to check.

    Returns
    -------
    tiff_files : list of lists
        List of lists tiff files.
    """
    tiff_files = []
    # check if there are tiff files in the folder
    if glob(os.path.join(folder, '*.tif')) == []:
        # list subfolders
        subfolders = [f.path for f in os.scandir(folder) if f.is_dir()]
        if subfolders == []:
            parser.error("No tiff files or subfolder found in the folder.")
        else:
            logging.info("Found subfolders in the folder.")
            for index, subfolder in enumerate(subfolders):
                if glob(os.path.join(subfolder, '*.tif')) == []:
                    parser.error("No tiff files found in the subfolder.")
                else:
                    tiff_files.append(sorted(glob(os.path.join(subfolder,
                                                               '*.tif'))))
    elif len([f.path for f in os.scandir(folder) if f.is_dir()]) != 0:
        parser.error("Both tiff files and subfolders found in the folder.")
    else:
        tiff_files = sorted(glob(os.path.join(folder, '*.tif')))
        logging.info("Found tiff files in the folder.")

    # check if all subfolders contain the same number of files
    it = iter(tiff_files)
    the_len = len(next(it))
    if not all(len(l) == the_len for l in it):
        parser.error('Not all subfolders contain the same number of files.')

    return tiff_files


def process_volume(mosaic, vol, index_z, tile_size=None):
    """
    Process a volume and add it to the mosaic.

    Parameters
    ----------
    mosaic : zarr.core.Array
        Mosaic grid.
    vol : list of str
        List of tiff files.
    index_z : int
        Index of the z slice.
    tile_size : tuple of int, optional
        Size of the tiles. The default is None.
    """
    for index_c, curr_vol in enumerate(vol):
        curr_vol = imread(curr_vol)
        if tile_size:
            curr_vol = resize(curr_vol,
                              tile_size,
                              anti_aliasing=True,
                              order=1,
                              preserve_range=True)

        mosaic[index_c, index_z, :, :] = curr_vol[0, 0, :, :]


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    tiff_files = check_folders(parser, args.in_folder)
    logging.info("Found {} channels and {} slices in z.".format(len(tiff_files),
                                                                len(tiff_files[0])))

    # Get first image to get the resolution
    volume = imread(tiff_files[0][0])
    volume = np.array(volume)

    logging.info("Initial shape: {} ".format(volume.shape[2:]))
    logging.info("Initial resolution: {} x {} x {} um (X, Y, Z)".format(args.in_dimensions[0],
                                                                        args.in_dimensions[1],
                                                                        args.in_dimensions[2]))

    if args.resolution:  # Resampling
        resolution = [args.in_dimensions[2]/1000,
                      args.resolution/1000,
                      args.resolution/1000]
        # Create a mosaic grid
        volume_shape = [int(volume.shape[2] * resolution[0] * 1000 / args.resolution),
                        int(volume.shape[3] * resolution[0] * 1000 / args.resolution)]
        mosaic_shape = [len(tiff_files),
                        len(tiff_files[0]),
                        volume_shape[0],
                        volume_shape[1]]
        logging.info("Output shape: {}".format(tuple(mosaic_shape[2:])))
        logging.info("Output resolution: {} x {} x {} um (X, Y, Z)".format(args.resolution,
                                                                           args.resolution,
                                                                           args.in_dimensions[2]))
    else:
        logging.info("No resampling.")
        resolution = [args.in_dimensions[2]/1000,
                      args.in_dimensions[0]/1000,
                      args.in_dimensions[1]/1000]
        # Create a mosaic grid
        mosaic_shape = [len(tiff_files),
                        len(tiff_files[0]),
                        volume.shape[2],
                        volume.shape[3]]

    zarr_store = create_tempstore(dir=args.zarr_root, suffix=".zarr")
    mosaic = zarr.open(zarr_store, mode="w", shape=mosaic_shape,
                       dtype=np.float32, chunks=[1, 1, 128, 128])

    for index_z in range(len(tiff_files[0])):
        process_volume(mosaic, [item[index_z] for item in tiff_files],
                       index_z, [1, 1] + mosaic_shape[2:])

    mosaic_dask = da.from_zarr(mosaic)
    save_omezarr(mosaic_dask, args.out_zarr,
              voxel_size=resolution,
              chunks=args.chunks,
              n_levels=args.n_levels)


if __name__ == '__main__':
    main()
