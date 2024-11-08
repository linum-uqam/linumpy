#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Convert 3D OCT tiles to zarr"""
import argparse
from pathlib import Path

import dask.array as da
import numpy as np
import zarr
from tqdm.auto import tqdm
import shutil

from skimage.transform import resize
from linumpy import reconstruction
from linumpy.io.zarr import save_zarr
from linumpy.microscope.oct import OCT

# Tasks
# TODO: flag to choose between zarr for reconstruction (small chunks), or for visualization (smaller chunks)
# TODO: use dask and data loader to reduce IO and reduce memory usage.
# TODO: add option to keep all the scan or crop the galvo return (this assumes that the galvo return is in the right position)
# TODO: add option to choose the smallest resolution (downsample) instead of using the maximum resolution
# TODO: add option to crop


def _build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument('input_tiles',
                   help='Input tiles directory.')
    p.add_argument('output_zarr',
                   help='Output OME-zarr file.')
    p.add_argument('depth', type=int,
                   help='Depth (z) of tiles of interest.')
    p.add_argument('--res', type=float,
                   help='Output resolution in microns. Default is the raw data resolution.')
    p.add_argument('--n_levels', type=int, default=5,
                   help='Number of levels in pyramid. [%(default)s]')
    return p


def preprocess_volume(oct) -> np.ndarray:
    vol = oct.load_image(crop=False)
    vol = np.rot90(vol, k=3, axes=(1, 2))
    vol = np.flip(vol, axis=1)
    return vol


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Prepare the zarr information
    chunk_size = 128
    chunks = tuple([chunk_size] * 3)

    # Analyze the tiles
    tiles, tiles_pos = reconstruction.get_tiles_ids(args.input_tiles, z=args.depth)
    mx = [tiles_pos[i][0] for i in range(len(tiles_pos))]
    my = [tiles_pos[i][1] for i in range(len(tiles_pos))]
    mz = [tiles_pos[i][2] for i in range(len(tiles_pos))]
    mx_min = min(mx)
    mx_max = max(mx)
    my_min = min(my)
    my_max = max(my)
    mz_min = min(mz)
    mz_max = max(mz)
    n_mx = mx_max - mx_min + 1
    n_my = my_max - my_min + 1
    n_mz = mz_max - mz_min + 1

    # Prepare the mosaic_grid and chunk size
    oct = OCT(tiles[0])
    vol = preprocess_volume(oct)
    resolution = [oct.resolution[2], oct.resolution[0], oct.resolution[1]]

    # Compute the rescaled tile size based on the minimum target output resolution
    if args.res is not None:
        target_resolution = [args.res] * 3
        tile_size = [int(vol.shape[i] * resolution[i] * 1000 / target_resolution[i]) for i in range(3)]
    else:
        tile_size = vol.shape
    mosaic_shape = [tile_size[0], n_mx * tile_size[1], n_my * tile_size[2]]

    # Assemble an initial zarr
    store = zarr.TempStore()
    root = zarr.group(store=store, overwrite=True)

    # Add a group
    mosaic = root.zeros(f"slice_{args.depth:02d}", shape=mosaic_shape, chunks=tile_size, dtype=np.float32)

    for i in tqdm(range(len(tiles)), desc="Reading tiles"):
        f = tiles[i]
        mx, my, mz = tiles_pos[i]
        oct = OCT(f)
        vol = preprocess_volume(oct)

        # Rescale the volume
        if args.res is not None:
            vol = resize(vol, tile_size, anti_aliasing=True, order=1, preserve_range=True)

        rmin = (mx - mx_min) * vol.shape[1]
        cmin = (my -my_min) * vol.shape[2]
        rmax = rmin + vol.shape[1]
        cmax = cmin + vol.shape[2]
        mosaic[:, rmin:rmax, cmin:cmax] = vol

    # Convert to ome-zarr
    mosaic_dask = da.from_zarr(mosaic)
    save_zarr(mosaic_dask, args.output_zarr, scales=resolution,
              chunks=chunks, n_levels=args.n_levels)


if __name__ == '__main__':
    main()
