#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Convert 3D OCT tiles to zarr"""

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

# Parameters
tiles_directory = Path("/Users/jlefebvre/Downloads/tiles_z25_for_Joel")
tmp_zarr_directory = Path("/Users/jlefebvre/Downloads/tiles_z25_for_Joel.zarr")
output_zarr = Path("/Users/jlefebvre/Downloads/tiles_z25_for_Joel.ome_zarr")
z = 25
n_rot = 3
flip_lr = False
flip_ud = False
crop = False
n_levels = 5
output_resolution = 10.0 # micron per pixel (minimum resolution). If -1, use keep the raw data resolution


def preprocess_volume(oct) -> np.ndarray:
    vol = oct.load_image(crop=False)
    vol = np.rot90(vol, k=3, axes=(1, 2))
    vol = np.flip(vol, axis=1)
    return vol

# Prepare the zarr information
chunk_size = 128
chunks = tuple([chunk_size] * 3)

# Analyze the tiles
tiles, tiles_pos = reconstruction.get_tiles_ids(tiles_directory, z=z)
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
if output_resolution != -1:
    target_resolution = [output_resolution, output_resolution, output_resolution]
    tile_size = [int(vol.shape[i] * resolution[i] * 1000 / target_resolution[i]) for i in range(3)]
else:
    tile_size = vol.shape
mosaic_shape = [tile_size[0], n_mx * tile_size[1], n_my * tile_size[2]]


# Assemble an initial zarr
store = zarr.NestedDirectoryStore(tmp_zarr_directory)
root = zarr.group(store=store, overwrite=True)
ome_root = zarr.group(store=zarr.NestedDirectoryStore(tmp_zarr_directory), overwrite=True)

# Add a group
mosaic = root.zeros(f"slice_{z:02d}", shape=mosaic_shape, chunks=tile_size, dtype=np.float32)

for i in tqdm(range(len(tiles)), desc="Reading tiles"):
    f = tiles[i]
    mx, my, mz = tiles_pos[i]
    oct = OCT(f)
    vol = preprocess_volume(oct)

    # Rescale the volume
    if output_resolution != -1:
        vol = resize(vol, tile_size, anti_aliasing=True, order=1, preserve_range=True)

    rmin = mx * vol.shape[1]
    cmin = my * vol.shape[2]
    rmax = rmin + vol.shape[1]
    cmax = cmin + vol.shape[2]
    mosaic[:, rmin:rmax, cmin:cmax] = vol

# Convert to ome-zarr
mosaic_dask = da.from_zarr(mosaic)
save_zarr(mosaic_dask, output_zarr, scales=resolution, chunks=chunks, n_levels=n_levels)

# Remove the temporary zarr
shutil.rmtree(tmp_zarr_directory)
