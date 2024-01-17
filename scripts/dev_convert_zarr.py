# Convert the 3D tiles for a slice into a 3D zarr mosaic grid

# Tasks
# TODO: use dask directly instead of using zarr, then dask, then om-zarr

import zarr
from linumpy import reconstruction
from pathlib import Path
import numpy as np
from linumpy.microscope.oct import OCT
from tqdm import tqdm
from linumpy.conversion import save_zarr
import dask.array as da
import shutil

# Parameters
tiles_directory = Path("G:/frans/2024-01-12-S9-Coronal/dev_tiles_z05")
zarr_directory = Path("G:/frans/2024-01-12-S9-Coronal/dev_tiles_z05.zarr")
omezarr_directory = Path("G:/frans/2024-01-12-S9-Coronal/dev_tiles_z05.ome-zarr")

# Options
# keep_galvo_return = False
# REscaling for the minimum size (ex: 3, 5, 10 or 25 micron reconstruction)
# Create a zarr.zip?
# Read the mosaic metadat and use for ome-zarr
chunk_size = tuple([128] * 3)
n_levels = 4

# Analyze the tiles
tiles, tiles_pos = reconstruction.get_tiles_ids(tiles_directory)

# Get a list of z slice to process
slices = np.unique([x[-1] for x in tiles_pos])

# Analyze the dataset
mosaic_info = reconstruction.get_mosaic_info(tiles_directory, z=slices[0], overlap_fraction=0.0)

# Prepare the mosaic_grid and chunck size
oct = OCT(tiles[0])
vol = oct.load_image()
mosaic_shape = [vol.shape[0], mosaic_info['mosaic_nrows'], mosaic_info['mosaic_ncols']]
store = zarr.NestedDirectoryStore(zarr_directory)
root = zarr.group(store=store, overwrite=True)

# Add a group
this_slice = root.create_group(f"slice_{slices[0]:02d}")
mosaic = this_slice.zeros('mosaic', shape=mosaic_shape, chunks=vol.shape)

for i in tqdm(range(len(mosaic_info['tiles']))):
    f = mosaic_info['tiles'][i]
    oct = OCT(f)
    vol = oct.load_image()
    rmin = mosaic_info['tiles_pos_px'][i][0]
    cmin = mosaic_info['tiles_pos_px'][i][1]
    rmax = rmin + oct.shape[0]
    cmax = cmin + oct.shape[1]
    mosaic[:, rmin:rmax, cmin:cmax] = np.log(vol)

# Convert from zarr to ome-zarr
mosaic_dask = da.from_array(mosaic, chunks=chunk_size)
save_zarr(mosaic_dask, omezarr_directory, scales=oct.resolution, chunks=chunk_size, n_levels=n_levels)

# Remove the original zarr
shutil.rmtree(zarr_directory)
