#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Convert 3D OCT tiles to a 3D mosaic grid"""

import argparse
import multiprocessing
import shutil
from pathlib import Path

import numpy as np
import zarr
from skimage.transform import resize

from linumpy import reconstruction
from linumpy.microscope.oct import OCT
from linumpy.io.thorlabs import ThorOCT
from tqdm.auto import tqdm


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("tiles_directory",
                   help="Full path to a directory containing the tiles to process")
    p.add_argument("output_zarr",
                   help="Full path to the output zarr file")
    p.add_argument("-r", "--resolution", type=float, default=10.0,
                   help="Output isotropic resolution in micron per pixel. (default=%(default)s)")
    p.add_argument("-z", "--slice", type=int, default=0,
                   help="Slice to process (default=%(default)s)")
    p.add_argument("--keep_galvo_return", action="store_true",
                   help="Keep the galvo return signal (default=%(default)s)")
    p.add_argument("--data_type", type = str, default='OCT',choices=['OCT', 'PSOCT'],
                   help="Type of the data to process (default=%(default)s)")
    p.add_argument('--polarization', type = int, default = 1, choices = [1,2],
                   help="Polarization index to process (In case of PSOCT data type)")
    p.add_argument('--angle_index', type = int, default = 0,
                   help="Angle index to process (In case of PSOCT data type)")
    p.add_argument('--return_complex', type = bool, default = False,
                   help="Return Complex64 or Float32 data type (In case of PSOCT data type)")
    return p


def preprocess_volume(vol: np.ndarray) -> np.ndarray:
    """Preprocess the volume by rotating and flipping it."""
    vol = np.rot90(vol, k=3, axes=(1, 2))
    vol = np.flip(vol, axis=1)
    return vol


def process_tile(params: dict):
    """Process a tile and add it to the mosaic"""
    f = params["file"]
    mx, my, mz = params["tile_pos"]
    crop = params["crop"]
    tile_size = params["tile_size"]
    mosaic = params["mosaic"]
    data_type = params["data_type"]
    pol_index = params["pol_index"]
    return_complex = params["return_complex"]
    # Load the tile
    if data_type == 'OCT':
        oct = OCT(f)
        vol = oct.load_image(crop=crop)
        vol = preprocess_volume(vol)
    elif data_type == 'PSOCT':
        oct = ThorOCT(f)
        if pol_index == 0:
            oct.load(erase_polarization_2=True, return_complex = return_complex)
            vol = oct.polarization1
        else:
            oct.load(erase_polarization_1=True, return_complex = return_complex)
            vol = oct.polarization2
        vol = ThorOCT.preprocess_volume_PSOCT(vol)
    # Rescale the volume
    vol = resize(vol, tile_size, anti_aliasing=True, order=1, preserve_range=True)
    # Compute the tile position
    rmin = mx * vol.shape[1]
    cmin = my * vol.shape[2]
    rmax = rmin + vol.shape[1]
    cmax = cmin + vol.shape[2]
    mosaic[:, rmin:rmax, cmin:cmax] = vol

def main():
    # Parse arguments
    p = _build_arg_parser()
    args = p.parse_args()

    # Parameters
    tiles_directory = Path(args.tiles_directory)
    zarr_file = Path(args.output_zarr)
    z = args.slice
    output_resolution = args.resolution
    crop = not args.keep_galvo_return
    n_cpus = multiprocessing.cpu_count() - 1
    data_type = args.data_type
    pol_index = args.polarization
    angle_index = args.angle_index
    return_complex = args.return_complex

    # Analyze the tiles
    if data_type == 'OCT':
        tiles, tiles_pos = reconstruction.get_tiles_ids(tiles_directory, z=z)
    elif data_type == 'PSOCT':
        tiles, tiles_pos = ThorOCT.get_PSOCT_tiles_ids(tiles_directory)
        tiles = tiles[angle_index]
    mx = [tiles_pos[i][0] for i in range(len(tiles_pos))]
    my = [tiles_pos[i][1] for i in range(len(tiles_pos))]
    mx_min = min(mx)
    mx_max = max(mx)
    my_min = min(my)
    my_max = max(my)
    n_mx = mx_max - mx_min + 1
    n_my = my_max - my_min + 1

    # Prepare the mosaic_grid
    if data_type == 'OCT':
        oct = OCT(tiles[0])
        vol = oct.load_image(crop=crop)
        vol = preprocess_volume(vol)
        resolution = [oct.resolution[2], oct.resolution[0], oct.resolution[1]]
    elif data_type == 'PSOCT':
        oct = ThorOCT(tiles[0])
        if pol_index == 0:
            oct.load(erase_polarization_2=True, return_complex = return_complex)
            vol = oct.polarization1
        else:
            oct.load(erase_polarization_1=True, return_complex = return_complex)
            vol = oct.polarization2
        vol = ThorOCT.preprocess_volume_PSOCT(vol)
        resolution = [oct.resolution[2], oct.resolution[0], oct.resolution[1]]
        print(f"Resolutoin: z = {resolution[0]} , x = {resolution[1]} , y = {resolution[2]} ") 
    

    # Compute the rescaled tile size based on the minimum target output resolution
    if output_resolution == -1:
        tile_size = vol.shape
    else:
        tile_size = [int(vol.shape[i] * resolution[i] * 1000 / output_resolution) for i in range(3)]     
    mosaic_shape = [tile_size[0], n_mx * tile_size[1], n_my * tile_size[2]]
    print("Tile size:", tile_size)
    
    # Create the zarr persistent array
    process_sync_file = str(zarr_file).replace(".zarr", ".sync")
    synchronizer = zarr.ProcessSynchronizer(process_sync_file)
    if return_complex == True:
        mosaic = zarr.open(zarr_file, mode="w", shape=mosaic_shape, dtype=np.complex64, chunks=tile_size,
                       synchronizer=synchronizer)
    else:
        mosaic = zarr.open(zarr_file, mode="w", shape=mosaic_shape, dtype=np.float32, chunks=tile_size,
                       synchronizer=synchronizer)

    # Create a params dictionary for every tile
    params = []
    for i in range(len(tiles)):
        params.append({
            "file": tiles[i],
            "tile_pos": tiles_pos[i],
            "crop": crop,
            "tile_size": tile_size,
            "mosaic": mosaic,
            "data_type": data_type,
            "pol_index": pol_index,
            "return_complex": return_complex
        })

    # Process the tiles in parallel
    with multiprocessing.Pool(n_cpus) as pool:
        results = tqdm(pool.imap(process_tile, params), total=len(params))
        tuple(results)

    # Remove the process sync file
    shutil.rmtree(process_sync_file)


if __name__ == "__main__":
    main()
