#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Convert 3D OCT tiles to a 3D mosaic grid"""

import argparse
import multiprocessing
from pathlib import Path

import numpy as np
import dask.array as da
import zarr
from skimage.transform import resize
from tqdm.auto import tqdm
from linumpy.io.zarr import save_omezarr
from linumpy import reconstruction
from linumpy.microscope.oct import OCT
from linumpy.io.thorlabs import ThorOCT, PreprocessingConfig


from linumpy.utils.io import parse_processes_arg, add_processes_arg


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("output_zarr",
                   help="Full path to the output zarr file")
    p.add_argument("--data_type", type = str, default='OCT',choices=['OCT', 'PSOCT'],
                   help="Type of the data to process (default=%(default)s)")
    input_g = p.add_argument_group("input")
    input_mutex_g = input_g.add_mutually_exclusive_group(required=True)
    input_mutex_g.add_argument("--from_root_directory",
                               help="Full path to a directory containing the tiles to process.")
    input_mutex_g.add_argument("--from_tiles_list", nargs='+',
                               help='List of tiles to assemble (argument --slice is ignored).')
    options_g = p.add_argument_group("other options")
    options_g.add_argument("-r", "--resolution", type=float, default=10.0,
                           help="Output isotropic resolution in micron per pixel. [%(default)s]")
    options_g.add_argument("-z", "--slice", type=int,
                           help="Slice to process.")
    options_g.add_argument("--keep_galvo_return", action="store_true",
                           help="Keep the galvo return signal [%(default)s]")
    options_g.add_argument('--n_levels', type=int, default=5,
                           help='Number of levels in pyramid representation.')
    options_g.add_argument('--zarr_root',
                           help='Path to parent directory under which the zarr'
                                ' temporary directory will be created [/tmp/].')
    add_processes_arg(options_g)
    psoct_options_g = p.add_argument_group("PS-OCT options")  
    psoct_options_g.add_argument('--polarization', type = int, default = 1, choices = [0,1],
                   help="Polarization index to process")
    psoct_options_g.add_argument('--number_of_angles', type = int, default = 1,
                   help="Angle index to process")
    psoct_options_g.add_argument('--angle_index', type = int, default = 0,
                   help="Angle index to process")
    psoct_options_g.add_argument('--return_complex', type = bool, default = False,
                   help="Return Complex64 or Float32 data type")
    psoct_options_g.add_argument('--crop_first_index', type=int, default=320,
                   help="First index for cropping on the z axis (default=%(default)s)")
    psoct_options_g.add_argument('--crop_second_index', type=int, default=750,
                   help="Second index for cropping on the z axis (default=%(default)s)")
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
    psoct_config = params["psoct_config"]
    mx_min, my_min = params["mxy_min"]

    # Load the tile
    if data_type == 'OCT':
        oct = OCT(f)
        vol = oct.load_image(crop=crop)
        vol = preprocess_volume(vol)
    elif data_type == 'PSOCT':
        oct = ThorOCT(f, config=psoct_config)
        if psoct_config.erase_polarization_2:
            oct.load()
            vol = oct.first_polarization
        else:
            oct.load()
            vol = oct.second_polarization
        vol = ThorOCT.orient_volume_psoct(vol)
    # Rescale the volume
    if np.iscomplexobj(vol):
        vol = (
            resize(vol.real, tile_size, anti_aliasing=True, order=1, preserve_range=True) +
            1j * resize(vol.imag, tile_size, anti_aliasing=True, order=1, preserve_range=True)
            )
    else:
        vol = resize(vol, tile_size, anti_aliasing=True, order=1, preserve_range=True)
    # Compute the tile position
    rmin = (mx - mx_min) * vol.shape[1]
    cmin = (my - my_min) * vol.shape[2]
    rmax = rmin + vol.shape[1]
    cmax = cmin + vol.shape[2]
    mosaic[0:tile_size[0], rmin:rmax, cmin:cmax] = vol

def main():
    # Parse arguments
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Parameters
    output_resolution = args.resolution
    crop = not args.keep_galvo_return

    data_type = args.data_type
    angle_index = args.angle_index
    n_cpus = parse_processes_arg(args.n_processes)
    psoct_config = PreprocessingConfig()
    psoct_config.crop_first_index = args.crop_first_index
    psoct_config.crop_second_index = args.crop_second_index
    psoct_config.erase_polarization_1 = not args.polarization == 1
    psoct_config.erase_polarization_2 = not psoct_config.erase_polarization_1
    psoct_config.return_complex = args.return_complex

    # Analyze the tiles
    if data_type == 'OCT':
        if args.from_root_directory:
            z = args.slice
            tiles_directory = args.from_root_directory
            tiles, tiles_pos = reconstruction.get_tiles_ids(tiles_directory, z=z)
        else:
            if args.slice is not None:
                parser.error('Argument --slice is incompatible with --from_tiles_list.')
            tiles = [Path(d) for d in args.from_tiles_list]
            tiles_pos = reconstruction.get_tiles_ids_from_list(tiles)
    elif data_type == 'PSOCT':
        tiles, tiles_pos = ThorOCT.get_psoct_tiles_ids(
            tiles_directory,
            number_of_angles=args.number_of_angles
            )
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
        oct = ThorOCT(tiles[0], config=psoct_config)
        if psoct_config.erase_polarization_2:
            oct.load()
            vol = oct.first_polarization
        else:
            oct.load()
            vol = oct.second_polarization
        vol = ThorOCT.orient_volume_psoct(vol)
        resolution = [oct.resolution[2], oct.resolution[0], oct.resolution[1]]
        print(f"Resolutoin: z = {resolution[0]} , x = {resolution[1]} , y = {resolution[2]} ")   

    # Compute the rescaled tile size based on
    # the minimum target output resolution
    if output_resolution == -1:
        tile_size = vol.shape
        output_resolution = resolution
    else:
        tile_size = [int(vol.shape[i] * resolution[i] * 1000 / output_resolution) for i in range(3)]
        output_resolution = [output_resolution / 1000.0] * 3
    mosaic_shape = [tile_size[0], n_mx * tile_size[1], n_my * tile_size[2]]
    
    # Create the zarr persistent array
    zarr_store = zarr.TempStore(dir=args.zarr_root, suffix=".zarr")
    mosaic = zarr.open(zarr_store, mode="w", shape=mosaic_shape,
                       dtype=np.complex64 if args.return_complex else np.float32,
                       chunks=tile_size)

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
            "psoct_config": psoct_config,
            "mxy_min": (mx_min, my_min)
        })

    if n_cpus > 1:  # process in parallel
        with multiprocessing.Pool(n_cpus) as pool:
            results = tqdm(pool.imap(process_tile, params), total=len(params))
            tuple(results)
    else:  # Process the tiles sequentially
        for p in tqdm(params):
            process_tile(p)

    # Convert to ome-zarr
    mosaic_dask = da.from_zarr(mosaic)
    save_omezarr(mosaic_dask, args.output_zarr, voxel_size=output_resolution,
              chunks=tile_size, n_levels=args.n_levels)


if __name__ == "__main__":
    main()
