#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert 3D OCT tiles to a 3D mosaic grid.

GPU-accelerated version using linumpy.gpu module for:
- Volume resampling/resizing (5-12x speedup)

Note: Galvo shift detection uses CPU (no GPU benefit).
Falls back to CPU if GPU is not available.
"""

# Configure thread limits before numpy/scipy imports
import linumpy._thread_config  # noqa: F401

import argparse
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from linumpy import reconstruction
from linumpy.gpu import GPU_AVAILABLE, print_gpu_info
from linumpy.gpu.interpolation import resize
from linumpy.io.thorlabs import ThorOCT, PreprocessingConfig
from linumpy.io.zarr import OmeZarrWriter
from linumpy.microscope.oct import OCT
from linumpy.utils.io import parse_processes_arg, add_processes_arg

# Global flag for GPU usage (set in main, used in process_tile)
_USE_GPU = True


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("output_zarr",
                   help="Full path to the output zarr file")
    p.add_argument("--data_type", type=str, default='OCT', choices=['OCT', 'PSOCT'],
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
    options_g.add_argument("--axial_resolution", type=float, default=3.5,
                           help='Axial resolution of the raw data in microns. [%(default)s]')
    options_g.add_argument("-z", "--slice", type=int,
                           help="Slice to process.")
    options_g.add_argument("--keep_galvo_return", action="store_true",
                           help="Keep the galvo return signal [%(default)s]")
    options_g.add_argument('--n_levels', type=int, default=5,
                           help='Number of levels in pyramid representation.')
    options_g.add_argument('--zarr_root',
                           help='Path to parent directory under which the zarr'
                                ' temporary directory will be created [/tmp/].')
    options_g.add_argument('--fix_galvo_shift', default=True,
                           action=argparse.BooleanOptionalAction,
                           help='Fix the galvo shift. [%(default)s]')
    options_g.add_argument('--fix_camera_shift', default=False,
                           action=argparse.BooleanOptionalAction,
                           help='Fix the camera shift. [%(default)s]')
    options_g.add_argument('--preprocess', default=False,
                           action=argparse.BooleanOptionalAction,
                           help='Apply preprocessing (rotate/flip) for legacy data. [%(default)s]')
    options_g.add_argument('--galvo_threshold', type=float, default=0.6,
                           help='Galvo detection confidence threshold. [%(default)s]')
    options_g.add_argument('--sharding_factor', type=int, default=1,
                           help='A sharding factor of N will result '
                                'in N**2 tiles per shard. [%(default)s]')
    options_g.add_argument('--use_gpu', default=True,
                           action=argparse.BooleanOptionalAction,
                           help='Use GPU acceleration if available. [%(default)s]')
    options_g.add_argument('--verbose', '-v', action='store_true',
                           help='Print GPU information')
    add_processes_arg(options_g)
    psoct_options_g = p.add_argument_group("PS-OCT options")
    psoct_options_g.add_argument('--polarization', type=int, default=1, choices=[0, 1],
                                 help="Polarization index to process")
    psoct_options_g.add_argument('--number_of_angles', type=int, default=1,
                                 help="Angle index to process")
    psoct_options_g.add_argument('--angle_index', type=int, default=0,
                                 help="Angle index to process")
    psoct_options_g.add_argument('--return_complex', type=bool, default=False,
                                 help="Return Complex64 or Float32 data type")
    psoct_options_g.add_argument('--crop_first_index', type=int, default=320,
                                 help="First index for cropping on the z axis (default=%(default)s)")
    psoct_options_g.add_argument('--crop_second_index', type=int, default=750,
                                 help="Second index for cropping on the z axis (default=%(default)s)")
    return p


def preprocess_volume(vol: np.ndarray, apply: bool = True) -> np.ndarray:
    """Preprocess the volume by rotating and flipping it (for legacy data)."""
    if not apply:
        return vol
    vol = np.rot90(vol, k=3, axes=(1, 2))
    vol = np.flip(vol, axis=1)
    return vol


def load_single_tile(params: dict) -> tuple:
    """Load a single tile from disk. Used for parallel I/O.
    
    Returns
    -------
    tuple
        (params, volume) where volume is the loaded numpy array
    """
    f = params["file"]
    crop = params["crop"]
    galvo_shift = params["galvo_shift"]
    fix_camera_shift = params["fix_camera_shift"]
    preprocess = params["preprocess"]
    data_type = params["data_type"]
    psoct_config = params["psoct_config"]
    
    if data_type == 'OCT':
        oct = OCT(f)
        vol = oct.load_image(crop=crop, fix_galvo_shift=galvo_shift,
                             fix_camera_shift=fix_camera_shift)
        vol = preprocess_volume(vol, apply=preprocess)
    elif data_type == 'PSOCT':
        oct = ThorOCT(f, config=psoct_config)
        if psoct_config.erase_polarization_2:
            oct.load()
            vol = oct.first_polarization
        else:
            oct.load()
            vol = oct.second_polarization
        vol = ThorOCT.orient_volume_psoct(vol)
    else:
        raise ValueError(f"Unknown data type: {data_type}")
    
    return (params, vol)


def _load_shard_data(proc_params: dict) -> list:
    """Load all tiles for a shard from disk (I/O stage of the pipeline).

    For shards with multiple tiles (sharding_factor > 1) loads them in
    parallel with a ThreadPoolExecutor; otherwise loads the single tile
    directly to avoid threading overhead.

    Returns a list of (params, volume) tuples, one per tile.
    """
    tiles_params = proc_params['params']
    n_tiles = len(tiles_params)
    if n_tiles > 1:
        with ThreadPoolExecutor(max_workers=min(4, n_tiles)) as executor:
            return list(executor.map(load_single_tile, tiles_params))
    else:
        return [load_single_tile(tiles_params[0])]


def _resize_and_write_shard(proc_params: dict, loaded_tiles: list):
    """GPU-resize pre-loaded tiles and write the shard to zarr (compute/write stage).

    Separated from disk I/O so that _run_pipelined can overlap loading the
    next shard with GPU work on the current one.
    """
    mosaic = proc_params['mosaic']
    shard_shape = proc_params['shard_shape']
    tiles_params = proc_params['params']
    use_gpu = proc_params.get('use_gpu', _USE_GPU)

    shard = np.zeros(shard_shape, dtype=mosaic.dtype)

    mx_min = min(p["tile_pos"][0] for p in tiles_params)
    my_min = min(p["tile_pos"][1] for p in tiles_params)

    vol = None
    for params, vol in loaded_tiles:
        mx, my = params["tile_pos"]
        tile_size = params["tile_size"]

        tile_size_tuple = tuple(tile_size)
        if np.iscomplexobj(vol):
            real_resized = resize(vol.real, tile_size_tuple, order=1,
                                  anti_aliasing=True, use_gpu=use_gpu)
            imag_resized = resize(vol.imag, tile_size_tuple, order=1,
                                  anti_aliasing=True, use_gpu=use_gpu)
            vol = real_resized + 1j * imag_resized
        else:
            vol = resize(vol, tile_size_tuple, order=1, anti_aliasing=True, use_gpu=use_gpu)

        rmin = (mx - mx_min) * vol.shape[1]
        cmin = (my - my_min) * vol.shape[2]
        rmax = rmin + vol.shape[1]
        cmax = cmin + vol.shape[2]

        shard[0:tile_size[0], rmin:rmax, cmin:cmax] = vol

    mx_min *= vol.shape[1]
    my_min *= vol.shape[2]
    output_extent_x = min(shard_shape[1], mosaic.shape[1] - mx_min)
    output_extent_y = min(shard_shape[2], mosaic.shape[2] - my_min)
    mosaic[0:tile_size[0], mx_min:mx_min + output_extent_x, my_min:my_min + output_extent_y] = shard[
        :, :output_extent_x, :output_extent_y]


def _run_pipelined(params: list):
    """Process shards with a prefetch pipeline.

    A single background thread fetches the next shard's tiles from disk
    while the main thread runs GPU resize and zarr write for the current
    shard.  This hides most of the per-tile disk I/O latency behind GPU
    compute and largely eliminates the three-way sequential stall of

        disk read → GPU → zarr write → disk read → GPU → zarr write …

    replacing it with the overlapped pattern

        disk(i+1) ║ GPU+write(i)
    """
    if not params:
        return

    with ThreadPoolExecutor(max_workers=1) as prefetch_executor:
        # Submit the first shard's I/O immediately so it can start before
        # the loop begins.
        pending_load = prefetch_executor.submit(_load_shard_data, params[0])

        for i, p in enumerate(tqdm(params)):
            # Block until current shard's tiles are in RAM.
            loaded_tiles = pending_load.result()

            # Fire off disk I/O for the next shard before doing any GPU
            # work, so they overlap as much as possible.
            if i + 1 < len(params):
                pending_load = prefetch_executor.submit(_load_shard_data, params[i + 1])

            # GPU resize + zarr write for the current shard.
            # This runs concurrently with the prefetch of params[i+1].
            _resize_and_write_shard(p, loaded_tiles)


def process_tile_gpu(proc_params: dict):
    """Process a shard: load tiles from disk, GPU resize, write to zarr.

    This is the entry point used by the CPU multiprocessing pool.  For GPU
    mode the pipelined path (_run_pipelined) is preferred because it
    overlaps disk I/O with GPU work across shards.
    """
    loaded_tiles = _load_shard_data(proc_params)
    _resize_and_write_shard(proc_params, loaded_tiles)


def main():
    global _USE_GPU

    # Parse arguments
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Parameters
    output_resolution = args.resolution
    crop = not args.keep_galvo_return
    fix_galvo_shift = args.fix_galvo_shift
    fix_camera_shift = args.fix_camera_shift
    preprocess = args.preprocess
    galvo_threshold = args.galvo_threshold

    _USE_GPU = args.use_gpu and GPU_AVAILABLE

    if args.verbose:
        print_gpu_info()
        print(f"Using GPU: {_USE_GPU}")

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

    # Prepare the mosaic_grid
    if data_type == 'OCT':
        oct = OCT(tiles[0], args.axial_resolution)
        vol = oct.load_image(crop=crop)
        vol = preprocess_volume(vol, apply=preprocess)
        resolution = [oct.resolution[2], oct.resolution[0], oct.resolution[1]]
        n_extra = oct.info.get('n_extra', 0)
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
        n_extra = 0  # PSOCT doesn't have galvo return
    print(f"Resolution: z = {resolution[0]} , x = {resolution[1]} , y = {resolution[2]} ")

    # Detect galvo shift by sampling multiple tiles
    # galvo_shift: 0 = no fix, >0 = shift amount to apply
    galvo_shift = 0
    if fix_galvo_shift and data_type == 'OCT' and n_extra > 0:
        from linumpy.preproc.xyzcorr import detect_galvo_for_slice
        print(f"Running galvo detection on {len(tiles)} tiles with threshold={galvo_threshold}")
        galvo_shift, confidence = detect_galvo_for_slice(
            tiles, n_extra, threshold=galvo_threshold, axial_resolution=args.axial_resolution
        )
        print(f"Galvo detection result: shift={galvo_shift}, confidence={confidence:.3f}")
        if galvo_shift > 0:
            print(f"Galvo shift detected: shift={galvo_shift}, confidence={confidence:.3f} - will apply fix")
        else:
            print(f"Galvo shift not significant: confidence={confidence:.3f} - skipping fix")

    # tiles position in the mosaic grid
    pos_xy = np.asarray(tiles_pos)[:, :2]
    pos_xy = pos_xy - np.min(pos_xy, axis=0)
    nb_tiles_xy = np.max(pos_xy, axis=0) + 1

    # Compute the rescaled tile size based on
    # the minimum target output resolution
    if output_resolution == -1:
        tile_size = vol.shape
        output_resolution = resolution
    else:
        tile_size = [int(vol.shape[i] * resolution[i] * 1000 / output_resolution) for i in range(3)]
        output_resolution = [output_resolution / 1000.0] * 3
    mosaic_shape = [tile_size[0], nb_tiles_xy[0] * tile_size[1], nb_tiles_xy[1] * tile_size[2]]

    # sharding will lower the number of files stored on disk but increase
    # RAM usage for writing the data (an entire shard must fit in memory)
    shards = (tile_size[0],
              args.sharding_factor * tile_size[1],
              args.sharding_factor * tile_size[2])
    nb_shards_xy = np.ceil(nb_tiles_xy / float(args.sharding_factor)).astype(int)

    # Create the zarr writer
    writer = OmeZarrWriter(args.output_zarr, shape=mosaic_shape,
                           dtype=np.complex64 if args.return_complex else np.float32,
                           chunk_shape=tile_size, shards=shards, overwrite=True)

    # Create a params dictionary for every tile
    params_grid = np.full((nb_shards_xy[0], nb_shards_xy[1]), None, dtype=object)
    for i in range(len(tiles)):
        shard_pos = (pos_xy[i] / args.sharding_factor).astype(int)

        if params_grid[shard_pos[0], shard_pos[1]] is None:
            params_grid[shard_pos[0], shard_pos[1]] = {
                'params': [],
                'mosaic': writer,
                'shard_shape': shards if shards is not None else tile_size,
                'use_gpu': _USE_GPU,
            }

        params_grid[shard_pos[0], shard_pos[1]]['params'].append({
            "file": tiles[i],
            "tile_pos": pos_xy[i],
            "crop": crop,
            "galvo_shift": galvo_shift,
            "fix_camera_shift": fix_camera_shift,
            "preprocess": preprocess,
            "tile_size": tile_size,
            "data_type": data_type,
            "psoct_config": psoct_config,
        })

    # each item in params is a dictionary
    params = [params_grid[i, j]
              for i in range(nb_shards_xy[0])
              for j in range(nb_shards_xy[1])
              if params_grid[i, j] is not None]

    if n_cpus > 1 and not _USE_GPU:
        # CPU parallel processing
        from linumpy._thread_config import worker_initializer
        with multiprocessing.Pool(n_cpus, initializer=worker_initializer) as pool:
            results = tqdm(pool.imap(process_tile_gpu, params), total=len(params))
            tuple(results)
    else:
        # GPU mode: pipeline disk I/O with GPU compute + zarr write so that
        # reading the next shard from disk overlaps with GPU work on the current one.
        _run_pipelined(params)

    # Convert to ome-zarr
    writer.finalize(output_resolution, args.n_levels)


if __name__ == "__main__":
    main()
