#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detect and fix the lateral illumination inhomogeneities for each
3D tiles of a mosaic grid.
GPU-accelerated version using JAX/CUDA for BaSiCPy.
For CPU-only processing, use linum_fix_illumination_3d.py
"""
# Configure thread limits before numpy/scipy imports
import linumpy._thread_config  # noqa: F401
import os
import ctypes
import site
# When using multiprocessing with pqdm, we need to limit threads per worker
# to prevent thread oversubscription. The number of threads per worker should be
# calculated based on total CPUs and number of parallel processes.
# This is set dynamically in main() after parsing arguments.
# For now, we preserve any existing OMP_NUM_THREADS setting from Nextflow,
# or default to 1 for safety when multiprocessing.
if 'OMP_NUM_THREADS' not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "1"
# Configure JAX/XLA thread limits for BaSiCPy
# Must be set BEFORE importing jax/basicpy
if 'XLA_FLAGS' not in os.environ:
    omp_threads = os.environ.get('OMP_NUM_THREADS', '1')
    os.environ['XLA_FLAGS'] = f'--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads={omp_threads}'
if 'OMP_NUM_THREADS' not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "1"
def _preload_cuda_libraries():
    """Preload CUDA libraries before JAX import for GPU acceleration.
    JAX 0.4.23 requires specific library versions from nvidia-xxx-cu12 packages.
    These must be loaded via ctypes BEFORE JAX is imported so the symbols are
    available when XLA initializes.
    """
    # Get site-packages paths
    sp_paths = site.getsitepackages()
    ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    # Build list of CUDA library search paths (pip packages + LD_LIBRARY_PATH)
    search_paths = []
    for sp in sp_paths:
        for lib_dir in ['nvidia/cublas/lib', 'nvidia/cuda_runtime/lib',
                        'nvidia/cusolver/lib', 'nvidia/cusparse/lib',
                        'nvidia/cufft/lib', 'nvidia/cudnn/lib',
                        'nvidia/nvjitlink/lib']:
            path = os.path.join(sp, lib_dir)
            if os.path.isdir(path):
                search_paths.append(path)
    search_paths.extend(ld_path.split(':'))
    # Libraries to preload (order matters - dependencies first)
    # These are the .so versions from pinned nvidia-xxx-cu12 packages
    libs = [
        'libcudart.so.12',
        'libcublas.so.12',
        'libcublasLt.so.12',
        'libcusolver.so.11',  # JAX 0.4.23 needs .so.11
        'libcusparse.so.12',
        'libcufft.so.11',     # JAX 0.4.23 needs .so.11
    ]
    loaded = []
    for lib in libs:
        for path in search_paths:
            lib_path = os.path.join(path, lib)
            if os.path.exists(lib_path):
                try:
                    ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
                    loaded.append(lib)
                except Exception:
                    pass
                break
    if loaded:
        # Update LD_LIBRARY_PATH so child processes can find libraries too
        new_paths = ':'.join(search_paths)
        if ld_path:
            os.environ['LD_LIBRARY_PATH'] = f'{new_paths}:{ld_path}'
        else:
            os.environ['LD_LIBRARY_PATH'] = new_paths
        return True
    return False
# Preload CUDA libraries BEFORE importing JAX/basicpy
_cuda_available = _preload_cuda_libraries()
if not _cuda_available:
    print("Warning: CUDA libraries not found, JAX will use CPU fallback")
import argparse
import tempfile
from pathlib import Path
from basicpy import BaSiC
import dask.array as da
import zarr
from tqdm.auto import tqdm
import imageio as io
import numpy as np
from linumpy.io.zarr import save_omezarr, read_omezarr, create_tempstore
from linumpy.utils.io import add_processes_arg, parse_processes_arg
# TODO: add option to export the flatfields and darkfields
def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input_zarr",
                   help="Full path to the input zarr file")
    p.add_argument("output_zarr",
                   help="Full path to the output zarr file")
    p.add_argument("--max_iterations", type=int, default=500,
                   help='Maximum number of iterations for BaSiC. [%(default)s]')
    p.add_argument("--percentile_max", type=float,
                   help="Values above this percentile will be clipped when\n"
                        "estimating the flatfield (inside range [0-100]).")
    add_processes_arg(p)
    return p
def process_tile(params: dict):
    """Process a tile and add it to the output mosaic.

    GPU version processes tiles sequentially in the main process.
    """
    file = params["slice_file"]
    z = params["z"]
    tile_shape = params["tile_shape"]
    max_iterations = params["max_iterations"]
    p_upper = params["p_upper"]
    vol = io.v3.imread(str(file))
    file_output = Path(file).parent / file.name.replace(".tiff", "_corrected.tiff")
    # Get the number of tiles
    nx = vol.shape[0] // tile_shape[0]
    ny = vol.shape[1] // tile_shape[1]
    # Extract the tiles for this slice
    tiles = []
    for i in range(nx):
        for j in range(ny):
            rmin = i * tile_shape[0]
            rmax = (i + 1) * tile_shape[0]
            cmin = j * tile_shape[1]
            cmax = (j + 1) * tile_shape[1]
            tiles.append(vol[rmin:rmax, cmin:cmax])
    # Estimate the illumination bias
    tiles_for_fit = np.asarray(tiles)
    if np.iscomplexobj(tiles[0]):  # if input is complex, take amplitude of signal
        tiles_for_fit = np.abs(tiles_for_fit).astype(np.float64)
    if p_upper is not None:
        tiles_for_fit = np.clip(tiles_for_fit, None, p_upper)
    optimizer = BaSiC(get_darkfield=False, max_iterations=max_iterations)
    optimizer.fit(tiles_for_fit)
    # apply correction to tiles
    try:
        # Check if tiles contain complex values
        # TODO: Hasn't been validated for complex input since basicpy has replaced pybasic
        if np.iscomplexobj(tiles[0]):
            # Separate real and imaginary parts
            tiles_real = [t.real for t in tiles]
            tiles_imag = [t.imag for t in tiles]
            # Store the original signs before applying BaSic as it requires positive values
            sign_real = [np.sign(t) for t in tiles_real]
            sign_imag = [np.sign(t) for t in tiles_imag]
            # Run BaSiC
            tiles_real_corr = optimizer.transform(np.asarray(tiles_real))
            tiles_imag_corr = optimizer.transform(np.asarray(tiles_imag))
            # Apply correction and reconstruct complex result with original signs
            tiles_corrected = [
                (t_real * s_real) + 1j * (t_imag * s_imag)
                for t_real, t_imag, s_real, s_imag in zip(
                    tiles_real_corr, tiles_imag_corr, sign_real, sign_imag)
            ]
        else:
            # Process normally if tiles are real
            # Apply correction to original (not clipped) tiles
            tiles_corrected = optimizer.transform(np.asarray(tiles))
    except RuntimeError:
        print(f'Got runtime error at z={z}')
        tiles_corrected = np.asarray(tiles)
    # Fill the output mosaic
    vol_output = np.zeros_like(vol)
    for i in range(nx):
        for j in range(ny):
            rmin = i * tile_shape[0]
            rmax = (i + 1) * tile_shape[0]
            cmin = j * tile_shape[1]
            cmax = (j + 1) * tile_shape[1]
            t = tiles_corrected[i * ny + j]
            if np.isnan(t).any():
                print(f"NaN values found in tile {i}, {j} at z={z}. Replacing with zeros.")
                t = np.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
            vol_output[rmin:rmax, cmin:cmax] = t
    io.imsave(str(file_output), vol_output)
    return z, file_output
def main():
    # Parse arguments
    p = _build_arg_parser()
    args = p.parse_args()
    # Parameters
    input_zarr = Path(args.input_zarr)
    output_zarr = Path(args.output_zarr)
    n_cpus = parse_processes_arg(args.n_processes)
    # Log GPU status
    try:
        import jax
        devices = jax.devices()
        has_gpu = any('cuda' in str(d).lower() for d in devices)
        if has_gpu:
            print(f"JAX GPU acceleration enabled: {devices}")
        else:
            print(f"JAX running on CPU: {devices}")
    except Exception as e:
        print(f"JAX device check failed: {e}")
    # Prepare the data for the parallel processing
    vol, resolution = read_omezarr(input_zarr, level=0)
    p_upper = None
    if args.percentile_max is not None:
        p_upper = np.percentile(vol[:], args.percentile_max)
    n_slices = vol.shape[0]
    tmp_dir = tempfile.TemporaryDirectory(
        suffix="_linum_fix_illumination_3d_slices", dir=output_zarr.parent)
    params_list = []
    for z in tqdm(range(n_slices), "Preprocessing slices"):
        slice_file = Path(tmp_dir.name) / f"slice_{z:03d}.tiff"
        img = vol[z]
        io.imsave(str(slice_file), img)
        params = {
            "z": z,
            "slice_file": slice_file,
            "tile_shape": vol.chunks[1:],
            "max_iterations": args.max_iterations,
            "p_upper": p_upper
        }
        params_list.append(params)
    if n_cpus > 1:
        # GPU version: Process sequentially - CUDA contexts don't work with multiprocessing
        # The GPU speedup comes from JAX running on GPU, not from multiple processes
        print(f"Note: GPU mode uses sequential processing (ignoring n_processes={n_cpus})")

    # Process tiles sequentially
    corrected_files = []
    for param in tqdm(params_list, desc="Processing tiles"):
        corrected_files.append(process_tile(param))
    # Retrieve the results and fix the volume
    temp_store = create_tempstore(suffix=".zarr")
    vol_output = zarr.open(temp_store, mode="w", shape=vol.shape,
                           dtype=vol.dtype, chunks=vol.chunks)
    # TODO: Rebuilding volume step could be faster
    for z, f in tqdm(corrected_files, "Rebuilding volume"):
        slice_vol = io.v3.imread(str(f))
        vol_output[z] = slice_vol[:]
    out_dask = da.from_zarr(vol_output)
    min_value = out_dask.min().compute()
    if min_value < 0:
        print(f"Minimum value in the output volume is {min_value}. Clipping at 0.")
        out_dask = da.clip(out_dask, 0., None)
    save_omezarr(out_dask, output_zarr, voxel_size=resolution,
                 chunks=vol.chunks)
    # Remove the temporary slice files used by the parallel processes
    tmp_dir.cleanup()
if __name__ == "__main__":
    main()
