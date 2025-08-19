#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Convert all 3D OCT tiles in a directory to 2D mosaic grids"""

import argparse
import subprocess
from tqdm.auto import tqdm
from linumpy import reconstruction


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("tiles_directory",
                   help="Full path to a directory containing the tiles to process")
    p.add_argument("output_directory",
                   help="Full path to the output directory")
    p.add_argument("-r", "--resolution", type=float, default=-1,
                   help="Output isotropic resolution in micron per pixel. (Use -1 to keep the original resolution). (default=%(default)s)")
    p.add_argument("-e", "--extension", default=".tiff", choices=[".tiff", ".zarr"],
                     help="Output extension (default=%(default)s)")
    p.add_argument("--n_cpus", type=int, default=-1,
                     help="Number of CPUs to use for parallel processing (default=%(default)s). If -1, all CPUs - 1 are used.")

    return p

def main():
    # Parse arguments
    p = _build_arg_parser()
    args = p.parse_args()

    # Parameters
    input_directory = args.tiles_directory
    output_directory = args.output_directory
    resolution = args.resolution
    extension = args.extension
    n_cpus = args.n_cpus

    # Get a list of slices to process
    tiles, tiles_id = reconstruction.get_tiles_ids(input_directory)
    slices = list(set([t[2] for t in tiles_id]))

    for z in tqdm(slices, desc="Creating mosaic grids", unit="slice", leave=True):
        output_file = f"{output_directory}/mosaic_grid_z{z:02d}{extension}"
        cmd = f"linum_create_mosaic_grid_2d.py {input_directory} {output_file} --slice {z} --resolution {resolution} --n_cpus {n_cpus}"
        subprocess.run(cmd, shell=True)

if __name__ == "__main__":
    main()
