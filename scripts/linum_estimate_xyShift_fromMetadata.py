#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Estimate the mosaic grid positions from the tiles metadata"""

import argparse
import csv
from pathlib import Path

import numpy as np
from tqdm.contrib.concurrent import process_map

from linumpy.reconstruction import get_mosaic_info, get_tiles_ids


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("directory",
                   help="Tiles directory")
    p.add_argument("output_file",
                   help="Output CSV file")
    return p


def process_slice(z, tiles_directory):
    mosaic_info = get_mosaic_info(tiles_directory, z, use_stage_positions=True)
    return mosaic_info['mosaic_xmin_mm'], mosaic_info['mosaic_ymin_mm'], mosaic_info['tile_resolution']


def main():
    # Parse arguments
    p = _build_arg_parser()
    args = p.parse_args()

    # Extract the parameters
    tiles_directory = Path(args.directory)
    output_file = Path(args.output_file)

    # Get slice ids
    tiles, tile_ids = get_tiles_ids(tiles_directory)
    z_values = np.unique([ids[2] for ids in tile_ids])
    n_slices = len(z_values)

    # Prepare the parameters
    tiles_directory_list = [tiles_directory] * n_slices

    # Extract the metadata
    results = process_map(process_slice, z_values, tiles_directory_list, unit="slice", desc="Computing Mosaic Info",
                          position=0, leave=True)

    xmin_mm, ymin_mm, tile_resolutions = zip(*results)

    # Compute the shift between slices in mm
    x_shifts_mm = []
    y_shifts_mm = []
    for i in range(n_slices - 1):
        dx = xmin_mm[i] - xmin_mm[i + 1]
        dy = ymin_mm[i] - ymin_mm[i + 1]
        x_shifts_mm.append(dx)
        y_shifts_mm.append(dy)

    tile_resolution = tile_resolutions[0]
    # Convert the shifts in pixel, using the xy resolution
    x_shift_px = np.array(x_shifts_mm) / tile_resolution[0]
    y_shift_px = np.array(y_shifts_mm) / tile_resolution[1]

    # Save the shifts to a csv file
    shifts = np.array(
        [list(range(n_slices - 1)), list(range(1, n_slices)), x_shift_px, y_shift_px, x_shifts_mm, y_shifts_mm]).T
    with open(output_file, "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=",")
        writer.writerow(["fixed_id", "moving_id", "x_shift", "y_shift", "x_shift_mm", "y_shift_mm"])
        writer.writerows(shifts)


if __name__ == "__main__":
    main()
