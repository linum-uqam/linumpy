#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Estimate the inter-slice XY shifts from tile stage positions.

The shift between consecutive slices is inferred by comparing the stage
positions of the mosaic grid boundaries.  Because the acquisition software
can expand the mosaic grid between slices (to cover newly-exposed tissue),
the left/right or top/bottom boundary may shift independently of any real
tissue displacement.  To minimise this bias, the script compares *both*
boundaries (min and max) for each axis and uses whichever shifted less as
the reference: that boundary is more likely to reflect true inter-slice
tissue drift rather than a mosaic repositioning event.

When the mosaic expands symmetrically (or no expansion occurs), both
boundaries give consistent estimates and the result is the same as before.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from tqdm.contrib.concurrent import process_map
from linumpy.utils.io import add_processes_arg, parse_processes_arg

from linumpy.reconstruction import get_mosaic_info, get_tiles_ids


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("directory",
                   help="Tiles directory")
    p.add_argument("output_file",
                   help="Output CSV file")
    add_processes_arg(p)
    return p


def process_slice(z, tiles_directory):
    mosaic_info = get_mosaic_info(tiles_directory, z, use_stage_positions=True)
    return (mosaic_info['mosaic_xmin_mm'], mosaic_info['mosaic_xmax_mm'],
            mosaic_info['mosaic_ymin_mm'], mosaic_info['mosaic_ymax_mm'],
            mosaic_info['tile_resolution'],
            mosaic_info['mosaic_grid_shape'])


def main():
    # Parse arguments
    p = _build_arg_parser()
    args = p.parse_args()
    n_processes = parse_processes_arg(args.n_processes)

    # Extract the parameters
    tiles_directory = Path(args.directory)
    output_file = Path(args.output_file)

    # Get slice ids
    _, tile_ids = get_tiles_ids(tiles_directory)
    z_values = np.unique([ids[2] for ids in tile_ids])
    n_slices = len(z_values)

    # Prepare the parameters
    tiles_directory_list = [tiles_directory] * n_slices

    # Extract the metadata
    results = process_map(process_slice, z_values, tiles_directory_list,
                          unit="slice", desc="Computing Mosaic Info",
                          position=0, leave=True, max_workers=n_processes)

    xmin_mm, xmax_mm, ymin_mm, ymax_mm, tile_resolutions, grid_shapes = zip(*results)

    # Compute the shift between slices in mm.
    # For each axis, compare both boundaries (min and max).  Mosaic expansion
    # displaces the expanding boundary by roughly one tile step while the
    # opposite boundary stays put.  Using the boundary that moved less gives a
    # better estimate of true inter-slice tissue drift.
    x_shifts_mm = []
    y_shifts_mm = []
    reliable_flags = []
    for i in range(n_slices - 1):
        dx_from_min = xmin_mm[i] - xmin_mm[i + 1]
        dx_from_max = xmax_mm[i] - xmax_mm[i + 1]
        dy_from_min = ymin_mm[i] - ymin_mm[i + 1]
        dy_from_max = ymax_mm[i] - ymax_mm[i + 1]
        # Use the boundary that shifted less — it reflects tissue drift rather
        # than mosaic repositioning.  When neither boundary is dominant, both
        # estimates are consistent and either is equally valid.
        dx = dx_from_min if abs(dx_from_min) <= abs(dx_from_max) else dx_from_max
        dy = dy_from_min if abs(dy_from_min) <= abs(dy_from_max) else dy_from_max
        x_shifts_mm.append(dx)
        y_shifts_mm.append(dy)
        # Mark this transition as unreliable if the mosaic grid changed size,
        # indicating a grid expansion event where the metadata shift estimate
        # may not reflect the true tissue drift.
        grid_changed = (grid_shapes[i] != grid_shapes[i + 1])
        reliable_flags.append(0 if grid_changed else 1)

    tile_resolution = tile_resolutions[0]
    # Convert the shifts in pixel, using the xy resolution
    x_shift_px = np.array(x_shifts_mm) / tile_resolution[0]
    y_shift_px = np.array(y_shifts_mm) / tile_resolution[1]

    # Save the shifts to a csv file
    shifts = np.array(
        [z_values[:-1], z_values[1:], x_shift_px, y_shift_px, x_shifts_mm, y_shifts_mm,
         reliable_flags]).T
    with open(output_file, "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=",")
        writer.writerow(["fixed_id", "moving_id", "x_shift", "y_shift",
                         "x_shift_mm", "y_shift_mm", "reliable"])
        writer.writerows(shifts)


if __name__ == "__main__":
    main()
