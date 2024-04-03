#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Estimate the affine transform used to compute the tile position given a 2D mosaic grid."""

import argparse
import numpy as np
import SimpleITK as sitk
from linumpy.utils import registration
from linumpy.utils import mosaic_grid
from skimage.filters import threshold_otsu
from skimage.exposure import match_histograms
from pathlib import Path
import random
import zarr

def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input_images", nargs="+",
                   help="Full path to a 2D mosaic grid image.")
    p.add_argument("output_transform",
                   help="Output affine transform filename (must be a npy)")
    p.add_argument("--initial_overlap", type=float, default=0.3,
                   help="Initial overlap fraction between 0 and 1 for the optimization. (default=%(default)s)")
    p.add_argument("-t", "--tile_shape", nargs="+", type=int, default=400,
                   help="Tile shape in pixel. You can provide both the row and col shape if different. Additional "
                        "shapes will be ignored. Note that this will be ignored if a zarr is provided. The zarr chunks will be used instead. (default=%(default)s)")
    p.add_argument("--maximum_empty_fraction", type=float, default=0.9,
                   help="Maximum empty pixel fraction within an overlap to tolerate (default=%(default)s)")
    p.add_argument("--n_samples", type=int, default=512,
                   help="Maximum number of tile pairs to use for the optimization. (default=%(default)s)")
    p.add_argument("--seed", type=int,
                   help="Seed value for the random random number generator")

    return p

def main():
    # Parse arguments
    p = _build_arg_parser()
    args = p.parse_args()

    # Parameters
    input_images = args.input_images
    if isinstance(input_images, str):
        input_images = [input_images]
    output_transform = Path(args.output_transform)
    max_empty_fraction = args.maximum_empty_fraction

    # Compute the tile shape
    tile_shape = args.tile_shape
    if isinstance(tile_shape, int):
        tile_shape = [tile_shape] * 2
    elif len(tile_shape) == 1:
        tile_shape = [tile_shape[0]] * 2
    elif len(tile_shape) > 2:
        tile_shape = tile_shape[0:2]
    if input_images[0].endswith(".zarr"):
        img = zarr.open(input_images[0], mode="r")
        tile_shape = img.chunks

    # Check the output filename extensions
    assert output_transform.name.endswith(".npy"), "output_transform must be a .npy file"

    # Load all input images
    mosaics = []
    thresholds = []
    for file in input_images:
        if file.endswith(".zarr"):
            img = zarr.open(str(file), mode="r")
            image = img[:]
        else:
            image = sitk.GetArrayFromImage(sitk.ReadImage(str(file)))
        mosaic = mosaic_grid.MosaicGrid(image, tile_shape=tile_shape, overlap_fraction=args.initial_overlap)
        mosaics.append(mosaic)

        # Compute an intensity threshold
        thresholds.append(threshold_otsu(mosaic.image))

    # Perform registration for all nonempty overlaps
    rows = []
    rows_px = []
    cols = []
    cols_px = []
    tile_count = 0

    # Loop over mosaics (random order)
    if args.seed is not None:
        random.seed=args.seed
    mosaic_idx = list(range(len(mosaics)))
    random.shuffle(mosaic_idx)
    for m_id in mosaic_idx:
        mosaic = mosaics[m_id]
        thresh = thresholds[m_id]

        # Loop over tiles
        for i in range(mosaic.n_tiles_x):
            for j in range(mosaic.n_tiles_y):
                # Stop if max tile count is reached
                if tile_count > args.n_samples:
                    break

                # Loop over neighborhood tiles
                neighbors, tiles = mosaic.get_neighbors_around_tile(i, j)
                for n, t in zip(neighbors, tiles):
                    r = t[0] - i
                    c = t[1] - j

                    # Extract overlap
                    o1, o2, p1, p2 = mosaic.get_neighbor_overlap_from_pos((i, j), t)

                    # Check if one of the overlap is empty
                    o1_empty = np.sum(o1 <= thresh) > max_empty_fraction * o1.size
                    o2_empty = np.sum(o2 <= thresh) > max_empty_fraction * o2.size
                    if o1_empty or o2_empty:
                        continue

                    # Match histogram
                    o2 = match_histograms(o2, o1)

                    # Perform pairwise registration
                    dx, dy = registration.pairWisePhaseCorrelation(o1, o2)

                    # Compute the tile position
                    if r == -1:
                        r_px = p1[2] - mosaic.tile_size_x + dx
                    else:
                        r_px = p1[0] + dx
                    if c == -1:
                        c_px = p1[3] - mosaic.tile_size_y + dy
                    else:
                        c_px = p1[1] + dy

                    # Updating the rows/cols and rows_px/cols_px
                    rows.append(r)
                    cols.append(c)
                    rows_px.append(r_px)
                    cols_px.append(c_px)

                    # Count the number of tiles used
                    tile_count += 1

    # Estimate the transform matrix from this analysis
    a = np.zeros((len(rows) * 2, 4))
    b = np.zeros((len(rows) * 2, 1))
    for i in range(len(rows)):
        a[2 * i, :] = [rows[i], cols[i], 0, 0]
        b[2 * i, 0] = rows_px[i]
        a[2 * i + 1, :] = [0, 0, rows[i], cols[i]]
        b[2 * i + 1, 0] = cols_px[i]

    # Solve this
    transform = np.linalg.lstsq(a, b, rcond=None)[0].reshape((2, 2))

    # Save the transform
    output_transform.parent.mkdir(exist_ok=True, parents=True)
    np.save(str(output_transform), transform)


if __name__ == "__main__":
    main()
