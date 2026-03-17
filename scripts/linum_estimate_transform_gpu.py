#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Estimate the affine transform used to compute tile positions in a 2D mosaic grid.

GPU-accelerated version using linumpy.gpu module for phase correlation.
Falls back to CPU if GPU is not available.

Two modes are available:
1. Registration-based (default): Uses phase correlation to find optimal tile positions
2. Motor-position-based (--use_motor_positions): Uses expected tile spacing based on
   overlap fraction, corresponding to precise motor/stage positions from acquisition
"""

# Configure thread limits before numpy/scipy imports
import linumpy._thread_config  # noqa: F401

import argparse
import logging
import random
from pathlib import Path

import SimpleITK as sitk
import numpy as np
import zarr
from skimage.exposure import match_histograms
from skimage.filters import threshold_otsu

from linumpy.gpu import GPU_AVAILABLE, print_gpu_info
from linumpy.gpu.fft_ops import phase_correlation
from linumpy.io.zarr import read_omezarr
from linumpy.utils import mosaic_grid

# Configure all libraries (especially SimpleITK) to respect thread limits
from linumpy._thread_config import configure_all_libraries
configure_all_libraries()

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input_images", nargs="+",
                   help="Full path to a 2D mosaic grid image.")
    p.add_argument("output_transform",
                   help="Output affine transform filename (must be a npy)")
    p.add_argument("--initial_overlap", type=float, default=0.2,
                   help="Initial/expected overlap fraction between 0 and 1. (default=%(default)s)")
    p.add_argument("-t", "--tile_shape", nargs="+", type=int, default=400,
                   help="Tile shape in pixel. You can provide both the row and col shape if different. Additional "
                        "shapes will be ignored. Note that this will be ignored if a zarr is provided. The zarr chunks will be used instead. (default=%(default)s)")
    p.add_argument("--maximum_empty_fraction", type=float, default=0.9,
                   help="Maximum empty pixel fraction within an overlap to tolerate (default=%(default)s)")
    p.add_argument("--n_samples", type=int, default=512,
                   help="Maximum number of tile pairs to use for the optimization. (default=%(default)s)")
    p.add_argument("--seed", type=int,
                   help="Seed value for the random number generator")
    p.add_argument('--use_gpu', default=True,
                   action=argparse.BooleanOptionalAction,
                   help='Use GPU acceleration if available. [%(default)s]')
    p.add_argument('--verbose', '-v', action='store_true',
                   help='Print GPU information')

    # Motor position mode
    p.add_argument("--use_motor_positions", action="store_true",
                   help="Use motor positions (expected tile spacing) instead of image registration.\n"
                        "This creates a transform based purely on the overlap fraction,\n"
                        "corresponding to the precise motor/stage positions from acquisition.\n"
                        "Recommended when motor positions are reliable.")
    return p


def compute_motor_transform(tile_shape, overlap_fraction):
    """
    Compute the transform matrix for motor-based tile positions.
    """
    tile_size_y = tile_shape[0]  # rows (height)
    tile_size_x = tile_shape[1]  # cols (width)

    step_y = tile_size_y * (1.0 - overlap_fraction)
    step_x = tile_size_x * (1.0 - overlap_fraction)

    transform = np.array([
        [step_y, 0],
        [0, step_x]
    ])

    return transform


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
    use_gpu = args.use_gpu and GPU_AVAILABLE

    if args.verbose:
        print_gpu_info()
        print(f"Using GPU: {use_gpu}")

    # Compute the tile shape
    tile_shape = args.tile_shape
    if isinstance(tile_shape, int):
        tile_shape = [tile_shape] * 2
    elif len(tile_shape) == 1:
        tile_shape = [tile_shape[0]] * 2
    elif len(tile_shape) > 2:
        tile_shape = tile_shape[0:2]

    if input_images[0].rstrip('/').endswith('.ome.zarr'):
        img, _ = read_omezarr(input_images[0], level=0)
        tile_shape = list(img.chunks[-2:])  # Get last 2 dimensions (Y, X)
    elif input_images[0].rstrip('/').endswith(".zarr"):
        img = zarr.open(input_images[0], mode="r")
        tile_shape = list(img.chunks[-2:])

    # Check the output filename extensions
    assert output_transform.name.endswith(".npy"), "output_transform must be a .npy file"

    if args.use_motor_positions:
        # Motor-position mode: compute transform from expected overlap
        logger.info(f"Using motor positions with {args.initial_overlap*100:.1f}% overlap")
        logger.info(f"Tile shape: {tile_shape}")

        transform = compute_motor_transform(tile_shape, args.initial_overlap)
        residuals = np.array([0.0])
        tile_count = 0

        logger.info(f"Motor-based transform:")
        logger.info(f"  Step Y: {transform[0, 0]:.1f} px")
        logger.info(f"  Step X: {transform[1, 1]:.1f} px")

    else:
        # Registration mode: use phase correlation
        logger.info(f"Using image-based registration (GPU: {use_gpu})")

        # Load all input images
        mosaics = []
        thresholds = []
        for file in input_images:
            if file.rstrip('/').endswith(".ome.zarr"):
                img, _ = read_omezarr(str(file), level=0)
                image = img[:]
            elif file.rstrip('/').endswith(".zarr"):
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
            random.seed = args.seed
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

                        # GPU-accelerated phase correlation
                        result = phase_correlation(o1, o2, use_gpu=use_gpu)
                        if isinstance(result, tuple):
                            (dx, dy), _ = result
                        else:
                            dx, dy = result

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
        result = np.linalg.lstsq(a, b, rcond=None)
        transform = result[0].reshape((2, 2))
        residuals = result[1] if len(result[1]) > 0 else np.array([0.0])

        logger.info(f"Registration-based transform (from {tile_count} tile pairs):")
        logger.info(f"  Step Y: {transform[0, 0]:.1f} px (expected: {tile_shape[0] * (1 - args.initial_overlap):.1f})")
        logger.info(f"  Step X: {transform[1, 1]:.1f} px (expected: {tile_shape[1] * (1 - args.initial_overlap):.1f})")

        # Compare with expected motor positions
        expected_step_y = tile_shape[0] * (1 - args.initial_overlap)
        expected_step_x = tile_shape[1] * (1 - args.initial_overlap)
        diff_y = (transform[0, 0] - expected_step_y) / expected_step_y * 100
        diff_x = (transform[1, 1] - expected_step_x) / expected_step_x * 100

        if abs(diff_y) > 1 or abs(diff_x) > 1:
            logger.warning(f"Registration differs from motor positions by Y={diff_y:.1f}%, X={diff_x:.1f}%")
            logger.warning("Consider using --use_motor_positions if motor positions are reliable")

    # Save the transform
    output_transform.parent.mkdir(exist_ok=True, parents=True)
    np.save(str(output_transform), transform)
    logger.info(f"Transform saved to {output_transform}")

    # Collect metrics using helper function
    from linumpy.utils.metrics import collect_xy_transform_metrics
    collect_xy_transform_metrics(
        transform=transform,
        tile_pairs_used=tile_count,
        tile_shape=tuple(tile_shape),
        residuals=residuals,
        output_path=output_transform,
        input_paths=input_images,
        params={
            'initial_overlap': args.initial_overlap,
            'use_gpu': use_gpu,
            'use_motor_positions': args.use_motor_positions
        }
    )


if __name__ == "__main__":
    main()
