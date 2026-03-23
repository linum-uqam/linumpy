#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Estimate the affine transform used to compute tile positions in a 2D mosaic grid.

Two modes are available:
1. Registration-based (default): Uses phase correlation to find optimal tile positions
2. Motor-position-based (--use_motor_positions): Uses expected tile spacing based on
   overlap fraction, corresponding to precise motor/stage positions from acquisition

The output transform is a 2x2 matrix that maps tile indices (i, j) to pixel positions.
"""

# Configure thread limits before numpy/scipy imports
import linumpy._thread_config  # noqa: F401

import argparse
import numpy as np
import SimpleITK as sitk
from linumpy.stitching.registration import compute_motor_transform, estimate_mosaic_transform
from linumpy.utils import mosaic_grid
from linumpy.utils.metrics import collect_xy_transform_metrics
from pathlib import Path
import zarr
from linumpy.io.zarr import read_omezarr
import logging

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

    # Motor position mode
    p.add_argument("--use_motor_positions", action="store_true",
                   help="Use motor positions (expected tile spacing) instead of image registration.\n"
                        "This creates a transform based purely on the overlap fraction,\n"
                        "corresponding to the precise motor/stage positions from acquisition.\n"
                        "Recommended when motor positions are reliable.")

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
        logger.info("Using image-based registration (phase correlation)")

        # Load all input images
        mosaics = []
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

        # Estimate transform
        transform, residuals, tile_count = estimate_mosaic_transform(
            mosaics, max_empty_fraction, args.n_samples, args.seed
        )

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

    # Determine grid dimensions for accumulated error computation
    n_tiles_x = None
    n_tiles_y = None
    if args.use_motor_positions:
        # img may be defined if input was a zarr
        try:
            # img.shape[-2] = rows, img.shape[-1] = cols
            n_tiles_y = img.shape[-2] // tile_shape[0]
            n_tiles_x = img.shape[-1] // tile_shape[1]
        except NameError:
            pass  # non-zarr input; tile counts unknown
    else:
        if mosaics:
            n_tiles_x = mosaics[0].n_tiles_x
            n_tiles_y = mosaics[0].n_tiles_y

    # Collect metrics using helper function
    collect_xy_transform_metrics(
        transform=transform,
        tile_pairs_used=tile_count,
        tile_shape=tuple(tile_shape),
        residuals=residuals,
        output_path=output_transform,
        input_paths=input_images,
        params={
            'initial_overlap': args.initial_overlap,
            'use_motor_positions': args.use_motor_positions
        },
        n_tiles_x=n_tiles_x,
        n_tiles_y=n_tiles_y,
    )


if __name__ == "__main__":
    main()
