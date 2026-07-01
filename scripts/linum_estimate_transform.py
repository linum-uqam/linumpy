#!/usr/bin/env python3

"""
Estimate the affine transform used to compute tile positions in a 2D mosaic grid.

Two modes are available:
1. Registration-based (default): Uses phase correlation to find optimal tile positions
2. Motor-position-based (--use_motor_positions): Uses expected tile spacing based on
   overlap fraction, corresponding to precise motor/stage positions from acquisition

The output transform is a 2x2 matrix that maps tile indices (i, j) to pixel positions.
"""

# Configure thread limits before numpy/scipy imports
import argparse
import logging
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import zarr

import linumpy.config.threads  # noqa: F401

# Configure all libraries (especially SimpleITK) to respect thread limits
from linumpy.config.threads import configure_all_libraries
from linumpy.io.zarr import read_omezarr
from linumpy.metrics import collect_xy_transform_metrics
from linumpy.mosaic import grid as mosaic_grid
from linumpy.registration.transforms import compute_motor_transform, estimate_mosaic_transform

configure_all_libraries()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input_images", type=Path, nargs="+", help="Full path to a 2D mosaic grid image.")
    p.add_argument("output_transform", type=Path, help="Output affine transform filename (must be a npy)")
    p.add_argument(
        "--initial_overlap",
        type=float,
        default=0.2,
        help="Initial/expected overlap fraction between 0 and 1. (default=%(default)s)",
    )
    p.add_argument(
        "-t",
        "--tile_shape",
        nargs="+",
        type=int,
        default=400,
        help="Tile shape in pixel. You can provide both the row and col shape if different. Additional "
        "shapes will be ignored. Note that this will be ignored if a zarr is provided. The zarr chunks will be used instead."
        " (default=%(default)s)",
    )
    p.add_argument(
        "--maximum_empty_fraction",
        type=float,
        default=0.9,
        help="Maximum empty pixel fraction within an overlap to tolerate (default=%(default)s)",
    )
    p.add_argument(
        "--n_samples",
        type=int,
        default=512,
        help="Maximum number of tile pairs to use for the optimization. (default=%(default)s)",
    )
    p.add_argument("--seed", type=int, help="Seed value for the random number generator")

    # Motor position mode
    p.add_argument(
        "--use_motor_positions",
        action="store_true",
        help="Use motor positions (expected tile spacing) instead of image registration.\n"
        "This creates a transform based purely on the overlap fraction,\n"
        "corresponding to the precise motor/stage positions from acquisition.\n"
        "Recommended when motor positions are reliable.",
    )

    return p


def main() -> None:
    """Run the mosaic transform estimation script."""
    # Parse arguments
    p = _build_arg_parser()
    args = p.parse_args()

    # Parameters
    input_images = [str(path) for path in (args.input_images if isinstance(args.input_images, list) else [args.input_images])]
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

    img: zarr.Array | None = None
    if input_images[0].rstrip("/").endswith(".ome.zarr"):
        img, _ = read_omezarr(input_images[0], level=0)
        tile_shape = list(img.chunks[-2:])  # Get last 2 dimensions (Y, X)
    elif input_images[0].rstrip("/").endswith(".zarr"):
        _zarr = zarr.open(input_images[0], mode="r")
        assert isinstance(_zarr, zarr.Array)
        img = _zarr
        tile_shape = list(img.chunks[-2:])

    # Check the output filename extensions
    assert output_transform.name.endswith(".npy"), "output_transform must be a .npy file"

    mosaics: list = []
    if args.use_motor_positions:
        # Motor-position mode: compute transform from expected overlap
        logger.info("Using motor positions with %.1f%% overlap", args.initial_overlap * 100)
        logger.info("Tile shape: %s", tile_shape)

        transform = compute_motor_transform(tile_shape, args.initial_overlap)
        residuals = np.array([0.0])
        tile_count = 0

        logger.info("Motor-based transform:")
        logger.info("  Step Y: %.1f px", transform[0, 0])
        logger.info("  Step X: %.1f px", transform[1, 1])

    else:
        # Registration mode: use phase correlation
        logger.info("Using image-based registration (phase correlation)")

        # Load all input images
        for file in input_images:
            if file.rstrip("/").endswith(".ome.zarr"):
                img, _ = read_omezarr(file, level=0)
                image = img[:]
            elif file.rstrip("/").endswith(".zarr"):
                _zarr2 = zarr.open(str(file), mode="r")
                assert isinstance(_zarr2, zarr.Array)
                image = _zarr2[:]
            else:
                image = sitk.GetArrayFromImage(sitk.ReadImage(str(file)))
            mosaic = mosaic_grid.MosaicGrid(
                np.asarray(image), tile_shape=tuple(tile_shape), overlap_fraction=args.initial_overlap
            )
            mosaics.append(mosaic)

        # Estimate transform
        transform, residuals, tile_count = estimate_mosaic_transform(mosaics, max_empty_fraction, args.n_samples, args.seed)

        logger.info("Registration-based transform (from %d tile pairs):", tile_count)
        logger.info(
            "  Step Y: %.1f px (expected: %.1f)",
            transform[0, 0],
            tile_shape[0] * (1 - args.initial_overlap),
        )
        logger.info(
            "  Step X: %.1f px (expected: %.1f)",
            transform[1, 1],
            tile_shape[1] * (1 - args.initial_overlap),
        )

        # Compare with expected motor positions
        expected_step_y = tile_shape[0] * (1 - args.initial_overlap)
        expected_step_x = tile_shape[1] * (1 - args.initial_overlap)
        diff_y = (transform[0, 0] - expected_step_y) / expected_step_y * 100
        diff_x = (transform[1, 1] - expected_step_x) / expected_step_x * 100

        if abs(diff_y) > 1 or abs(diff_x) > 1:
            logger.warning("Registration differs from motor positions by Y=%.1f%%, X=%.1f%%", diff_y, diff_x)
            logger.warning("Consider using --use_motor_positions if motor positions are reliable")

    # Save the transform
    output_transform.parent.mkdir(exist_ok=True, parents=True)
    np.save(str(output_transform), transform)
    logger.info("Transform saved to %s", output_transform)

    # Determine grid dimensions for accumulated error computation
    n_tiles_x = None
    n_tiles_y = None
    if args.use_motor_positions:
        # img may be defined if input was a zarr
        if img is not None:
            n_tiles_y = img.shape[-2] // tile_shape[0]
            n_tiles_x = img.shape[-1] // tile_shape[1]
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
        params={"initial_overlap": args.initial_overlap, "use_motor_positions": args.use_motor_positions},
        n_tiles_x=n_tiles_x,
        n_tiles_y=n_tiles_y,
    )


if __name__ == "__main__":
    main()
