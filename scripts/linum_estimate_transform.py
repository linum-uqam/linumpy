#!/usr/bin/env python3

"""
Estimate the affine transform used to compute tile positions in a 2D mosaic grid.

GPU acceleration is used when available (--use_gpu, default on) for phase
correlation. Falls back to CPU if no GPU is detected or --no-use_gpu is passed.

Two modes are available:
1. Registration-based (default): Uses phase correlation to find optimal tile positions
2. Motor-position-based (--use_motor_positions): Uses expected tile spacing based on
   overlap fraction, corresponding to precise motor/stage positions from acquisition

The output transform is a 2x2 matrix that maps tile indices (i, j) to pixel positions.
"""

# Configure thread limits before numpy/scipy imports
import linumpy.config.threads  # noqa: F401

# Configure all libraries (especially SimpleITK) to respect thread limits
from linumpy.config.threads import configure_all_libraries

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import zarr
from skimage.exposure import match_histograms
from skimage.filters import threshold_otsu

from linumpy.gpu import GPU_AVAILABLE, print_gpu_info
from linumpy.gpu.fft_ops import phase_correlation
from linumpy.io.zarr import read_omezarr
from linumpy.metrics import collect_xy_transform_metrics
from linumpy.mosaic import grid as mosaic_grid
from linumpy.registration.transforms import compute_motor_transform

configure_all_libraries()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input_images", nargs="+", help="Full path to a 2D mosaic grid image.")
    p.add_argument("output_transform", help="Output affine transform filename (must be a npy)")
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
        "shapes will be ignored. Note that this will be ignored if a zarr is provided. "
        "The zarr chunks will be used instead. (default=%(default)s)",
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
    p.add_argument(
        "--use_motor_positions",
        action="store_true",
        help="Use motor positions (expected tile spacing) instead of image registration.\n"
        "This creates a transform based purely on the overlap fraction,\n"
        "corresponding to the precise motor/stage positions from acquisition.\n"
        "Recommended when motor positions are reliable.",
    )
    p.add_argument(
        "--use_gpu",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use GPU acceleration if available. [%(default)s]",
    )
    p.add_argument("--verbose", "-v", action="store_true", help="Print GPU information.")
    return p


def main() -> None:
    """Run function."""
    p = _build_arg_parser()
    args = p.parse_args()

    input_images = args.input_images
    if isinstance(input_images, str):
        input_images = [input_images]
    output_transform = Path(args.output_transform)
    max_empty_fraction = args.maximum_empty_fraction
    use_gpu = args.use_gpu and GPU_AVAILABLE

    if args.verbose:
        print_gpu_info()
        print(f"Using GPU: {use_gpu}")
    if args.use_gpu and not GPU_AVAILABLE:
        logger.info("GPU requested but not available, falling back to CPU phase correlation")

    tile_shape = args.tile_shape
    if isinstance(tile_shape, int):
        tile_shape = [tile_shape] * 2
    elif len(tile_shape) == 1:
        tile_shape = [tile_shape[0]] * 2
    elif len(tile_shape) > 2:
        tile_shape = tile_shape[0:2]

    img = None
    if input_images[0].rstrip("/").endswith(".ome.zarr"):
        img, _ = read_omezarr(input_images[0], level=0)
        tile_shape = list(img.chunks[-2:])
    elif input_images[0].rstrip("/").endswith(".zarr"):
        img = zarr.open_array(input_images[0], mode="r")
        tile_shape = list(img.chunks[-2:])

    assert output_transform.name.endswith(".npy"), "output_transform must be a .npy file"

    n_tiles_x = None
    n_tiles_y = None

    if args.use_motor_positions:
        logger.info("Using motor positions with %.1f%% overlap", args.initial_overlap * 100)
        logger.info("Tile shape: %s", tile_shape)

        transform = compute_motor_transform(tile_shape, args.initial_overlap)
        residuals = np.array([0.0])
        tile_count = 0

        logger.info("Motor-based transform:")
        logger.info("  Step Y: %.1f px", transform[0, 0])
        logger.info("  Step X: %.1f px", transform[1, 1])

        if img is not None:
            n_tiles_y = img.shape[-2] // tile_shape[0]
            n_tiles_x = img.shape[-1] // tile_shape[1]

    else:
        logger.info("Using image-based registration (phase correlation, GPU=%s)", use_gpu)

        mosaics = []
        thresholds = []
        for file in input_images:
            if file.rstrip("/").endswith(".ome.zarr"):
                img, _ = read_omezarr(Path(file), level=0)
                image = img[:]
            elif file.rstrip("/").endswith(".zarr"):
                img = zarr.open_array(str(file), mode="r")
                image = np.asarray(img[:])
            else:
                image = sitk.GetArrayFromImage(sitk.ReadImage(str(file)))
            mosaic = mosaic_grid.MosaicGrid(image, tile_shape=tile_shape, overlap_fraction=args.initial_overlap)
            mosaics.append(mosaic)
            thresholds.append(threshold_otsu(mosaic.image))

        rows = []
        rows_px = []
        cols = []
        cols_px = []
        tile_count = 0

        if args.seed is not None:
            random.seed(args.seed)
        mosaic_idx = list(range(len(mosaics)))
        random.shuffle(mosaic_idx)

        for m_id in mosaic_idx:
            mosaic = mosaics[m_id]
            thresh = thresholds[m_id]

            for i in range(mosaic.n_tiles_x):
                for j in range(mosaic.n_tiles_y):
                    if tile_count > args.n_samples:
                        break

                    neighbors, tiles = mosaic.get_neighbors_around_tile(i, j)
                    for _n, t in zip(neighbors, tiles, strict=False):
                        r = t[0] - i
                        c = t[1] - j

                        o1, o2, p1, _p2 = mosaic.get_neighbor_overlap_from_pos((i, j), t)

                        o1_empty = np.sum(o1 <= thresh) > max_empty_fraction * o1.size
                        o2_empty = np.sum(o2 <= thresh) > max_empty_fraction * o2.size
                        if o1_empty or o2_empty:
                            continue

                        o2 = match_histograms(o2, o1)

                        result = phase_correlation(o1, o2, use_gpu=use_gpu)
                        if isinstance(result, tuple):
                            (dx, dy), _ = result
                        else:
                            dx, dy = result

                        r_px = p1[2] - mosaic.tile_size_x + dx if r == -1 else p1[0] + dx
                        c_px = p1[3] - mosaic.tile_size_y + dy if c == -1 else p1[1] + dy

                        rows.append(r)
                        cols.append(c)
                        rows_px.append(r_px)
                        cols_px.append(c_px)

                        tile_count += 1

        a = np.zeros((len(rows) * 2, 4))
        b = np.zeros((len(rows) * 2, 1))
        for i in range(len(rows)):
            a[2 * i, :] = [rows[i], cols[i], 0, 0]
            b[2 * i, 0] = rows_px[i]
            a[2 * i + 1, :] = [0, 0, rows[i], cols[i]]
            b[2 * i + 1, 0] = cols_px[i]

        result = np.linalg.lstsq(a, b, rcond=None)
        transform = result[0].reshape((2, 2))
        residuals = result[1] if len(result[1]) > 0 else np.array([0.0])

        logger.info("Registration-based transform (from %s tile pairs):", tile_count)
        logger.info("  Step Y: %.1f px (expected: %.1f)", transform[0, 0], tile_shape[0] * (1 - args.initial_overlap))
        logger.info("  Step X: %.1f px (expected: %.1f)", transform[1, 1], tile_shape[1] * (1 - args.initial_overlap))

        expected_step_y = tile_shape[0] * (1 - args.initial_overlap)
        expected_step_x = tile_shape[1] * (1 - args.initial_overlap)
        diff_y = (transform[0, 0] - expected_step_y) / expected_step_y * 100
        diff_x = (transform[1, 1] - expected_step_x) / expected_step_x * 100

        if abs(diff_y) > 1 or abs(diff_x) > 1:
            logger.warning("Registration differs from motor positions by Y=%.1f%%, X=%.1f%%", diff_y, diff_x)
            logger.warning("Consider using --use_motor_positions if motor positions are reliable")

        if mosaics:
            n_tiles_x = mosaics[0].n_tiles_x
            n_tiles_y = mosaics[0].n_tiles_y

    output_transform.parent.mkdir(exist_ok=True, parents=True)
    np.save(str(output_transform), transform)
    logger.info("Transform saved to %s", output_transform)

    collect_xy_transform_metrics(
        transform=transform,
        tile_pairs_used=tile_count,
        tile_shape=tuple(tile_shape),
        residuals=residuals,
        output_path=output_transform,
        input_paths=input_images,
        params={"initial_overlap": args.initial_overlap, "use_gpu": use_gpu, "use_motor_positions": args.use_motor_positions},
        n_tiles_x=n_tiles_x,
        n_tiles_y=n_tiles_y,
    )


if __name__ == "__main__":
    main()
