#!/usr/bin/env python3

"""Convert a zarr file to an ome-zarr file."""

# Configure thread limits before numpy/scipy imports
import linumpy.config.threads  # noqa: F401

import argparse
from pathlib import Path

import dask.array as da
import zarr

from linumpy.io.zarr import save_omezarr


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input", type=Path, help="Full path to a zarr file (.zarr)")
    p.add_argument("output", type=Path, help="Full path to the output ome-zarr file (.ome-zarr)")
    p.add_argument(
        "-r",
        "--resolution",
        nargs="+",
        type=float,
        default=[1.0],
        help="Resolution of the image in microns. (default=%(default)s)",
    )
    p.add_argument("--n_levels", type=int, default=5, help="Number of levels in pyramidal decomposition. [%(default)s]")

    return p


def main() -> None:
    """Run the zarr-to-OME-Zarr conversion script."""
    # Parse arguments
    p = _build_arg_parser()
    args = p.parse_args()

    # Parameters
    input_file = Path(args.input)
    output_file = Path(args.output)
    resolution = args.resolution  # in microns

    assert len(resolution) in [1, 3], "Resolution must be a single value or a tuple of 3 values"

    # Convert the resolution to mm
    scales = []
    scales = [resolution[0] * 0.001] * 3 if len(resolution) == 1 else [r * 0.001 for r in resolution]

    foo = zarr.open(input_file, mode="r")
    out_dask = da.from_zarr(foo)
    save_omezarr(out_dask, output_file, voxel_size=scales, overwrite=True, n_levels=args.n_levels)


if __name__ == "__main__":
    main()
