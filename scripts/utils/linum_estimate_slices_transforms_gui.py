#!/usr/bin/env python3
"""
Manually align slices from a 2.5D volume using a graphical user interface.

The GUI can also be used to correct slice intensities across slices.
By default, the script rescales each slice intensities between their
minimum and maximum values to the range [0, 1]. When launched, a matplotlib
window opens, enabling the user to manipulate the slices as they wish. The
resulting transformations are saved as soon as the window is closed.
"""

# Configure thread limits before numpy/scipy imports
import linumpy.config.threads  # noqa: F401

import argparse
from pathlib import Path

import numpy as np
import zarr

from linumpy.registration.manual import ManualImageCorrection


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("in_zarr", type=Path, help="Input zarr file to align.")
    p.add_argument("resolution", nargs=3, type=float, help="Voxel size in microns.")
    p.add_argument("out_result", type=Path, help="Output result file in .npz format.")
    p.add_argument(
        "--downsample_factor",
        type=int,
        default=8,
        help="Downsample factor for rendering whole resolution image [%(default)s].",
    )
    p.add_argument("--checkpoint_file", type=Path, help="Result file (.npz) to use as initial parameters.")
    p.add_argument("-f", dest="overwrite", action="store_true", help="Overwrite output file.")
    return p


def main() -> None:
    """Run the GUI for estimating inter-slice transforms."""
    parser = _build_arg_parser()
    args = parser.parse_args()
    in_zarr = zarr.open(args.in_zarr, mode="r")
    ext = Path(args.out_result).suffix
    if not ext not in ["", "npz"]:
        parser.error("Invalid extension for output result. Extension should be .npz.")

    if Path(args.out_result).exists() and not args.overwrite:
        parser.error("Output file exists, use option -f to overwrite.")
    else:
        path = Path(args.out_result).parent
        if not path.exists():
            path.mkdir(parents=True)

    custom_ranges = None
    transforms = None
    if args.checkpoint_file:
        checkpoint = np.load(args.checkpoint_file)
        custom_ranges = checkpoint["custom_ranges"]
        transforms = checkpoint["transforms"]

    image_registration = ManualImageCorrection(
        np.asarray(in_zarr), args.resolution, args.downsample_factor, transforms, custom_ranges
    )

    if image_registration.start():
        # true when figure is closed
        image_registration.save_results(args.out_result)


if __name__ == "__main__":
    main()
