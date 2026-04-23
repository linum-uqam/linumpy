#!/usr/bin/env python3
"""
Generate orthogonal view screenshots from an OME-Zarr volume.

Creates a figure with three panels showing XY, XZ, and YZ views
through the center of the volume (or at specified slice indices).
"""

# Configure thread limits before numpy/scipy imports
import linumpy.config.threads  # noqa: F401

import argparse
from pathlib import Path

from linumpy.imaging.visualization import save_orthogonal_views
from linumpy.io.zarr import read_omezarr


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("in_zarr", help="Full path to a zarr file.")
    p.add_argument("out_figure", help="Full path to the output figure")
    p.add_argument("--z_slice", type=int, help="Slice index along first axis.")
    p.add_argument("--x_slice", type=int, help="Slice index along the second axis.")
    p.add_argument("--y_slice", type=int, help="Slice index along the last axis.")
    p.add_argument("--cmap", default="magma", help="Colormap for the figure (default: magma).")
    return p


def main() -> None:
    """Run function."""
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Validate input path
    in_path = Path(args.in_zarr)
    if not in_path.exists():
        parser.error(f"Input file not found: {args.in_zarr}")

    # Resolve symlinks (common in Nextflow work directories)
    in_path = in_path.resolve()

    image, _ = read_omezarr(Path(in_path))

    save_orthogonal_views(
        image, args.out_figure, z_slice=args.z_slice, x_slice=args.x_slice, y_slice=args.y_slice, cmap=args.cmap
    )


if __name__ == "__main__":
    main()
