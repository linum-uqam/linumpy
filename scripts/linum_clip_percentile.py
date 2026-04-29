#!/usr/bin/env python3
"""Clip .ome.zarr volume intensities between lower and upper percentile."""

# Configure thread limits before numpy/scipy imports
import linumpy.config.threads  # noqa: F401

import argparse
from pathlib import Path

import dask.array as da

from linumpy.io.zarr import read_omezarr, save_omezarr


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("in_volume", type=Path, help="Input volume .ome.zarr.")
    p.add_argument("out_volume", type=Path, help="Output volume .ome.zarr.")
    p.add_argument(
        "--percentile_lower", default=0, type=float, help="Percentile below which values will be clipped [%(default)s]."
    )
    p.add_argument(
        "--percentile_upper", default=99.9, type=float, help="Percentile above which values will be clipped [%(default)s]."
    )
    p.add_argument("--rescale", action="store_true", help="Rescale volume intensities after clipping.")
    return p


def main() -> None:
    """Run the percentile clipping script."""
    parser = _build_arg_parser()
    args = parser.parse_args()

    vol, res = read_omezarr(args.in_volume)
    darr = da.from_zarr(vol)
    p_lower = float(da.percentile(darr.ravel(), args.percentile_lower).compute()[0])
    p_upper = float(da.percentile(darr.ravel(), args.percentile_upper).compute()[0])
    darr = da.clip(darr, p_lower, p_upper)

    if args.rescale:
        darr = darr - darr.flatten().min()
        darr = darr / darr.flatten().max()

    save_omezarr(darr, args.out_volume, res, vol.chunks)


if __name__ == "__main__":
    main()
