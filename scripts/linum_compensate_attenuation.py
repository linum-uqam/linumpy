#!/usr/bin/env python
"""Compensate the tissue attenuation using a precomputed attenuation bias field."""

# Configure thread limits before numpy/scipy imports
import linumpy.config.threads  # noqa: F401

import argparse
from pathlib import Path

import dask.array as da

from linumpy.io.zarr import read_omezarr, save_omezarr


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input", type=Path, help="Input volume (.ome.zarr)")
    p.add_argument("bias", type=Path, help="Attenuation bias field (.ome.zarr)")
    p.add_argument("output", type=Path, help="Compensated volume (.ome.zarr)")

    return p


def main() -> None:
    """Run the attenuation compensation script."""
    # Parse arguments
    p = _build_arg_parser()
    args = p.parse_args()

    # Load volume and bias field
    vol, res = read_omezarr(args.input, level=0)
    bias, _ = read_omezarr(args.bias, level=0)
    chunks = vol.chunks

    # Apply correction
    vol_dask = da.from_zarr(vol)
    bias_dask = da.from_zarr(bias)
    vol_dask /= bias_dask

    # Save the output
    save_omezarr(vol_dask.astype(da.float32), args.output, voxel_size=res, chunks=chunks)


if __name__ == "__main__":
    main()
