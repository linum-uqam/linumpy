#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compensate the tissue attenuation using a precomputed attenuation
bias field.
"""

import argparse
import dask.array as da

from linumpy.io.zarr import save_zarr, read_omezarr


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input",
                   help="Input volume (.ome.zarr)")
    p.add_argument("bias",
                   help="Attenuation bias field (.ome.zarr)")
    p.add_argument("output",
                   help="Compensated volume (.ome.zarr)")

    return p


def main():
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
    save_zarr(vol_dask.astype(da.float32), args.output, scales=res, chunks=chunks)


if __name__ == "__main__":
    main()
