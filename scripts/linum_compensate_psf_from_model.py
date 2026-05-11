#!/usr/bin/env python3
"""Compensate PSF blurring using a parametric model.

Notes
-----
The default initial Rayleigh length (``--zr_initial 610`` µm) is tuned for a
3x objective.

For the 10x objective (Mitutoyo M Plan Apo NIR 10X, NA = 0.26, WD = 30.5 mm,
used with a water immersion cap), the confocal Rayleigh length has not yet
been characterised. Phantom calibration (linum-microscopes-soct/
psf_analysis.ipynb) constrains the axial coherence FWHM (~15 µm) and the
axial pixel size, but those are independent of ``zr_0`` (which governs the
focal-depth-dependent broadening, not the bandwidth-limited axial response).
Until a 10x value is fitted from real data, pass a measured
``--zr_initial`` rather than relying on the 3x default.
"""

# Configure thread limits before numpy/scipy imports
import linumpy.config.threads  # noqa: F401

import argparse
from pathlib import Path

import numpy as np

from linumpy.io.zarr import read_omezarr, save_omezarr
from linumpy.psf.extract import extract_psf_parameters_from_mosaic
from linumpy.psf.synthetic import synthesize_3d_psf


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("in_zarr", type=Path, help="Input stitched 3D slice (OME-zarr).")
    p.add_argument("out_zarr", type=Path, help="Output volume corrected for beam PSF (OME-zarr).")
    p.add_argument("--out_psf", type=Path, help="Optional output PSF filename.")
    p.add_argument("--nz", type=int, default=25, help='The "nz" first voxels belonging to background [%(default)s].')
    p.add_argument("--n_profiles", type=int, default=10, help="Number of intensity profiles to use [%(default)s].")
    p.add_argument("--n_iterations", type=int, default=15, help="Number of iterations [%(default)s].")
    p.add_argument("--smooth", type=float, default=0.01, help="Smoothing factor as a fraction of volume depth [%(default)s].")
    p.add_argument(
        "--zr_initial",
        type=float,
        default=610.0,
        help="Initial Rayleigh length in micron used to bootstrap the confocal-PSF fit.\n"
        "Default is calibrated for a 3x objective; for other objectives (e.g. 10x)\n"
        "supply a measured estimate. [%(default)s]",
    )
    return p


def main() -> None:
    """Run the model-based PSF compensation script."""
    parser = _build_arg_parser()
    args = parser.parse_args()

    # 1. load stitched tissue slice
    vol, res = read_omezarr(args.in_zarr, level=0)
    chunks = vol.chunks
    vol = np.moveaxis(np.asarray(vol), (0, 1, 2), (2, 1, 0))
    res = res[::-1]
    res_axial_microns = res[2] * 1000

    # 2. estimate psf
    zf, zr = extract_psf_parameters_from_mosaic(
        vol,
        n_profiles=args.n_profiles,
        res=res_axial_microns,
        f=args.smooth,
        n_iterations=args.n_iterations,
        zr_0=args.zr_initial,
    )
    psf_3d = synthesize_3d_psf(zf, zr, res_axial_microns, vol.shape)

    # Compensate by the PSF
    background = np.mean(vol[..., : args.nz])
    output = (vol - background) / psf_3d + background

    # remove negative values
    output -= output.min()

    # TODO: Use dask arrays
    output = np.moveaxis(output, (0, 1, 2), (2, 1, 0))
    res = res[::-1]

    if args.out_psf:
        psf_3d = np.moveaxis(psf_3d, (0, 1, 2), (2, 1, 0))
        # when there are too many levels it'll break the viewer for some reason
        save_omezarr(psf_3d.astype(np.float32), args.out_psf, voxel_size=res, chunks=chunks)

    save_omezarr(output.astype(np.float32), args.out_zarr, voxel_size=res, chunks=chunks)


if __name__ == "__main__":
    main()
