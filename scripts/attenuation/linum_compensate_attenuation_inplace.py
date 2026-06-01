#!/usr/bin/env python
r"""Compensate per-slice OCT depth attenuation in a single pass.

Combines the three legacy steps (compute attenuation -> integrate to bias
field -> divide) into one in-memory operation, avoiding intermediate
ome.zarr files. Produces a corrected volume of the same shape as the
input.

Pipeline
--------
1. Build a tissue mask via :func:`compute_tissue_mask` (Otsu + closing).
2. Estimate the local effective attenuation with one of four
   depth-resolved methods (``--method``):

   * ``smith`` (default) -- Smith 2015 extension of Vermeer 2014 with a
     log-gradient :math:`\hat\mu_E` for the finite-range constant.
   * ``vermeer`` -- the bare Vermeer 2014 estimator (``C = 0``).
   * ``liu`` -- Liu 2019 with the exact-form regularization
     ``C = I[imax] / (exp(2 mu_E dz) - 1)`` and curve-fit ``mu_E``.
   * ``li`` -- Li 2020: noise-floor subtraction + per-A-line SNR
     truncation, then Liu-style regularization on the cleaned signal.

   See :mod:`linumpy.intensity.attenuation` for full references.
3. Integrate cumulatively along Z to obtain the round-trip optical depth,
   convert to a multiplicative bias field ``bias = exp(-2 * OD)``.
4. Clamp the bias from below by ``--min_bias`` to prevent runaway gain in
   deep, low-signal voxels and divide the input volume by the clamped
   bias.

Inputs are expected to be per-slice OME-Zarr volumes already cropped to
the tissue interface (i.e. the output of ``linum_crop_3d_mosaic_below_interface``).
"""

# Configure thread limits before numpy/scipy imports
import linumpy.config.threads  # noqa: F401

import argparse
from pathlib import Path

import dask.array as da
import numpy as np
from scipy.integrate import cumulative_trapezoid

from linumpy.intensity.attenuation import (
    get_attenuation_li2020,
    get_attenuation_liu2019,
    get_attenuation_smith2015,
    get_attenuation_vermeer2013,
)
from linumpy.intensity.bias_field import compute_tissue_mask
from linumpy.io.zarr import read_omezarr, save_omezarr

METHODS = ("smith", "vermeer", "liu", "li")


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input", type=Path, help="Input per-slice volume (.ome.zarr)")
    p.add_argument("output", type=Path, help="Compensated volume (.ome.zarr)")
    p.add_argument(
        "--min_bias",
        type=float,
        default=0.05,
        help="Floor applied to the bias field before division.\n"
        "Caps the maximum gain at 1/min_bias and prevents amplification\n"
        "of noise in deep, low-signal voxels. [%(default)s]",
    )
    p.add_argument(
        "--mask_smoothing_sigma",
        type=float,
        default=2.0,
        help="Gaussian sigma (XY voxels) for the Otsu tissue mask. [%(default)s]",
    )
    p.add_argument(
        "--k",
        type=int,
        default=10,
        help="XY median-filter kernel (voxels) applied before the Vermeer attenuation\n"
        "estimation. 0 disables denoising. Larger values stabilise the per-Aline\n"
        "fit at the cost of lateral resolution. [%(default)s]",
    )
    p.add_argument(
        "--zshift",
        type=int,
        default=3,
        help="Number of voxels under the water/tissue interface to ignore when fitting\n"
        "the exponential signal extension. Smaller values use more of the shallow\n"
        "tissue (often improves correction at the surface). [%(default)s]",
    )
    p.add_argument(
        "--strength",
        type=float,
        default=0.3,
        help="Multiplicative scale on the optical-depth correction (0..1).\n"
        "Vermeer's single-scattering model overestimates effective attenuation\n"
        "in scattering brain tissue because the multiple-scattering signal floor\n"
        "violates the geometric-tail assumption. Values <1 attenuate the\n"
        "correction; the empirical sweet spot for cropped 600 um sub-22 slices\n"
        "is ~0.30 (yields a near-flat depth profile). Set to 1.0 for the\n"
        "textbook formula. [%(default)s]",
    )
    p.add_argument(
        "--method",
        choices=METHODS,
        default="li",
        help="Depth-resolved attenuation estimator. ``li`` (default) adds\n"
        "noise-floor subtraction and SNR-based A-line truncation on top\n"
        "of the Liu 2019 exact-form regularization, and produces the\n"
        "flattest axial profile on real OCT data. ``liu`` is the same\n"
        "without noise handling. ``smith`` reproduces the historical\n"
        "linumpy behaviour. ``vermeer`` is the bare estimator with no\n"
        "regularization. [%(default)s]",
    )
    p.add_argument(
        "--snr_threshold_db",
        type=float,
        default=6.0,
        help="Per-voxel SNR threshold (dB) for A-line truncation in the ``li`` method.\n"
        "Voxels where signal / noise_floor < 10^(snr_threshold_db/10) are excluded\n"
        "from the attenuation fit. 6 dB is the Li 2020 paper value. [%(default)s]",
    )
    p.add_argument("--n_levels", type=int, default=0, help="Pyramid levels in the output. [%(default)s]")
    return p


def main() -> None:
    """Run the in-place attenuation compensation."""
    args = _build_arg_parser().parse_args()

    vol_zarr, res = read_omezarr(args.input, level=0)
    chunks = vol_zarr.chunks
    vol = np.asarray(vol_zarr).astype(np.float32)  # (Z, Y, X)

    # Tissue mask (Z, Y, X)
    mask_zyx = compute_tissue_mask(vol, smoothing_sigma=args.mask_smoothing_sigma)

    # Vermeer expects (X, Y, Z) ordering.
    vol_xyz = np.moveaxis(vol, (0, 1, 2), (2, 1, 0))
    mask_xyz = np.moveaxis(mask_zyx, (0, 1, 2), (2, 1, 0)).astype(bool)

    res_axial_microns = float(res[0]) * 1000.0  # res is in mm

    # Depth-resolved attenuation in 1/cm.
    if args.method == "smith":
        attn_cm = get_attenuation_smith2015(
            vol_xyz,
            mask=mask_xyz,
            k=args.k,
            res=res_axial_microns,
            fill_holes=True,
            zshift=args.zshift,
        )
    elif args.method == "liu":
        attn_cm = get_attenuation_liu2019(
            vol_xyz,
            mask=mask_xyz,
            k=args.k,
            res=res_axial_microns,
            fill_holes=True,
            zshift=args.zshift,
        )
    elif args.method == "li":
        attn_cm = get_attenuation_li2020(
            vol_xyz,
            mask=mask_xyz,
            k=args.k,
            res=res_axial_microns,
            fill_holes=True,
            zshift=args.zshift,
            snr_threshold_db=args.snr_threshold_db,
        )
    else:  # vermeer
        # The bare Vermeer estimator returns 1/cm directly; reuse smith's
        # XY-median pre-filter for fairness across methods.
        if args.k > 0:
            import SimpleITK as sitk

            vol_xyz = sitk.GetArrayFromImage(sitk.Median(sitk.GetImageFromArray(vol_xyz), (0, args.k, args.k)))
        attn_cm = get_attenuation_vermeer2013(
            vol_xyz,
            dz=res_axial_microns * 1e-6,
            mask=mask_xyz,
        )

    # 1/cm -> 1/voxel: cm^-1 * 100 = m^-1; * 1e-6 = um^-1; * res_um = voxel^-1
    attn_per_voxel = attn_cm * 100.0 * 1.0e-6 * res_axial_microns

    # Cumulative round-trip optical depth along the Aline (axis=2 in XYZ).
    optical_depth = cumulative_trapezoid(attn_per_voxel, axis=2, initial=0)
    bias_xyz = np.exp(-2.0 * args.strength * optical_depth).astype(np.float32)
    np.maximum(bias_xyz, args.min_bias, out=bias_xyz)

    # Back to (Z, Y, X) and apply.
    bias_zyx = np.moveaxis(bias_xyz, (0, 1, 2), (2, 1, 0))
    corrected = (vol / bias_zyx).astype(np.float32)

    save_omezarr(da.from_array(corrected), args.output, voxel_size=res, chunks=chunks, n_levels=args.n_levels)


if __name__ == "__main__":
    main()
