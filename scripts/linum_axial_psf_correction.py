#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Axial beam profile correction. The script estimates the beam profile
from agarose voxels and then applies the inverse profile to each a-line.
"""

import argparse

import numpy as np
import dask.array as da
from skimage.filters import threshold_otsu
from linumpy.io.zarr import save_omezarr, read_omezarr
from linumpy.preproc.xyzcorr import findTissueInterface, maskUnderInterface

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input_zarr",
                   help="Path to file (.ome.zarr) containing the 3D mosaic grid.")
    p.add_argument("output_zarr",
                   help="Corrected 3D mosaic grid file path (.ome.zarr).")
    p.add_argument('--n_levels', type=int, default=5,
                   help='Number of levels in pyramid representation.')
    p.add_argument('--fit_gaussian', action='store_true',
                   help='Fit a gaussian on the beam profile.')
    p.add_argument('--output_plot',
                   help='Optional output plot filename.')
    return p


def main():
    # Parse the arguments
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Load ome-zarr data
    vol, res = read_omezarr(args.input_zarr, level=0)
    aip = np.mean(vol[:], axis=0)
    otsu = threshold_otsu(aip)
    agarose_mask = aip < otsu

    interface = findTissueInterface(vol[:])
    mask = maskUnderInterface(vol[:], interface, returnMask=True)

    # Exclude out of bounds columns
    mask_all = mask.all(axis=0)  # True where mask is True for every voxel along the aline
    agarose_mask = np.logical_and(agarose_mask, ~mask_all)

    vol_data = vol[:]

    profile = np.reshape(vol_data, (len(vol_data), -1))
    profile = np.array([profile[i] for i in range(len(profile)) if agarose_mask.reshape(-1)[i]])
    profile = np.mean(profile, axis=-1)
    profile = np.clip(profile, np.min(profile[profile > 0.0]), None)

    background = np.min(profile)
    psf = (profile - background) / background

    if args.fit_gaussian:
        psf_max = np.max(psf)
        psf_mu = np.argmax(psf)
        half_max = psf_max / 2
        half_max_right = psf_mu
        for i in range(psf_mu, len(profile)):
            if psf[i] < half_max:
                half_max_right = i
                break
        fwhm = (half_max_right - psf_mu) * 2.0
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        psf = psf_max*np.exp(-((np.arange(len(profile)) - psf_mu) ** 2) / (2 * sigma ** 2))

    if args.output_plot is not None:
        fig, ax = plt.subplots(1, 3)

        ax[0].imshow(agarose_mask, cmap='gray')
        ax[0].set_title('Agarose mask')
        ax[1].plot(np.arange(len(profile)), profile)
        ax[1].plot(np.repeat(background, len(profile)))
        ax[1].set_title('Agarose profile')
        ax[2].plot(np.arange(len(profile)), psf)
        ax[2].set_title('Estimated PSF')
        fig.set_size_inches(12, 5)
        fig.savefig(args.output_plot)

    # apply correction
    vol_corr = vol_data / (1.0 + psf.reshape((-1, 1, 1)))

    # save to ome-zarr
    dask_arr = da.from_array(vol_corr)
    save_omezarr(dask_arr, args.output_zarr, voxel_size=res,
                 chunks=vol.chunks, n_levels=args.n_levels)


if __name__ == "__main__":
    main()
