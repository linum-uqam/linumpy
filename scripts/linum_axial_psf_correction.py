#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Model-free axial PSF correction."""

import argparse

import numpy as np
import dask.array as da
from scipy.ndimage import convolve
from skimage.filters import threshold_otsu
from skimage.restoration import richardson_lucy
from linumpy.io.zarr import save_omezarr, read_omezarr
from linumpy.preproc.xyzcorr import findTissueInterface, maskUnderInterface
import zarr

import matplotlib.pyplot as plt


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input_zarr",
                   help="Path to file (.ome.zarr) containing the 3D mosaic grid.")
    p.add_argument("output_zarr",
                   help="Corrected 3D mosaic grid file path (.ome.zarr).")
    p.add_argument('--dont_mask_output', action='store_true',
                   help='Option for disabling masking of the corrected output.')
    p.add_argument('--n_levels', type=int, default=5,
                   help='Number of levels in pyramid representation.')
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

    # Generate mask Under interface
    mask = maskUnderInterface(vol[:], interface, returnMask=True)

    # Exclude out of bounds columns
    mask_all = mask.all(axis=0)  # True where mask is True for every voxel along the aline
    agarose_mask = np.logical_and(agarose_mask, ~mask_all)

    vol_data = vol[:]
    profile = np.reshape(vol_data, (len(vol_data), -1))
    profile = np.array([profile[i] for i in range(len(profile)) if agarose_mask.reshape(-1)[i]])
    print(profile.shape)
    profile = np.mean(profile, axis=-1)
    z = np.polyfit(np.arange(len(profile)), profile, deg=16)
    p = np.poly1d(z)
    xp = np.linspace(0, len(profile) - 1, 4*len(profile))
    profile_fit = p(xp)

    psf_max = np.max(profile_fit)
    psf_mu = np.argmax(profile_fit)
    half_max = psf_max / 2
    half_max_left = None
    half_max_right = None
    for i in range(psf_mu, 0, -1):
        if profile_fit[i] < half_max:
            half_max_left = xp[i]
            break
    for i in range(psf_mu, len(profile_fit)):
        if profile_fit[i] < half_max:
            half_max_right = xp[i]
            break
    if half_max_left is None or half_max_right is None:
        raise ValueError("Could not find half maximum in the PSF profile.")
    fwhm = half_max_right - half_max_left
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    psf = 1.0/(sigma*np.sqrt(2.0*np.pi))*np.exp(-((np.arange(len(profile)) - xp[psf_mu]) ** 2) / (2 * sigma ** 2))
    background = np.mean(profile)
    corr = convolve(vol_data, 1.0/psf.reshape((-1, 1, 1)), mode='constant')

    save_omezarr(da.from_array(corr), args.output_zarr, voxel_size=res,
                 chunks=vol.chunks, n_levels=args.n_levels)
    return
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2)

    ax[0].imshow(agarose_mask, cmap='gray')
    ax[0].set_title('Agarose mask')
    ax[1].plot(np.arange(len(profile)), profile)
    ax[1].plot(xp, profile_fit)
    ax[1].plot(x_psf, psf)
    ax[1].set_title('Agarose profile')
    plt.show()
    exit()

    # Extract the tile shape from the filename
    tile_shape = vol.chunks

    # otsu threshold for identifying agarose voxels
    bg = vol[:]
    bg = np.ma.masked_array(bg, mask[:] > 0)
    bg_curve = np.mean(bg, axis=(1, 2))

    temp_store = zarr.TempStore()
    vol_corr = zarr.open(temp_store, mode="w", shape=vol.shape,
                         dtype=np.float32, chunks=tile_shape)

    vol_corr[:] = vol[:]
    vol_corr[:] /= (np.reshape(bg_curve, (-1, 1, 1)))
    vol_corr[:] *= np.mean(bg_curve)
    if not args.dont_mask_output:
        vol_corr[:] = vol_corr[:] * mask[:]

    # save to ome-zarr
    dask_arr = da.from_zarr(vol_corr)
    save_omezarr(dask_arr, args.output_zarr, scales=res, chunks=tile_shape,
                 n_levels=args.n_levels)


if __name__ == "__main__":
    main()
