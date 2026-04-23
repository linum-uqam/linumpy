"""Extract PSF parameters (focal depth, Rayleigh length) from a stitched mosaic."""

import numpy as np
from scipy.ndimage import binary_dilation, binary_fill_holes, gaussian_filter
from scipy.stats import zscore
from skimage.filters import threshold_li
from skimage.morphology import disk

from linumpy.geometry.interface import find_tissue_interface
from linumpy.intensity.psf_model import confocal_psf, fit_tissue_confocal_model


# TODO: Fine-tune default values for 10x microscope or give heuristic
# for fixing them.
def extract_psf_parameters_from_mosaic(
    vol: np.ndarray,
    f: float = 0.01,
    n_profiles: int = 10,
    zr_0: float = 610.0,
    res: float = 6.5,
    n_iterations: int = 15,
) -> tuple[float, float]:
    """Compute the confocal PSF from a slice.

    Parameters
    ----------
    vol : ndarray
        A stitched tissue slice with axes in order (x, y, z).
    f : float
        Smoothing factor (in fraction of image size).
    n_profiles : int
        Number of intensity profile to use.
    zr_0 : float
        Initial Rayleigh length to use in micron (default=%(default)s for a 3X objective)
    res : float
        Z resolution (in micron).
    n_iterations : int
        Number of fitting iterations.

    Returns
    -------
    (2,) tuple
        Focal depth (zf) and Rayleigh length (zr) in micron

    """
    nx, ny, nz = vol.shape
    k = int(0.5 * f * (nx + ny))
    aip = vol.mean(axis=2)

    # Compute water-tissue interface
    interface = find_tissue_interface(vol).astype(int)

    # Compute the agarose mask with the li thresholding method
    thresh = threshold_li(aip)
    mask_tissue = binary_fill_holes(aip > thresh)
    mask_agarose = ~binary_fill_holes(binary_dilation(mask_tissue, disk(k)))
    mask_agarose[aip == 0] = 0
    del mask_tissue

    # Get min and max interface depth for the agarose
    zmin = np.percentile(interface[mask_agarose], 2.5)

    # Get the average iProfile / interface depth
    profile_per_interface_depth = np.zeros((n_profiles, nz))
    for ii in range(n_profiles):
        for z in range(nz):
            profile_per_interface_depth[ii, z] = np.mean(vol[:, :, z][mask_agarose * (interface == zmin + ii)])

    # Detect outliers
    i_profile_gradient = np.abs(gaussian_filter(profile_per_interface_depth, sigma=(0, 2), order=1))
    profile_mask = np.abs(zscore(i_profile_gradient, axis=1)) <= 1.0
    for ii in range(n_profiles):
        profile_mask[ii, 0 : int(zmin + ii)] = 0

    z = np.linspace(0, nz * res, nz)
    zf_list = []
    zr_list = []
    total_err = []
    for z0 in range(n_profiles):
        # Find the coarse alignment of the focus based on
        # pre-established Rayleigh length from thorlab
        err_list = []
        for zf in range(nz):
            a = profile_per_interface_depth[z0, zf]
            synthetic_signal = confocal_psf(z, zf, zr_0, a)
            err = np.abs(synthetic_signal - profile_per_interface_depth[z0, :])
            err = np.mean(err[profile_mask[z0, :]])
            err_list.append(err)

        err_list = np.array(err_list)
        zf = np.argmin(err_list) * res
        a = profile_per_interface_depth[z0, int(zf / res)]
        zr: float = float(zr_0)
        output: dict = {}

        if not (np.isnan(a)):
            last_zr = zr_0
            for _ in range(n_iterations):
                # Optimize the model (without using attenuation)
                i_profile = profile_per_interface_depth[z0, :]
                output = fit_tissue_confocal_model(
                    i_profile,
                    int(z0 + zmin),
                    last_zr,
                    res,
                    return_parameters=True,
                    return_full_model=True,
                    use_bump_model=True,
                )
                zf = output["parameters"]["zf"]
                zr = output["parameters"]["zr"]
                last_zr = zr

            zf_list.append(zf)
            zr_list.append(zr)
            err_fit = (output["tissue_psf"] - profile_per_interface_depth[z0, :]) ** 2.0
            total_err.append(np.mean(err_fit))

    min_err = np.argmin(total_err)
    zf_final = zf_list[min_err]
    zr_final = zr_list[min_err]

    return zf_final, zr_final
