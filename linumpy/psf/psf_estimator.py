import numpy as np
from linumpy.preproc.xyzcorr import findTissueInterface
from linumpy.preproc.icorr import confocalPSF, fit_TissueConfocalModel

from scipy.ndimage import binary_dilation, binary_fill_holes, gaussian_filter
from scipy.stats import zscore
from skimage.filters import threshold_li
from skimage.morphology import disk

def extract_psfParametersFromMosaic(
    vol, f=0.01, nProfiles=10, zr_0=610.0, res=6.5, nIterations=15
):
    """Computes the confocal PSF from a slice

    Parameters
    ----------
    vol : ndarray
        A stitched tissue slice with axes in order (x, y, z).
    f : float
        Smoothing factor (in fraction of image size).
    nProfiles : int
        Number of intensity profile to use.
    zr_0 : float
        Initial Rayleigh length to use in micron (default=%(default)s for a 3X objective)
    res : float
        Z resolution (in micron).

    Returns
    -------
    (2,) tuple
        Focal depth (zf) and Rayleigh length (zr) in micron

    """

    nx, ny, nz = vol.shape
    k = int(0.5 * f * (nx + ny))
    aip = vol.mean(axis=2)

    # Compute water-tissue interface
    interface = findTissueInterface(vol).astype(int)

    # Compute the agarose mask with the li thresholding method
    thresh = threshold_li(aip)
    mask_tissue = binary_fill_holes(aip > thresh)
    mask_agarose = ~binary_fill_holes(binary_dilation(mask_tissue, disk(k)))
    mask_agarose[aip == 0] = 0
    del mask_tissue

    # Get min and max interface depth for the agarose
    zmin = np.percentile(interface[mask_agarose], 2.5)

    # Get the average iProfile / interface depth
    profilePerInterfaceDepth = np.zeros((nProfiles, nz))
    for ii in range(nProfiles):
        for z in range(nz):
            profilePerInterfaceDepth[ii, z] = np.mean(
                vol[:, :, z][mask_agarose * (interface == zmin + ii)]
            )

    # Detect outliers
    iProfile_gradient = np.abs(
        gaussian_filter(profilePerInterfaceDepth, sigma=(0, 2), order=1)
    )
    profile_mask = np.abs(zscore(iProfile_gradient, axis=1)) <= 1.0
    for ii in range(nProfiles):
        profile_mask[ii, 0 : int(zmin + ii)] = 0

    z = np.linspace(0, nz * res, nz)
    zf_list = list()
    zr_list = list()
    total_err = list()
    for z0 in range(nProfiles):
        # Find the coarse alignment of the focus based on pre-established Rayleigh length from thorlab)
        errList = list()
        for zf in range(nz):
            a = profilePerInterfaceDepth[z0, zf]
            synthetic_signal = confocalPSF(z, zf, zr_0, a)
            err = np.abs(synthetic_signal - profilePerInterfaceDepth[z0, :])
            err = np.mean(err[profile_mask[z0, :]])
            errList.append(err)

        errList = np.array(errList)
        zf = np.argmin(errList) * res
        a = profilePerInterfaceDepth[z0, int(zf / res)]

        if not (np.isnan(a)):
            last_zr = zr_0
            for _ in range(nIterations):
                # Optimize the model (without using attenuation)
                iProfile = profilePerInterfaceDepth[z0, :]
                output = fit_TissueConfocalModel(
                    iProfile,
                    int(z0 + zmin),
                    last_zr,
                    res,
                    returnParameters=True,
                    return_fullModel=True,
                    useBumpModel=True,
                )
                zf = output["parameters"]["zf"]
                zr = output["parameters"]["zr"]
                last_zr = zr

            zf_list.append(zf)
            zr_list.append(zr)
            err_fit = (output["tissue_psf"] - profilePerInterfaceDepth[z0, :]) ** 2.0
            total_err.append(np.mean(err_fit))

    min_err = np.argmin(total_err)
    zf_final = zf_list[min_err]
    zr_final = zr_list[min_err]

    return zf_final, zr_final


def get_3dPSF(zf, zr, res, volshape):
    """Generate a 3D PSF based on Gaussian beam parameters.

    Parameters
    ----------
    zf : float
        Focal depth in microns
    zr : float
        Rayleigh length in microns
    res : float
        Axial resolution in micron / pixel
    volshape : (3,) list of int
        Output volume shape in pixel

    Returns
    -------
    ndarray
        3D PSF of shape 'volshape'
    """
    # TODO: Invert axes to agree with OME-zarr convention?
    nx, ny, nz = volshape[0:3]
    z = np.linspace(0, res * nz, nz)
    psf = confocalPSF(z, zf, zr)
    psf = np.tile(np.reshape(psf, (1, 1, nz)), (nx, ny, 1))

    return psf
