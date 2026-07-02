"""Attenuation estimation and profile recovery models for OCT volumes."""

import itertools
import multiprocessing
from typing import Literal, overload

import numpy as np
import SimpleITK as sitk
from dipy.segment.mask import median_otsu
from scipy.interpolate import interp1d
from scipy.ndimage import (
    binary_fill_holes,
    gaussian_filter,
    median_filter,
    uniform_filter,
)
from scipy.optimize import minimize
from skimage.filters import threshold_li

from linumpy.cli.args import get_available_cpus
from linumpy.config.threads import worker_initializer
from linumpy.geometry.crop import mask_under_interface
from linumpy.geometry.interface import find_tissue_interface, get_interface_depth_from_mask
from linumpy.intensity.normalize import eqhist, normalize


def get_attenuation_vermeer2013(
    vol: np.ndarray, dz: float = 6.5e-6, mask: np.ndarray | None = None, C: np.ndarray | int | float | None = None
) -> np.ndarray:
    """Estimates the attenuation coefficient using the Vermeer2013 model.

    Parameters
    ----------
    vol : ndarray
        3D Reflectivity OCT data
    dz : float
        Axial resolution (in microns/pixel)
    mask : ndarray
        Tissue mask (optional). Every pixel above the mask will be attributed null attenuation,
        and pixel under the mask will be set to the last computed Aline attenuation.
    C: ndarray, int or float
        The bottom constant (DEV)

    Return
    ------
    ndarray
        Estimated attenuation coefficient map (same size as vol)

    Notes
    -----
    - This algorithm is inspired by an ultrasound attenuation compensation method.

    References
    ----------
    - Vermeer et al. Depth-resolved model-based reconstruction of attenuation
      coefficients in optical coherence tomography. Biomed. Opt. Exp., vol5,
      no1, pp332-337, 2013

    """
    # Prepare the bottom constant (to better consider the finite Bscan dimension)
    if C is None:
        C = np.zeros(vol.shape)
    elif isinstance(C, (int, float)):
        C = np.ones(vol, dtype=float) * C
    elif C.ndim == 2:
        C = np.tile(np.reshape(C, (C.shape[0], C.shape[1], 1)), (1, 1, vol.shape[2]))

    # Use mask to ignore tissue layers above / under the mask
    if mask is None:
        mask = np.ones_like(vol, dtype=bool)

    # Compensation profile with depth for each A-line
    vol_p = np.ma.masked_array(vol, ~mask)
    profile = np.cumsum(vol_p[:, :, ::-1], axis=-1) + C
    profile = profile[:, :, ::-1]

    # Estimating the attenuation coefficient for each A-line
    mu = np.zeros_like(vol)
    mu[profile > 0] = vol[profile > 0] / profile[profile > 0]
    mu[profile > 0] = np.log(1 + mu[profile > 0]) / (2.0 * dz)

    # Convert attenuation in cm-1
    mu /= 100.0

    # Masking the result
    if mask is not None:
        interface = get_interface_depth_from_mask(mask)
        mask_above_interface = ~mask_under_interface(mask, interface, return_mask=True)
        mu[mask_above_interface] = 0

        # Find bottom interface
        nx, ny, nz = mask.shape
        bottom_interface = (nz - get_interface_depth_from_mask(mask[:, :, ::-1]) - 1).astype(int)
        xx, yy = np.meshgrid(list(range(nx)), list(range(ny)), indexing="ij")
        bottom_mu = mu[xx, yy, bottom_interface]
        bottom_mu = np.tile(np.reshape(bottom_mu, (nx, ny, 1)), (1, 1, nz))
        bottom_mask = mask_under_interface(mask, bottom_interface, return_mask=True)
        mu[bottom_mask] = bottom_mu[bottom_mask]

    return mu


def get_extended_attenuation_vermeer2013(
    vol: np.ndarray,
    mask: np.ndarray | None = None,
    k: int = 10,
    sigma: int = 5,
    sigma_bottom: int = 3,
    dz: int = 1,
    res: float = 6.5,
    zshift: int = 3,
    fill_holes: bool = False,
) -> np.ndarray:
    """Compute the local effective tissue attenuation using the extended Vermeer model.

    Parameters
    ----------
    vol: ndarray
        OCT volume to process
    mask: ndarray
        Optional tissue mask. If none is given the water/tissue
        interface will be detected from the data.
    k: int
        Median filter kernel size (px) applied before the attenuation
        coefficient computation (applied in the XY direction).
    sigma : int
        Gaussian filter kernel size (px) applied axially before the
        exponential signal fit used to extend the Alines for the extended
        Vermeer signal evalution.
    sigma_bottom : int
        Gaussian filter kernel size (px) applied axially on the bottom slice
        signal before the extension fit.
    fill_holes : bool
        If True, fill holes in the tissue mask before computing attenuation.
    dz : int
        Number of axial pixel to consider when computing the bottom slice
        signal for the signal extension.
    res: float
        Axial resolution in micron / pixel
    zshift: int
        Number of pixel under the water-tissue interface to ignore while
        fitting the exponential function for signal extension.

    Returns
    -------
    ndarray
        Computed attenuation coefficients.
    """
    # First the slice is denoised with a small median filter
    if k > 0:
        vol = sitk.GetArrayFromImage(sitk.Median(sitk.GetImageFromArray(vol), (0, k, k)))

    # Computing tissue mask
    if mask is None:
        # Detecting the water / tissue interface
        interface = find_tissue_interface(vol, s_xy=3, s_z=1, order=1, use_log=True)
        mask = mask_under_interface(vol, interface + zshift, return_mask=True)

    # Lets fit an exponential function on each Aline to extend the tissue slice.
    exp_fit = get_gradient_attenuation(gaussian_filter(vol, (0, 0, sigma)))
    exp_fit = np.ma.masked_array(exp_fit, ~mask).mean(axis=2)

    # Fill holes left by NaN values
    exp_fit[np.isnan(exp_fit)] = 0

    # Get the signal at the interface for each Aline
    interface_bottom = vol.shape[2] - get_interface_depth_from_mask(mask[:, :, ::-1]) - 1 - dz
    mask_bottom = mask_under_interface(vol, interface_bottom, return_mask=True)
    mask_bottom = (mask_bottom * mask).astype(bool)
    i0 = np.ma.masked_array(vol, ~mask_bottom).mean(axis=2)

    # Compute the end-of-scan bias
    epsilon = 1e-3
    C = np.zeros_like(i0)
    C[exp_fit > epsilon] = i0[exp_fit > epsilon] / exp_fit[exp_fit > epsilon]
    C = gaussian_filter(C, sigma_bottom)

    # Compute the attenuation
    attn_cropped = get_attenuation_vermeer2013(vol, dz=res * 1e-06, mask=mask, C=C)

    # Remove NaN
    attn_cropped[np.isnan(attn_cropped)] = 0

    # Only keep attn within the mask
    attn_cropped[~mask.astype(bool)] = 0

    # Fill holes
    if fill_holes:
        attn_cropped = sitk.GetArrayFromImage(sitk.GrayscaleFillhole(sitk.GetImageFromArray(attn_cropped)))

    return attn_cropped


def get_attenuation_faber2004(
    vol: np.ndarray, mask: np.ndarray | None = None, dz: float = 6.5e-6, N: int = 4
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimates the attenuation coefficient using the Faber2004 model.

    Parameters
    ----------
    vol : ndarray
        3D Reflectivity OCT data
    mask : ndarray
        Tissue mask to control which points to use in each A-lines. (if None, the whole A-line is used)
    dz : float
        Axial resolution (in microns/pixel)
    N : int
        Size of the XY uniform filter used to average A-lines together

    Return
    ------
    ndarray
        Estimated attenuation coefficient map (computes a single mu_t per A-line)

    Notes
    -----
    - This algorithm uses the confocal PSF and an single-scattering photon model.
    - Assumes a 4X objective setup for now.

    References
    ----------
    - Faber et al. Quantitative measurement of attenuation coefficients of weakly
     scattering media using optical coherence tomography. Opt. Express 12, 4353–4365 (2004).
    """
    # Average the A-line together in the XY plane
    if N > 0:
        vol = uniform_filter(vol, size=(N, N, 0))

    # Computing the confocal psf parameters
    alpha = 2  # 1 : specular scattering; 2 : diffuse scattering
    n = 1.33  # Index of refraction
    w0 = 4.88e-6  # micron : Ligth beam width at the focal plane (need to validate)
    l0 = 1.030e-6  # micron : Ligth central wavelenth
    zr = np.pi * alpha * n * w0**2.0 / l0  # Apparent Rayleigh length (micron)

    # Defining the objective function for the minimization
    def f(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
        return np.sum((y - oct_signal_faber2004_model(z, mu_t=x[2], zR=x[1], z0=x[0])) ** 2.0)

    # Loop over all A-lines and computing attenuation
    attn = np.zeros((vol.shape[0], vol.shape[1]))
    r_length = np.zeros((vol.shape[0], vol.shape[1]))
    z_focus = np.zeros((vol.shape[0], vol.shape[1]))
    z = np.arange(0.0, dz * vol.shape[2], dz)
    for x in range(vol.shape[0]):
        for y in range(vol.shape[1]):
            mask_aline = mask[x, y, :] if mask is not None else np.ones((vol.shape[2],)).astype(bool)

            if np.any(mask_aline):
                p0 = [0.0, 100.0, 0.001]
                zp = np.where(mask_aline)[0][0]
                data = vol[x, y, :][mask_aline]
                data /= 1.0 * data.max()
                p_opt = minimize(
                    f,
                    p0,
                    args=(data, z[zp::] - z[zp]),
                    bounds=((None, None), (zr / 2.0, 2.0 * zr), (0.0, None)),
                )
                if p_opt.success:
                    attn[x, y] = p_opt.x[2]
                    r_length[x, y] = p_opt.x[1]
                    z_focus[x, y] = p_opt.x[0] + z[zp]
                else:
                    print(("No convergence for (x,y)=", x, y))

    return attn, r_length, z_focus


# Modele du signal utilisant la PSF confocale et single-scattering photons


def oct_signal_faber2004_model(z: np.ndarray, mu_t: float = 1.0, zR: float = 200.0, z0: float = 100.0) -> np.ndarray:
    """Model the oct signal using a single-scattered photons and the confocal PSF.

    Parameters
    ----------
    z : (N,) ndarray
        Depth Position along an A-line at which the signal must be computed (in micron)
    mu_t : float or (N,) ndarray
        Attenuation coefficient (either a single value for the whole column or N values for each position)
    zR : float
        Apparent Rayleigh length (in micron)
    z0 : float
        Focal plane depth (in micron)

    Returns
    -------
    ndarray
        OCT backscattering signal estimated at each z positions

    Notes
    -----
    - This is based on the single-scattering photon model from Faber2004.

    References
    ----------
    - https://www.osapublishing.org/oe/abstract.cfm?uri=oe-12-19-4353

    """
    psf = 1.0 / (((z - z0) / float(zR)) ** 2.0 + 1.0)
    iz = np.sqrt(psf * np.exp(-2 * mu_t * z))  # Should I fit the sqrt or not
    return iz


def _aline_fit(data: np.ndarray) -> float:
    """Aline fit to extract the attenuation coefficient."""

    # Defining the attenuation model (biexponential) with x=[A, mu_t], y=data, z=depths
    def f_attn(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
        return np.sum((y - x[0] * np.exp(-2 * x[1] * z)) ** 2.0)

    z = np.linspace(0, len(data), len(data))
    aline = np.array(data)
    p0 = [1.0, 0.001]  # Initial condition
    popt = minimize(f_attn, p0, args=(aline, z), bounds=((0, None), (0, None)))
    return popt.x[1]


def split_aline(data: np.ndarray, mask: np.ndarray) -> tuple[list, list]:
    """Split an A-line into contiguous masked segments."""
    data_list = []
    z_list = []
    this_aline = []
    this_z = []
    for elem, m, z in zip(data, mask, list(range(len(data))), strict=False):
        if m:
            this_aline.append(elem)
            this_z.append(z)
        else:
            if len(this_aline) > 0:
                data_list.append(this_aline)
                this_aline = []
                z_list.append(this_z)
                this_z = []
    if len(this_aline) > 0:
        data_list.append(this_aline)
        z_list.append(this_z)

    return data_list, z_list


def _split_alines_worker(param: tuple) -> tuple[list, list]:
    return split_aline(param[0], param[1])


def get_aline_attenuation(vol: np.ndarray, k: int = 1, mask: np.ndarray | None = None) -> np.ndarray:
    """Compute the attenuation coefficient for each A-lines.

    Parameters
    ----------
    vol : ndarray
        Volume to analyze of size NxMxL

    k : int
        Number of points to compute

    mask : ndarray
        Tissue mask to limit the region where the fit is done.

    Returns
    -------
    ndarray
        Estimated attenuation of size NxMxk

    """
    # Computing the shape of this volume
    nx, ny, nz = vol.shape
    z = np.arange(nz)
    z_list = np.array_split(z, k)
    attn_vol = np.zeros((nx, ny, k))

    for z, ik in zip(z_list, list(range(k)), strict=False):
        # Selecting a subsample
        this_vol = vol[:, :, z[0] : z[-1]]
        this_mask = mask[:, :, z[0] : z[-1]] if mask is not None else None

        # Transforming this volume into a list
        a_lines = np.split(this_vol.flatten(), nx * ny)
        if mask is not None and this_mask is not None:
            mask_a_lines = np.split(this_mask.flatten(), nx * ny)
            for A, M, ii in zip(a_lines, mask_a_lines, list(range(nx * ny)), strict=False):
                a_lines[ii] = A[M]

        # Process each Alines in parallel
        nproc = get_available_cpus()
        p = multiprocessing.Pool(nproc, initializer=worker_initializer)
        result = p.map(_aline_fit, a_lines)
        p.close()
        p.join()

        # Reshaping the output
        attn_vol[:, :, ik] = np.reshape(result, (nx, ny))
    # Removing background
    attn_vol[vol.mean(axis=2) == 0] = 0

    return np.squeeze(attn_vol)


def get_gradient_attenuation(
    vol: np.ndarray,
    mask: np.ndarray | None = None,
    return_mask: bool = False,
    low_thresh: float = 0.0,
    fill_holes: bool = False,
    sz: int = 3,
    sxy: int = 0,
    res: float = 1.0,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Compute the attenuation coefficient using the log-gradient method."""
    vol_l = np.copy(vol)
    vol_l[vol > 0] = np.log(vol[vol > 0])
    attn = -0.5 * np.gradient(vol_l, res, axis=2)

    # Removing 0 values and masked values
    attn[vol == 0] = 0
    if mask is not None:
        attn[~mask] = 0

    # Removing negative attenuation values
    mask_attn = attn < 0
    attn[mask_attn] = 0

    # Removing also small attenuation values
    small_mask = attn < np.percentile(attn[attn > 0], low_thresh)
    mask_attn[small_mask] = True
    attn[mask_attn] = 0

    # Filling holes using morphological reconstruction.
    if fill_holes:
        vol_itk = sitk.GetImageFromArray(attn)
        cfilter = sitk.GrayscaleMorphologicalClosingImageFilter()
        cfilter.SetKernelType(sitk.sitkBall)
        cfilter.SetKernelRadius((sz, sxy, sxy))

        output_itk = cfilter.Execute(vol_itk)
        attn = sitk.GetArrayFromImage(output_itk)

    if return_mask:
        return attn, mask_attn
    else:
        return attn


def get_interface_mask(
    vol: np.ndarray, s: int = 0, mask_tissue: bool = True, mask_water_tissue_interface: bool = True
) -> np.ndarray:
    """Compute a mask excluding tissue interfaces and boundaries."""
    nx, ny, nz = vol.shape
    mask = np.ones_like(vol).astype(bool)

    # Get tissue mask
    if mask_tissue:
        tissue_mask = binary_fill_holes(median_otsu(eqhist(vol.mean(axis=2)), median_radius=5.0)[1])
        tissue_mask = np.tile(np.reshape(tissue_mask, (nx, ny, 1)), (1, 1, nz))
        mask *= tissue_mask

    # Get water/tissue 3D interface mask
    if mask_water_tissue_interface:
        # Computing the gradient
        vol_f = median_filter(vol, 5)
        gradient = np.gradient(vol_f)
        gm = gradient[0] ** 2.0 + gradient[1] ** 2.0 + gradient[2] ** 2.0
        gm = normalize(gm, high_thresh=99.5)

        # Thresholding the gradient
        thresh = threshold_li(gm)
        interfaces_g = gm >= thresh

        # Converting this into a water/tissue interface depth
        depths = np.zeros((nx, ny))
        for x, y in itertools.product(list(range(nx)), list(range(ny))):
            idx = np.where(interfaces_g[x, y, :])
            if len(idx[0]) > 0:
                depths[x, y] = idx[0][0]

        water_tissue_mask = mask_under_interface(vol, depths, return_mask=True)
        mask *= water_tissue_mask

    # Smoothing the volume
    if s > 0:
        vol = gaussian_filter(vol, sigma=(s, s, s))

    # Get tissue interface boundaries using a Canny filter
    vol_itk = sitk.GetImageFromArray(vol)
    canny_filter = sitk.CannyEdgeDetectionImageFilter()
    edges = sitk.GetArrayFromImage(canny_filter.Execute(vol_itk)).astype(bool)
    mask *= ~edges

    return mask


def find_interface_from_gradient(vol: np.ndarray, f: float = 0.005, remove_smooth: bool = False) -> np.ndarray:
    """Find the tissue-water interface depth map using gradient magnitude."""
    nx, ny = vol.shape[0:2]
    k = int(np.round(f * 0.5 * (nx + ny)))

    # Computing the gradient
    vol_f = gaussian_filter(vol, k)

    gradient = np.gradient(vol_f)
    gm = gradient[0] ** 2.0 + gradient[1] ** 2.0 + gradient[2] ** 2.0
    gm = normalize(gm, high_thresh=99.5)

    # Thresholding the gradient
    thresh = threshold_li(gm)
    interfaces_g = gm >= thresh

    # Converting this into a water/tissue interface depth
    depths = np.argmax(interfaces_g, axis=2)

    if remove_smooth:
        depths += k + 1

    return depths


def get_heterogeneous_attenuation(
    vol: np.ndarray, mask: np.ndarray | None = None, fill_holes: bool = False
) -> np.ndarray:  # TODO: adapt multiproc to available proc given by mpi4py
    """Compute heterogeneous attenuation coefficients for each A-line segment."""
    nx, ny, nz = vol.shape
    nproc = get_available_cpus()
    if mask is None:  # Compute the mask
        mask = get_interface_mask(vol)

    # Split the volume into Alines and Alines portions.
    print(f"Splitting volume into Alines portions (using {nproc} processors)")
    a_lines = np.split(vol.flatten(), nx * ny)
    a_lines_mask = np.split(mask.flatten(), nx * ny)
    a_lines_to_split = list(zip(a_lines, a_lines_mask, strict=False))
    n_alines = len(a_lines)

    # Process each Alines in parallel
    p = multiprocessing.Pool(nproc, initializer=worker_initializer)
    result = p.map(_split_alines_worker, a_lines_to_split)
    p.close()
    p.join()

    # Debug
    print(("Number of Alines : ", len(a_lines_to_split)))
    p_count = 0
    for portion in result:
        p_count += len(portion[0])
    print(("Number of Alines portions : ", p_count))

    # Compute the attenuation for each aline portions
    print(f"Computing attenuation for each Aline portion (using {nproc} processors)")
    aline_portions = []
    z_portions = []
    portion_idx = []
    for foo, idx in zip(result, list(range(n_alines)), strict=False):
        aline_portions.extend(foo[0])
        z_portions.extend(foo[1])
        portion_idx.extend([idx] * len(foo[0]))

    p = multiprocessing.Pool(nproc, initializer=worker_initializer)
    result = p.map(_aline_fit, aline_portions)
    p.close()
    p.join()

    # Reshape attenuation as an Aline list  # TODO: Parallelize this loop.
    print("Reshape attenuation as an Aline list")
    aline_attn = [np.zeros((nz,)) for i in range(n_alines)]
    for idx, z, mu in zip(portion_idx, z_portions, result, strict=False):
        aline_attn[idx][z] = mu

    # Combine local attenuations into a single volume
    attn_vol = np.reshape(aline_attn, (nx, ny, nz))

    # Fill attenuation holes (using 1d-linear interpolation) # TODO: User inverve distance weighted interpolation instead ?
    if fill_holes:
        print("Filling attenuation holes using 1D-linear interpolation")
        attn_fill = np.zeros_like(attn_vol)
        for x in range(nx):
            for y in range(ny):
                idx = attn_vol[x, y, :] > 0
                this_z = np.array(np.where(idx)[0])
                this_mu = attn_vol[x, y, :]
                this_mu = this_mu[idx]
                if np.sum(idx) == 1:
                    attn_fill[x, y, :] = attn_vol[x, y, :]
                elif np.sum(idx) > 1:
                    f = interp1d(this_z, this_mu, kind="linear", bounds_error=False, fill_value=0)
                    new_z = np.arange(nz)
                    new_mu = f(new_z)
                    attn_fill[x, y, :] = new_mu  # Debug, should replace the original volume

        return attn_fill
    else:
        return attn_vol


@overload
def get_flat_agarose_profile(vol: np.ndarray, return_mask_and_profile: Literal[False] = ...) -> np.ndarray: ...
@overload
def get_flat_agarose_profile(
    vol: np.ndarray, return_mask_and_profile: Literal[True]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...


def get_flat_agarose_profile(
    vol: np.ndarray, return_mask_and_profile: bool = False
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate and remove the flat agarose intensity profile from a volume."""
    nx, ny, nz = vol.shape

    # Get agarose mask for this slice and its intensity profile
    tissue_mask = binary_fill_holes(median_otsu(eqhist(vol.mean(axis=2)), median_radius=5.0)[1])
    tissue_mask = np.tile(np.reshape(tissue_mask, (nx, ny, 1)), (1, 1, nz))
    mask = (~tissue_mask).astype(bool)

    # Computing intensity profile for the agarose Alines
    i_profile = np.zeros((nz,))
    n_alines = np.sum(mask.max(axis=2))
    for x in range(nx):
        for y in range(ny):
            this_mask = mask[x, y, :]
            a_line = vol[x, y, :]
            if np.any(this_mask):
                i_profile += a_line / float(n_alines)

    # Correction profile
    agarose_norm = np.tile(np.reshape(i_profile, (1, 1, nz)), (nx, ny, 1))
    vol_p = vol / agarose_norm.astype(float)

    if return_mask_and_profile:
        return vol_p, mask, i_profile
    else:
        return vol_p


def get_signal_from_attenuation(
    attn: np.ndarray, i0: np.ndarray | None = None, nz: int = 120, mask: np.ndarray | None = None, res: float = 1.0
) -> np.ndarray:
    """Estimate the signal from the 2D A-Line attenuation map.

    Parameters
    ----------
    attn : ndarray
        2D attenuation coefficient map for each A-line
    i0 : ndarray
        2D map of the top slice signal (optional)
    nz : int
        Number of Aline points
    mask : ndarray
        Data mask (to limit where the oct signal is simulated)
    res : float
        Axial resolution in micron / pixel

    Returns
    -------
    ndarray
        Estimated OCT signal

    """
    nx, ny = attn.shape
    attn_vol = np.zeros((nx, ny, nz))

    def f_attn(x: float, z: np.ndarray) -> np.ndarray:
        return np.exp(-2 * x * z)

    for ix, iy in itertools.product(list(range(nx)), list(range(ny))):
        this_mask = mask[ix, iy, :].astype(bool) if mask is not None else np.ones((nz,), dtype=bool)

        z0 = np.where(this_mask)
        z0 = z0[0][0] if len(z0[0]) > 0 else 0

        A = i0[ix, iy] if i0 is not None else 1

        if np.any(this_mask):
            this_mu = attn[ix, iy]
            z = np.linspace(0, nz * res, nz) - z0 * res
            sim_profile = A * f_attn(this_mu, z)
            attn_vol[ix, iy, this_mask] = sim_profile[this_mask]

    return attn_vol
