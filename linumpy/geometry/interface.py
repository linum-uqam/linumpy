"""Tissue interface detection and fitting."""

import itertools
from typing import Literal, overload

import numpy as np
from scipy.ndimage import (
    binary_fill_holes,
    gaussian_filter1d,
    gaussian_gradient_magnitude,
    label,
    uniform_filter,
)
from scipy.optimize import curve_fit
from scipy.signal import argrelmax
from skimage.filters import threshold_li
from skimage.morphology import dilation, disk


def find_tissue_depth(vol: np.ndarray, zmin: int = 15, zmax: int = 100, agarose_intensity: int = 5000) -> int:
    """Detect the tissue interface depth in given volume.

    This algorithm first segments the volume into tissue vs background(agarose) using
    the Li thresholding method and user-defined agarose intensity value. It then
    computes the XZ mean projection of the tissue mask, and detects the main edge
    position of the tissue/water interface using morphological operations and relative
    maximum detection. Other operations are done on the data to reduce the effect
    of intensity noise and artefacts on the water/tissue interface depth detection.

    Parameters
    ----------
    vol : ndarray
        Volume to analyze
    zmin : int
        Minimum depth of interface in pixel
    zmax : int
        Maximum depth of interface in pixel
    agarose_intensity : int
        Agarose mean intensity value used to restrict analysis to tissue voxels

    Returns
    -------
    int
        Tissue interface depth

    Notes
    -----
    * The default depth is 40 px

    """
    z0 = 0  # Default value

    try:
        nx, ny, nz = vol.shape

        # Removing agarose (threshold selected empirically)
        mip_agarose_mask = np.mean(vol[:, :, zmin:zmax], axis=2) < agarose_intensity
        mip_agarose_mask = binary_fill_holes(mip_agarose_mask)
        mip_agarose_mask = np.reshape(mip_agarose_mask, (nx, ny, 1))
        agarose_mask = np.tile(mip_agarose_mask, [1, 1, nz])

        mask = vol > threshold_li(vol[:, :, zmin:zmax])
        mask[agarose_mask] = 0  # This is mostly background/agarose pixels.
        im = mask.max(axis=1)
        im = binary_fill_holes(im)

        # Labeling features and keeping the largest
        im_label, num_features = label(im)
        hist = [np.sum(im_label == i) for i in range(num_features)]
        main_feature = np.argmax(hist[1:]) + 1
        im[im_label != main_feature] = 0

        # Find edges based on morphological dilation
        edges = dilation(im, disk(3)) - im
        edges[:, 0:zmin] = 0  # We don't want top slices
        edges[:, zmax : edges.shape[1]] = 0  # We don't want bottom slices either
        z_profile = edges.sum(axis=0)
        peaks = argrelmax(z_profile, order=20)
        if len(peaks[0]) > 0:
            z0 = peaks[0][0]
    except Exception:
        pass
    return z0


def get_interface_depth_from_mask(vol: np.ndarray) -> np.ndarray:
    """Compute the interface depths from a 3D tissue mask.

    Parameters
    ----------
    vol : (NxMxK) ndarray
        Tissue mask

    Returns
    -------
    ndarray : (NxM)
        Interface depth (in pixel)

    """
    nx, ny, _ = vol.shape
    depths = np.zeros((nx, ny))
    for x, y in itertools.product(list(range(nx)), list(range(ny))):
        idx = np.where(vol[x, y, :])
        if len(idx[0]) > 0:
            depths[x, y] = idx[0][0]

    return depths


def find_tissue_interface(
    vol: np.ndarray,
    s_xy: int = 15,
    s_z: int = 2,
    use_log: bool = True,
    mask: np.ndarray | None = None,
    order: int = 1,
    detect_cutting_errors: bool = False,
) -> np.ndarray:
    """Detect the tissue interface.

    Parameters
    ----------
    vol : ndarray
        Containing the volume to analyze
    s_xy : int
        Uniform filter kernel size (xy)
    s_z : int
        1st order gaussian kernel size (z)
    use_log : bool
        If True, apply log transform before filtering.
    mask : ndarray, optional
        Optional mask restricting the analysis region.
    order : int
        Gaussian filter order.
    detect_cutting_errors : bool
        If True, detect and correct cutting artefacts.

    Returns
    -------
    ndarray
        Tissue interface depth

    """
    if use_log:
        vol_p = np.copy(vol)
        vol_p[vol > 0] = np.log(vol[vol > 0])
    else:
        vol_p = vol
    vol_p = uniform_filter(vol_p, (s_xy, s_xy, 0))
    if mask is not None:
        vol_g = np.zeros_like(vol_p)
        for x in range(vol_p.shape[0]):
            for y in range(vol_p.shape[1]):
                mask_aline = mask[x, y, :]
                aline = vol_p[x, y, :]
                vol_g[x, y, mask_aline] = gaussian_filter1d(aline[mask_aline], s_z, order=order)
    else:
        vol_g = gaussian_filter1d(vol_p, s_z, order=order)
    z0 = np.ceil(vol_g.argmax(axis=2) + s_z * 0.5).astype(int)

    # Check if tissue begins before the FOV
    if detect_cutting_errors:
        vol_p = gaussian_filter1d(vol_p, s_z, order=0)
        z0_p = np.abs(vol_p).argmax(axis=2)
        mask_max = z0_p < z0
        z0[mask_max] = z0_p[mask_max]

    return z0


def find_cutting_plane(
    vol: np.ndarray, z0map: np.ndarray, agarose_mean: float, agarose_std: float
) -> tuple[np.ndarray, np.ndarray, float]:
    """Find the cutting plane using agarose segmentation.

    Parameters
    ----------
    vol : ndarray
        Input volume.
    z0map : ndarray
        Interface depth map.
    agarose_mean : float
        Mean intensity of agarose.
    agarose_std : float
        Standard deviation of agarose intensity.

    Returns
    -------
    popt
    detected_interface : ndarray
    z0 : int

    """
    # Computing agarose mask
    mask_tissue = vol >= agarose_mean + 3 * agarose_std
    mask_tissue = binary_fill_holes(mask_tissue)

    # Removing zero background
    agarose_mask = ~mask_tissue
    agarose_mask[vol == 0] = 0
    agarose_mask = agarose_mask.astype(bool)

    # Removing z0 outliers
    z0_median = np.median(z0map[agarose_mask])
    z0_mad = np.median(np.abs(z0map[agarose_mask] - z0_median))
    if z0_mad != 0:
        z0_zscore = np.abs(0.6745 * (z0map - z0_median) / z0_mad)
        z0_outliers = z0_zscore > 3.5
        agarose_mask[z0_outliers] = 0

    xdata = np.where(agarose_mask)
    ydata = z0map[agarose_mask][:]

    popt, _ = curve_fit(_plane, xdata, ydata)

    # Getting surface fit array
    xx, yy = np.meshgrid(list(range(vol.shape[0])), list(range(vol.shape[1])), indexing="ij")
    detected_interface = xx * popt[0] + yy * popt[1] + popt[2]

    # Choosing z range for stitching
    # Making sure we are 5*6.5 = 32.5 microns below the interface
    z0 = np.round(detected_interface.max()) + 5

    return popt, detected_interface, z0


# Fitting plane on agarose z0 values


def _plane(pos: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    x = pos[0]
    y = pos[1]
    return a * x + b * y + c


def remove_z0_outliers(z0map: np.ndarray) -> np.ndarray:
    """Remove outlier interface depths from the z0 map using median absolute deviation."""
    data = np.ravel(z0map[0, 0, :])
    # Median depth
    med = np.median(data)

    # Median absolute deviation
    mad = np.median(np.abs(data - med))

    if mad != 0:
        d_zscore = np.abs(0.6745 * (data - med) / mad)
        outliers = d_zscore > 3.0  # was 3.5

        # Replacing outliers by median depth
        z0map[:, :, outliers] = np.median(data)

        # Printing outliers for information
        print(("Z0 outliers were removed for the slices : ", np.where(outliers)[0]))

        return z0map
    else:
        print("MAD = 0. No outliers")
        return z0map


@overload
def fit_interface(interface: np.ndarray, method: str = ..., return_center: Literal[False] = ...) -> np.ndarray: ...
@overload
def fit_interface(
    interface: np.ndarray, method: str = ..., return_center: Literal[True] = ...
) -> tuple[np.ndarray, tuple[float, float]]: ...


def fit_interface(
    interface: np.ndarray, method: str = "linear", return_center: bool = False
) -> np.ndarray | tuple[np.ndarray, tuple[float, float]]:
    """Fit a model on the given interface.

    Parameters
    ----------
    interface : ndarray
        Interface depth map to fit.
    method : str
        Fitting method: 'linear', 'quad', 'gauss', or 'sph'.
    return_center : bool
        If True, also return the center of the fitted surface.
    """
    xdata = np.where(interface)
    ydata = np.ravel(interface)
    xx, yy = np.meshgrid(list(range(interface.shape[0])), list(range(interface.shape[1])), indexing="ij")
    fitted_interface: np.ndarray = np.zeros_like(interface)
    center: tuple = (0.0, 0.0)
    if method == "linear":
        popt, _ = curve_fit(_plane, xdata, ydata)

        # Getting surface fit array
        fitted_interface = xx * popt[0] + yy * popt[1] + popt[2]
        center = (interface.shape[0] / 2, interface.shape[1] / 2)
    elif method == "quad":
        popt, _ = curve_fit(quadratic_interface, xdata, ydata)

        a, b, c, d, e, f, g, h = popt
        xx = xx - g
        yy = yy - h
        fitted_interface = a * xx + b * yy + c * xx * yy + d * xx**2 + e * yy**2 + f
        center = (g, h)

    elif method == "gauss":

        def f(x: np.ndarray, a: float, b: float, c: float, d: float, e: float, f: float) -> np.ndarray:
            return np.exp(-((x[0] - a) ** 2) / (2.0 * b**2.0) - (x[1] - c) ** 2 / (2.0 * d**2.0)) * e + f

        popt, _ = curve_fit(f, xdata, ydata)
        a, b, c, d, e, f = popt
        fitted_interface = np.exp(-((xx - a) ** 2) / (2.0 * b**2.0) - (yy - c) ** 2 / (2.0 * d**2.0)) * e + f
        center = (a, c)

    elif method == "sph":

        def f(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
            return c * (((x[0] - a) ** 2 + (x[1] - b) ** 2) ** 2.0) / 8.0

        popt, _ = curve_fit(f, xdata, ydata)
        fitted_interface = popt[2] * (((xx - popt[0]) ** 2 + (yy - popt[1]) ** 2) ** 2.0) / 8.0
        center = (popt[0], popt[1])

    if return_center:
        return fitted_interface, center
    else:
        return fitted_interface


# Quadratic model for interface fit


def quadratic_interface(
    pos: np.ndarray, a: float, b: float, c: float, d: float, e: float, f: float, g: float, h: float
) -> np.ndarray:
    """Evaluate a quadratic surface model for the tissue interface."""
    x = pos[0] - g
    y = pos[1] - h
    return a * x + b * y + c * x * y + d * x**2 + e * y**2 + f


def get_quadratic_interface(popt: np.ndarray, volshape: tuple[int, int, int] = (512, 512, 120)) -> np.ndarray:
    """Compute the tissue interface map from quadratic fit parameters."""
    xx, yy = np.meshgrid(list(range(volshape[0])), list(range(volshape[1])), indexing="ij")
    tmp = quadratic_interface(np.array([xx[:], yy[:]]), popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7])
    interface = np.zeros([volshape[0], volshape[1]])
    interface[xx[:], yy[:]] = tmp
    return interface


def linear_homogeneous_profile(z: np.ndarray, z0: float, dz: float, I0: float, Ib: float, sigma: float) -> np.ndarray:
    """Intensity profile based on a single homogeneous tissue Beer-Lambert model (covered by some amount of water).

    This will return the log(I).

    Parameters
    ----------
    z : ndarray
        Position where the intensity is evaluated
    z0 : float
        Water-tissue interface depth
    dz : float
        Interface Transition width
    I0 : float
        Top tissue slice intensity (physics notation)
    Ib : float
        Water intensity (physics notation)
    sigma : float
        Tissue Attenuation coefficient

    Returns
    -------
    ndarray
        Log(I) evaluated at each position z.

    """
    z_underz0 = z < z0 - dz
    z_betweenz0 = (z >= z0 - dz) * (z < z0)
    z_overz0 = z >= z0
    log_intensity = np.zeros((len(z),))
    log_intensity[z_underz0] = Ib
    log_intensity[z_betweenz0] = (I0 - Ib) / (1.0 * dz) * (z[z_betweenz0] - (z0 - dz)) + Ib
    log_intensity[z_overz0] = I0 - sigma * (z[z_overz0] - z0)
    return log_intensity


def estimate_lh_profile_parameters(
    vol: np.ndarray, s: int = 25
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimates the linear-homogeneous intensity profile parameters.

    Parameters
    ----------
    vol : ndarray
        Volume for which the LHP parameters are evaluated
    s : int
        Neighborhood used to average intensities at each depth

    Returns
    -------
    float
        z0 : Water-tissue interface depth
    float
        dz : Interface Transition width
    float
        I0 : Top tissue slice intensity
    float
        Ib : water intensity
    float
        sigma : Tissue Attenuation coefficient

    Note
    ----
    * This first version loops over all intensity profiles (x, y)

    """
    nx, ny, _ = vol.shape
    vol_p = np.log(vol + 1.1)  # 1.1 factor is to prevent log of 0
    vol_p = uniform_filter(vol_p, (s, s, 0))  # Averaging intensities over a small XY neigborhood
    vol_f = gaussian_filter1d(vol_p, sigma=1, axis=2)  # Smoothing the intensity profiles in Z
    vol_g = gaussian_gradient_magnitude(vol_p, [0, 0, 1])  # TODO: Computing gradient in z direction only ?

    # Finding max gradient position
    z0 = vol_g.argmax(axis=2)

    xx, yy = np.meshgrid(list(range(nx)), list(range(ny)), indexing="ij")
    vol_p[xx, yy, z0]

    test = np.zeros(vol_p.shape)
    test[xx, yy, z0] = 1
    import nibabel as nib

    nib.save(
        nib.Nifti1Image(test, np.eye(4)),
        "/home/local/LIOM/jlefebvre/tmp/interface_test.nii",
    )

    # Preparing variables
    z0 = np.zeros((nx, ny), dtype=np.uint)
    dz = np.zeros((nx, ny), dtype=np.uint)
    I0 = np.zeros((nx, ny))
    Ib = np.zeros((nx, ny))
    sigma = np.zeros((nx, ny))

    for x in range(nx):  # TODO: Accelerate this loop (multithreading ?)
        for y in range(ny):
            profile = vol_p[x, y, :]
            If = vol_f[x, y, :]
            I_g = np.gradient(If)

            this_z0 = np.where(I_g == I_g.max())[0][0]
            I_gm = profile[this_z0]
            indices = np.where(I_g / I_gm < 0.1)
            zlist_min = indices[0][indices[0] < this_z0]
            zlist_max = indices[0][indices[0] > this_z0]

            this_dz = zlist_max[0] - zlist_min[-1] if len(zlist_min) > 0 and len(zlist_max) > 0 else 1
            if len(zlist_max) > 0:
                this_z0 = zlist_max[0]

            zmax = np.where(If == If.max())[0][0]
            if zmax - this_z0 < -5:
                this_z0 = zmax

            this_I0 = profile[this_z0]
            this_sigma = -np.median(I_g[this_z0::])

            this_Ib = 1 if this_z0 == 0 or this_z0 - this_dz <= 0 else np.median(profile[0 : this_z0 - this_dz])

            z0[x, y] = this_z0
            dz[x, y] = this_dz
            I0[x, y] = this_I0
            Ib[x, y] = this_Ib
            sigma[x, y] = this_sigma

    return z0, dz, I0, Ib, sigma
