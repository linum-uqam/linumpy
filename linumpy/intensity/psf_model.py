"""Confocal PSF model fitting and volume-based PSF estimation."""

import contextlib
from typing import Any

import numpy as np
from scipy.ndimage import (
    binary_erosion,
    gaussian_filter,
    gaussian_filter1d,
)
from scipy.optimize import curve_fit, minimize
from sklearn import linear_model

from linumpy.mosaic.overlap import get_overlap


def get_average_volume(data: Any, z: int, mask: np.ndarray | None = None, s: int = 0) -> np.ndarray:
    """Compute an average volume for a specific slice.

    Parameters
    ----------
    data : Any
        The dataset for which the average volume is computed.
    z : int
        Slice number.
    mask : ndarray, optional
        Mask specifying which volume contributes to the computation.
    s : int, optional
        Smoothing kernel size in pixel, by default 0 (no smoothing).

    Returns
    -------
    ndarray
        Average volume.

    """
    average_vol = np.zeros(data.volshape, dtype=data.format)
    if mask is not None:
        n_vol = mask[:, :, z - 1].sum()
    else:
        nx, ny, nz = data.gridshape[:]
        n_vol = nx * ny
        mask = np.ones((nx, ny, nz))

    # Loop over volumes
    for vol in data.slice_iterator(z, mask=mask):
        if s > 0:  # Smoothing volume in xy
            vol = gaussian_filter(vol, sigma=(s, s, 0))

        average_vol += vol / (1.0 * n_vol)

    return average_vol


def find_focal_depth(vol: np.ndarray) -> int:
    """Detect the focal plane depth in a volume.

    Parameters
    ----------
    vol : ndarray
        Volume in which the focal plane is detected

    Returns
    -------
    int
        The focal plane depth

    """
    # Averaging intensity slice-by-slice
    intensity_profile = np.mean(np.mean(vol, axis=0), axis=0)

    # Focal plane depth
    fz = np.argmax(intensity_profile)

    return int(fz)


def i_profile_piece_wise_model(
    z: np.ndarray, I0: float, Imax: float, z0: float, zf: float, s: float, mu: float, k: float
) -> np.ndarray:
    """Evaluate the piece-wise OCT intensity profile model."""
    i_profile = np.zeros(z.shape)
    z1 = z <= z0
    z2 = (z > z0) * (z <= zf)
    z3 = z > zf

    # Water above tissue
    i_profile[z1] = I0 * np.exp(-k * z[z1])

    # Tissue -> Focal plane area
    i_profile[z2] = Imax * np.exp(-((z[z2] - zf) ** 2) / s**2)

    # Attenuation area
    i_profile[z3] = Imax * np.exp(-mu * (z[z3] - zf))

    return i_profile


def glm_volume_normalization(vol: np.ndarray, average_vol: np.ndarray) -> np.ndarray:
    """Volume intensity normalization using GLM fit.

    Parameters
    ----------
    vol : ndarray
        Volume to normalize

    average_vol : ndarray
        Average volume representing the background or objective profile without tissue

    Returns
    -------
    ndarray
        Normalized volume

    """
    nx, ny, nz = vol.shape
    vol_p = np.zeros_like(vol)

    # Preparing the input variables
    X = np.zeros((nx * ny * nz, 4))
    xx, yy, zz = np.meshgrid(list(range(nx)), list(range(ny)), list(range(nz)), indexing="ij")
    X[:, 0] = np.reshape(xx, (nx * ny * nz,))
    X[:, 1] = np.reshape(yy, (nx * ny * nz,))
    X[:, 2] = np.reshape(zz, (nx * ny * nz,))
    X[:, 3] = np.reshape(average_vol, (nx * ny * nz,))
    y = np.reshape(vol, (nx * ny * nz,))

    regr = linear_model.BayesianRidge()
    regr.fit(X, y)
    meanv_p = np.reshape(regr.predict(X), (nx, ny, nz))
    vol_p = vol / meanv_p

    return vol_p


def volume_normalization(vol: np.ndarray, average_vol: np.ndarray, epsilon: float = 0.05) -> np.ndarray:
    """Volume intensity normalization.

    Parameters
    ----------
    vol : ndarray
        Volume to normalize

    average_vol : ndarray
        Average volume representing the background or objective profile without tissue

    epsilon: float (optional, 0 < epsilon < 1)
        Small constant to prevent zero-division

    Returns
    -------
    ndarray
        Normalized volume.

    """
    # Adjust volume intensity to be in range [0,1]
    meanv = average_vol

    # Apply normalization (division)
    vol = vol / (meanv + epsilon) * meanv.mean()  #

    return vol


# Defining the radiometric transformation function


def T_r(p: np.ndarray, x: int | np.ndarray, y: int | np.ndarray) -> np.ndarray:
    """Radiometric transformation function.

    Parameters
    ----------
    p : tuple
        Radiometic function parameters to evaluate. (Should be a (6,) tuple)

    x : int or ndarray
        X position where the function is evaluated (int or result of meshgrid)

    y : int or ndarray
        Y position where the function is evaluated (int or result of meshgrid)

    Returns
    -------
    int or ndarray
        Function evaluated at given positoin

    """
    return p[0] * x**2 + p[1] * x * y + p[2] * y**2 + p[3] * x + p[4] * y + p[5]


# Defining objective function f_r


def f_r(x: np.ndarray, data: Any, z: int, pos: np.ndarray, i_mean: float) -> float:
    """Global radiometric objective function.

    Parameters
    ----------
    x : tuple
        Radiometic function parameters to evaluate. (Should be a (6,) tuple)

    data : (slicecode.utils.FileUtils.data object)
        Data object used to iterate over volumes or images.

    z : int
        Slice number over which the function is evaluated.

    pos : ndarray
        Volume absolute positions computed with the shift_oct module
    i_mean : float
        Average volume / image intensity across all tiles.

    Returns
    -------
    float
        Objective function evaluation for parameters x.

    Notes
    -----
        This method is an implementation of [Sun2006](http://onlinelibrary.wiley.com/doi/10.1111/j.1365-2818.2006.01687.x/full)

    """
    f = 0  # Initial value

    # Transform function
    nx, ny = data.volshape[:2]
    xx, yy = np.meshgrid(list(range(nx)), list(range(ny)), indexing="ij")
    T = T_r(x, xx, yy)

    # Loop over overlaps
    for vol1, vol2, pos1, pos2 in data.neighbor_slice_iterator(z, return_pos=True):
        # Getting overlap regions
        real_pos1 = pos[pos1[0] - 1, pos1[1] - 1, :]
        real_pos2 = pos[pos2[0] - 1, pos2[1] - 1, :]
        ov1, ov2, pov1, pov2 = get_overlap(vol1, vol2, real_pos1, real_pos2)

        # Getting AIP of overlap regions
        im1 = np.squeeze(ov1.mean(axis=2))
        im2 = np.squeeze(ov2.mean(axis=2))

        # Evaluation T for overlap regions
        T_im1 = T[pov1[0] : pov1[2], pov1[1] : pov1[3]]
        T_im2 = T[pov2[0] : pov2[2], pov2[1] : pov2[3]]

        # Updating function evaluation
        f += np.sum(np.abs(im1 * T_im1 - im2 * T_im2))

    # Loop over tiles
    for vol in data.slice_iterator(z):
        im = np.squeeze(vol.mean(axis=2))
        f += np.sum(np.abs(im * T - i_mean))

    return f


def confocal_psf(z: np.ndarray, zf: float, zR: float, A: float | None = None) -> np.ndarray:
    """Confocal PSF model using a gaussian beam.

    Parameters
    ----------
    z : ndarray
        Depths at which the psf is evaluated
    zf : float
        Focal plane depth
    zR : float
        Rayleigh Length
    A : float
        Normalization constant

    Return
    ------
    ndarray
        PSF evaluated at each z locations
    """
    psf = 1.0 / (((z - zf) / float(zR)) ** 2.0 + 1.0)
    if A is not None:
        psf *= A
    return psf


def get_slice_resolutions_from_psf(
    zf: float,
    zr: float,
    nz: int = 120,
    spacing: tuple[float, float, float] = (6.5, 6.5, 6.5),
    N: int = 512,
    lam: float = 1.030,
) -> np.ndarray:
    """Compute the lateral resolution at each depth using the confocal PSF model."""
    res = np.zeros((nz,))
    z = np.linspace(0, nz * spacing[2], nz)
    w0 = np.sqrt(zr * lam / np.pi)
    wz = w0 * np.sqrt(1 + ((z - zf) / float(zr)) ** 2.0)

    for ii in range(nz):
        zmin = -wz[ii] * 4
        zmax = wz[ii] * 4
        x = np.linspace(zmin, zmax, N)
        edge = x < 0
        psf = np.exp(-(x**2.0) / wz[ii] ** 2.0)

        edge = np.convolve(edge, psf, mode="same")
        edge /= edge.max()

        x90 = x[np.where(edge > 0.90)[0][-1]]
        x10 = x[np.where(edge < 0.1)[0][0]]
        this_res = x10 - x90
        res[ii] = this_res

    return res


def estimate_psf(
    agarose: np.ndarray,
    interface: np.ndarray | None = None,
    dz: float = 6.5,
    remove_saturation: bool = False,
    mask_interface: bool = False,
    zf: float | None = None,
    fit_attn: bool = False,
) -> tuple[float, ...]:
    """Estimates the confocal PSF assuming a gaussian beam.

    Parameters
    ----------
    agarose : ndarray
        Agarose volume
    interface : ndarray
        Water-Tissue interface map
    dz : float
        Z spacing in micron
    remove_saturation : bool
        If True, remove saturated pixels before fitting.
    mask_interface : bool
        If True, mask the tissue interface region before fitting.
    zf : float, optional
        Fixed focal plane depth. If None, it is estimated.
    fit_attn : bool
        If True, also fit the attenuation parameter.

    Returns
    -------
    float
        Focal plane depth
    float
        Rayleigh length
    float
        Normalization constant

    Notes
    -----
    * If no interface is given, the whole volume is used for the psf regression
    """
    i_profile = np.copy(agarose) if agarose.ndim == 1 else agarose.mean(axis=(0, 1))
    nz = len(i_profile)
    z = np.linspace(0, len(i_profile) * dz, len(i_profile))

    # Only fit the tissue intensity (if interface is given)
    if interface is not None:
        z0 = np.median(interface)
        i_profile = i_profile[z0::]
        z = z[z0::]
    else:
        z0 = 0

    if mask_interface:
        m = gaussian_filter1d(i_profile, sigma=1, order=1)
        m = m < 0
        m = binary_erosion(m, iterations=1)

        i_profile = i_profile[m]
        z = z[m]

    # Remove outliers in the tissue
    if remove_saturation:
        i_diff = np.diff(i_profile)
        i_diff = np.insert(i_diff, 0, 0)
        z_diff = np.copy(z)

        med = np.median(i_diff)
        MAD = np.median(np.abs(i_diff - med))
        if MAD != 0:
            Zscore = np.abs(0.6745 * (i_diff - med) / MAD)
            outliers = Zscore > 3.5
        else:
            outliers = np.zeros_like(i_diff)

        i_profile = i_profile[~outliers]
        z = z_diff[~outliers]

    # Normalization
    i_profile /= i_profile.max()

    # Fitting model
    if zf is None:
        if fit_attn:

            def fo_psf(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
                return np.sum((y - confocal_psf(z, x[0], x[1], x[2]) * np.exp(-2 * x[3] * (z - z[0]))) ** 2.0)
        else:

            def fo_psf(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:  # type: ignore[misc]
                return np.sum((y - confocal_psf(z, x[0], x[1], x[2])) ** 2.0)

        # 1st fit of the model
        zf = 0.5 * (nz * dz)
        zR = 250.0
        A = 1.0
        p0 = [zf, zR, A]  # Initial parameters
        param_bounds = [(0.0, zf * dz), (0.0, None), (0.0, 1.0)]
        if fit_attn:
            p0.append(0.0)
            param_bounds.append((0.0, None))
        popt_1 = minimize(fo_psf, p0, args=(i_profile, z), bounds=param_bounds)

        # Detect outliers
        if fit_attn:
            I_err = (
                i_profile - confocal_psf(z, popt_1.x[0], popt_1.x[1], popt_1.x[2]) * np.exp(-2 * popt_1.x[3] * (z - z[0]))
            ) ** 2.0
        else:
            I_err = (i_profile - confocal_psf(z, popt_1.x[0], popt_1.x[1], popt_1.x[2])) ** 2.0
        err_med = np.median(I_err)
        err_MAD = np.median(np.abs(I_err - err_med))
        if err_MAD != 0:
            err_Zscore = np.abs(0.6745 * (I_err - err_med) / err_MAD)
            err_outliers = err_Zscore > 3.5
        else:
            err_outliers = np.zeros_like(I_err)

        # 2nd fit without outliers
        p0 = popt_1.x
        popt_2 = minimize(
            fo_psf,
            p0,
            args=(np.array(i_profile)[~err_outliers], np.array(z)[~err_outliers]),
            bounds=param_bounds,
        )

        if fit_attn:
            return popt_2.x[0], popt_2.x[1], popt_2.x[2], popt_2.x[3]
        else:
            return popt_2.x[0], popt_2.x[1], popt_2.x[2]
    else:

        def fo_psf(x: np.ndarray, y: np.ndarray, z: np.ndarray, zf: float) -> float:  # type: ignore[misc]
            return np.sum((y - confocal_psf(z, zf, x[0], x[1])) ** 2.0)

        # 1st fit of the model
        zR = 250.0
        A = 1.0
        p0 = [zR, A]  # Initial parameters
        popt_1 = minimize(fo_psf, p0, args=(i_profile, z, zf), bounds=((0.0, None), (0.0, 1.0)))

        # Detect outliers
        I_err = (i_profile - confocal_psf(z, zf, popt_1.x[0], popt_1.x[1])) ** 2.0
        err_med = np.median(I_err)
        err_MAD = np.median(np.abs(I_err - err_med))
        if err_MAD != 0:
            err_Zscore = np.abs(0.6745 * (I_err - err_med) / err_MAD)
            err_outliers = err_Zscore > 3.5
        else:
            err_outliers = np.zeros_like(I_err)

        # 2nd fit without outliers
        p0 = popt_1.x
        popt_2 = minimize(
            fo_psf,
            p0,
            args=(np.array(i_profile)[~err_outliers], np.array(z)[~err_outliers], zf),
            bounds=((0.0, None), (0.0, 1.0)),
        )

        return zf, popt_2.x[0], popt_2.x[1]


def get_3d_psf(
    vol: np.ndarray,
    interface: np.ndarray,
    res: float = 6.5,
    use_average_rayleigh: bool = False,
    remove_interface: bool = True,
    zf: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a 3D PSF from a given uniform volume (e.g. agarose).

    Parameters
    ----------
    vol : ndarray
        Agarose / Background volume to use for the PSF computation
    interface : ndarray
        Water/Tissue interface depth map (in pixel)
    res : float
        Z axis resolution (in micron / pixel)
    use_average_rayleigh : bool
        Use average Rayleigh length instead of the Rayleigh map.
    remove_interface : bool
        If True, mask the tissue interface before computing the PSF.
    zf : ndarray, optional
        Fixed focal depth map. If None, it is estimated from data.

    Returns
    -------
    ndarray
        PSF
    ndarray
        Focal depth map
    ndarray
        Rayleigh length map
    """
    nx, ny, nz = vol.shape
    zf_map = np.zeros((nx, ny))
    zr_map = np.zeros((nx, ny))

    print("Computing PSF parameters for each aline")
    for ix in range(nx):
        for iy in range(ny):
            a_line = np.array(vol[ix, iy, :])
            a_line_interface = np.squeeze(interface[ix, iy])
            try:
                if zf is None:
                    params = estimate_psf(
                        a_line,
                        interface=a_line_interface,
                        dz=res,
                        mask_interface=remove_interface,
                    )
                else:
                    params = estimate_psf(
                        a_line,
                        interface=a_line_interface,
                        dz=res,
                        mask_interface=remove_interface,
                        zf=zf[ix, iy],
                    )
                zf_map[ix, iy] = params[0]
                zr_map[ix, iy] = params[1]
            except Exception:
                pass

    # Smoothing the zf and zr maps
    print("Smoothing the zf and zr maps")
    if zf is None:
        zf_map = gaussian_filter(zf_map, (nx * 0.1, ny * 0.1))
    zr_map = gaussian_filter(zr_map, (nx * 0.1, ny * 0.1))

    # Fit parabola on zf_map
    def f(x: np.ndarray, a: float, b: float, c: float, d: float, e: float, f: float) -> np.ndarray:
        return a * x[0] * x[1] + b * x[0] ** 2 + c * x[1] ** 2 + d + e * x[0] + f * x[1]

    xx, yy = np.meshgrid(list(range(nx)), list(range(ny)), indexing="ij")
    xdata = (np.ravel(yy), np.ravel(xx))
    ydata = np.ravel(zf_map)
    popt, _ = curve_fit(f, xdata, ydata)
    a, b, c, d, e, f = popt
    print(popt)
    zf_map = np.reshape(
        a * xdata[0] * xdata[1] + b * xdata[0] ** 2.0 + c * xdata[1] ** 2.0 + d + e * xdata[0] + f * xdata[1],
        (nx, ny),
    )

    # Computing this PSF
    print("Creating the 3D PSF")
    z = np.linspace(0, nz * res, nz)
    psf = np.zeros_like(vol)
    zr_ave = np.mean(zr_map)
    for ix in range(nx):
        for iy in range(ny):
            if use_average_rayleigh:
                psf[ix, iy, :] = confocal_psf(z, zf_map[ix, iy], zr_ave, 1.0)
            else:
                psf[ix, iy, :] = confocal_psf(z, zf_map[ix, iy], zr_map[ix, iy], 1.0)

    return psf, zf_map, zr_map


def fit_tissue_confocal_model(
    iprofile: np.ndarray,
    z0: int,
    zr_0: float = 400.0,
    res: float = 6.5,
    use_bump_model: bool = False,
    return_parameters: bool = False,
    return_full_model: bool = False,
    plot_profiles: bool = False,
    fix_zr: bool = False,
) -> dict:
    """Fit a confocal tissue intensity profile model to depth data."""
    nz = len(iprofile)
    z = np.linspace(0, nz * res, nz)

    # Removing water DC background
    dc = np.min(iprofile[0:z0])
    this_profile = iprofile - dc

    # Normalizing the intensity profile
    imax = this_profile.max()
    this_profile = this_profile / float(imax)

    # Fit an initial tissue model (sigmoid)
    def tissue_model(x: np.ndarray | list[float], z: np.ndarray) -> np.ndarray:
        c, z0, a = x[:]
        signal = a / (1 + np.exp(-c * (z - z0) / float(z[-1] - z[0]))).astype(float)
        return signal

    def fo_signal(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
        return np.sqrt(np.sum((y - tissue_model(x, z)) ** 2.0) / float(y.size))

    p0 = [50.0, z0 * res, 1.0]  # c, z0, a
    popt_tissue = minimize(fo_signal, x0=p0, args=(this_profile, z))
    c, z0, a = popt_tissue.x[:]
    syn_tissue = tissue_model([c, z0, 1.0], z)
    new_z0 = popt_tissue.x[1]

    # Fit a static PSF (fixed zr, moving zf)
    def confocal_model(x: np.ndarray | list[float], z: np.ndarray) -> np.ndarray:
        zf, zr, a = x[:]
        return confocal_psf(z, zf, zr, a)

    def fo_PSF(x: np.ndarray, y: np.ndarray, z: np.ndarray, tissue: np.ndarray) -> float:
        return np.sqrt(np.sum((y - tissue * confocal_model(x, z)) ** 2.0) / float(y.size))

    p0 = [z[-1] * 0.5, zr_0, 1.0]  # zf, zr, a
    param_bounds: list[tuple[float | None, float | None]] = [(z[0], z[-1]), (zr_0, zr_0), (0.0, None)]
    popt_firstpsf = minimize(fo_PSF, x0=p0, args=(this_profile, z, syn_tissue), bounds=param_bounds)
    zf, zr = popt_firstpsf.x[0:2]
    psf1 = confocal_model([zf, zr, 1.0], z)

    # Detect the specular reflection bump at the water / tissue interface
    if use_bump_model:

        def bump_model(signal: np.ndarray, w: float, b: float) -> np.ndarray:
            t_grad = -gaussian_filter1d(signal, w, order=2)
            t_grad[t_grad < 0] = 0
            if t_grad.max() > 0:
                with contextlib.suppress(BaseException):
                    t_grad /= float(t_grad.max())
            bump = b * t_grad
            return bump

        def bump_tissue_model(x: np.ndarray, z: np.ndarray) -> np.ndarray:
            z0, c, w, a, b = x[:]
            signal = tissue_model([c, z0, a], z)
            return signal + bump_model(signal, w, b)

        def fo_btm(x: np.ndarray, y: np.ndarray, z: np.ndarray, psf: np.ndarray) -> float:
            return np.sqrt(np.sum((y - psf * bump_tissue_model(x, z)) ** 2.0) / float(y.size))

        p0 = [new_z0, 60, 5, 1.0, 0.5]
        param_bounds = [(z[0], z[-1]), (0, 100), (1.0, 10), (0, None), (0, None)]
        popt_btm = minimize(fo_btm, x0=p0, args=(this_profile, z, psf1), bounds=param_bounds)
        z0, c, w, a, b = popt_btm.x[:]
        bump_tissue = bump_tissue_model(popt_btm.x, z)
        tissue = tissue_model([c, z0, a], z)
        bump = bump_model(tissue, w, b)
        bump_mask = (bump - bump[-1]) <= 0.1 * (bump.max() - bump[-1])
        bump_mask[z <= new_z0] = 0
        limit = np.where(bump_mask)[0][0] * res
        this_mask = z >= limit

        # Optimize the PSF model (zf, zr)
        # Normalize the signal to fit the tissue model.
        def normalize_profile(signal: np.ndarray, psf: np.ndarray) -> np.ndarray:
            normalized_signal = signal / psf
            return normalized_signal

        def fo_psf_normalized(x: np.ndarray, y: np.ndarray, z: np.ndarray, tissue: np.ndarray) -> float:
            return np.sqrt(np.sum((tissue - normalize_profile(y, confocal_model(x, z))) ** 2.0) / float(y.size))

        p0 = popt_firstpsf.x  # zf, zr, a
        if fix_zr:
            zr_0 = p0[1]
            param_bounds = [(z[0], z[-1]), (zr_0, zr_0), (0.0, None)]
        else:
            param_bounds = [(z[0], z[-1]), (0, None), (0.0, None)]
        popt_psf = minimize(
            fo_psf_normalized,
            x0=p0,
            args=(this_profile[this_mask], z[this_mask], bump_tissue[this_mask]),
            bounds=param_bounds,
        )
        psf_final = confocal_model(popt_psf.x, z)

        if plot_profiles:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(8, 5))
            plt.plot(
                z,
                iprofile,
                marker="o",
                linestyle="dashed",
                alpha=2,
                label="Original Data",
            )
            plt.plot(
                z,
                imax * psf_final * bump_tissue + dc,
                label="Tissue model",
                linewidth=2,
                color="r",
            )
            plt.plot(
                z,
                imax * psf_final / psf_final.max() + dc,
                linewidth=2,
                linestyle="dashed",
                color="k",
                label="Confocal PSF",
            )
            plt.legend(loc="best", shadow=True)
            plt.grid(True)
            plt.show()

        output: dict[str, Any] = {"psf": psf_final}
        if return_full_model:
            output["tissue"] = imax * bump_tissue + dc
            output["tissue_psf"] = imax * psf_final * bump_tissue + dc

        if return_parameters:
            output["parameters"] = {
                "zf": popt_psf.x[0],
                "zr": popt_psf.x[1],
                "a": popt_psf.x[2],
            }
            output["parameters"]["z0"] = popt_btm.x[0]
            output["parameters"]["c"] = popt_btm.x[1]
            output["parameters"]["w"] = popt_btm.x[2]
            output["parameters"]["b1"] = popt_btm.x[3]
            output["parameters"]["b2"] = popt_btm.x[4]
    else:
        if plot_profiles:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(8, 5))
            plt.plot(
                z,
                iprofile,
                marker="o",
                linestyle="dashed",
                alpha=0.6,
                label="Original Data",
            )
            plt.plot(
                z,
                imax * psf1 * syn_tissue * popt_firstpsf.x[2] + dc,
                label="Tissue model",
                linewidth=2,
                color="r",
            )
            plt.plot(
                z,
                imax * psf1 / psf1.max() + dc,
                linewidth=2,
                linestyle="dashed",
                color="k",
                label="Confocal PSF",
            )
            plt.plot(
                z,
                (iprofile - dc) * psf1.max() / psf1 + dc,
                marker="o",
                linewidth=2,
                linestyle="dashed",
                color="k",
                label="Compensated Data",
            )
            plt.legend(loc="best", shadow=True)
            plt.grid(True)
            plt.show()

        output: dict[str, Any] = {"psf": psf1}
        if return_full_model:
            output["tissue"] = imax * syn_tissue + dc
            output["tissue_psf"] = imax * psf1 * syn_tissue + dc

        if return_parameters:
            output["parameters"] = {
                "zf": popt_firstpsf.x[0],
                "zr": popt_firstpsf.x[1],
                "a": popt_firstpsf.x[2],
            }
            output["parameters"]["z0"] = popt_tissue.x[0]
            output["parameters"]["c"] = popt_tissue.x[1]
            output["parameters"]["w"] = popt_tissue.x[2]

    return output
