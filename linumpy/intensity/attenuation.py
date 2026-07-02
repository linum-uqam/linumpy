"""Attenuation estimation and profile recovery models for OCT volumes."""

from linumpy.config.threads import worker_initializer

import itertools
import multiprocessing
from typing import Any, Literal, overload

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
from linumpy.geometry.crop import mask_under_interface
from linumpy.geometry.interface import find_tissue_interface, get_interface_depth_from_mask
from linumpy.intensity.normalize import eqhist, normalize


def get_attenuation_vermeer2013(
    vol: np.ndarray, dz: float = 6.5e-6, mask: np.ndarray | None = None, C: np.ndarray | int | float | None = None
) -> np.ndarray:
    r"""Estimate the depth-resolved attenuation coefficient (Vermeer 2014).

    Implements the *exact* discretized depth-resolved estimator (DRE)

    .. math::

        \hat\mu_{\mathrm{OCT}}[i] = \frac{1}{2\Delta}\,
            \ln\!\left(1 + \frac{I[i]}{\sum_{j=i+1}^{N-1} I[j] + C}\right)

    where :math:`I[i]` is the OCT intensity at pixel ``i``, :math:`\Delta`
    is the axial pixel size, ``N`` is the number of pixels in the A-line,
    and ``C`` is an optional finite-range regularization term that
    represents the (unobserved) signal beyond the data range. ``C = 0``
    reproduces Vermeer's original formulation; passing the tail integral
    estimate corresponds to the Smith 2015 / Liu 2019 improvements (use
    :func:`get_attenuation_smith2015` or :func:`get_attenuation_liu2019`).

    .. warning::
        The denominator excludes ``I[i]``. The widely circulated linearized
        form (Neubrand 2023, Eq. 18) reads ``I[i] / (2*dz * sum_{j=i+1..} I[j])``
        and systematically over-estimates :math:`\mu` by about
        :math:`\mu^2 \Delta`; we use the logarithmic form throughout.

    Parameters
    ----------
    vol : ndarray
        3D OCT intensity volume in ``(X, Y, Z)`` order. Z is the axial
        (depth) axis along which the cumulative tail is computed.
    dz : float
        Axial pixel size in metres. The output is rescaled to ``1/cm``.
    mask : ndarray, optional
        Boolean tissue mask. Voxels above the water/tissue interface get
        :math:`\mu = 0`; voxels below the bottom interface inherit the
        last in-mask attenuation value of their A-line.
    C : ndarray, int, float, optional
        Finite-range regularization constant (Eq. 17 of Vermeer 2014).
        ``None`` (default) sets ``C = 0`` and yields the original
        formulation. A 2-D array (one value per A-line) is broadcast
        along the depth axis.

    Returns
    -------
    ndarray
        Depth-resolved attenuation coefficient in ``1/cm``, same shape as
        ``vol``.

    Notes
    -----
    For a pure exponential ``I[i] = exp(-2 mu dz i)`` and ``C = 0``, the
    estimator recovers ``mu`` exactly except for the unavoidable boundary
    blow-up at ``i = N-1`` where the tail sum is zero.

    The single-scattering assumption breaks down at depth in scattering
    tissue (multiple-scattering signal floor), which causes the textbook
    estimator to *over-correct* real brain data. Use
    :func:`get_attenuation_li2020` for noise-floor-aware estimation.

    References
    ----------
    Vermeer K. A., Mo J., Weda J. J. A., Lemij H. G., de Boer J. F.
    "Depth-resolved model-based reconstruction of attenuation coefficients
    in optical coherence tomography." Biomed. Opt. Express 5(1): 322-337
    (2014). https://doi.org/10.1364/BOE.5.000322

    Neubrand L. B., van Leeuwen T. G., Faber D. J. "Accuracy and precision
    of depth-resolved estimation of attenuation coefficients in optical
    coherence tomography." J. Biomed. Opt. 28(6): 066001 (2023).
    https://doi.org/10.1117/1.JBO.28.6.066001  --  see Appendix B for the
    derivation of Eq. 17 (used here) versus the linearized Eq. 18.
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

    # Compensation profile with depth for each A-line. See the docstring
    # (and Vermeer 2014 / Neubrand 2023 Appendix B) for the derivation.
    vol_p = np.ma.masked_array(vol, ~mask)
    cum_rev = np.cumsum(vol_p[:, :, ::-1], axis=-1)
    tail = np.zeros_like(cum_rev)
    tail[:, :, 1:] = cum_rev[:, :, :-1]  # exclude I[i]: shift by one A-line pixel
    profile = tail[:, :, ::-1] + C

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


def get_attenuation_smith2015(
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
    r"""Smith 2015 depth-resolved attenuation: Vermeer + automated tail extension.

    Augments the Vermeer 2014 estimator with two additions from Smith
    et al. (2015): an automated water/tissue interface mask and a
    *finite-range regularization* that estimates the unobserved signal
    beyond the data range.

    The end-of-scan constant ``C`` (Eq. 17 of Vermeer 2014) is

    .. math::

        C \;=\; \sum_{j=i_{\max}+1}^{\infty} I[j]
            \;\approx\; \frac{I[i_{\max}]}{2\,\hat\mu_E\,\Delta}

    where :math:`\hat\mu_E` is an *independent* estimate of the
    attenuation at the bottom of the data range. Smith obtains
    :math:`\hat\mu_E` from a log-gradient fit averaged inside the tissue
    mask (the per-A-line ``exp_fit`` term below). The factor of ``2`` in
    the denominator comes from approximating the geometric tail
    ``sum_{k>=1} exp(-2 mu dz k) ~ 1/(2 mu dz)`` for ``2 mu dz << 1``.

    Parameters
    ----------
    vol : ndarray
        OCT volume in ``(X, Y, Z)`` order.
    mask : ndarray, optional
        Tissue mask. If ``None``, the water/tissue interface is detected
        automatically with :func:`find_tissue_interface`.
    k : int
        XY median-filter kernel (voxels) applied before the Vermeer
        estimation; ``0`` disables denoising.
    sigma : int
        Axial Gaussian sigma (voxels) applied before the log-gradient fit
        used to estimate :math:`\hat\mu_E`.
    sigma_bottom : int
        XY Gaussian sigma applied to the per-A-line ``C`` map for spatial
        smoothing.
    dz : int
        Axial integration window (voxels) used to average the
        bottom-of-volume signal :math:`I[i_{\max}]`.
    res : float
        Axial resolution in ``micron / pixel``.
    zshift : int
        Number of voxels under the water/tissue interface to ignore when
        building the mask used for the gradient fit.
    fill_holes : bool
        If ``True``, fill morphological holes in the resulting attenuation
        map with :func:`SimpleITK.GrayscaleFillhole`.

    Returns
    -------
    ndarray
        Depth-resolved attenuation in ``1/cm``, same shape as ``vol``.

    See Also
    --------
    get_attenuation_vermeer2013 : core estimator (no regularization).
    get_attenuation_liu2019 : exact-form regularization with curve-fit
        :math:`\hat\mu_E`.
    get_attenuation_li2020 : noise-floor-aware variant.

    References
    ----------
    Smith G. T., Dwork N., O'Connor D., Sikora U., Lurie K. L., Pauly J. M.,
    Ellerbee A. K. "Automated, depth-resolved estimation of the attenuation
    coefficient from optical coherence tomography data." IEEE Trans. Med.
    Imaging 34(12): 2592-2602 (2015).
    https://doi.org/10.1109/TMI.2015.2450197
    """
    # First the slice is denoised with a small median filter
    if k > 0:
        vol = sitk.GetArrayFromImage(sitk.Median(sitk.GetImageFromArray(vol), (0, k, k)))

    # Computing tissue mask
    if mask is None:
        # Detecting the water / tissue interface
        interface = find_tissue_interface(vol, s_xy=3, s_z=1, order=1, use_log=True)
        mask = mask_under_interface(vol, interface + zshift, return_mask=True)

    # Per-A-line slope estimate mu*dz (round-trip), averaged inside the mask.
    exp_fit = get_gradient_attenuation(gaussian_filter(vol, (0, 0, sigma)))
    exp_fit = np.ma.masked_array(exp_fit, ~mask).mean(axis=2)
    exp_fit[np.isnan(exp_fit)] = 0

    # Mean intensity in a window of `dz` voxels above the bottom interface.
    interface_bottom = vol.shape[2] - get_interface_depth_from_mask(mask[:, :, ::-1]) - 1 - dz
    mask_bottom = mask_under_interface(vol, interface_bottom, return_mask=True)
    mask_bottom = (mask_bottom * mask).astype(bool)
    i0 = np.ma.masked_array(vol, ~mask_bottom).mean(axis=2)

    # End-of-scan tail integral C ~= I[imax] / (2 * mu * dz). exp_fit is
    # mu*dz already (the factor 0.5 in get_gradient_attenuation cancels the
    # 2 in 2*mu*dz), so the linearized tail is i0 / (2 * exp_fit).
    epsilon = 1e-3
    C = np.zeros_like(i0)
    C[exp_fit > epsilon] = i0[exp_fit > epsilon] / (2.0 * exp_fit[exp_fit > epsilon])
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


# Backwards-compatible alias for the previous name.
def get_extended_attenuation_vermeer2013(*args: Any, **kwargs: Any) -> np.ndarray:
    """Forward to :func:`get_attenuation_smith2015` (deprecated alias)."""
    import warnings

    warnings.warn(
        "get_extended_attenuation_vermeer2013 has been renamed to "
        "get_attenuation_smith2015 (Smith et al., IEEE TMI 2015). The "
        "old name is kept for backwards compatibility and will be "
        "removed in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )
    return get_attenuation_smith2015(*args, **kwargs)


def get_attenuation_liu2019(
    vol: np.ndarray,
    mask: np.ndarray | None = None,
    k: int = 10,
    res: float = 6.5,
    zshift: int = 3,
    tail_fit_voxels: int = 20,
    fill_holes: bool = False,
) -> np.ndarray:
    r"""Liu 2019 optimized depth-resolved attenuation estimator.

    Equivalent to Vermeer 2014 with the *exact* (non-linearized)
    finite-range regularization derived in Neubrand 2023, Appendix B:

    .. math::

        C \;=\; \frac{I[i_{\max}]}{\exp\!\bigl(2\,\hat\mu_E\,\Delta\bigr) - 1}

    where :math:`\hat\mu_E` is an independent estimate of the attenuation
    at the bottom of the data range. The improvement over Smith 2015 is
    twofold:

    1. ``C`` uses the exact denominator ``exp(2 mu_E dz) - 1`` instead of
       its first-order Taylor approximation ``2 mu_E dz``. The two agree
       to within a percent for ``2 mu_E dz < 0.1`` but diverge for the
       low-resolution slices common in benchtop OCT.
    2. :math:`\hat\mu_E` is obtained from a least-squares exponential fit
       of the bottom ``tail_fit_voxels`` of each A-line rather than from
       a log-gradient (which is dominated by speckle noise at the tail).

    Parameters
    ----------
    vol : ndarray
        OCT volume in ``(X, Y, Z)`` order.
    mask : ndarray, optional
        Tissue mask; auto-detected if ``None``.
    k : int
        XY median-filter kernel (voxels) applied before estimation.
    res : float
        Axial resolution in ``micron / pixel``.
    zshift : int
        Voxels to skip below the water/tissue interface when
        auto-generating the mask.
    tail_fit_voxels : int
        Number of voxels at the bottom of each A-line used to fit
        :math:`\hat\mu_E`. Must satisfy ``2 <= tail_fit_voxels <= nz``.
    fill_holes : bool
        Apply morphological hole-filling to the output.

    Returns
    -------
    ndarray
        Depth-resolved attenuation in ``1/cm``, same shape as ``vol``.

    References
    ----------
    Liu J., Ding N., Yu Y., Yuan X., Luo S., Luan J., Zhao Y., Wang Y.,
    Ma Z. "Optimized depth-resolved estimation to measure optical
    attenuation coefficients from optical coherence tomography and its
    application in cerebral damage determination." J. Biomed. Opt. 24(3):
    035002 (2019). https://doi.org/10.1117/1.JBO.24.3.035002

    Neubrand L. B., van Leeuwen T. G., Faber D. J. "Accuracy and precision
    of depth-resolved estimation of attenuation coefficients in optical
    coherence tomography." J. Biomed. Opt. 28(6): 066001 (2023).
    """
    if k > 0:
        vol = sitk.GetArrayFromImage(sitk.Median(sitk.GetImageFromArray(vol), (0, k, k)))

    if mask is None:
        interface = find_tissue_interface(vol, s_xy=3, s_z=1, order=1, use_log=True)
        mask = mask_under_interface(vol, interface + zshift, return_mask=True)

    nz = vol.shape[2]
    n_tail = int(np.clip(tail_fit_voxels, 2, nz))
    dz_m = res * 1e-6

    # Per-A-line bottom-tail least-squares fit of ln I = a - 2*mu_E*dz*z.
    # Use a per-A-line relative floor so that A-lines that decay below
    # the absolute float epsilon (typical for clean exponentials with
    # mu*z >> 1) still produce an unbiased slope. A blanket 1e-12 floor
    # would clip large fractions of the tail and bias mu_E toward zero.
    z_idx = np.arange(n_tail, dtype=float)
    bot = vol[:, :, -n_tail:].astype(np.float64)
    bot_max = bot.max(axis=2, keepdims=True)
    floor = np.maximum(bot_max * 1e-12, 1e-30)
    log_bot = np.log(np.maximum(bot, floor))
    z_mean = z_idx.mean()
    log_mean = log_bot.mean(axis=2)
    cov = ((z_idx - z_mean) * (log_bot - log_mean[..., None])).sum(axis=2)
    var = ((z_idx - z_mean) ** 2).sum()
    slope = cov / var  # dimensionless ln(I) per voxel
    mu_E_per_m = np.maximum(-slope / (2.0 * dz_m), 0.0)

    # Exact regularization C = I[imax] / (exp(2*mu_E*dz) - 1).
    i_max = vol[:, :, -1].astype(np.float64)
    denom = np.expm1(2.0 * mu_E_per_m * dz_m)
    C = np.zeros_like(i_max)
    valid = denom > 1e-12
    C[valid] = i_max[valid] / denom[valid]

    attn = get_attenuation_vermeer2013(vol, dz=dz_m, mask=mask, C=C.astype(np.float32))
    attn[np.isnan(attn)] = 0
    attn[~mask.astype(bool)] = 0
    if fill_holes:
        attn = sitk.GetArrayFromImage(sitk.GrayscaleFillhole(sitk.GetImageFromArray(attn)))
    return attn


def get_attenuation_li2020(
    vol: np.ndarray,
    mask: np.ndarray | None = None,
    k: int = 10,
    res: float = 6.5,
    zshift: int = 3,
    snr_threshold_db: float = 6.0,
    noise_floor: float | None = None,
    tail_fit_voxels: int = 20,
    fill_holes: bool = False,
) -> np.ndarray:
    r"""Li 2020 robust depth-resolved attenuation: noise-floor + tail truncation.

    The textbook Vermeer estimator under-estimates :math:`\mu` at
    shallow depth and over-estimates it at the bottom of the volume
    when the OCT signal has decayed close to the noise floor. Li et al.
    (2020) addresses both biases in two steps:

    1. Subtract a constant noise floor :math:`\langle\zeta\rangle` from
       the intensity volume (estimated from the deepest few voxels of
       the volume if not supplied).
    2. Truncate each A-line at the depth where the local SNR drops
       below ``snr_threshold_db`` (default ``6 dB``, i.e. signal at
       least 4x the noise variance), and fit :math:`\hat\mu_E` from the
       last ``tail_fit_voxels`` of the truncated A-line.

    The resulting profile is fed through the regularized DRE
    (:func:`get_attenuation_liu2019`) on the truncated A-lines. Voxels
    below the truncation depth are set to ``0`` and (optionally) filled
    by morphological hole-filling.

    Parameters
    ----------
    vol : ndarray
        OCT volume in ``(X, Y, Z)`` order.
    mask : ndarray, optional
        Tissue mask; auto-detected if ``None``.
    k : int
        XY median-filter kernel (voxels).
    res : float
        Axial resolution in ``micron / pixel``.
    zshift : int
        Voxels to skip below the auto-detected interface.
    snr_threshold_db : float
        Per-voxel SNR (relative to the estimated noise floor) below
        which the A-line is truncated. ``6 dB`` is the value used in
        Li 2020.
    noise_floor : float, optional
        Constant noise floor to subtract; if ``None`` it is estimated
        as the median of the out-of-mask region of the volume (or, as
        a fallback, the median of the bottom 5 voxels).
    tail_fit_voxels : int
        Number of voxels at the bottom of each truncated A-line used
        for the :math:`\hat\mu_E` fit.
    fill_holes : bool
        Apply morphological hole-filling to the output.

    Returns
    -------
    ndarray
        Depth-resolved attenuation in ``1/cm``, same shape as ``vol``.

    References
    ----------
    Li K., Liang W., Yang Z., Liang Y., Wan S. "Robust, accurate
    depth-resolved attenuation characterization in optical coherence
    tomography." Biomed. Opt. Express 11(2): 672-687 (2020).
    https://doi.org/10.1364/BOE.382493
    """
    if k > 0:
        vol = sitk.GetArrayFromImage(sitk.Median(sitk.GetImageFromArray(vol), (0, k, k)))

    if mask is None:
        interface = find_tissue_interface(vol, s_xy=3, s_z=1, order=1, use_log=True)
        mask = mask_under_interface(vol, interface + zshift, return_mask=True)
    mask = mask.astype(bool)

    # Step 1: estimate and subtract the noise floor.
    if noise_floor is None:
        out_of_mask = vol[~mask]
        noise_floor = float(np.median(out_of_mask)) if out_of_mask.size > 100 else float(np.median(vol[:, :, -5:]))
    vol_clean = np.clip(vol.astype(np.float32) - noise_floor, 0.0, None)

    # Step 2: per-A-line truncation depth where SNR drops below threshold.
    snr_signal_threshold = max(noise_floor, 1e-6) * (10.0 ** (snr_threshold_db / 10.0))
    above = vol > snr_signal_threshold
    nz = vol.shape[2]
    z_idx = np.arange(nz)
    z_above = np.where(above, z_idx[None, None, :], -1)
    cutoff = z_above.max(axis=2)
    cutoff = np.where(cutoff >= 0, cutoff, nz - 1)

    # Truncation mask combined with tissue mask.
    z_grid = np.broadcast_to(z_idx[None, None, :], vol.shape)
    trunc_mask = (z_grid <= cutoff[..., None]) & mask

    # Step 3: regularized DRE on the noise-subtracted, truncated volume.
    return get_attenuation_liu2019(
        vol_clean,
        mask=trunc_mask,
        k=0,
        res=res,
        zshift=zshift,
        tail_fit_voxels=tail_fit_voxels,
        fill_holes=fill_holes,
    )


def get_attenuation_faber2004(
    vol: np.ndarray, mask: np.ndarray | None = None, dz: float = 6.5e-6, N: int = 4
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Faber 2004 single-scattering attenuation estimate (one mu_t per A-line).

    Fits the curve

    .. math::

        I(z) \;\propto\; \frac{1}{1 + \bigl((z-z_0)/z_R\bigr)^2}\,
            \exp\!\bigl(-2\,\mu_t\,z\bigr)

    to each A-line by least-squares minimization over
    :math:`(z_0, z_R, \mu_t)`. The model couples the confocal PSF
    (Lorentzian in :math:`(z-z_0)/z_R`) to the single-scattering
    Beer-Lambert decay; in contrast to the Vermeer/Smith/Liu/Li family
    this returns a single :math:`\mu_t` per A-line rather than a
    depth-resolved profile.

    Parameters
    ----------
    vol : ndarray
        OCT volume in ``(X, Y, Z)`` order.
    mask : ndarray, optional
        Per-voxel boolean mask. When provided, only the masked samples
        of each A-line are passed to the optimizer; the depth axis is
        rebuilt from ``z[mask_aline]`` so the optimizer receives data
        and depth arrays of matching length even when the mask has
        gaps. If ``None``, the full A-line is used.
    dz : float
        Axial pixel size in metres.
    N : int
        XY uniform-filter window applied before fitting (``0`` to skip).

    Returns
    -------
    attn : ndarray
        Per-A-line attenuation coefficient :math:`\mu_t` (1/m).
    r_length : ndarray
        Per-A-line apparent Rayleigh length :math:`z_R` (m).
    z_focus : ndarray
        Per-A-line focal-plane depth :math:`z_0` (m).

    Notes
    -----
    Hard-coded for a 4x objective (``w0 = 4.88 um``, central wavelength
    1030 nm, refractive index 1.33). Adjust those constants in the body
    if your setup differs.

    References
    ----------
    Faber D. J., van der Meer F. J., Aalders M. C. G., van Leeuwen T. G.
    "Quantitative measurement of attenuation coefficients of weakly
    scattering media using optical coherence tomography."
    Opt. Express 12(19): 4353-4365 (2004).
    https://doi.org/10.1364/OPEX.12.004353
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
                # Use the masked depth samples so `data` and `z_aligned`
                # have the same length even when the mask is non-contiguous.
                z_aligned = z[mask_aline] - z[zp]
                p_opt = minimize(
                    f,
                    p0,
                    args=(data, z_aligned),
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
