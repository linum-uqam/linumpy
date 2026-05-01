"""GPU N4 bias field correction.

Implements the Tustison 2010 N4 algorithm using the B-spline primitive
in :mod:`linumpy.gpu.bspline` and a CuPy/NumPy-shared histogram
sharpening routine.

Each fitting level loops over:

1.  Compute the log-residual ``r = log(v) - log_bias`` on masked voxels.
2.  Sharpen the residual histogram by Wiener-deconvolving it with a
    Gaussian PSF (Sled 1998 / Tustison 2010), producing a LUT mapping
    observed log-intensity to expected (unbiased) log-intensity.
3.  Voxel-wise, compute the per-voxel log-bias update
    ``delta = log(v) - LUT(log(v) - log_bias)``.
4.  Fit a tensor-product cubic B-spline to ``delta`` on a regular
    control grid, evaluate at full resolution, and add to ``log_bias``.

The next fitting level doubles the number of control points per axis.

Memory budget (per N4 call):

    ~6 x volume_size x 4 bytes

i.e. ~12 GB for a (256, 1024, 1024) float32 volume.
"""

from typing import Any

import numpy as np

from linumpy.gpu import GPU_AVAILABLE, get_array_module
from linumpy.gpu.bspline import (
    _build_axis_basis,
    _is_gpu_array,
    bspline_evaluate,
    bspline_fit,
    bspline_fit_precompute,
)

# ---------------------------------------------------------------------------
# Histogram sharpening
# ---------------------------------------------------------------------------


def _build_log_psf(n_bins: int, bin_width: float, fwhm: float, xp: Any) -> Any:
    """Return a centred Gaussian PSF over *n_bins* bins.

    Parameters
    ----------
    n_bins : int
        Histogram bin count.
    bin_width : float
        Histogram bin width in log-intensity units.
    fwhm : float
        Full-width-half-maximum of the Gaussian PSF, log-intensity units.
    xp : module
        Array module.
    """
    sigma = fwhm / 2.3548200450309493  # 2 sqrt(2 ln 2)
    centre = n_bins // 2
    x = (xp.arange(n_bins, dtype=xp.float32) - centre) * bin_width
    psf = xp.exp(-0.5 * (x / sigma) ** 2)
    psf = psf / psf.sum()
    return psf


def sharpen_residual(
    log_v: np.ndarray,
    mask: np.ndarray | None,
    *,
    n_bins: int = 200,
    fwhm_log: float = 0.15,
    wiener_noise: float = 0.01,
    use_gpu: bool = True,
    mask_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Return the per-voxel sharpened log-intensity (LUT-mapped).

    Implements the Sled/Tustison histogram sharpening: build the
    weighted log-intensity histogram restricted to *mask*, deconvolve
    it by a Gaussian PSF (Wiener-regularised), and return the LUT
    ``E[v_true | v_obs]`` evaluated at every voxel in *log_v*.

    Parameters
    ----------
    log_v : np.ndarray
        Log-intensity volume (any shape, float32).
    mask : np.ndarray or None
        Boolean mask; only masked voxels contribute to the histogram.
        When ``None``, all voxels are used.
    n_bins : int
        Histogram bin count.
    fwhm_log : float
        Full-width-half-maximum of the Gaussian PSF in log-intensity
        units.  Controls how much sharpening is applied (smaller FWHM
        means less sharpening, since the deconvolution kernel is
        narrower).  N4 default is approximately 0.15.
    wiener_noise : float
        Wiener regularisation term.  Larger values stabilise the
        deconvolution at the expense of sharpening.
    use_gpu : bool
        Use CuPy when available.
    mask_weights : np.ndarray, optional
        Float32 view of *mask* (``mask.astype(float32)``).  When provided,
        skips the per-call cast -- a full-volume allocation that the N4
        fit loop otherwise repeats every iteration.

    Returns
    -------
    np.ndarray
        Sharpened log-intensity, same shape and dtype as *log_v*.
        Outside the mask, the input log-intensity is returned unchanged.
    """
    xp = get_array_module(use_gpu=use_gpu and GPU_AVAILABLE)

    log_v_xp = xp.asarray(log_v, dtype=xp.float32)
    mask_xp = xp.ones_like(log_v_xp, dtype=xp.bool_) if mask is None else xp.asarray(mask, dtype=xp.bool_)

    # Compute masked min/max without materialising the masked subset
    # (boolean indexing is a slow scatter-gather on GPU).  We use
    # +/-inf sentinels outside the mask so reductions ignore them.
    pos_inf = xp.float32(np.inf)
    neg_inf = xp.float32(-np.inf)
    r_min = float(xp.where(mask_xp, log_v_xp, pos_inf).min())
    r_max = float(xp.where(mask_xp, log_v_xp, neg_inf).max())
    if not np.isfinite(r_min) or not np.isfinite(r_max):
        return log_v_xp if _is_gpu_array(log_v) else np.asarray(log_v).astype(np.float32)
    if r_max - r_min < 1e-8:
        # Degenerate distribution -- no sharpening possible.
        return log_v_xp if _is_gpu_array(log_v) else np.asarray(log_v).astype(np.float32)

    bin_width = (r_max - r_min) / float(n_bins - 1)
    bin_centres = xp.linspace(r_min, r_max, n_bins, dtype=xp.float32)

    # Quantise the FULL volume once.  bin_idx_full feeds both the
    # weighted histogram (via bincount) AND the per-voxel LUT lookup,
    # so we avoid a second pass over the volume and the
    # boolean-indexed copy of the masked subset.
    bin_idx_full = xp.clip(((log_v_xp - r_min) / bin_width + 0.5).astype(xp.int64), 0, n_bins - 1)
    mask_w = mask_xp.astype(xp.float32) if mask_weights is None else mask_weights
    hist = xp.bincount(bin_idx_full.reshape(-1), weights=mask_w.reshape(-1), minlength=n_bins).astype(xp.float32)

    # Gaussian PSF (centred); FFT-shift to align with FFT convention.
    # Zero-pad histogram and PSF to ``n_pad = 2 * n_bins`` so the FFT
    # convolutions are linear, not circular.  Without padding, mass in
    # the top bins (typically white matter for OCT) wraps into the
    # bottom-bin LUT entries (and vice-versa), pulling WM intensities
    # downward and visibly muting bright tissue.
    n_pad = 2 * n_bins
    psf = _build_log_psf(n_bins, bin_width, fwhm_log, xp)
    psf_padded = xp.zeros(n_pad, dtype=xp.float32)
    psf_padded[:n_bins] = psf
    psf_shifted = xp.roll(psf_padded, -(n_bins // 2))

    hist_padded = xp.zeros(n_pad, dtype=xp.float32)
    hist_padded[:n_bins] = hist

    psf_fft = xp.fft.rfft(psf_shifted)
    hist_fft = xp.fft.rfft(hist_padded)

    # Wiener deconvolution: H_sharp = H * conj(G) / (|G|^2 + noise).
    psf_mag2 = (psf_fft * xp.conj(psf_fft)).real
    sharp_fft = hist_fft * xp.conj(psf_fft) / (psf_mag2 + wiener_noise)
    hist_sharp = xp.fft.irfft(sharp_fft, n=n_pad)[:n_bins]
    hist_sharp = xp.maximum(hist_sharp, 0.0)

    # LUT: for each output bin i, E[r | r_obs = bin_centres[i]]
    #   = sum_j r_j * hist_sharp[j] * G(i - j) / sum_j hist_sharp[j] * G(i - j)
    # i.e. (bin_centres * hist_sharp) (*) G  /  hist_sharp (*) G.
    # Pad to n_pad as well so the LUT convolution is linear.
    weighted = bin_centres * hist_sharp
    weighted_padded = xp.zeros(n_pad, dtype=xp.float32)
    weighted_padded[:n_bins] = weighted
    hist_sharp_padded = xp.zeros(n_pad, dtype=xp.float32)
    hist_sharp_padded[:n_bins] = hist_sharp
    num_fft = xp.fft.rfft(weighted_padded)
    den_fft = xp.fft.rfft(hist_sharp_padded)
    num = xp.fft.irfft(num_fft * psf_fft, n=n_pad)[:n_bins]
    den = xp.fft.irfft(den_fft * psf_fft, n=n_pad)[:n_bins]
    lut = num / xp.maximum(den, 1e-12)

    # Apply LUT to every voxel; outside mask, leave intensity unchanged.
    sharpened = lut[bin_idx_full]
    sharpened = xp.where(mask_xp, sharpened, log_v_xp).astype(xp.float32)

    if _is_gpu_array(log_v):
        return sharpened
    if xp is np:
        return sharpened
    import cupy as cp

    return cp.asnumpy(sharpened).astype(np.float32)


# ---------------------------------------------------------------------------
# N4 driver
# ---------------------------------------------------------------------------


def n4_correct_gpu(
    vol: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    shrink_factor: int = 4,
    n_iterations: list[int] | None = None,
    spline_distance_mm: float = 10.0,
    voxel_size_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
    n_bins: int = 200,
    fwhm_log: float = 0.15,
    wiener_noise: float = 0.01,
    convergence_tol: float = 1e-3,
    use_gpu: bool = True,
    out: np.ndarray | None = None,
    bias_out: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """GPU-accelerated N4 bias field correction.

    Faithful CuPy/NumPy port of the Tustison 2010 N4 algorithm: at each
    fitting level, alternate Sled-style histogram sharpening and tensor
    cubic B-spline scattered-data fitting until convergence.  The
    B-spline control mesh is fixed across levels (matching SimpleITK's
    behaviour); ``n_iterations`` only controls per-level iteration
    counts and the residual is composed across levels.

    Parameters mirror :func:`linumpy.intensity.bias_field.n4_correct` so the
    two backends are interchangeable.  Extra knobs (``n_bins``,
    ``fwhm_log``, ``wiener_noise``) tune the sharpening histogram.

    Parameters
    ----------
    vol : np.ndarray
        Float32 input volume (Z, Y, X).
    mask : np.ndarray or None
        Boolean tissue mask.  ``None`` = full volume.
    shrink_factor : int
        Isotropic spatial subsampling factor for the fit (>=1).
    n_iterations : list of int or None
        Max iterations per fitting level.  Length sets the number of
        levels.  Default ``[20, 20, 20]``.  Fewer iterations than the
        SimpleITK CPU backend because the GPU PSDB residual update has
        no internal multilevel dampening, so each iteration has full
        effect; more than ~20 per level causes the bias field to absorb
        true tissue contrast (verified empirically on live OCT).
    spline_distance_mm : float
        Approximate distance between B-spline control knots at level 0.
    voxel_size_mm : 3-tuple of float
        Voxel size (z, y, x) in mm.
    n_bins, fwhm_log, wiener_noise : sharpening parameters
        See :func:`sharpen_residual`.
    convergence_tol : float
        Per-iteration convergence threshold on the relative L2 change of
        ``log_bias``.  Iterations stop early when the change drops below
        this value.
    use_gpu : bool
        Use CuPy when available.
    out : np.ndarray, optional
        Destination buffer for the corrected output (full ``vol.shape``,
        float32).  When provided, the streaming Z-tile loop writes
        results directly into this buffer instead of allocating a fresh
        array.  May safely alias the input ``vol`` (the host buffer is
        not read after the initial H2D upload at function entry), saving
        a full-volume float32 allocation on large mosaics.
    bias_out : np.ndarray, optional
        Destination buffer for the bias-field output, same shape and
        dtype constraints as *out*.

    Returns
    -------
    corrected : np.ndarray
        Bias-corrected float32 volume (Z, Y, X), full resolution.
    bias_field : np.ndarray
        Estimated multiplicative bias field, float32, full resolution.
    """
    if n_iterations is None:
        n_iterations = [25, 25, 25]
    n_levels = len(n_iterations)

    xp = get_array_module(use_gpu=use_gpu and GPU_AVAILABLE)
    on_gpu = xp is not np

    # Single host -> device transfer.  All intermediates remain on `xp`.
    vol_xp = xp.asarray(vol, dtype=xp.float32)
    full_shape: tuple[int, int, int] = (int(vol_xp.shape[0]), int(vol_xp.shape[1]), int(vol_xp.shape[2]))
    mask_xp = xp.ones(full_shape, dtype=xp.bool_) if mask is None else xp.asarray(mask, dtype=xp.bool_)

    # Spatial subsampling for fit (stride-subsample, on device).
    if shrink_factor > 1:
        # Materialise the subsampled volume/mask as contiguous copies so we
        # can release the full-resolution `mask_xp` (and any temporaries
        # that share storage with it) during the fit loop.  The fit only
        # touches the small arrays; `vol_xp` is kept live for the
        # full-resolution evaluation pass at the bottom of the function.
        vol_small = xp.ascontiguousarray(vol_xp[::shrink_factor, ::shrink_factor, ::shrink_factor])
        mask_small = xp.ascontiguousarray(mask_xp[::shrink_factor, ::shrink_factor, ::shrink_factor])
    else:
        vol_small = vol_xp
        mask_small = mask_xp

    # Free the full-resolution mask now -- only `mask_small` is used in the
    # fit loop, and a fresh full mask is not needed for the evaluation pass.
    del mask_xp
    if on_gpu:
        import cupy as _cp_free

        _cp_free.get_default_memory_pool().free_all_blocks()

    log_v = xp.log(xp.maximum(vol_small, 1e-6)).astype(xp.float32)

    # Base control-point grid sized to physical extent.  ITK's spline order is
    # 3, so we need at least 4 control points per axis.  We keep this grid
    # FIXED across all fitting levels: SimpleITK's N4 reuses one B-spline
    # mesh and accumulates residual composition across levels.  Doubling the
    # grid per level (as earlier versions did) yields an effectively
    # per-voxel control mesh at level 2-3 on typical OCT slabs, which
    # absorbs true tissue contrast and produces a visibly jagged bias
    # estimate.
    extents_mm = tuple(full_shape[i] * float(voxel_size_mm[i]) for i in range(3))
    n_ctrl_base = tuple(max(4, round(e / spline_distance_mm)) for e in extents_mm)
    small_shape: tuple[int, int, int] = (
        int(vol_small.shape[0]),
        int(vol_small.shape[1]),
        int(vol_small.shape[2]),
    )
    n_ctrl: tuple[int, int, int] = (
        max(4, min(n_ctrl_base[0], small_shape[0])),
        max(4, min(n_ctrl_base[1], small_shape[1])),
        max(4, min(n_ctrl_base[2], small_shape[2])),
    )

    # Build the three (n_voxels, n_control) cubic-B-spline basis matrices
    # once and reuse them across every level/iteration for both the fit
    # (forward) and evaluate (transpose-shaped) contractions.
    bases = (
        _build_axis_basis(small_shape[0], n_ctrl[0], xp),
        _build_axis_basis(small_shape[1], n_ctrl[1], xp),
        _build_axis_basis(small_shape[2], n_ctrl[2], xp),
    )

    log_bias = xp.zeros_like(vol_small, dtype=xp.float32)
    weights = mask_small.astype(xp.float32)
    # Accumulate control coefficients so the final full-resolution bias
    # field can be obtained by a single B-spline evaluation rather than
    # by upsampling the coarse field with a different kernel.
    coeff_total = xp.zeros(n_ctrl, dtype=xp.float32)

    # The B-spline fit denominator S(p) and the squared/cubed per-axis basis
    # matrices depend only on `bases`, which we hold fixed across all
    # fitting levels.  Precompute them once so each iteration's
    # bspline_fit call skips a small_vol-sized allocation of S plus six
    # basis-power multiplies.
    fit_precomputed = bspline_fit_precompute(bases)
    # Reuse the float32 mask cast across iterations of sharpen_residual
    # (it is identical every iter for a given level).
    sharpen_mask_weights = weights

    for level in range(n_levels):
        for _ in range(n_iterations[level]):
            # current = log_v - log_bias  (small_vol-sized intermediate; the
            # CuPy memory pool reuses the slot every iteration).
            current = log_v - log_bias
            sharpened = sharpen_residual(
                current,
                mask_small,
                n_bins=n_bins,
                fwhm_log=fwhm_log,
                wiener_noise=wiener_noise,
                use_gpu=use_gpu,
                mask_weights=sharpen_mask_weights,
            )
            # `weights == mask_small.astype(float32)`, so multiplying by it
            # zeros the residual outside the mask without an extra
            # ``xp.where`` call.
            current -= sharpened
            current *= weights
            del sharpened

            coeffs = bspline_fit(
                current,
                weights=weights,
                mask=None,  # weights already encodes the mask
                n_control_points=n_ctrl,
                use_gpu=use_gpu,
                bases=bases,
                precomputed=fit_precomputed,
            )
            del current
            update = bspline_evaluate(
                coeffs,
                target_shape=small_shape,
                use_gpu=use_gpu,
                bases=bases,
            ).astype(xp.float32)

            log_bias += update
            coeff_total += coeffs
            del coeffs

            # Convergence: ||update|| / ||log_bias|| < tol.  Compute on
            # device and issue a single D2H sync so each iteration of the
            # fit loop has at most one host round-trip (instead of two
            # ``float(xp.linalg.norm(...))`` calls).
            update_sq = (update * update).sum()
            del update
            bias_sq = (log_bias * log_bias).sum()
            converged = bool(xp.logical_and(bias_sq > 0, update_sq < (convergence_tol * convergence_tol) * bias_sq))
            if converged:
                break

    # Evaluate the accumulated B-spline at full resolution directly,
    # using the same cubic basis as the coarse-grid fits.  This replaces
    # the previous separable Catmull-Rom upsample of the coarse log-bias
    # field (different kernel -> ~2-3% spatial mismatch vs the ITK
    # reference, which evaluates the spline analytically on the fine
    # grid).
    #
    # The final stage materializes (log_bias_full, bias_field, corrected)
    # at full volume size.  For large volumes that dwarfs GPU memory, so
    # we drop the fit-time intermediates first and stream the evaluation
    # in Z-tiles back to host.
    del log_v, log_bias, weights, mask_small, vol_small, bases, fit_precomputed, sharpen_mask_weights
    if on_gpu:
        import cupy as cp

        cp.get_default_memory_pool().free_all_blocks()
    else:
        cp = None

    full_bases = (
        _build_axis_basis(full_shape[0], n_ctrl[0], xp),
        _build_axis_basis(full_shape[1], n_ctrl[1], xp),
        _build_axis_basis(full_shape[2], n_ctrl[2], xp),
    )
    M_z_full, M_y_full, M_x_full = full_bases

    # Pick a Z-tile that keeps the per-tile working set small relative to
    # vol_xp (which we keep on device for the per-voxel division).  Each
    # tile allocates ~3x its float32 size on GPU (log_bias_chunk, bias,
    # corrected).  Aim for ~2 GB total per tile.
    tile_bytes_target = 2 * 1024**3
    bytes_per_z = full_shape[1] * full_shape[2] * 4 * 3
    z_tile = max(1, min(full_shape[0], tile_bytes_target // max(bytes_per_z, 1)))

    if out is None:
        corrected_host = np.empty(full_shape, dtype=np.float32)
    else:
        if out.shape != full_shape or out.dtype != np.float32:
            raise ValueError(f"out must be {full_shape} float32, got {out.shape} {out.dtype}")
        corrected_host = out
    if bias_out is None:
        bias_host = np.empty(full_shape, dtype=np.float32)
    else:
        if bias_out.shape != full_shape or bias_out.dtype != np.float32:
            raise ValueError(f"bias_out must be {full_shape} float32, got {bias_out.shape} {bias_out.dtype}")
        bias_host = bias_out

    for z0 in range(0, full_shape[0], z_tile):
        z1 = min(z0 + z_tile, full_shape[0])
        log_bias_chunk = bspline_evaluate(
            coeff_total,
            target_shape=(z1 - z0, full_shape[1], full_shape[2]),
            use_gpu=use_gpu,
            bases=(M_z_full[z0:z1], M_y_full, M_x_full),
        )
        bias_chunk = xp.exp(log_bias_chunk).astype(xp.float32)
        del log_bias_chunk
        corrected_chunk = (vol_xp[z0:z1] / xp.maximum(bias_chunk, 1e-6)).astype(xp.float32)

        if on_gpu:
            corrected_host[z0:z1] = cp.asnumpy(corrected_chunk)
            bias_host[z0:z1] = cp.asnumpy(bias_chunk)
        else:
            corrected_host[z0:z1] = corrected_chunk
            bias_host[z0:z1] = bias_chunk
        del bias_chunk, corrected_chunk
        if on_gpu:
            cp.get_default_memory_pool().free_all_blocks()

    return corrected_host, bias_host
