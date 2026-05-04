"""Tensor-product cubic B-spline scattered-data approximation.

Provides a simple GPU/CPU primitive for fitting a smooth 3-D field to
scattered (weighted) voxel samples on a regular control-point lattice
and evaluating the resulting field at arbitrary voxel grids.

Used by :mod:`linumpy.gpu.n4` for the bias-field B-spline update step,
but kept generic so other smoothing/warp primitives can reuse it.

The fit implements the single-level Lee-Wolberg-Shin (1997) B-spline
approximation that ITK uses inside ``BSplineScatteredDataPointSetToImageFilter``
(the engine of N4).  For each scattered sample p with value v_p the
locally-optimal value at surrounding control point c is::

    phi_c(p) = w_c(p) * v_p / sum_d w_d(p)^2

and the per-control-point coefficient is the squared-weight average::

    coeff[c] = sum_p w_c(p)^2 * phi_c(p) / sum_p w_c(p)^2
             = sum_p gamma_p * w_c(p)^3 * v_p / S(p)
               -------------------------------------
                       sum_p gamma_p * w_c(p)^2

where ``S(p) = sum_d w_d(p)^2`` and gamma_p folds in the per-voxel
mask/weight.  Because the tensor-product basis is separable,
``w_c(p)^k`` factorises across axes and S(p) factorises into a product
of per-axis sums of squared basis weights, so the fit reduces to three
contiguous tensor contractions -- one through ``B^3`` for the numerator
and one through ``B^2`` for the denominator.  This matches the ITK
behaviour while remaining a single GPU-friendly tensordot chain.

An earlier implementation used a Nadaraya-Watson kernel regression
(``coeff[c] = sum_p w_c(p) * v_p / sum_p w_c(p)``).  That form has no
implicit smoothness penalty and, at the dense control grids reached by
later N4 fitting levels, lets the fit absorb tissue-scale features
(e.g. white-matter contrast) into the bias estimate.  PSDB's squared
weights regularise short-range support and recover the contrast.
"""

from typing import Any

import numpy as np

from linumpy.gpu import GPU_AVAILABLE, get_array_module


def _is_gpu_array(arr: Any) -> bool:
    """Return True if *arr* is a CuPy ndarray (so callers can keep results on GPU)."""
    try:
        import cupy as cp
    except ImportError:
        return False
    return isinstance(arr, cp.ndarray)


# ---------------------------------------------------------------------------
# Cubic B-spline basis
# ---------------------------------------------------------------------------


def _cubic_bspline_basis(t: Any, xp: Any) -> Any:
    """Return the four uniform cubic B-spline basis weights at offset *t*.

    Parameters
    ----------
    t : array-like
        Fractional offset(s) in [0, 1).  Any shape.
    xp : module
        Array module (numpy or cupy).

    Returns
    -------
    array
        Stack of shape ``t.shape + (4,)`` with weights ``[B0, B1, B2, B3]``.
        Weights sum to 1 along the last axis.
    """
    t = xp.asarray(t, dtype=xp.float32)
    t2 = t * t
    t3 = t2 * t
    one_m_t = 1.0 - t
    b0 = (one_m_t * one_m_t * one_m_t) / 6.0
    b1 = (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0
    b2 = (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0
    b3 = t3 / 6.0
    return xp.stack([b0, b1, b2, b3], axis=-1)


# ---------------------------------------------------------------------------
# Coordinate mapping
# ---------------------------------------------------------------------------


def _voxel_to_control_coords(n_voxels: int, n_control: int, xp: Any) -> Any:
    """Map ``[0, n_voxels-1]`` voxel indices to control-grid coordinates.

    Voxel 0 maps to control coordinate 0; voxel ``n_voxels - 1`` maps to
    ``n_control - 3``.  This leaves one control-point of padding on each
    side so the 4-tap cubic B-spline kernel has full support at the
    boundaries.
    """
    if n_voxels == 1:
        return xp.zeros(1, dtype=xp.float32)
    span = float(n_control - 3)
    if span <= 0:
        raise ValueError(f"n_control={n_control} too small; need at least 4 control points to host a cubic B-spline.")
    return xp.arange(n_voxels, dtype=xp.float32) * (span / float(n_voxels - 1))


# ---------------------------------------------------------------------------
# Per-axis basis matrix
# ---------------------------------------------------------------------------


def _build_axis_basis(n_voxels: int, n_control: int, xp: Any) -> Any:
    """Return the dense (n_voxels, n_control) cubic-B-spline basis matrix.

    Row ``i`` contains exactly four non-zero entries -- the four basis
    weights at offsets ``-1, 0, 1, 2`` around ``floor(u_i)``, with OOB
    stencil indices clamped to ``[0, n_control - 1]`` (boundary
    partition-of-unity preservation, matching the original scattered
    formulation).

    The matrix is small (axes are at most a few hundred voxels by a few
    dozen control points) so a dense layout is cheap and lets us turn
    the fit/evaluate into three contiguous tensor contractions.
    """
    u = _voxel_to_control_coords(n_voxels, n_control, xp)
    iu = xp.floor(u).astype(xp.int32)
    t = u - iu.astype(xp.float32)
    b = _cubic_bspline_basis(t, xp)  # (n_voxels, 4)

    M = xp.zeros((n_voxels, n_control), dtype=xp.float32)
    rows = xp.arange(n_voxels, dtype=xp.int32)
    for d in range(4):
        cols = xp.clip(iu + (d - 1), 0, n_control - 1)
        # Multiple stencil offsets may map to the same column at the
        # boundary; accumulate so partition-of-unity is preserved.
        if xp is np:
            np.add.at(M, (rows, cols), b[:, d])
        else:
            xp.add.at(M, (rows, cols), b[:, d])
    return M


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------


def bspline_fit_precompute(
    bases: tuple[Any, Any, Any],
) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    """Build the iteration-invariant constants used by :func:`bspline_fit`.

    The squared/cubed per-axis basis matrices and the separable per-voxel
    denominator ``S(p) = (sum_c M_z[z,c]^2)(sum_c M_y[y,c]^2)(sum_c M_x[x,c]^2)``
    depend only on *bases*, so callers that issue many fits at the same shape
    (e.g. the N4 fitting loop) can build them once and pass them in via
    :func:`bspline_fit`'s ``precomputed`` argument.

    Returns ``(M_z2, M_y2, M_x2, M_z3, M_y3, M_x3, S)``.
    """
    M_z, M_y, M_x = bases
    M_z2 = M_z * M_z
    M_y2 = M_y * M_y
    M_x2 = M_x * M_x
    M_z3 = M_z2 * M_z
    M_y3 = M_y2 * M_y
    M_x3 = M_x2 * M_x
    s_z = M_z2.sum(axis=1)
    s_y = M_y2.sum(axis=1)
    s_x = M_x2.sum(axis=1)
    S = s_z[:, None, None] * s_y[None, :, None] * s_x[None, None, :]
    return M_z2, M_y2, M_x2, M_z3, M_y3, M_x3, S


def bspline_fit(
    values: np.ndarray,
    weights: np.ndarray | None,
    mask: np.ndarray | None,
    n_control_points: tuple[int, int, int],
    *,
    use_gpu: bool = True,
    eps: float = 1e-8,
    bases: tuple[Any, Any, Any] | None = None,
    precomputed: tuple[Any, Any, Any, Any, Any, Any, Any] | None = None,
) -> np.ndarray:
    """Fit a tensor-product cubic B-spline to scattered voxel samples.

    Parameters
    ----------
    values : np.ndarray
        Sample values, shape (Z, Y, X), float32.
    weights : np.ndarray or None
        Per-voxel non-negative weights (same shape).  ``None`` = all ones.
    mask : np.ndarray or None
        Boolean mask selecting which voxels participate in the fit.
        ``None`` = all voxels.
    n_control_points : tuple of int
        Control-grid size ``(Cz, Cy, Cx)``.  Each value must be ``>= 4``.
    use_gpu : bool
        Use CuPy when available; falls back to NumPy.
    eps : float
        Floor on the kernel-weight denominator to avoid division by zero
        for control points with no support.
    bases : tuple of arrays, optional
        Pre-built per-axis basis matrices ``(M_z, M_y, M_x)`` from
        :func:`_build_axis_basis` matching ``values.shape`` and
        ``n_control_points``.  When provided, skips the per-call build;
        useful when the caller (e.g. an N4 fitting level) issues many
        fits at the same shape.
    precomputed : tuple of arrays, optional
        Output of :func:`bspline_fit_precompute` for the same *bases*.
        When provided, skips rebuilding the squared/cubed bases and the
        separable denominator ``S`` -- a per-iteration full-volume
        allocation in the N4 fit loop.

    Returns
    -------
    np.ndarray
        Control coefficients, shape ``n_control_points``, float32 NumPy
        array (always returned on the host).
    """
    if values.ndim != 3:
        raise ValueError(f"values must be 3-D, got shape {values.shape}")
    cz, cy, cx = n_control_points
    if min(cz, cy, cx) < 4:
        raise ValueError(f"n_control_points must each be >= 4, got {n_control_points}")

    xp = get_array_module(use_gpu=use_gpu and GPU_AVAILABLE)

    vals = xp.asarray(values, dtype=xp.float32)
    w = xp.ones_like(vals) if weights is None else xp.asarray(weights, dtype=xp.float32)
    if mask is not None:
        w = w * xp.asarray(mask, dtype=xp.float32)

    z_n, y_n, x_n = vals.shape

    # Build dense per-axis basis matrices: M_axis[i, c] is the cubic
    # B-spline weight that voxel ``i`` deposits onto control point ``c``.
    # The 3-D scattered-data fit factorises along axes because the basis
    # is separable, so the whole accumulation is three contiguous tensor
    # contractions instead of 64 scatter-adds.  Bases can be precomputed
    # by the caller (e.g. once per N4 level) and reused across many
    # fit/evaluate calls to avoid rebuilding the same small matrices.
    if bases is None:
        M_z = _build_axis_basis(z_n, cz, xp)
        M_y = _build_axis_basis(y_n, cy, xp)
        M_x = _build_axis_basis(x_n, cx, xp)
    else:
        M_z, M_y, M_x = bases

    # PSDB: separable tensor-product implementation of the Lee-Wolberg-Shin
    # single-level scattered-data B-spline approximation.
    #
    #   coeff[c] = sum_p gamma_p * w_c(p)^3 * v_p / S(p)
    #             ------------------------------------------
    #                      sum_p gamma_p * w_c(p)^2
    #
    # Squared and cubed per-axis basis matrices fold the per-control-point
    # weight powers into separable contractions.  S(p) factorises as the
    # product of per-axis sums of squared basis weights.  These derive only
    # from the bases, so an N4 fitting loop can build them once via
    # :func:`bspline_fit_precompute` and pass them in.
    if precomputed is None:
        M_z2, M_y2, M_x2, M_z3, M_y3, M_x3, S = bspline_fit_precompute((M_z, M_y, M_x))
    else:
        M_z2, M_y2, M_x2, M_z3, M_y3, M_x3, S = precomputed

    psi = (w * vals) / xp.maximum(S, eps)  # (Z, Y, X)

    # num[Cz, Cy, Cx] = sum_{z,y,x} M_z3[z,Cz] M_y3[y,Cy] M_x3[x,Cx] * psi
    num = xp.tensordot(psi, M_x3, axes=([2], [0]))  # (Nz, Ny, Cx)
    num = xp.tensordot(num, M_y3, axes=([1], [0]))  # (Nz, Cx, Cy)
    num = xp.tensordot(num, M_z3, axes=([0], [0]))  # (Cx, Cy, Cz)
    num = xp.transpose(num, (2, 1, 0))  # (Cz, Cy, Cx)

    # den[Cz, Cy, Cx] = sum_{z,y,x} M_z2[z,Cz] M_y2[y,Cy] M_x2[x,Cx] * w
    den = xp.tensordot(w, M_x2, axes=([2], [0]))
    den = xp.tensordot(den, M_y2, axes=([1], [0]))
    den = xp.tensordot(den, M_z2, axes=([0], [0]))
    den = xp.transpose(den, (2, 1, 0))

    coeff = (num / xp.maximum(den, eps)).astype(xp.float32)

    # Preserve caller's array module: cupy in -> cupy out, numpy in -> numpy out.
    if _is_gpu_array(values):
        return coeff
    if xp is np:
        return coeff
    import cupy as cp

    return cp.asnumpy(coeff).astype(np.float32)


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------


def bspline_evaluate(
    control_coeffs: np.ndarray,
    target_shape: tuple[int, int, int],
    *,
    use_gpu: bool = True,
    bases: tuple[Any, Any, Any] | None = None,
) -> np.ndarray:
    """Evaluate a cubic B-spline given control coefficients on a regular grid.

    Inverse of :func:`bspline_fit`'s coordinate mapping: target voxel 0
    maps to control coordinate 0; target voxel ``N - 1`` maps to
    ``Cn - 3``.

    Parameters
    ----------
    control_coeffs : np.ndarray
        Control-grid coefficients, shape ``(Cz, Cy, Cx)``.
    target_shape : tuple of int
        Output volume shape ``(Z, Y, X)``.
    use_gpu : bool
        Use CuPy when available.
    bases : tuple of arrays, optional
        Pre-built per-axis basis matrices ``(M_z, M_y, M_x)`` matching
        ``target_shape`` and ``control_coeffs.shape``.  When provided,
        skips the per-call build.

    Returns
    -------
    np.ndarray
        Evaluated field, shape ``target_shape``, float32.
    """
    xp = get_array_module(use_gpu=use_gpu and GPU_AVAILABLE)

    coeff = xp.asarray(control_coeffs, dtype=xp.float32)
    cz, cy, cx = coeff.shape
    z_n, y_n, x_n = target_shape

    if bases is None:
        M_z = _build_axis_basis(z_n, cz, xp)  # (Nz, Cz)
        M_y = _build_axis_basis(y_n, cy, xp)
        M_x = _build_axis_basis(x_n, cx, xp)
    else:
        M_z, M_y, M_x = bases

    # out[z, y, x] = sum_{Z,Y,X} M_z[z,Z] M_y[y,Y] M_x[x,X] * coeff[Z,Y,X]
    out = xp.tensordot(coeff, M_x, axes=([2], [1]))  # (Cz, Cy, Nx)
    out = xp.tensordot(out, M_y, axes=([1], [1]))  # (Cz, Nx, Ny)
    out = xp.tensordot(out, M_z, axes=([0], [1]))  # (Nx, Ny, Nz)
    out = xp.transpose(out, (2, 1, 0)).astype(xp.float32)  # (Nz, Ny, Nx)

    if _is_gpu_array(control_coeffs):
        return out
    if xp is np:
        return out
    import cupy as cp

    return cp.asnumpy(out).astype(np.float32)
