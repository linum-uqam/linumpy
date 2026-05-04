"""N4 bias field correction for serial OCT stacks.

Provides CPU-based N4 correction via SimpleITK and helpers to run it
per serial section in parallel via :mod:`multiprocessing`.

Typical two-pass usage::

    from linumpy.intensity.bias_field import compute_tissue_mask, n4_correct_per_section, n4_correct

    mask = compute_tissue_mask(vol)
    vol_ps, _ = n4_correct_per_section(vol, n_serial_slices=50, mask=mask, n_processes=48)
    vol_out, _ = n4_correct(vol_ps, mask)
"""

import multiprocessing
from typing import Any

import numpy as np
import SimpleITK as sitk

from linumpy.intensity.normalization import _chunk_boundaries

# ---------------------------------------------------------------------------
# Tissue mask
# ---------------------------------------------------------------------------


def _compute_tissue_mask_gpu(
    vol: np.ndarray,
    smoothing_sigma: float,
    smoothing_sigma_z: float,
    n_serial_slices: int,
    closing_radius: int,
    z_closing_sections: int,
) -> np.ndarray:
    """GPU implementation of :func:`compute_tissue_mask`.

    Keeps the full pipeline (gaussian → Otsu → threshold → per-Z hole
    fill + closing → final Z-closing) resident on GPU. Only the final
    bool mask crosses PCIe (8x smaller than a float32 D2H of the
    smoothed volume). One section per H2D round trip; if a single
    section exceeds GPU memory, we fall back to the CPU path.
    """
    import cupy as cp
    from cupyx.scipy.ndimage import (
        binary_closing as cp_binary_closing,
    )
    from cupyx.scipy.ndimage import (
        binary_fill_holes as cp_binary_fill_holes,
    )
    from cupyx.scipy.ndimage import (
        gaussian_filter as cp_gaussian_filter,
    )
    from skimage.morphology import disk

    sigma_zyx = (smoothing_sigma_z, smoothing_sigma, smoothing_sigma)
    structuring_g = cp.asarray(disk(closing_radius), dtype=bool) if closing_radius > 0 else None

    bounds = _chunk_boundaries(vol.shape[0], n_serial_slices)
    mask = np.zeros(vol.shape, dtype=bool)

    for s, e in bounds:
        section_g = cp.asarray(vol[s:e], dtype=cp.float32)
        smoothed_g = cp_gaussian_filter(section_g, sigma=sigma_zyx)
        del section_g

        # Otsu on the GPU section using cupy.histogram on nonzero voxels.
        nonzero_g = smoothed_g[smoothed_g > 0]
        if nonzero_g.size < 100:
            mask[s:e] = True
            del smoothed_g, nonzero_g
            cp.get_default_memory_pool().free_all_blocks()
            continue
        thresh = float(_otsu_threshold_gpu(nonzero_g))
        del nonzero_g

        section_mask_g = smoothed_g > thresh
        del smoothed_g

        # Per-Z hole filling and closing (oblique masks differ across Z).
        for z in range(section_mask_g.shape[0]):
            plane_g = cp_binary_fill_holes(section_mask_g[z])
            if structuring_g is not None:
                plane_g = cp_binary_closing(plane_g, structure=structuring_g)
            section_mask_g[z] = plane_g

        mask[s:e] = cp.asnumpy(section_mask_g)
        del section_mask_g
        cp.get_default_memory_pool().free_all_blocks()

    # Bridge step artifacts at section boundaries by closing along Z.
    if z_closing_sections > 0 and n_serial_slices > 1:
        z_struct = np.ones((2 * z_closing_sections + 1, 1, 1), dtype=bool)
        # The full bool mask is 8x smaller than vol; usually fits on a single
        # GPU. If it does not, fall back to CPU for this final step.
        mask_bytes = int(mask.size)
        free_mem, _ = cp.cuda.runtime.memGetInfo()
        if mask_bytes * 4 < free_mem:  # 4x headroom for kernel scratch
            mask_g = cp.asarray(mask)
            struct_g = cp.asarray(z_struct)
            mask_g = cp_binary_closing(mask_g, structure=struct_g)
            mask = cp.asnumpy(mask_g)
            del mask_g, struct_g
            cp.get_default_memory_pool().free_all_blocks()
        else:
            from scipy.ndimage import binary_closing as np_binary_closing

            mask = np_binary_closing(mask, structure=z_struct)

    return mask


def _otsu_threshold_gpu(values: Any, nbins: int = 256) -> float:
    """Compute Otsu's threshold on a 1-D CuPy array via histogram search."""
    import cupy as cp

    lo = float(values.min().item())
    hi = float(values.max().item())
    if hi <= lo:
        return lo
    hist, edges = cp.histogram(values, bins=nbins, range=(lo, hi))
    # Mirror skimage.filters.threshold_otsu: minimize within-class variance
    # equivalent to maximizing between-class variance.
    centers = 0.5 * (edges[:-1] + edges[1:])
    hist = hist.astype(cp.float64)
    weight1 = cp.cumsum(hist)
    weight2 = cp.cumsum(hist[::-1])[::-1]
    mean1 = cp.cumsum(hist * centers) / cp.maximum(weight1, 1.0)
    mean2 = (cp.cumsum((hist * centers)[::-1]) / cp.maximum(weight2[::-1], 1.0))[::-1]
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    idx = int(cp.argmax(variance12).item())
    return float(centers[idx].item())


def compute_tissue_mask(
    vol: np.ndarray,
    smoothing_sigma: float = 2.0,
    n_serial_slices: int = 1,
    closing_radius: int = 3,
    z_closing_sections: int = 2,
    smoothing_sigma_z: float = 1.0,
    use_gpu: bool = False,
) -> np.ndarray:
    """Return a 3-D boolean mask where *True* indicates tissue (not agarose).

    The volume is lightly smoothed with an anisotropic 3-D Gaussian
    (``smoothing_sigma`` in XY, ``smoothing_sigma_z`` in Z) and a single
    Otsu threshold is computed per serial section from the smoothed
    voxel histogram (background-zero voxels excluded).  The threshold is
    then applied per voxel, so the mask follows tissue shape through Z
    and correctly handles oblique sections (e.g. 45° acquisitions),
    where the tissue footprint shifts across Z within a section.

    Each Z-plane is post-processed with hole-filling and morphological
    closing to remove internal speckle (e.g. dark white-matter or
    ventricle voxels falling below the Otsu threshold).  Finally the
    stacked 3-D mask is closed along Z to bridge step artifacts at
    section boundaries.

    Parameters
    ----------
    vol : np.ndarray
        3-D volume (Z, Y, X), any float dtype.
    smoothing_sigma : float
        Gaussian smoothing sigma in XY (pixels) before thresholding.
    n_serial_slices : int
        Number of serial sections in the volume.  When 1 (default), one
        global Otsu threshold is used.
    closing_radius : int
        Radius (pixels) of the 2-D disk used for morphological closing
        on each Z-plane mask.  0 disables 2-D closing.
    z_closing_sections : int
        Number of adjacent sections to bridge with a 3-D closing pass on
        the stacked mask.  0 disables Z-direction closing.
    smoothing_sigma_z : float
        Gaussian smoothing sigma along Z (voxels) before thresholding.
        Small values (1-2) denoise without blurring oblique edges.
    use_gpu : bool
        If True, run the dominant 3-D ``gaussian_filter`` on GPU via
        CuPy (Z-chunked for memory safety). Falls back to CPU silently
        if CuPy is unavailable. Otsu and morphology stay on CPU. When
        *vol* is already a ``cupy.ndarray`` the GPU path is used
        regardless and slabs are read with no host round-trip.

    Returns
    -------
    np.ndarray
        Boolean array of shape (Z, Y, X) -- True where tissue is present.
    """
    from scipy.ndimage import binary_closing, binary_fill_holes, gaussian_filter
    from skimage.filters import threshold_otsu
    from skimage.morphology import disk

    from linumpy.gpu import is_cupy_array

    # Cupy input always goes through the GPU path; the CPU fallback uses
    # numpy/scipy ops that don't accept cupy arrays.
    if use_gpu or is_cupy_array(vol):
        try:
            return _compute_tissue_mask_gpu(
                vol,
                smoothing_sigma=smoothing_sigma,
                smoothing_sigma_z=smoothing_sigma_z,
                n_serial_slices=n_serial_slices,
                closing_radius=closing_radius,
                z_closing_sections=z_closing_sections,
            )
        except ImportError:
            if is_cupy_array(vol):
                raise  # cupy installed but cupyx missing — surface clearly
            # CuPy missing -- fall back to CPU below.

    # Anisotropic 3-D smoothing: stronger in XY, light in Z to preserve
    # oblique tissue boundaries without per-Z Otsu noise.
    sigma_zyx = (smoothing_sigma_z, smoothing_sigma, smoothing_sigma)
    smoothed = gaussian_filter(vol.astype(np.float32), sigma=sigma_zyx)

    bounds = _chunk_boundaries(vol.shape[0], n_serial_slices)
    mask = np.zeros(vol.shape, dtype=bool)
    structuring = disk(closing_radius) if closing_radius > 0 else None
    for s, e in bounds:
        section_smooth = smoothed[s:e]
        nonzero = section_smooth[section_smooth > 0]
        if nonzero.size < 100:
            mask[s:e] = True
            continue
        thresh = threshold_otsu(nonzero)
        section_mask = section_smooth > thresh
        # Per-Z hole filling and closing (oblique masks differ across Z).
        for z in range(section_mask.shape[0]):
            plane = binary_fill_holes(section_mask[z])
            if structuring is not None:
                plane = binary_closing(plane, structure=structuring)
            section_mask[z] = plane
        mask[s:e] = section_mask

    # Bridge step artifacts at section boundaries by closing along Z.
    if z_closing_sections > 0 and n_serial_slices > 1:
        z_struct = np.ones((2 * z_closing_sections + 1, 1, 1), dtype=bool)
        mask = binary_closing(mask, structure=z_struct)

    return mask


# ---------------------------------------------------------------------------
# N4 core
# ---------------------------------------------------------------------------


def n4_correct(
    vol: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    shrink_factor: int = 4,
    n_iterations: list[int] | None = None,
    spline_distance_mm: float = 10.0,
    voxel_size_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
    backend: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Run N4 bias field correction on a 3-D volume.

    The N4 fit is performed on a spatially downsampled copy (``shrink_factor``);
    the bias field is then upsampled back to full resolution before division.

    Parameters
    ----------
    vol : np.ndarray
        Float32 input volume (Z, Y, X).
    mask : np.ndarray or None
        Boolean tissue mask (Z, Y, X) -- same shape as *vol*.  A full-volume
        mask is used when *None*.
    shrink_factor : int
        Isotropic spatial downsampling factor for the N4 fit.
    n_iterations : list of int or None
        Max iterations per fitting level; its length sets the number of fitting
        levels.  Defaults to ``[50, 50, 50, 50]`` (4 levels).
    spline_distance_mm : float
        Approximate distance (in mm) between B-spline control-point knots.
    voxel_size_mm : 3-tuple of float
        Voxel size (z, y, x) in mm -- sets physical spacing for SimpleITK.
    backend : {"cpu", "gpu", "auto"}
        Backend selector.  ``"cpu"`` (default) uses SimpleITK's N4
        implementation.  ``"gpu"`` dispatches to
        :func:`linumpy.gpu.n4.n4_correct_gpu` (CuPy-accelerated when CUDA is
        available, NumPy fallback otherwise).  ``"auto"`` picks ``"gpu"`` when
        CuPy + CUDA are available and ``"cpu"`` otherwise.

    Returns
    -------
    corrected : np.ndarray
        Bias-corrected float32 volume, same shape as *vol*.
    bias_field : np.ndarray
        Estimated bias field (multiplicative), float32, same shape as *vol*.
    """
    if backend not in ("cpu", "gpu", "auto"):
        raise ValueError(f"backend must be 'cpu', 'gpu', or 'auto', got {backend!r}")

    if backend == "auto":
        from linumpy.gpu import GPU_AVAILABLE

        backend = "gpu" if GPU_AVAILABLE else "cpu"

    if backend == "gpu":
        from linumpy.gpu.n4 import n4_correct_gpu

        return n4_correct_gpu(
            vol,
            mask,
            shrink_factor=shrink_factor,
            n_iterations=n_iterations,
            spline_distance_mm=spline_distance_mm,
            voxel_size_mm=voxel_size_mm,
            use_gpu=True,
        )

    vol_f32 = vol.astype(np.float32)

    if n_iterations is None:
        n_iterations = [50, 50, 50, 50]

    # Build SimpleITK images -- ITK convention is (x, y, z), so transpose (Z,Y,X)→(X,Y,Z)
    sitk_vol = sitk.GetImageFromArray(vol_f32.transpose(2, 1, 0))
    sitk_vol.SetSpacing((float(voxel_size_mm[2]), float(voxel_size_mm[1]), float(voxel_size_mm[0])))

    if mask is not None:
        sitk_mask = sitk.GetImageFromArray(mask.astype(np.uint8).transpose(2, 1, 0))
        sitk_mask.CopyInformation(sitk_vol)
    else:
        sitk_mask = None

    # Shrink for fast fit
    shrinker = sitk.ShrinkImageFilter()
    shrinker.SetShrinkFactors([shrink_factor] * 3)
    sitk_vol_shrunk = shrinker.Execute(sitk_vol)
    sitk_mask_shrunk = shrinker.Execute(sitk_mask) if sitk_mask is not None else None

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations(n_iterations)

    # Per-axis control points = physical extent (mm) / spline_distance (mm).
    # SimpleITK expects (x, y, z) order while voxel_size_mm / vol.shape are (z, y, x).
    min_control_points = corrector.GetSplineOrder() + 1  # ITK requires n_pts > spline_order
    extents_mm_zyx = [vol_f32.shape[i] * float(voxel_size_mm[i]) for i in range(3)]
    n_pts_zyx = [max(min_control_points, round(e / spline_distance_mm)) for e in extents_mm_zyx]
    corrector.SetNumberOfControlPoints([n_pts_zyx[2], n_pts_zyx[1], n_pts_zyx[0]])

    if sitk_mask_shrunk is not None:
        corrector.Execute(sitk_vol_shrunk, sitk_mask_shrunk)
    else:
        corrector.Execute(sitk_vol_shrunk)

    # Reconstruct full-resolution bias field
    log_bias_shrunk = corrector.GetLogBiasFieldAsImage(sitk_vol_shrunk)
    log_bias_full = sitk.Resample(
        log_bias_shrunk,
        sitk_vol,
        sitk.Transform(),
        sitk.sitkLinear,
        0.0,
        sitk.sitkFloat32,
    )
    log_bias_arr = sitk.GetArrayFromImage(log_bias_full).transpose(2, 1, 0)  # back to (Z,Y,X)
    bias_field = np.exp(log_bias_arr).astype(np.float32)

    corrected = apply_bias_field(vol_f32, bias_field)
    return corrected, bias_field


# ---------------------------------------------------------------------------
# Bias field application
# ---------------------------------------------------------------------------


def apply_bias_field(vol: np.ndarray, bias_field: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    """Divide *vol* element-wise by *bias_field*, guarding against near-zero divisors.

    Parameters
    ----------
    vol : np.ndarray
        Input volume, any shape.
    bias_field : np.ndarray
        Multiplicative bias field, same shape as *vol*.
    floor : float
        Minimum divisor value (prevents division by zero).

    Returns
    -------
    np.ndarray
        Corrected float32 array.
    """
    divisor = np.maximum(bias_field.astype(np.float32), floor)
    return (vol.astype(np.float32) / divisor).astype(np.float32)


# ---------------------------------------------------------------------------
# Per-section parallel N4
# ---------------------------------------------------------------------------


def _n4_section_worker(args: tuple[Any, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Worker function for :func:`n4_correct_per_section` (picklable top-level)."""
    chunk_vol, chunk_mask, kwargs = args
    return n4_correct(chunk_vol, chunk_mask, **kwargs)


def n4_correct_per_section(
    vol: np.ndarray,
    n_serial_slices: int,
    mask: np.ndarray | None = None,
    *,
    n_processes: int = 1,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Run N4 bias field correction independently on each serial section.

    Splits the volume along Z into *n_serial_slices* chunks and corrects each
    chunk independently (serial sections have independent optical attenuation).
    Chunks are dispatched to a :class:`multiprocessing.Pool` when
    *n_processes* > 1.

    Parameters
    ----------
    vol : np.ndarray
        Float32 3-D volume (Z, Y, X).
    n_serial_slices : int
        Number of serial tissue sections stacked along Z.
    mask : np.ndarray or None
        Boolean tissue mask (Z, Y, X).  Sliced alongside *vol*.
    n_processes : int
        Number of parallel worker processes.  1 runs serially.
    **kwargs
        Extra keyword arguments forwarded to :func:`n4_correct`
        (e.g. ``shrink_factor``, ``spline_distance_mm``).

    Returns
    -------
    corrected : np.ndarray
        Bias-corrected float32 volume, same shape as *vol*.
    bias_field : np.ndarray
        Per-section bias field stitched into a single (Z, Y, X) array.
    """
    bounds = _chunk_boundaries(vol.shape[0], n_serial_slices)

    # GPU backend cannot be parallelised across processes (single device);
    # force serial execution.
    backend = kwargs.get("backend", "cpu")
    if backend == "auto":
        from linumpy.gpu import GPU_AVAILABLE

        effective_gpu = GPU_AVAILABLE
    else:
        effective_gpu = backend == "gpu"

    if effective_gpu and n_processes != 1:
        import logging

        logging.getLogger(__name__).warning(
            "GPU N4 backend cannot be parallelised across processes (single device); "
            "forcing n_processes=1 (was %d). Per-section sections will run serially on GPU.",
            n_processes,
        )
        n_processes = 1

    if n_processes == 1:
        # Serial fast path: write each section's output straight into a
        # pre-allocated buffer.  Avoids the 76x ``.copy()`` of host slabs
        # (~36 GB on a typical OCT mosaic) and the final ``np.concatenate``
        # (another ~72 GB peak with both chunk lists alive).
        corrected = np.empty_like(vol)
        bias_field = np.empty_like(vol, dtype=np.float32)
        for s, e in bounds:
            chunk_mask = mask[s:e] if mask is not None else None
            corr_chunk, bias_chunk = n4_correct(vol[s:e], chunk_mask, **kwargs)
            corrected[s:e] = corr_chunk
            bias_field[s:e] = bias_chunk
            del corr_chunk, bias_chunk
        return corrected, bias_field

    # Parallel path: workers need pickled, independent slabs.
    work_items = [(vol[s:e].copy(), mask[s:e].copy() if mask is not None else None, kwargs) for s, e in bounds]
    with multiprocessing.Pool(processes=n_processes) as pool:
        results = pool.map(_n4_section_worker, work_items)

    corrected_chunks, bias_chunks = zip(*results, strict=True)
    corrected = np.concatenate(corrected_chunks, axis=0)
    bias_field = np.concatenate(bias_chunks, axis=0)
    return corrected, bias_field
