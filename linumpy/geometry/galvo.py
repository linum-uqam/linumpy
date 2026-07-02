"""Galvanometric XY shift detection and correction."""

import numpy as np
from scipy.ndimage import median_filter


def detect_galvo_band_in_tile(tile_aip: np.ndarray, min_drop_ratio: float = 0.40) -> tuple:
    """Detect a galvo return dark band in the AIP of a single assembled mosaic tile.

    Companion to :func:`detect_galvo_shift` for use when only the assembled
    OME-Zarr mosaic is available and the raw ``.bin`` tiles no longer exist.
    Each zarr chunk corresponds to one OCT tile (the zarr chunk shape equals the
    tile size), so this function can be run per chunk to detect and characterise
    any unfixed galvo artifact.

    Parameters
    ----------
    tile_aip : np.ndarray
        2-D average intensity projection of a single tile,
        shape ``(n_alines, n_bscans)``.
    min_drop_ratio : float
        Minimum relative intensity drop compared to the surrounding tissue
        baseline to be classified as a dark band.  Default 0.40 (40 % drop).

    Returns
    -------
    tuple
        ``(band_start, band_width, confidence)`` -- pixel coordinates of the
        detected band within the tile (along the A-line axis) and a confidence
        score in [0, 1].  Returns ``(0, 0, 0.0)`` when no band is detected.
    """
    n_alines = tile_aip.shape[0]
    profile = median_filter(tile_aip.mean(axis=1), size=5)

    baseline = float(np.percentile(profile, 75))
    if baseline <= 1.0:
        return 0, 0, 0.0

    threshold = baseline * (1.0 - min_drop_ratio)
    dark_mask = profile < threshold

    if not dark_mask.any():
        return 0, 0, 0.0

    dark_idx = np.where(dark_mask)[0]
    gaps = np.where(np.diff(dark_idx) > 2)[0]
    groups = np.split(dark_idx, gaps + 1) if len(gaps) else [dark_idx]

    best_group = max(groups, key=lambda g: float(np.sum(threshold - profile[g].clip(max=threshold))))

    band_start = int(best_group[0])
    band_end = int(best_group[-1]) + 1
    band_width = band_end - band_start

    if band_width > n_alines * 0.20:
        return 0, 0, 0.0

    confidence = _compute_dark_band_confidence(tile_aip, band_start, band_end)
    return band_start, band_width, float(confidence)


def detect_galvo_shift(aip: np.ndarray, n_pixel_return: int = 40) -> tuple:
    """Detect galvo shift artifact in an average intensity projection.

    The galvo return region creates a dark horizontal band in OCT data.
    This function locates the band by finding gradient pairs separated by
    n_pixel_return pixels, then validates using dark band consistency.

    Parameters
    ----------
    aip : np.ndarray
        Average intensity projection of shape (n_alines, n_bscans).
    n_pixel_return : int
        Width of galvo return region in pixels (from acquisition metadata).

    Returns
    -------
    tuple
        (shift, confidence) where shift is the circular shift needed to move
        the galvo region to the edge, and confidence (0-1) indicates detection
        reliability. Apply fix when confidence >= 0.5.
    """
    n_alines = aip.shape[0]

    profile = median_filter(aip.mean(axis=1), 5)
    gradient = np.abs(np.diff(profile))

    n = len(gradient) - n_pixel_return
    if n <= 0:
        return 0, 0.0

    similarities = gradient[:n] * gradient[n_pixel_return : n_pixel_return + n]
    shift_idx = np.argmax(similarities)
    shift = n_alines - shift_idx - n_pixel_return

    boundary_pos = shift_idx
    boundary_end = boundary_pos + n_pixel_return

    confidence = _compute_dark_band_confidence(aip, int(boundary_pos), int(boundary_end))

    return int(shift), float(confidence)


def detect_galvo_for_slice(
    tiles: list,
    n_extra: int,
    threshold: float = 0.6,
    n_samples: int = 5,
    axial_resolution: float | None = None,
    min_intensity: float = 20.0,
) -> tuple:
    """Detect galvo shift for a slice by sampling multiple tiles.

    Parameters
    ----------
    tiles : list
        List of tile paths for the slice.
    n_extra : int
        Number of extra A-lines (galvo return pixels) from acquisition metadata.
    threshold : float
        Confidence threshold for applying fix (default: 0.6).
    n_samples : int
        Maximum number of tiles to sample (default: 5).
    axial_resolution : float, optional
        Axial resolution for OCT loading.
    min_intensity : float
        Minimum mean intensity for a tile to be considered valid.

    Returns
    -------
    tuple
        (shift, confidence) where shift is 0 if confidence < threshold.
    """
    from linumpy.microscope.oct import OCT

    if not tiles or n_extra <= 0:
        return 0, 0.0

    n_tiles = len(tiles)

    center_start = int(n_tiles * 0.2)
    center_end = int(n_tiles * 0.8)
    sample_indices = np.linspace(center_start, max(center_end - 1, center_start), min(n_samples, n_tiles), dtype=int)
    sample_indices = list(dict.fromkeys(sample_indices))

    detections = []
    for idx in sample_indices:
        if len(detections) >= n_samples:
            break

        oct_obj = OCT(tiles[idx], axial_resolution) if axial_resolution else OCT(tiles[idx])
        vol = oct_obj.load_image(crop=False, fix_galvo_shift=False, fix_camera_shift=False)
        aip = vol.mean(axis=0)

        if np.mean(aip) < min_intensity:
            continue

        shift, conf = detect_galvo_shift(aip, n_pixel_return=n_extra)
        detections.append((shift, conf))

    if not detections:
        return 0, 0.0

    shifts = np.array([d[0] for d in detections])
    confidences = np.array([d[1] for d in detections])

    best_idx = np.argmax(confidences)
    best_shift = shifts[best_idx]
    best_confidence = confidences[best_idx]

    if len(shifts) > 1:
        shift_tolerance = max(n_extra // 4, 5)
        n_consistent = np.sum(np.abs(shifts - best_shift) <= shift_tolerance)
        consistency_factor = (n_consistent / len(shifts)) ** 0.5
        best_confidence *= consistency_factor

    if best_confidence >= threshold:
        return int(best_shift), float(best_confidence)
    return 0, float(best_confidence)


def _compute_dark_band_confidence(aip: np.ndarray, boundary_pos: int, boundary_end: int) -> float:
    """Compute confidence that a dark band exists at the detected position.

    Real galvo artifacts create a consistent dark horizontal band visible
    across all B-scans. This is the key discriminator vs tissue boundaries.

    Parameters
    ----------
    aip : np.ndarray
        Average intensity projection of shape (n_alines, n_bscans).
    boundary_pos : int
        Start position of detected galvo region.
    boundary_end : int
        End position of detected galvo region.

    Returns
    -------
    float
        Confidence score (0-1).
    """
    n_alines, n_bscans = aip.shape
    n_pixel_return = boundary_end - boundary_pos

    if boundary_pos < 0 or boundary_end > n_alines or n_pixel_return < 5:
        return 0.0

    margin = max(10, n_pixel_return // 2)
    before_start = max(0, boundary_pos - margin * 2)
    before_end = boundary_pos
    after_start = boundary_end
    after_end = min(n_alines, boundary_end + margin * 2)

    if before_end <= before_start or after_end <= after_start:
        return 0.0

    n_check = min(n_bscans, 20)
    column_indices = np.linspace(0, n_bscans - 1, n_check, dtype=int)

    cols = aip[:, column_indices]
    before_vals = cols[before_start:before_end, :].mean(axis=0)
    galvo_vals = cols[boundary_pos:boundary_end, :].mean(axis=0)
    after_vals = cols[after_start:after_end, :].mean(axis=0)
    surrounding = (before_vals + after_vals) / 2

    valid_mask = surrounding >= 10
    valid_cols = int(np.sum(valid_mask))

    if valid_cols == 0:
        return 0.0

    surrounding_v = surrounding[valid_mask]
    galvo_v = galvo_vals[valid_mask]

    drop_mask = galvo_v < surrounding_v
    drop_count = int(np.sum(drop_mask))
    rel_drops = np.where(drop_mask, (surrounding_v - galvo_v) / surrounding_v, 0.0)
    total_drop = float(np.sum(rel_drops))
    significant_drops = int(np.sum(rel_drops > 0.10))

    consistency = drop_count / valid_cols
    significant_ratio = significant_drops / valid_cols
    avg_drop = total_drop / max(drop_count, 1)

    if consistency < 0.5:
        return consistency * 0.3

    score = consistency * 0.40 + significant_ratio * 0.35 + min(avg_drop / 0.3, 1.0) * 0.25

    return float(np.clip(score, 0.0, 1.0))


def fix_galvo_shift(vol: np.ndarray, shift: int = 0, axis: int = 1) -> np.ndarray:
    """Apply circular shift to move galvo return region to edge of volume.

    Parameters
    ----------
    vol : np.ndarray
        OCT volume data.
    shift : int
        Number of pixels to shift.
    axis : int
        Axis along which to shift (default: 1 for A-line axis).

    Returns
    -------
    np.ndarray
        Shifted volume. Crop with vol[:, :n_alines, :] to remove galvo region.
    """
    if shift == 0:
        return vol
    return np.roll(vol, shift, axis=axis)


def aggregate_band_detections(
    detections: list[tuple[float, float, float]],
    chunk_size: float,
    verbose: bool = False,
) -> tuple[int, int, float]:
    """Combine per-chunk galvo-band detections into one confidence-weighted estimate.

    Companion to :func:`detect_galvo_band_in_tile` / :func:`detect_galvo_shift` for
    callers that sample several chunks/tiles (e.g. assembled OME-Zarr mosaics where
    the raw ``.bin`` tiles are unavailable) and need to reconcile the per-chunk
    ``(band_start, band_width, confidence)`` results into a single robust estimate.

    Parameters
    ----------
    detections : list of (band_start, band_width, confidence)
        Per-chunk detections, e.g. one entry per sampled zarr chunk.
    chunk_size : float
        Width of one tile/chunk along the detection axis, used to size the
        consistency tolerance.
    verbose : bool
        If True, print the consistency penalty details.

    Returns
    -------
    tuple
        ``(band_start, band_width, confidence)`` -- rounded band position and
        combined confidence. Returns ``(0, 0, 0.0)`` when *detections* is empty.
    """
    if not detections:
        return 0, 0, 0.0

    confs = np.array([d[2] for d in detections])
    starts = np.array([d[0] for d in detections])
    widths = np.array([d[1] for d in detections])

    best_conf = float(confs.max())
    # Weighted median approximation: sort by start, pick at cumulative weight 0.5
    order = np.argsort(starts)
    cum_w = np.cumsum(confs[order])
    half = cum_w[-1] / 2.0
    med_idx = int(np.searchsorted(cum_w, half))
    med_start = float(starts[order[med_idx]])
    med_width = float(np.median(widths))

    # Penalise inconsistency across chunks.
    if len(detections) > 1:
        tol = max(chunk_size * 0.04, 3)
        n_consistent = int(np.sum(np.abs(starts - med_start) <= tol))
        consistency = n_consistent / len(detections)
        best_conf *= consistency**0.5
        if verbose:
            print(
                f"  Consistency: {n_consistent}/{len(detections)} chunks within "
                f"±{tol:.0f}px → confidence penalty factor {consistency**0.5:.3f}"
            )

    return round(med_start), round(med_width), best_conf


def decide_tile_shift(
    tile_aip: np.ndarray,
    default_shift: int,
    min_confidence: float,
    n_extra: int | None = None,
) -> tuple[int, float, bool]:
    """Decide the per-tile galvo roll shift for a single tile AIP.

    Uses the gradient-pair detector (:func:`detect_galvo_shift`) when *n_extra*
    is given, otherwise the threshold-based :func:`detect_galvo_band_in_tile`.
    Falls back to *default_shift* when the per-tile detection confidence is
    below *min_confidence*.

    Parameters
    ----------
    tile_aip : np.ndarray
        Average intensity projection of a single tile, shape (n_alines, n_bscans).
    default_shift : int
        Fallback shift to use when per-tile detection confidence is too low.
    min_confidence : float
        Minimum confidence required to use the per-tile detected shift.
    n_extra : int or None
        Number of galvo-return pixels from acquisition metadata; enables the
        gradient-pair detector when set.

    Returns
    -------
    tuple
        ``(shift, confidence, used_per_tile)`` -- the chosen roll shift, its
        detection confidence, and whether the per-tile value was used
        (``True``) or the fallback (``False``).
    """
    n_alines = tile_aip.shape[0]
    if n_extra:
        sh, cf = detect_galvo_shift(tile_aip, n_pixel_return=n_extra)
        sh = int(sh)
    else:
        bs, bw, cf = detect_galvo_band_in_tile(tile_aip)
        sh = n_alines - int(bs) - int(bw) if bw else default_shift
    if float(cf) >= min_confidence:
        return sh, float(cf), True
    return default_shift, float(cf), False
