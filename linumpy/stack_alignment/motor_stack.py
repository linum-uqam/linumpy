"""Core motor-position-based slice stacking logic.

Extracted from ``scripts/stacking/linum_stack_slices_motor.py``: loading
pairwise registration transforms, sizing the output XY canvas, and
accumulating pairwise registration translations into per-slice cumulative
offsets. The script itself remains a thin CLI wrapper around this module
plus ``linumpy.stack_alignment.io`` (shifts loading) and
``linumpy.stack_alignment.units`` (mm-to-pixel conversion, centering).
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk

logger = logging.getLogger(__name__)


def load_registration_transforms(
    transforms_dir: Path,
    slice_ids: Any,
    skip_error_status: bool = False,
    skip_warning_status: bool = False,
    load_min_zcorr: float = 0.0,
    load_max_rotation: float = 0.0,
) -> tuple[dict, dict]:
    """
    Load pairwise registration transforms from directory.

    Parameters
    ----------
    transforms_dir : Path
        Directory containing registration outputs (subdirs per slice)
    slice_ids : list
        List of slice IDs to load transforms for
    skip_error_status : bool
        If True, discard transforms whose pairwise_registration_metrics.json
        reports overall_status == 'error'.  These are typically registrations
        that failed (e.g. registered against an interpolated/synthetic slice)
        and would introduce spurious rotations into the stack.
    skip_warning_status : bool
        If True, also discard transforms with overall_status == 'warning'.
        Warning-status registrations hit the optimizer boundary (e.g. large
        translation or rotation) and their Z-offsets (fixed_z/moving_z) are
        unreliable, causing incorrect Z-overlap computation during stacking.
        Discarding them falls back to the default moving_z_first_index.
    load_min_zcorr : float
        When > 0 (together with load_max_rotation), use metric-based gating
        instead of status-based gating. Accept a transform if z_correlation
        >= load_min_zcorr AND rotation <= load_max_rotation. 0 = disabled.
    load_max_rotation : float
        Maximum rotation in degrees for metric-based gating. 0 = disabled.

    Returns
    -------
    tuple[dict, dict]
        First dict: mapping from slice_id to (transform, fixed_z, moving_z, confidence)
        or None for gated/missing slices.
        Second dict: mapping from slice_id to (tx, ty, zcorr) for ALL slices
        that have metrics, regardless of whether the transform was accepted.
        This allows translation accumulation to use translations from slices
        whose transforms were gated out (e.g. bad rotation but valid translation).
    """
    transforms_dir = Path(transforms_dir)
    transforms = {}
    all_pairwise_translations = {}
    use_metric_gating = load_min_zcorr > 0 and load_max_rotation > 0

    for slice_id in slice_ids[1:]:  # First slice has no transform
        # Find transform directory for this slice
        # Pattern: slice_z{id}_* or similar
        matching_dirs = list(transforms_dir.glob(f"*z{slice_id:02d}*")) + list(transforms_dir.glob(f"*z{slice_id}*"))

        if not matching_dirs:
            logger.warning("No transform found for slice %s", slice_id)
            transforms[slice_id] = None
            continue

        transform_dir = matching_dirs[0]

        # Load transform file
        tfm_files = list(transform_dir.glob("*.tfm"))
        offset_files = list(transform_dir.glob("*.txt"))

        if not tfm_files:
            logger.warning("No .tfm file in %s", transform_dir)
            transforms[slice_id] = None
            continue

        try:
            # Read registration quality metrics (always, to extract confidence score
            # and pairwise translations for accumulation)
            confidence = 1.0
            metrics_files = list(transform_dir.glob("pairwise_registration_metrics.json"))
            if metrics_files:
                with Path(metrics_files[0]).open() as f:
                    metrics_data = json.load(f)
                status = metrics_data.get("overall_status", "ok")
                try:
                    confidence = float(metrics_data["metrics"]["registration_confidence"]["value"])
                except KeyError, TypeError, ValueError:
                    confidence = 1.0  # fallback for older JSONs without confidence score

                # Always extract translations and zcorr for accumulation,
                # BEFORE gating -- so translations are available even for
                # slices whose transforms are skipped due to bad rotation.
                try:
                    metrics_tx = float(metrics_data["metrics"]["translation_x"]["value"])
                    metrics_ty = float(metrics_data["metrics"]["translation_y"]["value"])
                except KeyError, TypeError, ValueError:
                    metrics_tx, metrics_ty = 0.0, 0.0
                try:
                    metrics_zcorr = float(metrics_data["metrics"]["z_correlation"]["value"])
                except KeyError, TypeError, ValueError:
                    metrics_zcorr = 0.0
                all_pairwise_translations[slice_id] = (metrics_tx, metrics_ty, metrics_zcorr)

                if use_metric_gating:
                    # Metric-based gating: accept based on z_correlation and rotation
                    try:
                        zcorr = float(metrics_data["metrics"]["z_correlation"]["value"])
                    except KeyError, TypeError, ValueError:
                        zcorr = 0.0
                    try:
                        rot_deg = float(metrics_data["metrics"]["rotation"]["value"])
                    except KeyError, TypeError, ValueError:
                        rot_deg = 999.0
                    if zcorr < load_min_zcorr or abs(rot_deg) > load_max_rotation:
                        logger.warning(
                            "Slice %s: skipping transform (zcorr=%.3f < %s or rot=%.2f° > %s°)",
                            slice_id,
                            zcorr,
                            load_min_zcorr,
                            rot_deg,
                            load_max_rotation,
                        )
                        transforms[slice_id] = None
                        continue
                    logger.debug(
                        "Slice %s: accepting transform via metric gating (zcorr=%.3f, rot=%.2f°, status=%s)",
                        slice_id,
                        zcorr,
                        rot_deg,
                        status,
                    )
                else:
                    should_skip = (status == "error" and skip_error_status) or (status == "warning" and skip_warning_status)
                    if should_skip:
                        logger.warning(
                            "Slice %s: skipping transform with overall_status='%s' (unreliable registration)",
                            slice_id,
                            status,
                        )
                        transforms[slice_id] = None
                        continue

            tfm = sitk.ReadTransform(str(tfm_files[0]))

            # Load z-offsets if available
            # offsets.txt contains [fixed_z, moving_z]
            # - fixed_z: Z-index in fixed volume where overlap region starts
            # - moving_z: Z-index in moving volume where overlap region starts
            # These indicate WHERE the volumes overlap, not how much.
            fixed_z = None
            moving_z = None
            if offset_files:
                offsets = np.loadtxt(str(offset_files[0]))
                if len(offsets) >= 2:
                    fixed_z = int(offsets[0])
                    moving_z = int(offsets[1])
                    logger.debug("Slice %s: fixed_z=%s, moving_z=%s", slice_id, fixed_z, moving_z)

            transforms[slice_id] = (tfm, fixed_z, moving_z, confidence)
            logger.debug("Loaded transform for slice %s (confidence=%.2f)", slice_id, confidence)

        except Exception as e:
            logger.warning("Could not load transform for slice %s: %s", slice_id, e)
            transforms[slice_id] = None

    return transforms, all_pairwise_translations


def compute_output_shape(_slice_files: Any, cumsum_px: Any, first_vol_shape: Any) -> Any:
    """Compute output volume shape to fit all slices."""
    xmin, xmax, ymin, ymax = [0], [first_vol_shape[2]], [0], [first_vol_shape[1]]

    for dx, dy in cumsum_px.values():
        # Assuming all slices have similar XY dimensions
        xmin.append(dx)
        xmax.append(dx + first_vol_shape[2])
        ymin.append(dy)
        ymax.append(dy + first_vol_shape[1])

    x0 = min(xmin)
    y0 = min(ymin)
    nx = int(np.ceil(max(xmax) - x0))
    ny = int(np.ceil(max(ymax) - y0))

    return ny, nx, x0, y0


def accumulate_pairwise_translations(
    available_ids: list,
    registration_transforms: dict,
    all_pairwise_translations: dict,
    confidence_weight_translations: bool = False,
    max_pairwise_translation: float = 0.0,
    translation_smooth_sigma: float = 0.0,
    max_cumulative_drift_px: float = 0.0,
    translation_min_zcorr: float = 0.2,
) -> dict:
    """Accumulate pairwise registration translations into per-slice cumulative offsets.

    Filters translations by z-correlation confidence and boundary-clamped
    magnitude, accumulates them cumulatively (optionally confidence-weighted),
    then applies optional Gaussian smoothing and a cumulative drift cap.

    Parameters
    ----------
    available_ids : list
        Sorted slice IDs being stacked (first slice has no pairwise translation).
    registration_transforms : dict
        Mapping slice_id -> (transform, fixed_z, moving_z, confidence) or None,
        as returned by `load_registration_transforms`.
    all_pairwise_translations : dict
        Mapping slice_id -> (tx, ty, zcorr) for ALL slices with metrics,
        regardless of whether the transform itself was gated out.
    confidence_weight_translations : bool
        Weight each pairwise translation by its registration confidence before
        accumulating.
    max_pairwise_translation : float
        Translations at or above 95% of this boundary are treated as
        registration failures and excluded (zeroed) before accumulation.
        0 = disabled.
    translation_smooth_sigma : float
        Gaussian smoothing sigma (in slices) applied to the accumulated
        translations. 0 = disabled.
    max_cumulative_drift_px : float
        Clamp total accumulated drift (from motor baseline) to this magnitude.
        0 = disabled.
    translation_min_zcorr : float
        Minimum z_correlation required to use a slice's translation.
        0 = use all translations regardless of quality.

    Returns
    -------
    dict
        Mapping slice_id -> (cumulative_tx, cumulative_ty) for every slice in
        `available_ids[1:]`, after filtering, accumulation, smoothing and
        drift-capping (NOT yet merged with the motor baseline or centered --
        callers combine this with the pre-accumulation cumsum_px).
    """
    # First pass: extract + filter pairwise translations from metrics data.
    pairwise_translations = {}
    n_from_metrics = 0
    n_zcorr_skipped = 0
    for slice_id in available_ids[1:]:
        if slice_id in all_pairwise_translations:
            tx, ty, zcorr = all_pairwise_translations[slice_id]
            # Apply separate zcorr threshold for translations
            if translation_min_zcorr > 0 and zcorr < translation_min_zcorr:
                logger.debug(
                    "Slice %s: skipping translation (zcorr=%.3f < %s)",
                    slice_id,
                    zcorr,
                    translation_min_zcorr,
                )
                n_zcorr_skipped += 1
                continue
            pairwise_translations[slice_id] = (tx, ty)
            # Log whether this came from a loaded or gated-out transform
            if slice_id not in registration_transforms or registration_transforms[slice_id] is None:
                n_from_metrics += 1
                logger.debug(
                    "Slice %s: using translation from metrics (transform gated out) tx=%.1f, ty=%.1f, zcorr=%.3f",
                    slice_id,
                    tx,
                    ty,
                    zcorr,
                )
    if n_from_metrics > 0:
        logger.info("Recovered %s translations from gated-out transforms via metrics", n_from_metrics)
    if n_zcorr_skipped > 0:
        logger.info("Skipped %s translations due to low zcorr (< %s)", n_zcorr_skipped, translation_min_zcorr)

    # Filter unreliable translations before accumulation
    # Translations at the registration boundary are optimizer failures, not real corrections
    if pairwise_translations and max_pairwise_translation > 0:
        boundary = max_pairwise_translation * 0.95  # 95% of boundary = likely clamped
        n_excluded = 0
        for slice_id in list(pairwise_translations.keys()):
            tx, ty = pairwise_translations[slice_id]
            mag = np.sqrt(tx**2 + ty**2)
            if mag >= boundary:
                logger.warning(
                    "Slice %s: excluding boundary translation tx=%.1f, ty=%.1f (mag=%.1f >= %.1f)",
                    slice_id,
                    tx,
                    ty,
                    mag,
                    boundary,
                )
                pairwise_translations[slice_id] = (0.0, 0.0)
                n_excluded += 1
        n_total = len(pairwise_translations)
        logger.info("Translation filter: excluded %s/%s pairs at boundary (>= %.1f px)", n_excluded, n_total, boundary)

    # Second pass: accumulate filtered translations (NO cap yet -- cap applied after smoothing)
    # Optionally weight each translation by its confidence score
    cumulative_tx, cumulative_ty = 0.0, 0.0
    n_accumulated = 0
    accumulated_offsets: dict = {}  # Track per-slice cumulative offset for smoothing + cap
    for slice_id in available_ids[1:]:
        if slice_id in pairwise_translations:
            tx, ty = pairwise_translations[slice_id]
            # Confidence-weighted accumulation: attenuate low-confidence translations
            if confidence_weight_translations:
                confidence = 1.0
                if slice_id in registration_transforms and registration_transforms[slice_id] is not None:
                    confidence = registration_transforms[slice_id][3]
                tx *= confidence
                ty *= confidence
            cumulative_tx += tx
            cumulative_ty += ty
            if tx != 0 or ty != 0:
                n_accumulated += 1
            logger.debug(
                "Slice %s: pairwise tx=%.2f, ty=%.2f -> cumulative tx=%.2f, ty=%.2f",
                slice_id,
                tx,
                ty,
                cumulative_tx,
                cumulative_ty,
            )
        accumulated_offsets[slice_id] = (cumulative_tx, cumulative_ty)
    logger.info(
        "Accumulated translations for %s slices (final cumulative: tx=%.2f, ty=%.2f)",
        n_accumulated,
        cumulative_tx,
        cumulative_ty,
    )
    if confidence_weight_translations:
        logger.info("Confidence-weighted accumulation enabled")

    # Gaussian smoothing of accumulated translations (recommended over moving average).
    # Smooths only the pairwise-accumulated component, preserving motor baseline.
    # Applied BEFORE drift cap so the cap acts on the smoothed trend, not raw noise.
    ids_list = sorted(accumulated_offsets.keys())
    acc_x = np.array([accumulated_offsets[sid][0] for sid in ids_list])
    acc_y = np.array([accumulated_offsets[sid][1] for sid in ids_list])

    if translation_smooth_sigma > 0 and len(acc_x) >= 3:
        from scipy.ndimage import gaussian_filter1d

        acc_x_smooth = gaussian_filter1d(acc_x, sigma=translation_smooth_sigma)
        acc_y_smooth = gaussian_filter1d(acc_y, sigma=translation_smooth_sigma)

        max_correction = float(np.max(np.sqrt((acc_x_smooth - acc_x) ** 2 + (acc_y_smooth - acc_y) ** 2)))
        logger.info(
            "Gaussian-smoothed accumulated translations (sigma=%.1f, max correction: %.1f px)",
            translation_smooth_sigma,
            max_correction,
        )
        for j, sid in enumerate(ids_list):
            accumulated_offsets[sid] = (float(acc_x_smooth[j]), float(acc_y_smooth[j]))
        acc_x = acc_x_smooth
        acc_y = acc_y_smooth

    # Cumulative drift cap: clamp total drift from motor baseline (safety valve).
    # Now operates on smoothed values, so it only triggers for genuine large trends.
    if max_cumulative_drift_px > 0:
        n_clamped = 0
        for sid in ids_list:
            ox, oy = accumulated_offsets[sid]
            drift = np.sqrt(ox**2 + oy**2)
            if drift > max_cumulative_drift_px:
                scale = max_cumulative_drift_px / drift
                accumulated_offsets[sid] = (ox * scale, oy * scale)
                n_clamped += 1
        if n_clamped > 0:
            logger.warning("Drift cap: clamped %s slices to %.1f px", n_clamped, max_cumulative_drift_px)

    return accumulated_offsets
