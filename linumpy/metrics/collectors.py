"""Step-specific metric collectors for pipeline stages.

Each ``collect_*`` function records the relevant metrics for a single pipeline
step, saves the JSON file, and returns the populated :class:`PipelineMetrics`.
"""

from pathlib import Path
from typing import Any

import numpy as np

from linumpy.metrics.core import PipelineMetrics


def collect_normalization_metrics(
    vol_normalized: np.ndarray,
    agarose_mask: np.ndarray,
    otsu_threshold: float,
    background_thresholds: np.ndarray,
    output_path: Path,
    input_path: Path | None = None,
    params: dict | None = None,
) -> PipelineMetrics:
    """
    Collect metrics for intensity normalization step.

    Parameters
    ----------
    vol_normalized : np.ndarray
        The normalized volume.
    agarose_mask : np.ndarray
        The agarose mask used.
    otsu_threshold : float
        Otsu threshold computed.
    background_thresholds : np.ndarray
        Background thresholds per slice.
    output_path : str or Path
        Path to the output file.
    input_path : str, optional
        Path to the input image.
    params : dict, optional
        Dictionary of parameters used.

    Returns
    -------
    PipelineMetrics
        Metrics object (already saved).
    """
    output_path = Path(output_path)
    metrics = PipelineMetrics("normalize_intensities", str(output_path.parent))

    if input_path:
        metrics.add_info("input_volume", str(input_path), "Input volume path")
    metrics.add_info("output_volume", str(output_path), "Output volume path")
    metrics.add_info("volume_shape", list(vol_normalized.shape), "Volume shape")

    if params:
        for key, val in params.items():
            metrics.add_info(key, val, f"Parameter: {key}")

    # Agarose mask metrics
    agarose_coverage = float(np.sum(agarose_mask)) / agarose_mask.size
    metrics.add_metric(
        "agarose_coverage",
        agarose_coverage,
        description="Fraction of image classified as agarose",
        threshold_name="agarose_coverage",
    )
    metrics.add_metric("otsu_threshold", float(otsu_threshold), description="Otsu threshold used for agarose detection")

    # Background normalization metrics
    metrics.add_metric(
        "mean_background", float(np.mean(background_thresholds)), description="Mean background threshold across slices"
    )
    metrics.add_metric(
        "std_background",
        float(np.std(background_thresholds)),
        description="Std dev of background thresholds",
        threshold_name="std_background",
    )

    metrics.save(f"{output_path.stem}_metrics.json")
    metrics.log_issues()
    return metrics


def collect_xy_transform_metrics(
    transform: np.ndarray,
    tile_pairs_used: int,
    tile_shape: tuple[int, int],
    residuals: np.ndarray,
    output_path: Path,
    input_paths: list[str] | None = None,
    params: dict | None = None,
    n_tiles_x: int | None = None,
    n_tiles_y: int | None = None,
) -> PipelineMetrics:
    """
    Collect metrics for XY transform estimation step.

    Parameters
    ----------
    transform : np.ndarray
        The estimated 2x2 transform matrix.
    tile_pairs_used : int
        Number of tile pairs used for estimation.
    tile_shape : tuple
        Tile shape (rows, cols).
    residuals : np.ndarray
        Residuals from least squares fit.
    output_path : str or Path
        Path to the output transform file.
    input_paths : list, optional
        List of input image paths.
    params : dict, optional
        Dictionary of parameters used.
    n_tiles_x : int, optional
        Number of tiles in the X (column) direction.
    n_tiles_y : int, optional
        Number of tiles in the Y (row) direction.

    Returns
    -------
    PipelineMetrics
        Metrics object (already saved).
    """
    output_path = Path(output_path)
    metrics = PipelineMetrics("xy_transform_estimation", str(output_path.parent))

    if input_paths:
        metrics.add_info("input_images", input_paths, "Input mosaic images")
    metrics.add_info("tile_shape", list(tile_shape), "Tile shape in pixels")

    if n_tiles_x is not None:
        metrics.add_info("n_tiles_x", int(n_tiles_x), "Number of tiles in X direction")
    if n_tiles_y is not None:
        metrics.add_info("n_tiles_y", int(n_tiles_y), "Number of tiles in Y direction")

    if params:
        for key, val in params.items():
            metrics.add_info(key, val, f"Parameter: {key}")

    # Transform metrics
    metrics.add_metric("tile_pairs_used", tile_pairs_used, description="Number of tile pairs used for estimation")
    metrics.add_metric(
        "transform_00", float(transform[0, 0]), unit="pixels", description="Transform matrix element [0,0] (row scale)"
    )
    metrics.add_metric(
        "transform_01", float(transform[0, 1]), unit="pixels", description="Transform matrix element [0,1] (row shear)"
    )
    metrics.add_metric(
        "transform_10", float(transform[1, 0]), unit="pixels", description="Transform matrix element [1,0] (col shear)"
    )
    metrics.add_metric(
        "transform_11", float(transform[1, 1]), unit="pixels", description="Transform matrix element [1,1] (col scale)"
    )

    # Compute overlap fraction from the estimated transform
    estimated_overlap_x = 1.0 - abs(transform[0, 0]) / tile_shape[0]
    estimated_overlap_y = 1.0 - abs(transform[1, 1]) / tile_shape[1]
    metrics.add_metric(
        "estimated_overlap_x", float(estimated_overlap_x), description="Estimated overlap fraction in X direction"
    )
    metrics.add_metric(
        "estimated_overlap_y", float(estimated_overlap_y), description="Estimated overlap fraction in Y direction"
    )

    # Residual error from least squares fit
    rms_residual = None
    if len(residuals) > 0:
        rms_residual = float(np.sqrt(np.mean(residuals)))
        metrics.add_metric(
            "rms_residual",
            rms_residual,
            unit="pixels",
            description="RMS residual from least squares fit",
            threshold_name="rms_residual",
        )

    # Accumulated positioning error across the mosaic
    if n_tiles_x is not None and n_tiles_y is not None:
        expected_step_y = tile_shape[0] * (1.0 - (params or {}).get("initial_overlap", 0.2))
        expected_step_x = tile_shape[1] * (1.0 - (params or {}).get("initial_overlap", 0.2))
        systematic_err_y = abs(float(transform[0, 0]) - expected_step_y) * (n_tiles_y - 1)
        systematic_err_x = abs(float(transform[1, 1]) - expected_step_x) * (n_tiles_x - 1)
        accumulated_systematic_px = float(np.sqrt(systematic_err_y**2 + systematic_err_x**2))
        metrics.add_metric(
            "accumulated_systematic_error_px",
            accumulated_systematic_px,
            unit="pixels",
            description="Estimated accumulated systematic positioning error across mosaic",
        )
        if rms_residual is not None:
            accumulated_random_px = rms_residual * float(np.sqrt(max(n_tiles_x, n_tiles_y)))
            metrics.add_metric(
                "accumulated_random_error_px",
                accumulated_random_px,
                unit="pixels",
                description="Estimated accumulated random positioning error across mosaic",
            )

    metrics.save()
    metrics.log_issues()
    return metrics


def collect_pairwise_registration_metrics(
    registration_error: float,
    tx: float,
    ty: float,
    rotation_deg: float,
    best_z_index: int,
    expected_z_index: int,
    output_path: Path,
    fixed_path: Path | None = None,
    moving_path: Path | None = None,
    params: dict | None = None,
    z_correlation: float = 0.0,
) -> PipelineMetrics:
    """
    Collect metrics for pairwise registration step.

    Parameters
    ----------
    registration_error : float
        Registration error value.
    tx, ty : float
        Translation in X and Y.
    rotation_deg : float
        Rotation in degrees.
    best_z_index : int
        Best matching z-index.
    expected_z_index : int
        Expected z-index based on slice interval.
    output_path : str or Path
        Path to the output directory.
    fixed_path, moving_path : str, optional
        Paths to fixed and moving volumes.
    params : dict, optional
        Dictionary of parameters used.
    z_correlation : float, optional
        Normalized cross-correlation score from Z-matching (0-1). Higher values
        indicate a reliable Z-match between the two slices.

    Returns
    -------
    PipelineMetrics
        Metrics object (already saved).
    """
    output_path = Path(output_path)
    metrics = PipelineMetrics("pairwise_registration", str(output_path))

    if fixed_path:
        metrics.add_info("fixed_volume", str(fixed_path), "Path to fixed volume")
    if moving_path:
        metrics.add_info("moving_volume", str(moving_path), "Path to moving volume")
    metrics.add_info("best_z_offset", int(best_z_index), "Best matching z-index in fixed volume")

    if params:
        for key, val in params.items():
            metrics.add_info(key, val, f"Parameter: {key}")

    translation_magnitude = float(np.sqrt(tx**2 + ty**2))

    metrics.add_metric(
        "registration_error",
        float(registration_error),
        description="Registration error (lower is better)",
        threshold_name="registration_error",
    )
    metrics.add_metric("translation_x", float(tx), unit="pixels", description="Translation in X direction")
    metrics.add_metric("translation_y", float(ty), unit="pixels", description="Translation in Y direction")
    metrics.add_metric(
        "translation_magnitude",
        translation_magnitude,
        unit="pixels",
        description="Total translation magnitude",
        threshold_name="translation_magnitude",
    )
    metrics.add_metric(
        "rotation", float(rotation_deg), unit="degrees", description="Rotation angle", threshold_name="rotation_degrees"
    )
    metrics.add_metric(
        "z_drift", int(abs(best_z_index - expected_z_index)), unit="voxels", description="Deviation from expected z-index"
    )
    metrics.add_metric(
        "z_correlation",
        float(max(0.0, z_correlation)),
        unit="",
        description="Z-matching cross-correlation score (0-1; higher = more reliable)",
        threshold_name="correlation",
    )

    # Composite confidence score (0-1): combines Z-correlation, normalized translation
    # and normalized rotation.  Used downstream by adaptive transform degradation
    # in linum_stack_slices_motor.py to decide whether to apply the full transform,
    # rotation-only, or skip entirely.
    max_translation = float(params.get("max_translation_px", 50.0)) if params else 50.0
    max_rotation = float(params.get("max_rotation_deg", 5.0)) if params else 5.0
    norm_translation = min(translation_magnitude / max(max_translation, 1.0), 1.0)
    norm_rotation = min(abs(rotation_deg) / max(max_rotation, 1.0), 1.0)
    z_corr_score = float(max(0.0, z_correlation))
    confidence = float(np.clip(0.5 * z_corr_score + 0.3 * (1.0 - norm_translation) + 0.2 * (1.0 - norm_rotation), 0.0, 1.0))
    metrics.add_metric(
        "registration_confidence",
        confidence,
        unit="",
        description="Overall transform reliability score (0=unreliable, 1=reliable)",
        custom_thresholds={"warning": 0.4, "error": 0.3, "higher_is_better": True},
    )

    metrics.save()
    metrics.log_issues()
    return metrics


def collect_interface_crop_metrics(
    detected_interface: int,
    crop_depth_px: int,
    start_idx: int,
    end_idx: int,
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    resolution_um: float,
    output_path: Path,
    input_path: Path | None = None,
    padding_needed: bool = False,
) -> PipelineMetrics:
    """
    Collect metrics for interface cropping step.

    Parameters
    ----------
    detected_interface : int
        Detected interface depth in voxels.
    crop_depth_px : int
        Cropping depth in voxels.
    start_idx, end_idx : int
        Start and end indices for cropping.
    input_shape, output_shape : tuple
        Input and output volume shapes.
    resolution_um : float
        Resolution in microns.
    output_path : str or Path
        Path to the output file.
    input_path : str, optional
        Path to the input file.
    padding_needed : bool
        Whether padding was required.

    Returns
    -------
    PipelineMetrics
        Metrics object (already saved).
    """
    output_path = Path(output_path)
    metrics = PipelineMetrics("crop_interface", str(output_path.parent))

    if input_path:
        metrics.add_info("input_volume", str(input_path), "Input volume path")
    metrics.add_info("output_volume", str(output_path), "Output volume path")
    metrics.add_info("input_shape", list(input_shape), "Input volume shape")
    metrics.add_info("output_shape", list(output_shape), "Output volume shape")
    metrics.add_info("resolution_um", float(resolution_um), "Resolution in microns")

    metrics.add_metric(
        "detected_interface_depth", int(detected_interface), unit="voxels", description="Detected interface depth in voxels"
    )
    metrics.add_metric(
        "detected_interface_depth_um",
        float(detected_interface * resolution_um),
        unit="um",
        description="Detected interface depth in microns",
    )
    metrics.add_metric("crop_depth", int(crop_depth_px), unit="voxels", description="Cropping depth in voxels")
    metrics.add_metric("start_index", int(start_idx), unit="voxels", description="Start index for cropping")
    metrics.add_metric("end_index", int(end_idx), unit="voxels", description="End index for cropping")

    # Quality checks
    min_depth = PipelineMetrics.DEFAULT_THRESHOLDS["interface_min_depth_px"]["error"]
    max_fraction = PipelineMetrics.DEFAULT_THRESHOLDS["interface_max_depth_fraction"]["error"]
    if detected_interface < min_depth:
        metrics.add_metric(
            "interface_quality", "warning", description="Interface detected very close to start - may be incorrect"
        )
    elif detected_interface > input_shape[0] * max_fraction:
        metrics.add_metric("interface_quality", "warning", description="Interface detected past halfway - check detection")
    else:
        metrics.add_metric("interface_quality", "ok", description="Interface detection appears reasonable")

    metrics.add_info("padding_needed", padding_needed, "Whether padding was required")

    metrics.save(f"{output_path.stem}_metrics.json")
    metrics.log_issues()
    return metrics


def collect_psf_compensation_metrics(
    psf: np.ndarray,
    agarose_coverage: float,
    output_path: Path,
    input_path: Path | None = None,
    fit_gaussian: bool = False,
) -> PipelineMetrics:
    """
    Collect metrics for PSF compensation step.

    Parameters
    ----------
    psf : np.ndarray
        The estimated PSF profile.
    agarose_coverage : float
        Fraction of image classified as agarose.
    output_path : str or Path
        Path to the output file.
    input_path : str, optional
        Path to the input file.
    fit_gaussian : bool
        Whether Gaussian fit was used.

    Returns
    -------
    PipelineMetrics
        Metrics object (already saved).
    """
    output_path = Path(output_path)
    metrics = PipelineMetrics("psf_compensation", str(output_path.parent))

    if input_path:
        metrics.add_info("input_volume", str(input_path), "Input volume path")
    metrics.add_info("output_volume", str(output_path), "Output volume path")
    metrics.add_info("fit_gaussian", fit_gaussian, "Whether Gaussian fit was used")

    # PSF profile metrics
    psf_max = float(np.max(psf))
    psf_peak_index = int(np.argmax(psf))
    metrics.add_metric(
        "psf_max",
        psf_max,
        description="Maximum PSF value",
        custom_thresholds={"warning": 0.1, "error": 0.05, "higher_is_better": True},
    )
    metrics.add_metric("psf_peak_depth", psf_peak_index, unit="voxels", description="Depth of PSF peak")

    metrics.add_metric(
        "agarose_coverage",
        agarose_coverage,
        description="Fraction of image classified as agarose",
        threshold_name="agarose_coverage",
    )

    # Profile quality assessment
    if psf_max < 0.05:
        metrics.add_metric("profile_quality", "poor", description="PSF profile quality assessment - very low signal")
    elif psf_peak_index < 5 or psf_peak_index > len(psf) * 0.8:
        metrics.add_metric("profile_quality", "warning", description="PSF peak at unexpected depth")
    else:
        metrics.add_metric("profile_quality", "good", description="PSF profile appears reasonable")

    metrics.save(f"{output_path.stem}_metrics.json")
    metrics.log_issues()
    return metrics


def collect_stack_metrics(
    output_shape: tuple[int, ...],
    z_offsets: np.ndarray,
    num_slices: int,
    resolution: list[float],
    output_path: Path,
    blend_enabled: bool = False,
    normalize_enabled: bool = False,
    z_matches_df: Any = None,
    decisions_df: Any = None,
) -> PipelineMetrics:
    """
    Collect metrics for slice stacking step.

    Parameters
    ----------
    output_shape : tuple
        Final output shape.
    z_offsets : np.ndarray
        Z-offsets between consecutive slices.
    num_slices : int
        Number of slices stacked.
    resolution : list
        Output resolution.
    output_path : str or Path
        Path to the output file.
    blend_enabled : bool
        Whether blending was enabled.
    normalize_enabled : bool
        Whether normalization was enabled.
    z_matches_df : pandas.DataFrame, optional
        Per-pair z-match diagnostics with at least a ``correlation`` column.
    decisions_df : pandas.DataFrame, optional
        Per-slice stacking decisions with optional columns
        ``transform_loaded``, ``manual_override``, ``overlap_source``.

    Returns
    -------
    PipelineMetrics
        Metrics object (already saved).
    """
    output_path = Path(output_path)
    metrics = PipelineMetrics("stack_slices", str(output_path.parent))

    metrics.add_info("output_volume", str(output_path), "Output stacked volume path")
    metrics.add_info("num_slices", num_slices, "Number of slices stacked")
    metrics.add_info("output_shape", list(output_shape), "Final output shape")
    metrics.add_info("resolution", list(resolution), "Output resolution")
    metrics.add_info("blending_enabled", blend_enabled, "Whether blending was enabled")
    metrics.add_info("normalization_enabled", normalize_enabled, "Whether normalization was enabled")

    z_offsets = np.asarray(z_offsets)
    metrics.add_info("z_offsets", z_offsets.tolist(), "Z-offsets between consecutive slices")

    metrics.add_metric("total_z_depth", int(output_shape[0]), unit="voxels", description="Total Z depth of stacked volume")
    metrics.add_metric("mean_z_offset", float(np.mean(z_offsets)), unit="voxels", description="Mean Z-offset between slices")
    metrics.add_metric(
        "std_z_offset",
        float(np.std(z_offsets)),
        unit="voxels",
        description="Std dev of Z-offsets",
        threshold_name="z_offset_std",
    )

    z_offset_range = float(np.max(z_offsets) - np.min(z_offsets))
    metrics.add_metric(
        "z_offset_range",
        z_offset_range,
        unit="voxels",
        description="Range of Z-offsets (max - min)",
        threshold_name="z_offset_range",
    )

    if z_matches_df is not None and len(z_matches_df) > 0 and "correlation" in z_matches_df.columns:
        corr = np.asarray(z_matches_df["correlation"], dtype=float)
        corr = corr[np.isfinite(corr)]
        if corr.size > 0:
            metrics.add_info("num_z_match_pairs", int(corr.size), "Number of evaluated z-match pairs")
            metrics.add_metric(
                "mean_z_correlation",
                float(np.mean(corr)),
                description="Mean correlation across z-match pairs",
                threshold_name="correlation",
            )
            metrics.add_metric(
                "min_z_correlation",
                float(np.min(corr)),
                description="Minimum correlation across z-match pairs",
                threshold_name="correlation",
            )
            metrics.add_info("max_z_correlation", float(np.max(corr)), "Maximum correlation across z-match pairs")

    if decisions_df is not None and len(decisions_df) > 0:
        if "transform_loaded" in decisions_df.columns:
            loaded = decisions_df["transform_loaded"].astype(bool)
            metrics.add_info("n_transform_loaded", int(loaded.sum()), "Slices where pairwise transform was loaded")
            metrics.add_info("n_transform_missing", int((~loaded).sum()), "Slices where pairwise transform was unavailable")
        if "manual_override" in decisions_df.columns:
            metrics.add_info(
                "n_manual_override",
                int(decisions_df["manual_override"].astype(bool).sum()),
                "Slices with a manual stacking override",
            )
        if "overlap_source" in decisions_df.columns:
            counts = decisions_df["overlap_source"].astype(str).value_counts().to_dict()
            metrics.add_info(
                "overlap_source_counts",
                {str(k): int(v) for k, v in counts.items()},
                "Histogram of overlap source decisions",
            )

    metrics.save(f"{output_path.stem}_metrics.json")
    metrics.log_issues()
    return metrics


def collect_quality_assessment_metrics(
    output_path: Path,
    quality_results: dict[int, dict[str, Any]],
    excluded_ids: list[int],
    min_quality: float,
) -> PipelineMetrics:
    """Collect aggregate metrics for the slice-quality assessment step.

    Parameters
    ----------
    output_path : Path
        Path to the slice_config CSV that was written.
    quality_results : dict
        Mapping ``slice_id -> per-slice quality dict`` (with at least an
        ``overall`` field).
    excluded_ids : list of int
        Slices marked as excluded by the quality assessment.
    min_quality : float
        Threshold used to flag low-quality slices.
    """
    output_path = Path(output_path)
    metrics = PipelineMetrics("slice_quality_assessment", str(output_path.parent))

    metrics.add_info("output_file", str(output_path), "Slice config CSV with quality stamps")
    metrics.add_info("min_quality_threshold", float(min_quality), "Quality cutoff used")
    metrics.add_info("num_slices", len(quality_results), "Slices evaluated")
    metrics.add_info("num_excluded", len(excluded_ids), "Slices excluded by this stage")
    metrics.add_info("excluded_slice_ids", sorted(int(s) for s in excluded_ids), "IDs of excluded slices")

    overalls = np.array(
        [float(q.get("overall", 0.0)) for q in quality_results.values() if q.get("has_data", True)],
        dtype=float,
    )
    if overalls.size > 0:
        metrics.add_metric("mean_quality", float(np.mean(overalls)), description="Mean per-slice quality score")
        metrics.add_metric("min_quality", float(np.min(overalls)), description="Minimum per-slice quality score")
        metrics.add_info("max_quality", float(np.max(overalls)), "Maximum per-slice quality score")
        metrics.add_info(
            "n_below_threshold",
            int(np.sum(overalls < min_quality)) if min_quality > 0 else 0,
            "Slices below the quality threshold",
        )

    metrics.save("slice_quality_assessment_metrics.json")
    metrics.log_issues()
    return metrics


def collect_rehoming_metrics(
    output_path: Path,
    n_total_transitions: int,
    tile_corrected_indices: list[int],
    spike_corrected_indices: list[int],
    n_unreliable: int,
    max_correction_mm: float | None = None,
) -> PipelineMetrics:
    """Collect aggregate metrics for the rehoming-detection step."""
    output_path = Path(output_path)
    metrics = PipelineMetrics("rehoming_detection", str(output_path.parent))

    metrics.add_info("output_shifts", str(output_path), "Corrected shifts CSV")
    metrics.add_info("n_total_transitions", int(n_total_transitions), "Total pairwise transitions evaluated")
    metrics.add_info("n_tile_corrected", len(tile_corrected_indices), "Pass 1 tile-FOV multiple corrections applied")
    metrics.add_info("n_spike_corrected", len(spike_corrected_indices), "Pass 2 self-cancelling spike corrections applied")
    metrics.add_info("n_unreliable", int(n_unreliable), "Transitions still flagged reliable=0 after correction")
    if max_correction_mm is not None:
        metrics.add_info("max_correction_mm", float(max_correction_mm), "Largest absolute correction applied (mm)")

    metrics.save("rehoming_detection_metrics.json")
    metrics.log_issues()
    return metrics


def collect_auto_exclude_metrics(
    output_path: Path,
    num_total_slices: int,
    excluded_ids: list[int],
    cluster_count: int,
    z_corr_threshold: float,
    consecutive_threshold: int,
) -> PipelineMetrics:
    """Collect aggregate metrics for the auto-exclude step."""
    output_path = Path(output_path)
    metrics = PipelineMetrics("auto_exclude", str(output_path.parent))

    metrics.add_info("output_slice_config", str(output_path), "Slice config CSV stamped with auto_excluded")
    metrics.add_info("num_total_slices", int(num_total_slices), "Total slices considered")
    metrics.add_info("num_auto_excluded", len(excluded_ids), "Slices auto-excluded by this stage")
    metrics.add_info("auto_excluded_slice_ids", sorted(int(s) for s in excluded_ids), "IDs of auto-excluded slices")
    metrics.add_info("num_clusters", int(cluster_count), "Number of consecutive low-z_corr clusters detected")
    metrics.add_info("z_corr_threshold", float(z_corr_threshold), "Z-correlation threshold used")
    metrics.add_info("consecutive_threshold", int(consecutive_threshold), "Minimum cluster length")

    metrics.save("auto_exclude_metrics.json")
    metrics.log_issues()
    return metrics


def collect_common_space_metrics(
    output_dir: Path,
    n_selected_slices: int,
    n_excluded_slices: int,
    n_unreliable: int,
    n_refined_image_based: int,
    n_refined_rejected: int,
    refine_discrepancies_px: list[float] | None = None,
) -> PipelineMetrics:
    """Collect aggregate metrics for the common-space alignment step."""
    output_dir = Path(output_dir)
    metrics = PipelineMetrics("common_space_alignment", str(output_dir))

    metrics.add_info("output_dir", str(output_dir), "Aligned mosaics directory")
    metrics.add_info("n_selected_slices", int(n_selected_slices), "Slices retained for alignment")
    metrics.add_info("n_excluded_slices", int(n_excluded_slices), "Slices excluded by slice config")
    metrics.add_info("n_unreliable", int(n_unreliable), "Transitions flagged reliable=0 in shifts file")
    metrics.add_info("n_refined_image_based", int(n_refined_image_based), "Unreliable transitions refined via registration")
    metrics.add_info("n_refined_rejected", int(n_refined_rejected), "Image-based refinements rejected (NCC/discrepancy)")

    if refine_discrepancies_px:
        arr = np.asarray(refine_discrepancies_px, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size > 0:
            metrics.add_metric(
                "mean_refine_discrepancy_px",
                float(np.mean(arr)),
                unit="px",
                description="Mean discrepancy between motor and image-based estimates",
            )
            metrics.add_info("max_refine_discrepancy_px", float(np.max(arr)), "Maximum motor-vs-image discrepancy (px)")

    metrics.save("common_space_alignment_metrics.json")
    metrics.log_issues()
    return metrics


def collect_slice_interpolation_metrics(
    output_path: Path,
    n_fragments: int,
    interpolated_ids: list[str],
    failed_ids: list[str],
    fallback_reasons: dict[str, int] | None = None,
    method_counts: dict[str, int] | None = None,
) -> PipelineMetrics:
    """Collect aggregate metrics for the slice-interpolation finalise step."""
    output_path = Path(output_path)
    metrics = PipelineMetrics("slice_interpolation", str(output_path.parent))

    metrics.add_info("output_slice_config", str(output_path), "Final slice config CSV")
    metrics.add_info("n_fragments", int(n_fragments), "Number of per-slice fragments merged")
    metrics.add_info("n_interpolated", len(interpolated_ids), "Slices successfully reconstructed")
    metrics.add_info("n_failed", len(failed_ids), "Slices where interpolation failed")
    metrics.add_info("interpolated_slice_ids", sorted(interpolated_ids), "IDs of interpolated slices")
    metrics.add_info("failed_slice_ids", sorted(failed_ids), "IDs of slices that could not be interpolated")
    if fallback_reasons:
        metrics.add_info("fallback_reasons", dict(fallback_reasons), "Histogram of interpolation fallback reasons")
    if method_counts:
        metrics.add_info("method_counts", dict(method_counts), "Histogram of interpolation methods used")

    metrics.save("slice_interpolation_metrics.json")
    metrics.log_issues()
    return metrics


def collect_stitch_3d_metrics(
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    num_tiles: int,
    resolution: list[float],
    output_path: Path,
    input_path: Path | None = None,
    blending_method: str = "diffusion",
) -> PipelineMetrics:
    """
    Collect metrics for 3D tile stitching step.

    Parameters
    ----------
    input_shape : tuple
        Input mosaic grid shape.
    output_shape : tuple
        Output stitched volume shape.
    num_tiles : int
        Number of tiles stitched.
    resolution : list
        Output resolution.
    output_path : str or Path
        Path to the output file.
    input_path : str, optional
        Path to the input file.
    blending_method : str
        Blending method used.

    Returns
    -------
    PipelineMetrics
        Metrics object (already saved).
    """
    output_path = Path(output_path)
    metrics = PipelineMetrics("stitch_3d", str(output_path.parent))

    if input_path:
        metrics.add_info("input_volume", str(input_path), "Input mosaic grid path")
    metrics.add_info("output_volume", str(output_path), "Output stitched volume path")
    metrics.add_info("input_shape", list(input_shape), "Input mosaic shape")
    metrics.add_info("output_shape", list(output_shape), "Output stitched shape")
    metrics.add_info("num_tiles", num_tiles, "Number of tiles stitched")
    metrics.add_info("resolution", list(resolution), "Output resolution")
    metrics.add_info("blending_method", blending_method, "Blending method used")

    # Compute compression ratio (how much the stitching reduced overlap)
    input_pixels = np.prod(input_shape)
    output_pixels = np.prod(output_shape)
    overlap_reduction = 1.0 - (output_pixels / input_pixels) if input_pixels > 0 else 0.0
    metrics.add_metric(
        "overlap_reduction", float(overlap_reduction), description="Fraction of pixels removed by stitching (overlap)"
    )

    metrics.save(f"{output_path.stem}_metrics.json")
    metrics.log_issues()
    return metrics
