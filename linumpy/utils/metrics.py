#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Metrics module for collecting and saving quality metrics from pipeline steps.

This module provides utilities for recording, saving, and aggregating metrics
from various processing steps in the 3D reconstruction pipeline.

Usage:
    # Use step-specific collectors (recommended)
    from linumpy.utils.metrics import collect_mask_metrics, collect_pairwise_registration_metrics

    # In your script:
    mask = create_mask(vol, ...)
    collect_mask_metrics(mask, vol, output_path, params={'sigma': 5.0})
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class MetricsEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


class PipelineMetrics:
    """
    Class for collecting and managing metrics from pipeline steps.

    Each step can record multiple metrics with associated quality indicators.
    Metrics are saved as JSON files for later aggregation and report generation.
    """

    # Quality thresholds for common metrics (can be overridden)
    DEFAULT_THRESHOLDS = {
        'registration_error': {'warning': 0.05, 'error': 0.15},
        'translation_magnitude': {'warning': 30.0, 'error': 50.0},
        'rotation_degrees': {'warning': 1.0, 'error': 2.0},
        'correlation': {'warning': 0.7, 'error': 0.5, 'higher_is_better': True},
        'tissue_coverage': {'warning': 0.1, 'error': 0.05, 'higher_is_better': True},
        'mask_coverage': {'warning': 0.05, 'error': 0.01, 'higher_is_better': True},
        'agarose_coverage': {'warning': 0.05, 'error': 0.01, 'higher_is_better': True},
        'empty_fraction': {'warning': 0.5, 'error': 0.8},
        'interface_depth': {'warning': 50, 'error': 100},
        'profile_quality': {'warning': 0.5, 'error': 0.3, 'higher_is_better': True},
        'rms_residual': {'warning': 5.0, 'error': 15.0},
        'z_offset_std': {'warning': 10.0, 'error': 25.0},
        'z_offset_range': {'warning': 15.0, 'error': 30.0},
        'std_background': {'warning': 0.1, 'error': 0.25},
        'min_slice_coverage': {'warning': 0.02, 'error': 0.005, 'higher_is_better': True},
        'std_slice_coverage': {'warning': 0.15, 'error': 0.3},
    }

    def __init__(self, step_name: str, output_dir: Optional[str] = None):
        """
        Initialize metrics collector.

        Parameters
        ----------
        step_name : str
            Name of the processing step (e.g., 'pairwise_registration', 'stitch_3d')
        output_dir : str, optional
            Directory to save metrics file. If None, metrics won't be saved automatically.
        """
        self.step_name = step_name
        self.output_dir = Path(output_dir) if output_dir else None
        self.metrics: Dict[str, Any] = {}
        self.warnings: List[str] = []
        self.errors: List[str] = []
        self.timestamp = datetime.now().isoformat()

    def add_metric(self, name: str, value: Any,
                   unit: Optional[str] = None,
                   threshold_name: Optional[str] = None,
                   custom_thresholds: Optional[Dict] = None,
                   description: Optional[str] = None):
        """
        Add a metric with optional quality assessment.

        Parameters
        ----------
        name : str
            Name of the metric.
        value : Any
            Value of the metric.
        unit : str, optional
            Unit of measurement.
        threshold_name : str, optional
            Name of threshold to use from DEFAULT_THRESHOLDS.
        custom_thresholds : dict, optional
            Custom thresholds {'warning': val, 'error': val, 'higher_is_better': bool}
        description : str, optional
            Human-readable description of the metric.
        """
        metric_entry = {
            'value': value,
            'unit': unit,
            'description': description,
            'status': 'ok'
        }

        # Evaluate quality if thresholds are provided
        thresholds = custom_thresholds or self.DEFAULT_THRESHOLDS.get(threshold_name)
        if thresholds and value is not None:
            higher_is_better = thresholds.get('higher_is_better', False)
            warning_thresh = thresholds.get('warning')
            error_thresh = thresholds.get('error')

            if higher_is_better:
                if error_thresh is not None and value < error_thresh:
                    metric_entry['status'] = 'error'
                    self.errors.append(f"{name}: {value} < {error_thresh} (error threshold)")
                elif warning_thresh is not None and value < warning_thresh:
                    metric_entry['status'] = 'warning'
                    self.warnings.append(f"{name}: {value} < {warning_thresh} (warning threshold)")
            else:
                if error_thresh is not None and value > error_thresh:
                    metric_entry['status'] = 'error'
                    self.errors.append(f"{name}: {value} > {error_thresh} (error threshold)")
                elif warning_thresh is not None and value > warning_thresh:
                    metric_entry['status'] = 'warning'
                    self.warnings.append(f"{name}: {value} > {warning_thresh} (warning threshold)")

        self.metrics[name] = metric_entry

    def add_info(self, name: str, value: Any, description: Optional[str] = None):
        """
        Add informational data (not quality-assessed).

        Parameters
        ----------
        name : str
            Name of the info field.
        value : Any
            Value of the info field.
        description : str, optional
            Human-readable description.
        """
        self.metrics[name] = {
            'value': value,
            'description': description,
            'status': 'info'
        }

    def get_overall_status(self) -> str:
        """
        Get overall status based on all metrics.

        Returns
        -------
        str
            'error', 'warning', or 'ok'
        """
        if self.errors:
            return 'error'
        elif self.warnings:
            return 'warning'
        return 'ok'

    def to_dict(self) -> Dict:
        """
        Convert metrics to dictionary format.

        Returns
        -------
        dict
            Dictionary containing all metrics and metadata.
        """
        return {
            'step_name': self.step_name,
            'timestamp': self.timestamp,
            'overall_status': self.get_overall_status(),
            'metrics': self.metrics,
            'warnings': self.warnings,
            'errors': self.errors
        }

    def save(self, filename: Optional[str] = None) -> Path:
        """
        Save metrics to JSON file.

        Parameters
        ----------
        filename : str, optional
            Filename for metrics file. Defaults to '{step_name}_metrics.json'

        Returns
        -------
        Path
            Path to the saved metrics file.
        """
        if self.output_dir is None and filename is None:
            raise ValueError("No output directory or filename specified")

        if filename is None:
            filename = f"{self.step_name}_metrics.json"

        if self.output_dir:
            filepath = self.output_dir / filename
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            filepath = Path(filename)

        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, cls=MetricsEncoder)

        return filepath

    def log_issues(self):
        """Log any warnings or errors to the logger."""
        for w in self.warnings:
            logger.warning(f"Metric warning: {w}")
        for e in self.errors:
            logger.error(f"Metric error: {e}")


# =============================================================================
# Step-specific metric collectors
# =============================================================================

def collect_mask_metrics(mask: np.ndarray,
                         input_vol: np.ndarray,
                         output_path: Union[str, Path],
                         input_path: Optional[str] = None,
                         params: Optional[Dict] = None) -> PipelineMetrics:
    """
    Collect metrics for mask creation step.

    Parameters
    ----------
    mask : np.ndarray
        The created binary mask.
    input_vol : np.ndarray
        The input volume.
    output_path : str or Path
        Path to the output mask file.
    input_path : str, optional
        Path to the input image.
    params : dict, optional
        Dictionary of parameters used (sigma, selem_radius, min_size, etc.)

    Returns
    -------
    PipelineMetrics
        Metrics object (already saved).
    """
    output_path = Path(output_path)
    metrics = PipelineMetrics('create_masks', str(output_path.parent))

    # Info
    if input_path:
        metrics.add_info('input_image', str(input_path), 'Input image path')
    metrics.add_info('output_mask', str(output_path), 'Output mask path')
    metrics.add_info('volume_shape', list(input_vol.shape), 'Volume shape')

    if params:
        for key, val in params.items():
            metrics.add_info(key, val, f'Parameter: {key}')

    # Mask coverage metrics
    mask_coverage = float(np.sum(mask > 0)) / mask.size
    metrics.add_metric('mask_coverage', mask_coverage,
                       description='Fraction of volume covered by mask',
                       threshold_name='mask_coverage')

    # Per-slice coverage
    if mask.ndim == 3:
        per_slice_coverage = np.mean(mask > 0, axis=(1, 2))
        metrics.add_metric('mean_slice_coverage', float(np.mean(per_slice_coverage)),
                           description='Mean mask coverage per slice')
        metrics.add_metric('min_slice_coverage', float(np.min(per_slice_coverage)),
                           description='Minimum mask coverage across slices',
                           threshold_name='min_slice_coverage')
        metrics.add_metric('std_slice_coverage', float(np.std(per_slice_coverage)),
                           description='Std dev of mask coverage across slices',
                           threshold_name='std_slice_coverage')

    metrics.save(f"{output_path.stem}_metrics.json")
    metrics.log_issues()
    return metrics


def collect_normalization_metrics(vol_normalized: np.ndarray,
                                  agarose_mask: np.ndarray,
                                  otsu_threshold: float,
                                  background_thresholds: np.ndarray,
                                  output_path: Union[str, Path],
                                  input_path: Optional[str] = None,
                                  params: Optional[Dict] = None) -> PipelineMetrics:
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
    metrics = PipelineMetrics('normalize_intensities', str(output_path.parent))

    if input_path:
        metrics.add_info('input_volume', str(input_path), 'Input volume path')
    metrics.add_info('output_volume', str(output_path), 'Output volume path')
    metrics.add_info('volume_shape', list(vol_normalized.shape), 'Volume shape')

    if params:
        for key, val in params.items():
            metrics.add_info(key, val, f'Parameter: {key}')

    # Agarose mask metrics
    agarose_coverage = float(np.sum(agarose_mask)) / agarose_mask.size
    metrics.add_metric('agarose_coverage', agarose_coverage,
                       description='Fraction of image classified as agarose',
                       threshold_name='agarose_coverage')
    metrics.add_metric('otsu_threshold', float(otsu_threshold),
                       description='Otsu threshold used for agarose detection')

    # Background normalization metrics
    metrics.add_metric('mean_background', float(np.mean(background_thresholds)),
                       description='Mean background threshold across slices')
    metrics.add_metric('std_background', float(np.std(background_thresholds)),
                       description='Std dev of background thresholds',
                       threshold_name='std_background')

    metrics.save(f"{output_path.stem}_metrics.json")
    metrics.log_issues()
    return metrics


def collect_xy_transform_metrics(transform: np.ndarray,
                                 tile_pairs_used: int,
                                 tile_shape: Tuple[int, int],
                                 residuals: np.ndarray,
                                 output_path: Union[str, Path],
                                 input_paths: Optional[List[str]] = None,
                                 params: Optional[Dict] = None,
                                 n_tiles_x: Optional[int] = None,
                                 n_tiles_y: Optional[int] = None) -> PipelineMetrics:
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
    metrics = PipelineMetrics('xy_transform_estimation', str(output_path.parent))

    if input_paths:
        metrics.add_info('input_images', input_paths, 'Input mosaic images')
    metrics.add_info('tile_shape', list(tile_shape), 'Tile shape in pixels')

    if n_tiles_x is not None:
        metrics.add_info('n_tiles_x', int(n_tiles_x), 'Number of tiles in X direction')
    if n_tiles_y is not None:
        metrics.add_info('n_tiles_y', int(n_tiles_y), 'Number of tiles in Y direction')

    if params:
        for key, val in params.items():
            metrics.add_info(key, val, f'Parameter: {key}')

    # Transform metrics
    metrics.add_metric('tile_pairs_used', tile_pairs_used,
                       description='Number of tile pairs used for estimation')
    metrics.add_metric('transform_00', float(transform[0, 0]), unit='pixels',
                       description='Transform matrix element [0,0] (row scale)')
    metrics.add_metric('transform_01', float(transform[0, 1]), unit='pixels',
                       description='Transform matrix element [0,1] (row shear)')
    metrics.add_metric('transform_10', float(transform[1, 0]), unit='pixels',
                       description='Transform matrix element [1,0] (col shear)')
    metrics.add_metric('transform_11', float(transform[1, 1]), unit='pixels',
                       description='Transform matrix element [1,1] (col scale)')

    # Compute overlap fraction from the estimated transform
    estimated_overlap_x = 1.0 - abs(transform[0, 0]) / tile_shape[0]
    estimated_overlap_y = 1.0 - abs(transform[1, 1]) / tile_shape[1]
    metrics.add_metric('estimated_overlap_x', float(estimated_overlap_x),
                       description='Estimated overlap fraction in X direction')
    metrics.add_metric('estimated_overlap_y', float(estimated_overlap_y),
                       description='Estimated overlap fraction in Y direction')

    # Residual error from least squares fit
    rms_residual = None
    if len(residuals) > 0:
        rms_residual = float(np.sqrt(np.mean(residuals)))
        metrics.add_metric('rms_residual', rms_residual, unit='pixels',
                           description='RMS residual from least squares fit',
                           threshold_name='rms_residual')

    # Accumulated positioning error across the mosaic
    if n_tiles_x is not None and n_tiles_y is not None:
        expected_step_y = tile_shape[0] * (1.0 - (params or {}).get('initial_overlap', 0.2))
        expected_step_x = tile_shape[1] * (1.0 - (params or {}).get('initial_overlap', 0.2))
        systematic_err_y = abs(float(transform[0, 0]) - expected_step_y) * (n_tiles_y - 1)
        systematic_err_x = abs(float(transform[1, 1]) - expected_step_x) * (n_tiles_x - 1)
        accumulated_systematic_px = float(np.sqrt(systematic_err_y**2 + systematic_err_x**2))
        metrics.add_metric('accumulated_systematic_error_px', accumulated_systematic_px,
                           unit='pixels',
                           description='Estimated accumulated systematic positioning error across mosaic')
        if rms_residual is not None:
            accumulated_random_px = rms_residual * float(np.sqrt(max(n_tiles_x, n_tiles_y)))
            metrics.add_metric('accumulated_random_error_px', accumulated_random_px,
                               unit='pixels',
                               description='Estimated accumulated random positioning error across mosaic')

    metrics.save()
    metrics.log_issues()
    return metrics


def collect_pairwise_registration_metrics(registration_error: float,
                                          tx: float, ty: float, rotation_deg: float,
                                          best_z_index: int,
                                          expected_z_index: int,
                                          output_path: Union[str, Path],
                                          fixed_path: Optional[str] = None,
                                          moving_path: Optional[str] = None,
                                          params: Optional[Dict] = None) -> PipelineMetrics:
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

    Returns
    -------
    PipelineMetrics
        Metrics object (already saved).
    """
    output_path = Path(output_path)
    metrics = PipelineMetrics('pairwise_registration', str(output_path))

    if fixed_path:
        metrics.add_info('fixed_volume', str(fixed_path), 'Path to fixed volume')
    if moving_path:
        metrics.add_info('moving_volume', str(moving_path), 'Path to moving volume')
    metrics.add_info('best_z_offset', int(best_z_index), 'Best matching z-index in fixed volume')

    if params:
        for key, val in params.items():
            metrics.add_info(key, val, f'Parameter: {key}')

    translation_magnitude = float(np.sqrt(tx ** 2 + ty ** 2))

    metrics.add_metric('registration_error', float(registration_error),
                       description='Registration error (lower is better)',
                       threshold_name='registration_error')
    metrics.add_metric('translation_x', float(tx), unit='pixels',
                       description='Translation in X direction')
    metrics.add_metric('translation_y', float(ty), unit='pixels',
                       description='Translation in Y direction')
    metrics.add_metric('translation_magnitude', translation_magnitude, unit='pixels',
                       description='Total translation magnitude',
                       threshold_name='translation_magnitude')
    metrics.add_metric('rotation', float(rotation_deg), unit='degrees',
                       description='Rotation angle',
                       threshold_name='rotation_degrees')
    metrics.add_metric('z_drift', int(abs(best_z_index - expected_z_index)), unit='voxels',
                       description='Deviation from expected z-index')

    metrics.save()
    metrics.log_issues()
    return metrics


def collect_interface_crop_metrics(detected_interface: int,
                                   crop_depth_px: int,
                                   start_idx: int, end_idx: int,
                                   input_shape: Tuple[int, ...],
                                   output_shape: Tuple[int, ...],
                                   resolution_um: float,
                                   output_path: Union[str, Path],
                                   input_path: Optional[str] = None,
                                   padding_needed: bool = False) -> PipelineMetrics:
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
    metrics = PipelineMetrics('crop_interface', str(output_path.parent))

    if input_path:
        metrics.add_info('input_volume', str(input_path), 'Input volume path')
    metrics.add_info('output_volume', str(output_path), 'Output volume path')
    metrics.add_info('input_shape', list(input_shape), 'Input volume shape')
    metrics.add_info('output_shape', list(output_shape), 'Output volume shape')
    metrics.add_info('resolution_um', float(resolution_um), 'Resolution in microns')

    metrics.add_metric('detected_interface_depth', int(detected_interface), unit='voxels',
                       description='Detected interface depth in voxels')
    metrics.add_metric('detected_interface_depth_um', float(detected_interface * resolution_um), unit='um',
                       description='Detected interface depth in microns')
    metrics.add_metric('crop_depth', int(crop_depth_px), unit='voxels',
                       description='Cropping depth in voxels')
    metrics.add_metric('start_index', int(start_idx), unit='voxels',
                       description='Start index for cropping')
    metrics.add_metric('end_index', int(end_idx), unit='voxels',
                       description='End index for cropping')

    # Quality checks
    if detected_interface < 5:
        metrics.add_metric('interface_quality', 'warning',
                           description='Interface detected very close to start - may be incorrect')
    elif detected_interface > input_shape[0] * 0.5:
        metrics.add_metric('interface_quality', 'warning',
                           description='Interface detected past halfway - check detection')
    else:
        metrics.add_metric('interface_quality', 'ok',
                           description='Interface detection appears reasonable')

    metrics.add_info('padding_needed', padding_needed, 'Whether padding was required')

    metrics.save(f"{output_path.stem}_metrics.json")
    metrics.log_issues()
    return metrics


def collect_psf_compensation_metrics(psf: np.ndarray,
                                     agarose_coverage: float,
                                     output_path: Union[str, Path],
                                     input_path: Optional[str] = None,
                                     fit_gaussian: bool = False) -> PipelineMetrics:
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
    metrics = PipelineMetrics('psf_compensation', str(output_path.parent))

    if input_path:
        metrics.add_info('input_volume', str(input_path), 'Input volume path')
    metrics.add_info('output_volume', str(output_path), 'Output volume path')
    metrics.add_info('fit_gaussian', fit_gaussian, 'Whether Gaussian fit was used')

    # PSF profile metrics
    psf_max = float(np.max(psf))
    psf_peak_index = int(np.argmax(psf))
    metrics.add_metric('psf_max', psf_max,
                       description='Maximum PSF value',
                       custom_thresholds={'warning': 0.1, 'error': 0.05, 'higher_is_better': True})
    metrics.add_metric('psf_peak_depth', psf_peak_index, unit='voxels',
                       description='Depth of PSF peak')

    metrics.add_metric('agarose_coverage', agarose_coverage,
                       description='Fraction of image classified as agarose',
                       threshold_name='agarose_coverage')

    # Profile quality assessment
    if psf_max < 0.05:
        metrics.add_metric('profile_quality', 'poor',
                           description='PSF profile quality assessment - very low signal')
    elif psf_peak_index < 5 or psf_peak_index > len(psf) * 0.8:
        metrics.add_metric('profile_quality', 'warning',
                           description='PSF peak at unexpected depth')
    else:
        metrics.add_metric('profile_quality', 'good',
                           description='PSF profile appears reasonable')

    metrics.save(f"{output_path.stem}_metrics.json")
    metrics.log_issues()
    return metrics


def collect_stack_metrics(output_shape: Tuple[int, ...],
                          z_offsets: np.ndarray,
                          num_slices: int,
                          resolution: List[float],
                          output_path: Union[str, Path],
                          blend_enabled: bool = False,
                          normalize_enabled: bool = False) -> PipelineMetrics:
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

    Returns
    -------
    PipelineMetrics
        Metrics object (already saved).
    """
    output_path = Path(output_path)
    metrics = PipelineMetrics('stack_slices', str(output_path.parent))

    metrics.add_info('output_volume', str(output_path), 'Output stacked volume path')
    metrics.add_info('num_slices', num_slices, 'Number of slices stacked')
    metrics.add_info('output_shape', list(output_shape), 'Final output shape')
    metrics.add_info('resolution', list(resolution), 'Output resolution')
    metrics.add_info('blending_enabled', blend_enabled, 'Whether blending was enabled')
    metrics.add_info('normalization_enabled', normalize_enabled, 'Whether normalization was enabled')

    z_offsets = np.asarray(z_offsets)
    metrics.add_info('z_offsets', z_offsets.tolist(), 'Z-offsets between consecutive slices')

    metrics.add_metric('total_z_depth', int(output_shape[0]), unit='voxels',
                       description='Total Z depth of stacked volume')
    metrics.add_metric('mean_z_offset', float(np.mean(z_offsets)), unit='voxels',
                       description='Mean Z-offset between slices')
    metrics.add_metric('std_z_offset', float(np.std(z_offsets)), unit='voxels',
                       description='Std dev of Z-offsets',
                       threshold_name='z_offset_std')

    z_offset_range = float(np.max(z_offsets) - np.min(z_offsets))
    metrics.add_metric('z_offset_range', z_offset_range, unit='voxels',
                       description='Range of Z-offsets (max - min)',
                       threshold_name='z_offset_range')

    metrics.save(f"{output_path.stem}_metrics.json")
    metrics.log_issues()
    return metrics


def collect_stitch_3d_metrics(input_shape: Tuple[int, ...],
                              output_shape: Tuple[int, ...],
                              num_tiles: int,
                              resolution: List[float],
                              output_path: Union[str, Path],
                              input_path: Optional[str] = None,
                              blending_method: str = 'diffusion') -> PipelineMetrics:
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
    metrics = PipelineMetrics('stitch_3d', str(output_path.parent))

    if input_path:
        metrics.add_info('input_volume', str(input_path), 'Input mosaic grid path')
    metrics.add_info('output_volume', str(output_path), 'Output stitched volume path')
    metrics.add_info('input_shape', list(input_shape), 'Input mosaic shape')
    metrics.add_info('output_shape', list(output_shape), 'Output stitched shape')
    metrics.add_info('num_tiles', num_tiles, 'Number of tiles stitched')
    metrics.add_info('resolution', list(resolution), 'Output resolution')
    metrics.add_info('blending_method', blending_method, 'Blending method used')

    # Compute compression ratio (how much the stitching reduced overlap)
    input_pixels = np.prod(input_shape)
    output_pixels = np.prod(output_shape)
    overlap_reduction = 1.0 - (output_pixels / input_pixels) if input_pixels > 0 else 0.0
    metrics.add_metric('overlap_reduction', float(overlap_reduction),
                       description='Fraction of pixels removed by stitching (overlap)')

    metrics.save(f"{output_path.stem}_metrics.json")
    metrics.log_issues()
    return metrics


# =============================================================================
# Aggregation and reporting utilities
# =============================================================================

def load_metrics(filepath: Union[str, Path]) -> Dict:
    """
    Load metrics from a JSON file.

    Parameters
    ----------
    filepath : str or Path
        Path to the metrics JSON file.

    Returns
    -------
    dict
        Loaded metrics dictionary.
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def aggregate_metrics(metrics_dir: Union[str, Path],
                      pattern: str = "*_metrics.json") -> Dict[str, List[Dict]]:
    """
    Aggregate all metrics files from a directory.

    Parameters
    ----------
    metrics_dir : str or Path
        Directory containing metrics files.
    pattern : str
        Glob pattern to match metrics files.

    Returns
    -------
    dict
        Dictionary with step names as keys and lists of metrics as values.
    """
    metrics_dir = Path(metrics_dir)
    aggregated: Dict[str, List[Dict]] = {}

    for metrics_file in sorted(metrics_dir.rglob(pattern)):
        try:
            metrics = load_metrics(metrics_file)
            step_name = metrics.get('step_name', 'unknown')
            if step_name not in aggregated:
                aggregated[step_name] = []
            metrics['source_file'] = str(metrics_file)
            aggregated[step_name].append(metrics)
        except Exception as e:
            logger.warning(f"Could not load {metrics_file}: {e}")

    return aggregated


def compute_summary_statistics(metrics_list: List[Dict]) -> Dict:
    """
    Compute summary statistics for a list of metrics from the same step.

    Parameters
    ----------
    metrics_list : list
        List of metrics dictionaries from the same step.

    Returns
    -------
    dict
        Summary statistics for numerical metrics.
    """
    if not metrics_list:
        return {}

    # Collect all numerical values per metric name
    numerical_values: Dict[str, List[float]] = {}
    statuses: List[str] = []

    for m in metrics_list:
        statuses.append(m.get('overall_status', 'unknown'))
        for name, data in m.get('metrics', {}).items():
            value = data.get('value')
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if name not in numerical_values:
                    numerical_values[name] = []
                numerical_values[name].append(float(value))

    # Compute statistics
    summary: Dict[str, Any] = {
        'count': len(metrics_list),
        'status_counts': {
            'ok': statuses.count('ok'),
            'warning': statuses.count('warning'),
            'error': statuses.count('error')
        }
    }

    for name, values in numerical_values.items():
        if values:
            summary[name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values))
            }

    return summary
