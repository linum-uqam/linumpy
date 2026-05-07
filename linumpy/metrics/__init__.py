"""Pipeline metrics collection, aggregation, and reporting.

This package keeps a stable public API:

- :class:`PipelineMetrics` and :class:`MetricsEncoder` (core)
- ``collect_*_metrics`` functions for each pipeline step (collectors)
- :func:`aggregate_metrics`, :func:`compute_summary_statistics`,
  :func:`load_metrics` (aggregation/IO)

Usage:
    # Use step-specific collectors (recommended)
    from linumpy.metrics import collect_pairwise_registration_metrics
"""

from linumpy.metrics.aggregate import aggregate_metrics, compute_summary_statistics
from linumpy.metrics.collectors import (
    collect_auto_exclude_metrics,
    collect_common_space_metrics,
    collect_interface_crop_metrics,
    collect_normalization_metrics,
    collect_pairwise_registration_metrics,
    collect_psf_compensation_metrics,
    collect_quality_assessment_metrics,
    collect_rehoming_metrics,
    collect_slice_interpolation_metrics,
    collect_stack_metrics,
    collect_stitch_3d_metrics,
    collect_xy_transform_metrics,
)
from linumpy.metrics.core import MetricsEncoder, PipelineMetrics, load_metrics

__all__ = [
    "MetricsEncoder",
    "PipelineMetrics",
    "aggregate_metrics",
    "collect_auto_exclude_metrics",
    "collect_common_space_metrics",
    "collect_interface_crop_metrics",
    "collect_normalization_metrics",
    "collect_pairwise_registration_metrics",
    "collect_psf_compensation_metrics",
    "collect_quality_assessment_metrics",
    "collect_rehoming_metrics",
    "collect_slice_interpolation_metrics",
    "collect_stack_metrics",
    "collect_stitch_3d_metrics",
    "collect_xy_transform_metrics",
    "compute_summary_statistics",
    "load_metrics",
]
