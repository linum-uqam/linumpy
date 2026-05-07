"""Core metrics primitives: JSON encoder, PipelineMetrics, and IO."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

logger = logging.getLogger(__name__)


class MetricsEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy types."""

    def default(self, o: Any) -> Any:
        """Serialize numpy integer and float types to Python builtins."""
        if isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, np.bool_):
            return bool(o)
        elif isinstance(o, Path):
            return str(o)
        return super().default(o)


class PipelineMetrics:
    """
    Class for collecting and managing metrics from pipeline steps.

    Each step can record multiple metrics with associated quality indicators.
    Metrics are saved as JSON files for later aggregation and report generation.
    """

    # Quality thresholds for common metrics (can be overridden)
    DEFAULT_THRESHOLDS: ClassVar[dict] = {
        # Mean squared error of the registration transform (normalized, unitless)
        "registration_error": {"warning": 0.05, "error": 0.15},
        # Euclidean magnitude of the estimated translation vector (pixels)
        "translation_magnitude": {"warning": 30.0, "error": 50.0},
        # Rotation angle derived from the estimated transform (degrees)
        "rotation_degrees": {"warning": 1.0, "error": 2.0},
        # Normalized cross-correlation between registered image pairs (unitless, 0-1)
        "correlation": {"warning": 0.7, "error": 0.5, "higher_is_better": True},
        # Fraction of the volume voxels classified as tissue
        "tissue_coverage": {"warning": 0.1, "error": 0.05, "higher_is_better": True},
        # Fraction of the volume voxels covered by the binary mask
        "mask_coverage": {"warning": 0.05, "error": 0.01, "higher_is_better": True},
        # Fraction of the volume voxels classified as agarose (embedding medium)
        "agarose_coverage": {"warning": 0.05, "error": 0.01, "higher_is_better": True},
        # Fraction of the volume voxels that are empty (below background threshold)
        "empty_fraction": {"warning": 0.5, "error": 0.8},
        # Depth (in pixels) of the tissue-agarose interface from the top of the volume
        "interface_depth": {"warning": 50, "error": 100},
        # Quality score of the axial intensity profile fit (unitless, 0-1)
        "profile_quality": {"warning": 0.5, "error": 0.3, "higher_is_better": True},
        # Root-mean-square residual of the least-squares transform fit (pixels)
        "rms_residual": {"warning": 5.0, "error": 15.0},
        # Standard deviation of per-slice Z offsets across the mosaic (pixels)
        "z_offset_std": {"warning": 10.0, "error": 25.0},
        # Peak-to-peak range of per-slice Z offsets across the mosaic (pixels)
        "z_offset_range": {"warning": 15.0, "error": 30.0},
        # Standard deviation of the per-slice background thresholds (normalized)
        "std_background": {"warning": 0.1, "error": 0.25},
        # Minimum mask coverage fraction across all slices
        "min_slice_coverage": {"warning": 0.02, "error": 0.005, "higher_is_better": True},
        # Standard deviation of mask coverage fractions across slices
        "std_slice_coverage": {"warning": 0.15, "error": 0.3},
        # Minimum acceptable interface depth from the top of the volume (voxels)
        "interface_min_depth_px": {"error": 5},
        # Maximum acceptable interface depth as a fraction of the volume's Z size
        "interface_max_depth_fraction": {"error": 0.5},
    }

    def __init__(self, step_name: str, output_dir: str | None = None) -> None:
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
        self.metrics: dict[str, Any] = {}
        self.warnings: list[str] = []
        self.errors: list[str] = []
        self.timestamp = datetime.now().isoformat()

    def add_metric(
        self,
        name: str,
        value: Any,
        unit: str | None = None,
        threshold_name: str | None = None,
        custom_thresholds: dict | None = None,
        description: str | None = None,
    ) -> None:
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
        metric_entry = {"value": value, "unit": unit, "description": description, "status": "ok"}

        # Evaluate quality if thresholds are provided
        thresholds = custom_thresholds or self.DEFAULT_THRESHOLDS.get(threshold_name)
        if thresholds and value is not None:
            higher_is_better = thresholds.get("higher_is_better", False)
            warning_thresh = thresholds.get("warning")
            error_thresh = thresholds.get("error")

            if higher_is_better:
                if error_thresh is not None and value < error_thresh:
                    metric_entry["status"] = "error"
                    self.errors.append(f"{name}: {value} < {error_thresh} (error threshold)")
                elif warning_thresh is not None and value < warning_thresh:
                    metric_entry["status"] = "warning"
                    self.warnings.append(f"{name}: {value} < {warning_thresh} (warning threshold)")
            else:
                if error_thresh is not None and value > error_thresh:
                    metric_entry["status"] = "error"
                    self.errors.append(f"{name}: {value} > {error_thresh} (error threshold)")
                elif warning_thresh is not None and value > warning_thresh:
                    metric_entry["status"] = "warning"
                    self.warnings.append(f"{name}: {value} > {warning_thresh} (warning threshold)")

        self.metrics[name] = metric_entry

    def add_info(self, name: str, value: Any, description: str | None = None) -> None:
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
        self.metrics[name] = {"value": value, "description": description, "status": "info"}

    def add_params(self, params: dict | None) -> None:
        """Record each entry of ``params`` as an info field. No-op if ``params`` is falsy."""
        if not params:
            return
        for key, val in params.items():
            self.add_info(key, val, f"Parameter: {key}")

    def finalize(self, filename: str | None = None) -> PipelineMetrics:
        """Save metrics, log warnings/errors, and return self."""
        self.save(filename)
        self.log_issues()
        return self

    def get_overall_status(self) -> str:
        """
        Get overall status based on all metrics.

        Returns
        -------
        str
            'error', 'warning', or 'ok'
        """
        if self.errors:
            return "error"
        elif self.warnings:
            return "warning"
        return "ok"

    def to_dict(self) -> dict:
        """
        Convert metrics to dictionary format.

        Returns
        -------
        dict
            Dictionary containing all metrics and metadata.
        """
        return {
            "step_name": self.step_name,
            "timestamp": self.timestamp,
            "overall_status": self.get_overall_status(),
            "metrics": self.metrics,
            "warnings": self.warnings,
            "errors": self.errors,
        }

    def save(self, filename: str | None = None) -> Path:
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

        with filepath.open("w") as f:
            json.dump(self.to_dict(), f, indent=2, cls=MetricsEncoder)

        return filepath

    def log_issues(self) -> None:
        """Log any warnings or errors to the logger."""
        for w in self.warnings:
            logger.warning("Metric warning: %s", w)
        for e in self.errors:
            logger.error("Metric error: %s", e)


def load_metrics(filepath: Path) -> dict:
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
    with Path(filepath).open() as f:
        return json.load(f)
