"""Aggregation utilities for pipeline metrics files."""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from linumpy.metrics.core import load_metrics

logger = logging.getLogger(__name__)


def aggregate_metrics(metrics_dir: Path, pattern: str = "*_metrics.json") -> dict[str, list[dict]]:
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
    aggregated: dict[str, list[dict]] = defaultdict(list)

    for metrics_file in sorted(metrics_dir.rglob(pattern)):
        try:
            metrics = load_metrics(metrics_file)
        except Exception as e:
            logger.warning("Could not load %s: %s", metrics_file, e)
            continue
        step_name = metrics.get("step_name", "unknown")
        metrics["source_file"] = str(metrics_file)
        aggregated[step_name].append(metrics)

    return dict(aggregated)


def compute_summary_statistics(metrics_list: list[dict]) -> dict:
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
    numerical_values: dict[str, list[float]] = defaultdict(list)
    statuses: list[str] = []

    for m in metrics_list:
        statuses.append(m.get("overall_status", "unknown"))
        for name, data in m.get("metrics", {}).items():
            value = data.get("value")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numerical_values[name].append(float(value))

    summary: dict[str, Any] = {
        "count": len(metrics_list),
        "status_counts": {
            "ok": statuses.count("ok"),
            "warning": statuses.count("warning"),
            "error": statuses.count("error"),
        },
    }

    for name, values in numerical_values.items():
        if values:
            summary[name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "median": float(np.median(values)),
            }

    return summary
