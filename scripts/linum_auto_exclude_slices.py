#!/usr/bin/env python3
r"""Detect extended clusters of consecutive low-quality pairwise registrations.

Also stamps the affected slices as auto-excluded in ``slice_config.csv``.

Reads ``pairwise_registration_metrics.json`` files from the registration
output directory. Any cluster of consecutive slice pairs of length at least
``--consecutive_threshold`` whose ``z_correlation`` values are all below
``--z_corr_threshold`` marks *every* slice in that cluster (including the
endpoints) with ``auto_excluded=true`` / ``auto_exclude_reason=consecutive_low_z_corr``.
Downstream stacking then treats those slices as motor-only (``use=false`` OR
``auto_excluded=true`` → force-skip).

Usage
-----
    linum_auto_exclude_slices.py transforms/ slice_config_in.csv slice_config_out.csv \\
        --consecutive_threshold 3 --z_corr_threshold 0.4
"""

import argparse
import json
import logging
import operator
import os
import re
from pathlib import Path
from typing import Any

from linumpy.io import slice_config as slice_config_io

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Run function."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "transforms_dir",
        type=Path,
        help="Directory containing per-slice subdirectories with pairwise_registration_metrics.json files.",
    )
    p.add_argument(
        "slice_config_in",
        type=Path,
        help="Input slice_config.csv.",
    )
    p.add_argument(
        "slice_config_out",
        type=Path,
        help="Output slice_config.csv (stamped with auto_excluded / auto_exclude_reason).",
    )
    p.add_argument(
        "--consecutive_threshold",
        type=int,
        default=3,
        help="Minimum consecutive bad pairs to trigger exclusion. [%(default)s]",
    )
    p.add_argument(
        "--z_corr_threshold", type=float, default=0.4, help="z_correlation below this marks a pair as bad. [%(default)s]"
    )
    return p


def load_registration_metrics(transforms_dir: Path) -> Any:
    """Load z_correlation from each pairwise_registration_metrics.json.

    Returns a sorted list of ``(moving_slice_id: int, z_correlation: float)``.
    The moving slice ID is extracted from the directory name.
    """
    metrics = []
    pattern = re.compile(r"slice_z(\d+)")

    found_files = []
    for root, _dirs, files in os.walk(str(transforms_dir), followlinks=True):
        if "pairwise_registration_metrics.json" in files:
            found_files.append(Path(root) / "pairwise_registration_metrics.json")

    for metrics_file in sorted(found_files):
        m = pattern.search(metrics_file.parent.name)
        if not m:
            continue
        slice_id = int(m.group(1))
        with Path(metrics_file).open() as f:
            data = json.load(f)
        z_corr = data.get("metrics", {}).get("z_correlation", {}).get("value")
        if z_corr is not None:
            metrics.append((slice_id, float(z_corr)))

    metrics.sort(key=operator.itemgetter(0))
    return metrics


def find_bad_clusters(metrics: Any, consecutive_threshold: float, z_corr_threshold: float) -> Any:
    """Find clusters of consecutive slice pairs where z_corr < threshold.

    Returns a list of clusters, each being a list of ``(slice_id, z_corr)``.
    Only clusters with length ``>= consecutive_threshold`` are included.
    """
    clusters = []
    current_cluster = []

    for slice_id, z_corr in metrics:
        if z_corr < z_corr_threshold:
            current_cluster.append((slice_id, z_corr))
        else:
            if len(current_cluster) >= consecutive_threshold:
                clusters.append(current_cluster)
            current_cluster = []

    if len(current_cluster) >= consecutive_threshold:
        clusters.append(current_cluster)

    return clusters


def main() -> None:
    """Run function."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args()

    metrics = load_registration_metrics(args.transforms_dir)
    if not metrics:
        logger.warning("No registration metrics found in %s -- copying slice_config unchanged", args.transforms_dir)
        slice_config_io.stamp_many(args.slice_config_in, args.slice_config_out, {})
        return

    logger.info("Loaded %d registration metrics", len(metrics))

    clusters = find_bad_clusters(metrics, args.consecutive_threshold, args.z_corr_threshold)

    updates: dict[str, dict[str, object]] = {}
    for cluster in clusters:
        ids = [s[0] for s in cluster]
        corrs = [s[1] for s in cluster]
        logger.info(
            "Bad cluster: slices z%s-z%s (%d pairs, z_corr range %.3f-%.3f)",
            str(ids[0]).zfill(2),
            str(ids[-1]).zfill(2),
            len(cluster),
            min(corrs),
            max(corrs),
        )
        for slice_id, _z_corr in cluster:
            sid = slice_config_io.normalize_slice_id(slice_id)
            updates[sid] = {
                "auto_excluded": True,
                "auto_exclude_reason": "consecutive_low_z_corr",
            }

    slice_config_io.stamp_many(args.slice_config_in, args.slice_config_out, updates)

    logger.info(
        "Auto-exclude: %d slices in %d cluster(s) → %s",
        len(updates),
        len(clusters),
        args.slice_config_out,
    )


if __name__ == "__main__":
    main()
