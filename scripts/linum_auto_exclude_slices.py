#!/usr/bin/env python3
"""Detect extended clusters of consecutive low-quality pairwise registrations
and output a list of slice IDs whose transforms should be force-skipped during
stacking (motor-only positioning).

Reads ``pairwise_registration_metrics.json`` files from the registration output
directory.  When *N* or more consecutive slice pairs all have
``z_correlation < threshold``, the interior slices of that cluster are flagged
for force-skipping.

Usage
-----
    linum_auto_exclude_slices.py transforms/ auto_exclude.csv \
        --consecutive_threshold 3 --z_corr_threshold 0.4
"""

import argparse
import csv
import json
import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)


def build_parser():
    p = argparse.ArgumentParser(
        description="Detect consecutive low-quality registration clusters "
                    "and output a force-skip list for stacking.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("transforms_dir", type=Path,
                   help="Directory containing per-slice subdirectories with "
                        "pairwise_registration_metrics.json files.")
    p.add_argument("output_csv", type=Path,
                   help="Output CSV listing slice IDs to force-skip.")
    p.add_argument("--consecutive_threshold", type=int, default=3,
                   help="Minimum consecutive bad pairs to trigger exclusion. "
                        "[%(default)s]")
    p.add_argument("--z_corr_threshold", type=float, default=0.4,
                   help="z_correlation below this marks a pair as bad. "
                        "[%(default)s]")
    return p


def load_registration_metrics(transforms_dir: Path):
    """Load z_correlation from each pairwise_registration_metrics.json.

    Returns a sorted list of (moving_slice_id: int, z_correlation: float).
    The moving slice ID is extracted from the directory name.
    """
    metrics = []
    pattern = re.compile(r"slice_z(\d+)")

    # os.walk with followlinks=True is used instead of Path.rglob() because
    # Nextflow stages input directories as symlinks; rglob does not follow
    # symlinks in Python 3.12+.
    found_files = []
    for root, _dirs, files in os.walk(str(transforms_dir), followlinks=True):
        if "pairwise_registration_metrics.json" in files:
            found_files.append(Path(root) / "pairwise_registration_metrics.json")

    for metrics_file in sorted(found_files):
        m = pattern.search(metrics_file.parent.name)
        if not m:
            continue
        slice_id = int(m.group(1))
        with open(metrics_file) as f:
            data = json.load(f)
        z_corr = data.get("metrics", {}).get("z_correlation", {}).get("value")
        if z_corr is not None:
            metrics.append((slice_id, float(z_corr)))

    metrics.sort(key=lambda x: x[0])
    return metrics


def find_bad_clusters(metrics, consecutive_threshold, z_corr_threshold):
    """Find clusters of consecutive slice pairs where z_corr < threshold.

    Returns a list of clusters, each being a list of (slice_id, z_corr).
    Only clusters with length >= consecutive_threshold are included.
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

    # Don't forget the last cluster
    if len(current_cluster) >= consecutive_threshold:
        clusters.append(current_cluster)

    return clusters


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s: %(message)s")
    args = build_parser().parse_args()

    metrics = load_registration_metrics(args.transforms_dir)
    if not metrics:
        logger.warning("No registration metrics found in %s", args.transforms_dir)
        # Write empty CSV
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["slice_id", "z_correlation", "exclude_reason"])
        return

    logger.info("Loaded %d registration metrics", len(metrics))

    clusters = find_bad_clusters(
        metrics, args.consecutive_threshold, args.z_corr_threshold
    )

    exclude_slices = []
    for cluster in clusters:
        ids = [s[0] for s in cluster]
        corrs = [s[1] for s in cluster]
        logger.info(
            "Bad cluster: slices z%s–z%s (%d pairs, z_corr range %.3f–%.3f)",
            str(ids[0]).zfill(2), str(ids[-1]).zfill(2),
            len(cluster), min(corrs), max(corrs),
        )
        for slice_id, z_corr in cluster:
            exclude_slices.append({
                "slice_id": slice_id,
                "z_correlation": round(z_corr, 4),
                "exclude_reason": "consecutive_low_z_corr",
            })

    # Write output CSV
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["slice_id", "z_correlation", "exclude_reason"]
        )
        writer.writeheader()
        writer.writerows(exclude_slices)

    logger.info(
        "Auto-exclude: %d slices in %d cluster(s) → %s",
        len(exclude_slices), len(clusters), args.output_csv,
    )


if __name__ == "__main__":
    main()
