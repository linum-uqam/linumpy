#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detect acquisition segment breaks (sample remounting events) from common-space slices.

During serial-section acquisitions, the sample sometimes needs to be remounted
(e.g., water level drops, focus loss). This physically rotates and/or translates
the tissue block, causing a sudden orientation jump between two consecutive slices
that the normal per-slice registration correction (capped at 1°) cannot fix.

This script detects such events by running unclamped 2D rigid registration between
all consecutive slice pairs and identifying sudden rotation or translation jumps.

Strategy
--------
For each consecutive pair of common-space slices:
  1. Compute a 2D maximum-intensity projection (AIP) of each 3D volume.
  2. Run SimpleITK Euler2DTransform registration with a large search range
     (default ±45°, no clamping).
  3. Record the rotation angle and XY translation for every pair.

Break detection:
  A pair is flagged as a segment break when the absolute rotation magnitude
  exceeds ``--rotation_threshold`` (default 3°) *and* deviates from the local
  running median by more than that same threshold.  Using both conditions avoids
  falsely flagging samples that happen to be mounted with a global tilt.

Outputs (in ``out_directory``)
------------------------------
rotations.csv
    Per-pair measurements (fixed_id, moving_id, rotation_deg, tx_px, ty_px,
    z_correlation, is_break).

segment_corrections.json
    Correction transforms to apply to slices after each detected break, in order
    to realign them with the pre-break segment.  This file is consumed by
    ``linum_apply_segment_corrections.py``.

segment_breaks.png
    Visualisation of per-pair rotation angles with detected breaks marked.

Usage
-----
    linum_detect_segment_breaks.py <slices_dir> <out_directory> \\
        [--rotation_threshold 3.0] [--max_rotation_search 45.0] \\
        [--resolution 10.0]
"""

import linumpy._thread_config  # noqa: F401

import argparse
import json
import logging
import re
from pathlib import Path

import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk

from linumpy.io.zarr import read_omezarr, save_omezarr
from linumpy.utils.io import add_overwrite_arg

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_slices_dir',
                   help='Directory containing common-space slice .ome.zarr files '
                        '(output of bring_to_common_space).')
    p.add_argument('out_directory',
                   help='Output directory for detection results.')

    p.add_argument('--rotation_threshold', type=float, default=3.0,
                   help='Rotation jump that flags a segment break, in degrees. '
                        '[%(default)s]')
    p.add_argument('--max_rotation_search', type=float, default=45.0,
                   help='Maximum rotation search range for registration, in degrees. '
                        '[%(default)s]')
    p.add_argument('--resolution', type=float, default=None,
                   help='In-plane (XY) resolution in µm/pixel. '
                        'Read from OME metadata if not provided.')
    p.add_argument('--local_window', type=int, default=7,
                   help='Window half-width (slices) for local-median reference. '
                        '[%(default)s]')
    p.add_argument('--shrink_factors', type=int, nargs='+',
                   default=[8, 4, 2, 1],
                   help='Multi-resolution shrink factors for registration. '
                        '[%(default)s]')
    p.add_argument('--smoothing_sigmas', type=float, nargs='+',
                   default=[4.0, 2.0, 1.0, 0.0],
                   help='Gaussian smoothing sigmas per resolution level. '
                        '[%(default)s]')
    p.add_argument('--metric_threshold', type=float, default=-0.40,
                   help='Registration metric quality gate (correlation, range −1..0). '
                        'Pairs whose final metric is above this value are considered '
                        'unreliable (optimizer did not converge) and are excluded from '
                        'break detection. Default −0.40 rejects most failed '
                        'registrations while keeping well-converged pairs. '
                        '[%(default)s]')
    p.add_argument('--translation_threshold', type=float, default=0.0,
                   help='XY translation magnitude (pixels) that flags a segment break. '
                        'Only reliable pairs (metric < metric_threshold) are checked. '
                        '0 disables translation-based detection. [%(default)s]')
    p.add_argument('--refine_translations', action='store_true', default=False,
                   help='Use AIP registration tx/ty from ALL reliable pairs to '
                        'progressively correct XY drift, not just at detected breaks. '
                        'This corrects accumulated motor position errors across the '
                        'entire volume. Unreliable pairs (metric > metric_threshold) '
                        'are skipped. Rotation corrections still only apply at breaks.')

    add_overwrite_arg(p)
    return p


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_slice_id(path: Path) -> int:
    """Return the numeric slice ID from a path like slice_z36_normalize.ome.zarr."""
    m = re.search(r'z(\d+)', path.name)
    return int(m.group(1)) if m else -1


def _discover_slices(slices_dir: Path) -> list[Path]:
    """Return sorted list of .ome.zarr slice directories."""
    slices = sorted(
        [p for p in slices_dir.iterdir()
         if p.is_dir() and p.name.endswith('.ome.zarr') and re.search(r'z\d+', p.name)],
        key=_extract_slice_id
    )
    if not slices:
        raise FileNotFoundError(
            f"No .ome.zarr slice directories found in {slices_dir}")
    return slices


def _load_aip(zarr_path: Path) -> tuple[np.ndarray, list]:
    """Load a volume and return its maximum-intensity projection along Z, plus resolution."""
    vol, res = read_omezarr(str(zarr_path), level=0)
    arr = np.array(vol)
    aip = arr.max(axis=0).astype(np.float32)
    return aip, res


def _normalize(image: np.ndarray) -> np.ndarray:
    """Normalise to [0, 1] using 5th–95th percentile of non-zero values."""
    valid = image > 0
    if not np.any(valid):
        return np.zeros_like(image, dtype=np.float32)
    pmin = float(np.percentile(image[valid], 5))
    pmax = float(np.percentile(image[valid], 95))
    if pmax <= pmin:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - pmin) / (pmax - pmin), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_full_range(
    fixed_aip: np.ndarray,
    moving_aip: np.ndarray,
    max_rotation_deg: float = 45.0,
    shrink_factors: list[int] = [8, 4, 2, 1],
    smoothing_sigmas: list[float] = [4.0, 2.0, 1.0, 0.0],
) -> dict:
    """
    Run unclamped 2D rigid (Euler) registration between two AIP images.

    Returns a dict with keys: rotation_deg, tx_px, ty_px, metric.
    """
    fixed_n = _normalize(fixed_aip)
    moving_n = _normalize(moving_aip)

    fixed_itk = sitk.GetImageFromArray(fixed_n)
    moving_itk = sitk.GetImageFromArray(moving_n)

    # Initialise transform at geometry centre
    tx_init = sitk.CenteredTransformInitializer(
        fixed_itk, moving_itk,
        sitk.Euler2DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsCorrelation()
    reg.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=500,
        convergenceMinimumValue=1e-7,
        convergenceWindowSize=20,
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetInitialTransform(tx_init, inPlace=False)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetShrinkFactorsPerLevel(list(shrink_factors))
    reg.SetSmoothingSigmasPerLevel(list(smoothing_sigmas))

    try:
        result = reg.Execute(fixed_itk, moving_itk)
        metric = float(reg.GetMetricValue())

        # Unwrap composite transform if needed
        if result.GetName() == 'CompositeTransform':
            inner = result.GetNthTransform(0)
        else:
            inner = result

        euler = sitk.Euler2DTransform(inner)
        angle_deg = float(np.degrees(euler.GetAngle()))
        tx, ty = euler.GetTranslation()

        # Clamp only for reporting (detect even extreme angles up to search range)
        if abs(angle_deg) > max_rotation_deg:
            logger.warning(
                f"  Registration angle {angle_deg:.1f}° exceeds search range "
                f"±{max_rotation_deg}°; result may be unreliable.")

        return dict(rotation_deg=angle_deg, tx_px=float(tx), ty_px=float(ty),
                    metric=metric)

    except Exception as exc:
        logger.warning(f"  Registration failed: {exc}")
        return dict(rotation_deg=0.0, tx_px=0.0, ty_px=0.0, metric=float('inf'))


# ---------------------------------------------------------------------------
# Break detection
# ---------------------------------------------------------------------------

def detect_breaks(
    rotations: np.ndarray,
    rotation_threshold: float,
    local_window: int,
    metrics: np.ndarray | None = None,
    metric_threshold: float = -0.40,
    translations: np.ndarray | None = None,
    translation_threshold: float = 0.0,
) -> np.ndarray:
    """
    Return boolean mask of break positions in a rotation-per-pair array.

    A pair is a segment break when it meets the rotation condition OR the
    translation condition (both require reliable registration quality):

    Rotation break — ALL three conditions hold:
      1. metric < metric_threshold  (registration converged reliably)
      2. |rotation| > rotation_threshold
      3. |rotation - local_median| > rotation_threshold

    Translation break — ALL two conditions hold:
      1. metric < metric_threshold  (registration converged reliably)
      2. sqrt(tx² + ty²) > translation_threshold  (only when translation_threshold > 0)

    Condition 1 (metric gate) guards against pairs where the optimizer reached a
    poor local minimum — such pairs produce unreliable large estimates that must
    not be flagged as breaks.

    Condition 3 (for rotation) allows samples that are globally tilted (non-zero
    mean rotation) to be handled correctly — only *jumps* relative to recent
    history are flagged.
    """
    n = len(rotations)
    is_break = np.zeros(n, dtype=bool)

    for i in range(n):
        # Skip pairs where registration quality is too poor to trust
        if metrics is not None:
            m = metrics[i]
            if not np.isfinite(m) or m > metric_threshold:
                logger.debug(
                    f"  Pair {i}: skipped (metric={m:.4f} > threshold {metric_threshold})")
                continue

        lo = max(0, i - local_window)
        hi = min(n, i + local_window + 1)
        # Exclude self from local median
        neighbours = np.concatenate([rotations[lo:i], rotations[i + 1:hi]])
        local_med = float(np.median(neighbours)) if len(neighbours) > 0 else 0.0

        abs_rot = abs(rotations[i])
        abs_jump = abs(rotations[i] - local_med)

        if abs_rot > rotation_threshold and abs_jump > rotation_threshold:
            is_break[i] = True

        # Translation-based break detection (independent of rotation)
        if (not is_break[i] and translation_threshold > 0
                and translations is not None):
            trans_mag = float(translations[i])
            if trans_mag > translation_threshold:
                is_break[i] = True
                logger.debug(
                    f"  Pair {i}: translation break (mag={trans_mag:.1f}px > "
                    f"threshold {translation_threshold:.1f}px)")

    return is_break


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = _build_arg_parser()
    args = p.parse_args()

    slices_dir = Path(args.in_slices_dir)
    out_dir = Path(args.out_directory)

    if out_dir.exists() and not args.overwrite:
        p.error(f"Output directory already exists: {out_dir}. Use -f to overwrite.")
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Discover slices
    # -----------------------------------------------------------------------
    slices = _discover_slices(slices_dir)
    slice_ids = [_extract_slice_id(s) for s in slices]
    logger.info(f"Found {len(slices)} slices: z{slice_ids[0]:02d} – z{slice_ids[-1]:02d}")

    if len(slices) < 2:
        logger.warning("Need at least 2 slices — nothing to do.")
        return

    # -----------------------------------------------------------------------
    # Pairwise registration
    # -----------------------------------------------------------------------
    records = []
    logger.info("Running pairwise registrations (this may take a few minutes)…")

    for i in range(len(slices) - 1):
        fixed_path = slices[i]
        moving_path = slices[i + 1]
        fid = slice_ids[i]
        mid = slice_ids[i + 1]

        logger.info(f"  Registering z{fid:02d} → z{mid:02d}")

        fixed_aip, res = _load_aip(fixed_path)
        moving_aip, _ = _load_aip(moving_path)

        result = register_full_range(
            fixed_aip, moving_aip,
            max_rotation_deg=args.max_rotation_search,
            shrink_factors=args.shrink_factors,
            smoothing_sigmas=args.smoothing_sigmas,
        )

        records.append(dict(
            fixed_id=fid,
            moving_id=mid,
            rotation_deg=result['rotation_deg'],
            tx_px=result['tx_px'],
            ty_px=result['ty_px'],
            metric=result['metric'],
        ))

        logger.info(
            f"    rot={result['rotation_deg']:+.2f}°  "
            f"tx={result['tx_px']:+.1f}px  ty={result['ty_px']:+.1f}px  "
            f"metric={result['metric']:.4f}"
        )

    df = pd.DataFrame(records)

    # -----------------------------------------------------------------------
    # Break detection
    # -----------------------------------------------------------------------
    rotations_arr = df['rotation_deg'].to_numpy()
    metrics_arr = df['metric'].to_numpy()
    translations_arr = np.sqrt(df['tx_px']**2 + df['ty_px']**2).to_numpy()
    n_skipped = int(((~np.isfinite(metrics_arr)) | (metrics_arr > args.metric_threshold)).sum())
    if n_skipped:
        logger.info(
            f"Skipping {n_skipped} pair(s) with metric > {args.metric_threshold} "
            f"(registration did not converge reliably).")
    is_break = detect_breaks(rotations_arr, args.rotation_threshold, args.local_window,
                             metrics=metrics_arr, metric_threshold=args.metric_threshold,
                             translations=translations_arr,
                             translation_threshold=args.translation_threshold)
    df['is_break'] = is_break

    n_breaks = int(is_break.sum())
    logger.info(f"Detected {n_breaks} segment break(s).")
    if n_breaks > 0:
        break_pairs = df[df['is_break']][['fixed_id', 'moving_id', 'rotation_deg',
                                          'tx_px', 'ty_px']]
        for _, row in break_pairs.iterrows():
            logger.info(
                f"  Break z{int(row.fixed_id):02d}→z{int(row.moving_id):02d}: "
                f"rot={row.rotation_deg:+.2f}°  "
                f"tx={row.tx_px:+.1f}px  ty={row.ty_px:+.1f}px"
            )

    # -----------------------------------------------------------------------
    # Compute cumulative corrections
    #
    # Break-only mode (default):
    #   At each break the moving segment needs a correction of
    #   (-rotation, -tx, -ty) applied to all slices from moving_id onwards.
    #   When multiple breaks exist, corrections accumulate.
    #
    # Refine-translations mode (--refine_translations):
    #   Accumulate tx/ty corrections from ALL reliable pairs (metric ok),
    #   progressively correcting motor position drift.  Rotation corrections
    #   still only apply at detected breaks.
    # -----------------------------------------------------------------------
    breaks_info = []
    per_slice_corrections = {}
    cumulative_rot = 0.0
    cumulative_tx = 0.0
    cumulative_ty = 0.0

    if args.refine_translations:
        # Progressive drift correction from ALL reliable pairs
        n_refined = 0
        for _, row in df.iterrows():
            reliable = (np.isfinite(row['metric'])
                        and row['metric'] < args.metric_threshold)

            if reliable:
                cumulative_tx -= row['tx_px']
                cumulative_ty -= row['ty_px']
                n_refined += 1

            if row['is_break']:
                cumulative_rot -= row['rotation_deg']
                breaks_info.append(dict(
                    fixed_id=int(row['fixed_id']),
                    moving_id=int(row['moving_id']),
                    measured_rotation_deg=float(row['rotation_deg']),
                    measured_tx_px=float(row['tx_px']),
                    measured_ty_px=float(row['ty_px']),
                    correction_rotation_deg=cumulative_rot,
                    correction_tx_px=cumulative_tx,
                    correction_ty_px=cumulative_ty,
                ))

            # Every moving slice gets the current cumulative correction
            per_slice_corrections[int(row['moving_id'])] = dict(
                rotation_deg=cumulative_rot,
                tx_px=cumulative_tx,
                ty_px=cumulative_ty,
            )

        # Extend to the last slice (in case it's beyond the last pair's moving_id)
        if per_slice_corrections:
            last_correction = dict(
                rotation_deg=cumulative_rot,
                tx_px=cumulative_tx,
                ty_px=cumulative_ty,
            )
            for sid in range(int(df['moving_id'].max()) + 1, slice_ids[-1] + 1):
                per_slice_corrections[sid] = last_correction

        logger.info(
            f"Translation refinement: applied tx/ty from {n_refined}/{len(df)} "
            f"reliable pair(s).  Final cumulative shift: "
            f"tx={cumulative_tx:+.1f}px  ty={cumulative_ty:+.1f}px")
    else:
        # Original break-only mode
        for _, row in df[df['is_break']].iterrows():
            cumulative_rot -= row['rotation_deg']
            cumulative_tx -= row['tx_px']
            cumulative_ty -= row['ty_px']
            breaks_info.append(dict(
                fixed_id=int(row['fixed_id']),
                moving_id=int(row['moving_id']),
                measured_rotation_deg=float(row['rotation_deg']),
                measured_tx_px=float(row['tx_px']),
                measured_ty_px=float(row['ty_px']),
                correction_rotation_deg=cumulative_rot,
                correction_tx_px=cumulative_tx,
                correction_ty_px=cumulative_ty,
            ))

        if breaks_info:
            sorted_breaks = sorted(breaks_info, key=lambda b: b['moving_id'])
            for bid_idx, binfo in enumerate(sorted_breaks):
                start = binfo['moving_id']
                end = (sorted_breaks[bid_idx + 1]['moving_id']
                       if bid_idx + 1 < len(sorted_breaks) else slice_ids[-1] + 1)
                for sid in range(start, end):
                    per_slice_corrections[sid] = dict(
                        rotation_deg=binfo['correction_rotation_deg'],
                        tx_px=binfo['correction_tx_px'],
                        ty_px=binfo['correction_ty_px'],
                    )
            last_break = sorted_breaks[-1]
            for sid in range(last_break['moving_id'], slice_ids[-1] + 1):
                per_slice_corrections[sid] = dict(
                    rotation_deg=last_break['correction_rotation_deg'],
                    tx_px=last_break['correction_tx_px'],
                    ty_px=last_break['correction_ty_px'],
                )

    corrections_data = dict(
        n_breaks=n_breaks,
        rotation_threshold_deg=args.rotation_threshold,
        metric_threshold=args.metric_threshold,
        n_pairs_skipped_low_quality=n_skipped,
        breaks=breaks_info,
        per_slice_corrections={str(k): v for k, v in per_slice_corrections.items()},
    )

    # -----------------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------------
    csv_path = out_dir / 'rotations.csv'
    df.to_csv(csv_path, index=False)
    logger.info(f"Rotation table saved to {csv_path}")

    json_path = out_dir / 'segment_corrections.json'
    with open(json_path, 'w') as fh:
        json.dump(corrections_data, fh, indent=2)
    logger.info(f"Corrections saved to {json_path}")

    # -----------------------------------------------------------------------
    # Visualisation
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    pair_labels = [f"{int(r.fixed_id)}→{int(r.moving_id)}" for _, r in df.iterrows()]
    x = np.arange(len(df))

    # Rotation
    ax = axes[0]
    ax.bar(x, df['rotation_deg'], color='steelblue', alpha=0.7, label='Rotation')
    ax.axhline(args.rotation_threshold, color='red', linestyle='--', linewidth=1,
               label=f'Threshold ±{args.rotation_threshold}°')
    ax.axhline(-args.rotation_threshold, color='red', linestyle='--', linewidth=1)
    for xi, brk in enumerate(is_break):
        if brk:
            ax.axvline(xi, color='red', linewidth=2, alpha=0.6)
    ax.set_ylabel('Rotation (°)')
    ax.set_title('Pairwise Rotation Between Consecutive Common-Space Slices')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # TX
    ax = axes[1]
    ax.bar(x, df['tx_px'], color='darkorange', alpha=0.7, label='TX')
    for xi, brk in enumerate(is_break):
        if brk:
            ax.axvline(xi, color='red', linewidth=2, alpha=0.6)
    ax.set_ylabel('TX (pixels)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # TY
    ax = axes[2]
    ax.bar(x, df['ty_px'], color='seagreen', alpha=0.7, label='TY')
    for xi, brk in enumerate(is_break):
        if brk:
            ax.axvline(xi, color='red', linewidth=2, alpha=0.6, label='Break')
    ax.set_ylabel('TY (pixels)')
    ax.set_xlabel('Slice pair')
    ax.set_xticks(x[::max(1, len(x) // 20)])
    ax.set_xticklabels(
        [pair_labels[i] for i in range(0, len(x), max(1, len(x) // 20))],
        rotation=45, ha='right', fontsize=8)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    png_path = out_dir / 'segment_breaks.png'
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Plot saved to {png_path}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SEGMENT BREAK DETECTION SUMMARY")
    print("=" * 60)
    print(f"Slices processed : {len(slices)}")
    print(f"Pairs registered : {len(df)}")
    print(f"Threshold        : ±{args.rotation_threshold}°")
    print(f"Metric threshold : {args.metric_threshold} (pairs above excluded)")
    if args.translation_threshold > 0:
        print(f"Trans. threshold : {args.translation_threshold:.1f} px (XY translation break detection)")
    if args.refine_translations:
        print(f"Refine XY        : enabled (correct drift from all reliable pairs)")
    if n_skipped:
        print(f"Pairs excluded   : {n_skipped} (low registration quality)")
    print(f"Breaks detected  : {n_breaks}")
    if n_breaks > 0:
        print()
        print("Break details:")
        for b in breaks_info:
            print(f"  z{b['fixed_id']:02d}→z{b['moving_id']:02d}  "
                  f"rot={b['measured_rotation_deg']:+.2f}°  "
                  f"tx={b['measured_tx_px']:+.1f}px  ty={b['measured_ty_px']:+.1f}px  "
                  f"cumulative correction: {b['correction_rotation_deg']:+.2f}°  "
                  f"tx={b['correction_tx_px']:+.1f}px  ty={b['correction_ty_px']:+.1f}px")
    n_corrected = len(per_slice_corrections)
    if n_corrected:
        print(f"Slices corrected : {n_corrected}")
    print("=" * 60)


if __name__ == '__main__':
    main()
