#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Suggest 3D reconstruction pipeline parameters from raw input files.

Analyses the motor-positions file (shifts_xy.csv) and, optionally, the raw
data directory produced by the preprocessing pipeline to automatically
estimate suitable nextflow.config parameters.

Estimable parameters
--------------------
From shifts_xy.csv:
  stitch_rehoming_threshold_mm    — midpoint between max-normal and min-rehoming shift
  stitch_rehoming_enabled         — true if re-homing events are detected
  stitch_rehoming_use_motor       — always recommended true when re-homing is present
  max_shift_mm                    — IQR upper bound of normal inter-slice shifts
  common_space_max_step_mm        — 95th percentile of consecutive normal shift changes
  interpolate_missing_slices      — true when gaps (moving_id - fixed_id > 1) are found

From the raw data directory (--data_dir):
  registration_slicing_interval_mm  — from slice_thickness in metadata.json / state.json
  stitch_overlap_fraction           — from overlap_fraction in metadata.json / state.json
  resolution                        — smallest standard resolution >= native lateral px size
  crop_interface_out_depth          — depth below tissue interface; user must verify
  registration_max_translation      — tile width in pixels at target resolution

Parameters that cannot be estimated automatically:
  crop_interface_out_depth    — requires tissue-specific knowledge; an estimate is given
                                based on the raw OCT depth and focus position, but should
                                be verified by inspecting a cross-section preview.

Raw data directory layout (output of the preprocessing pipeline)
-----------------------------------------------------------------
  <data_dir>/
    state.json                          # global acquisition state
    slice_z00/
      metadata.json                     # per-slice acquisition parameters
      tiles/
        tile_x##_y##_z##/
          info.txt                      # per-tile OCT scan parameters
"""

# Configure thread limits before numpy/scipy imports (optional; skipped if
# linumpy is not installed into the current environment)
try:
    import linumpy._thread_config  # noqa: F401
except ImportError:
    pass

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Default OCT axial resolution in µm (hardware constant for Thorlabs PSOCT).
# Cannot be determined from metadata files; override with --axial_res_um if known.
OCT_AXIAL_RES_UM = 3.5


# =============================================================================
# CLI
# =============================================================================

def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument("shifts_file",
                   help="Motor-positions CSV file (shifts_xy.csv)")
    p.add_argument("output_dir",
                   help="Directory for the report and suggested config snippet")
    p.add_argument("--data_dir", default=None,
                   help="Raw data directory (contains state.json and\n"
                        "slice_z##/ subdirectories). Used to read slice\n"
                        "thickness, tile overlap, and tile dimensions.")
    p.add_argument("--n_calibration_slices", type=int, default=1,
                   help="Number of leading calibration slices to skip when\n"
                        "reading per-slice metadata (default: 1, i.e. skip\n"
                        "slice_z00 which is a calibration slice). [%(default)s]")
    p.add_argument("--axial_res_um", type=float, default=OCT_AXIAL_RES_UM,
                   help=f"OCT axial resolution in µm/pixel [%(default)s µm].\n"
                        f"Used to convert tile depth (pixels) → µm.")
    p.add_argument("--resolution_um", type=float, default=None,
                   help="Override target pipeline resolution in µm/pixel.\n"
                        "Derived automatically from tile dimensions if not given.")
    p.add_argument("-f", "--overwrite", action="store_true",
                   help="Overwrite existing output directory")
    return p


# =============================================================================
# Shifts loading and analysis
# =============================================================================

def load_shifts(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ['fixed_id', 'moving_id', 'x_shift_mm', 'y_shift_mm']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in shifts file: {missing}")
    return df


def detect_rehoming(magnitudes: np.ndarray, mad_k: float = 3.0) -> np.ndarray:
    """
    Robust re-homing event detection using Median Absolute Deviation (MAD).

    MAD is insensitive to the very outliers we are trying to detect, unlike
    plain IQR whose quartiles are pulled up by re-homing events.

    Returns boolean array, True where the shift is classified as re-homing.
    """
    med = np.median(magnitudes)
    mad = np.median(np.abs(magnitudes - med))
    # 1.4826 makes MAD a consistent estimator of σ for normally distributed data
    sigma_equiv = 1.4826 * mad if mad > 0 else 1e-9
    is_rehoming = magnitudes > (med + mad_k * sigma_equiv)

    # Fallback: IQR if MAD gives nothing (perfectly uniform shifts, no outliers)
    if not is_rehoming.any():
        q1, q3 = np.percentile(magnitudes, [25, 75])
        iqr = q3 - q1
        if iqr > 0:
            is_rehoming = magnitudes > q3 + mad_k * iqr

    return is_rehoming


def detect_slice_gaps(df: pd.DataFrame) -> list[dict]:
    """
    Detect missing slices in the shifts sequence.

    A gap occurs when moving_id - fixed_id > 1 (i.e. one or more slice IDs
    are absent from the sequence of consecutive pairs).

    Returns a list of dicts with keys 'fixed_id', 'moving_id', 'n_missing'.
    """
    gaps = []
    for _, row in df.iterrows():
        skip = int(row['moving_id']) - int(row['fixed_id']) - 1
        if skip > 0:
            gaps.append({
                'fixed_id': int(row['fixed_id']),
                'moving_id': int(row['moving_id']),
                'n_missing': skip,
            })
    return gaps


def analyze_shifts(df: pd.DataFrame) -> dict:
    """Compute all shift-derived parameter estimates."""
    mag = np.sqrt(df['x_shift_mm'] ** 2 + df['y_shift_mm'] ** 2).values
    df = df.copy()
    df['magnitude_mm'] = mag

    is_rehoming = detect_rehoming(mag)
    normal_mag = mag[~is_rehoming]
    rehoming_mag = mag[is_rehoming]

    # stitch_rehoming_threshold_mm: 60 % of the gap from max-normal to min-rehoming.
    # Positioned closer to the normal side for a comfortable safety margin.
    if is_rehoming.any():
        max_normal = normal_mag.max()
        min_rehoming = rehoming_mag.min()
        threshold = max_normal + 0.6 * (min_rehoming - max_normal)
    else:
        threshold = None

    # max_shift_mm: IQR upper bound of the normal (non-rehoming) shifts.
    if len(normal_mag) >= 4:
        nq1, nq3 = np.percentile(normal_mag, [25, 75])
        niqr = nq3 - nq1
        max_shift = nq3 + 1.5 * niqr
    else:
        max_shift = normal_mag.max() if len(normal_mag) else 0.5

    # common_space_max_step_mm: 95th pct of consecutive shift magnitude changes.
    normal_df = df[~is_rehoming].reset_index(drop=True)
    if len(normal_df) > 1:
        dx = normal_df['x_shift_mm'].diff().abs()
        dy = normal_df['y_shift_mm'].diff().abs()
        step_mag = np.sqrt(dx ** 2 + dy ** 2).dropna()
        max_step = float(np.percentile(step_mag, 95)) if len(step_mag) else 0.5
    else:
        max_step = 0.5

    # Slice gap detection
    gaps = detect_slice_gaps(df)

    return {
        'df': df,
        'is_rehoming': is_rehoming,
        'rehoming_rows': df[is_rehoming],
        'normal_rows': df[~is_rehoming],
        'n_rehoming': int(is_rehoming.sum()),
        'has_rehoming': bool(is_rehoming.any()),
        'rehoming_threshold_mm': threshold,
        'max_shift_mm': max_shift,
        'max_step_mm': max_step,
        'gaps': gaps,
        'has_gaps': bool(gaps),
        'normal_mag_stats': {
            'mean': float(normal_mag.mean()) if len(normal_mag) else 0.0,
            'std': float(normal_mag.std()) if len(normal_mag) else 0.0,
            'max': float(normal_mag.max()) if len(normal_mag) else 0.0,
            'p95': float(np.percentile(normal_mag, 95)) if len(normal_mag) else 0.0,
        },
    }


# =============================================================================
# Raw metadata reading
# =============================================================================

def _parse_info_txt(path: Path) -> dict:
    """
    Parse a tile info.txt file.

    The file may contain repeated sections (one per tile position).  Only the
    first value of each key is kept, since acquisition parameters are constant
    across positions.
    """
    info: dict = {}
    for line in path.read_text().splitlines():
        parts = line.split(": ", 1)
        if len(parts) != 2:
            continue
        key, val = parts[0].strip(), parts[1].strip()
        if key in info:
            continue  # keep first occurrence
        try:
            info[key] = int(val)
        except ValueError:
            try:
                info[key] = float(val)
            except ValueError:
                info[key] = val
    return info


def analyze_metadata(data_dir: str, axial_res_um: float,
                     n_calibration_slices: int = 1) -> dict:
    """
    Extract acquisition parameters from the raw data directory.

    Reads (in priority order):
      1. First tissue slice metadata.json (skipping calibration slices)
      2. <data_dir>/state.json            — fallback for missing fields
      3. First tile info.txt in the tissue slice — for OCT depth parameters

    Parameters
    ----------
    n_calibration_slices
        Number of leading slice directories to skip (default: 1 to skip
        slice_z00, which is always a calibration slice).

    Returns a dict with 'ok', 'sources', and the extracted parameters.
    """
    data_dir = Path(data_dir)
    result: dict = {'ok': False, 'sources': [], 'warnings': []}

    # ── 1. Per-slice metadata.json (skip calibration slices) ─────────────────
    slice_meta: dict = {}
    slice_dirs = sorted(data_dir.glob("slice_z*/"))
    tissue_dirs = slice_dirs[n_calibration_slices:]  # skip leading calibration slices
    if not tissue_dirs:
        result['warnings'].append(
            f"No tissue slice directories found after skipping "
            f"{n_calibration_slices} calibration slice(s) "
            f"(found {len(slice_dirs)} total). "
            f"Try --n_calibration_slices 0."
        )
        tissue_dirs = slice_dirs  # fall back to all dirs
    if tissue_dirs:
        meta_path = tissue_dirs[0] / "metadata.json"
        if meta_path.exists():
            slice_meta = json.loads(meta_path.read_text())
            result['sources'].append(str(meta_path))
        else:
            result['warnings'].append(
                f"metadata.json not found in {tissue_dirs[0].name}"
            )

    # ── 2. Global state.json ──────────────────────────────────────────────────
    state: dict = {}
    state_path = data_dir / "state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        result['sources'].append(str(state_path))

    if not slice_meta and not state:
        result['error'] = "No metadata.json or state.json found in data_dir"
        return result

    # ── 3. Tile info.txt (from the same tissue slice) ─────────────────────────
    tile_info: dict = {}
    if tissue_dirs:
        tiles_dir = tissue_dirs[0] / "tiles"
        tile_dirs = sorted(tiles_dir.glob("tile_*/")) if tiles_dir.exists() else []
        for td in tile_dirs:
            info_path = td / "info.txt"
            if info_path.exists():
                tile_info = _parse_info_txt(info_path)
                result['sources'].append(str(info_path))
                break

    # ── Extract parameters (metadata.json takes priority over state.json) ────
    def get(key, *sources):
        for src in sources:
            if key in src:
                return src[key]
        return None

    tile_size_um = get('tile_size_um', slice_meta, state)
    tile_n_samples = get('tile_n_samples', slice_meta, state)
    slice_thickness = get('slice_thickness', slice_meta, state)
    overlap_fraction = get('overlap_fraction', slice_meta, state)

    # Lateral pixel size
    if tile_size_um and tile_n_samples:
        native_lateral_um = tile_size_um / tile_n_samples
    else:
        native_lateral_um = None
        result['warnings'].append("Could not compute native lateral resolution "
                                  "(tile_size_um or tile_n_samples missing)")

    # Axial depth from tile info.txt
    top_z = tile_info.get('top_z')
    bottom_z = tile_info.get('bottom_z')
    focus_z = tile_info.get('focus_z')

    if bottom_z is not None and top_z is not None:
        n_depth_pixels = bottom_z - top_z + 1
        total_depth_um = n_depth_pixels * axial_res_um
    else:
        n_depth_pixels = state.get('z_max', None)
        if n_depth_pixels and state.get('z_min') is not None:
            n_depth_pixels = state['z_max'] - state['z_min']
            total_depth_um = n_depth_pixels * axial_res_um
        else:
            n_depth_pixels = total_depth_um = None
        result['warnings'].append("top_z/bottom_z not found in tile info.txt; "
                                  "fell back to state.json z_min/z_max")

    # Estimate crop depth: depth of tissue below interface.
    # focus_z marks where the tissue surface is in the OCT depth axis.
    # We take 30 % of the remaining usable depth as a conservative starting
    # point; the user must verify this against a cross-section preview.
    if focus_z is not None and bottom_z is not None:
        pixels_below_focus = bottom_z - focus_z
        crop_depth_um = int(round(pixels_below_focus * axial_res_um * 0.30 / 50) * 50)
        crop_depth_um = max(crop_depth_um, 200)  # floor at 200 µm
    else:
        crop_depth_um = None

    result.update({
        'ok': True,
        'slice_thickness_mm': slice_thickness,
        'overlap_fraction': overlap_fraction,
        'tile_size_um': tile_size_um,
        'tile_n_samples': tile_n_samples,
        'native_lateral_um': native_lateral_um,
        'n_depth_pixels': n_depth_pixels,
        'total_depth_um': total_depth_um,
        'focus_z': focus_z,
        'crop_depth_um': crop_depth_um,
        'axial_res_um': axial_res_um,
    })
    return result


# =============================================================================
# Rounding helpers
# =============================================================================

def ceil_to(value: float, step: float) -> float:
    return float(np.ceil(value / step) * step)


def suggest_target_resolution(native_xy_um: float) -> int:
    """Return the smallest standard pipeline resolution >= native lateral px size."""
    for candidate in [5, 10, 15, 20, 25, 50]:
        if candidate >= native_xy_um:
            return candidate
    return 25


# =============================================================================
# Report
# =============================================================================

def build_report(shift_stats: dict, acq: dict, shifts_path: str) -> str:
    df = shift_stats['df']
    s = shift_stats['normal_mag_stats']
    normal = shift_stats['normal_rows']

    lines = [
        "=" * 62,
        "PARAMETER ESTIMATION REPORT",
        f"Shifts file : {shifts_path}",
        "=" * 62,
        "",
        "ACQUISITION OVERVIEW",
        "-" * 40,
        f"  Slice pairs (rows):    {len(df)}",
        f"  Estimated # of slices: {len(df) + 1}",
        f"  Re-homing events:      {shift_stats['n_rehoming']}",
    ]

    # Acquisition metadata section
    if acq.get('ok'):
        lines += [""]
        lines += ["ACQUISITION METADATA", "-" * 40]
        if acq.get('slice_thickness_mm') is not None:
            lines.append(f"  Slice thickness :   {acq['slice_thickness_mm']:.3f} mm"
                         "  → registration_slicing_interval_mm")
        if acq.get('overlap_fraction') is not None:
            lines.append(f"  Tile overlap    :   {acq['overlap_fraction']:.0%}"
                         "           → stitch_overlap_fraction")
        if acq.get('native_lateral_um') is not None:
            lines.append(f"  Tile size       :   {acq['tile_size_um']:.0f} µm "
                         f"/ {acq['tile_n_samples']} px "
                         f"= {acq['native_lateral_um']:.2f} µm/px native lateral")
        if acq.get('total_depth_um') is not None:
            lines.append(f"  OCT depth       :   {acq['n_depth_pixels']} px "
                         f"× {acq['axial_res_um']:.1f} µm/px "
                         f"= {acq['total_depth_um']:.0f} µm total axial range")
        if acq.get('focus_z') is not None:
            below = acq['n_depth_pixels'] - acq['focus_z']
            below_um = below * acq['axial_res_um']
            lines.append(f"  Focus position  :   z={acq['focus_z']} px "
                         f"({below} px = {below_um:.0f} µm below interface)")
        if acq.get('crop_depth_um') is not None:
            lines.append(f"  → suggested crop_interface_out_depth ≈ {acq['crop_depth_um']} µm"
                         "  (30 % of depth below focus; verify from preview)")
        for w in acq.get('warnings', []):
            lines.append(f"  [!] {w}")
        lines.append(f"  Sources: {', '.join(Path(s).name for s in acq.get('sources', []))}")

    lines += [
        "",
        "NORMAL-SHIFT STATISTICS (re-homing excluded)",
        "-" * 40,
        f"  Mean magnitude : {s['mean']:.3f} mm",
        f"  Std            : {s['std']:.3f} mm",
        f"  95th percentile: {s['p95']:.3f} mm",
        f"  Maximum        : {s['max']:.3f} mm",
        f"  → suggested max_shift_mm = {shift_stats['max_shift_mm']:.3f} mm",
        f"    (IQR upper bound of non-re-homing shifts)",
    ]

    if shift_stats['has_rehoming']:
        rh = shift_stats['rehoming_rows']
        lines += ["", "RE-HOMING EVENTS DETECTED", "-" * 40]
        for _, row in rh.iterrows():
            lines.append(
                f"  Slice {int(row['fixed_id']):02d}→{int(row['moving_id']):02d}: "
                f"X={row['x_shift_mm']:+.3f}, Y={row['y_shift_mm']:+.3f} mm  "
                f"(mag={row['magnitude_mm']:.3f} mm)"
            )
        lines += [
            "",
            f"  Max normal magnitude : {normal['magnitude_mm'].max():.3f} mm",
            f"  Min re-homing mag    : {rh['magnitude_mm'].min():.3f} mm",
            f"  → suggested stitch_rehoming_threshold_mm = "
            f"{shift_stats['rehoming_threshold_mm']:.3f} mm",
        ]
    else:
        lines += [
            "",
            "NO RE-HOMING EVENTS DETECTED",
            "  stitch_rehoming_enabled = false",
        ]

    if shift_stats['has_gaps']:
        n_total = sum(g['n_missing'] for g in shift_stats['gaps'])
        lines += ["", "MISSING SLICES DETECTED", "-" * 40]
        for g in shift_stats['gaps']:
            label = "slice" if g['n_missing'] == 1 else "slices"
            lines.append(
                f"  Gap between slice {g['fixed_id']:02d} and {g['moving_id']:02d}: "
                f"{g['n_missing']} missing {label}"
            )
        lines += [
            f"  Total missing slices: {n_total}",
            "  → interpolate_missing_slices = true  (recommended)",
        ]
    else:
        lines += ["", "NO MISSING SLICES DETECTED",
                  "  interpolate_missing_slices = false"]

    lines += ["", "=" * 62]
    return "\n".join(lines)


# =============================================================================
# Config snippet
# =============================================================================

def build_config_snippet(shift_stats: dict, acq: dict, args) -> str:
    """Return a nextflow.config parameter block with estimated values."""

    # ── Resolution ───────────────────────────────────────────────────────────
    if args.resolution_um:
        res_um = int(args.resolution_um)
        res_comment = "// set by --resolution_um"
    elif acq.get('ok') and acq.get('native_lateral_um'):
        native = acq['native_lateral_um']
        res_um = suggest_target_resolution(native)
        res_comment = (f"// native lateral resolution = {native:.2f} µm/px → "
                       f"smallest standard res >= native")
    else:
        res_um = "TODO"
        res_comment = ("// set to the smallest standard resolution "
                       "(5/10/15/20/25/50) >= native pixel size in µm")

    # ── Crop depth ───────────────────────────────────────────────────────────
    if acq.get('ok') and acq.get('crop_depth_um'):
        crop_depth = acq['crop_depth_um']
        depth_comment = ("// 30 % of OCT depth below tissue interface — "
                         "verify against cross-section preview")
    else:
        crop_depth = "TODO"
        depth_comment = ("// depth in µm to keep below the tissue interface; "
                         "inspect a cross-section preview to set correctly")

    # ── Slicing interval and overlap (from metadata) ─────────────────────────
    if acq.get('ok') and acq.get('slice_thickness_mm') is not None:
        slicing = f"{acq['slice_thickness_mm']:.3f}"
        drifting = f"{acq['slice_thickness_mm'] / 2:.3f}"
        slice_src = "// from slice_thickness in metadata"
    else:
        slicing = "TODO  // ← set from acquisition protocol (e.g. 0.200)"
        drifting = "TODO  // ← typically half the slicing interval"
        slice_src = ""

    if acq.get('ok') and acq.get('overlap_fraction') is not None:
        overlap = f"{acq['overlap_fraction']:.2f}"
        overlap_src = "// from overlap_fraction in metadata"
    else:
        overlap = "TODO  // ← set from acquisition tile-overlap setting"
        overlap_src = ""

    # ── Shift-based params ────────────────────────────────────────────────────
    max_shift = ceil_to(shift_stats['max_shift_mm'], 0.05)
    max_step = float(np.clip(ceil_to(shift_stats['max_step_mm'], 0.05), 0.05, 2.0))

    # ── Registration max translation ─────────────────────────────────────────
    # The optimizer bound must comfortably exceed any real inter-slice
    # translation. Setting it to the tile size in pixels at the target
    # resolution ensures the optimizer is never clamped for whole-tile shifts.
    if acq.get('ok') and acq.get('tile_size_um') and isinstance(res_um, (int, float)):
        max_trans_px = int(np.ceil(acq['tile_size_um'] / res_um / 10) * 10)
        max_trans_comment = (
            f"// tile {acq['tile_size_um']:.0f} µm / {res_um} µm·px⁻¹ "
            f"= {acq['tile_size_um'] / res_um:.0f} px, rounded up to nearest 10"
        )
    else:
        max_trans_px = 200
        max_trans_comment = "// default — set to tile width in pixels at target resolution"

    # ── Missing slice interpolation ───────────────────────────────────────────
    if shift_stats['has_gaps']:
        interp_val = "true"
        gap_detail = ", ".join(
            f"{g['fixed_id']}→{g['moving_id']} ({g['n_missing']} missing)"
            for g in shift_stats['gaps']
        )
        interp_comment = f"// gaps detected: {gap_detail}"
    else:
        interp_val = "false"
        interp_comment = "// no gaps detected in shifts sequence"

    lines = [
        "// ================================================================",
        "// SUGGESTED PARAMETERS  (generated by linum_suggest_params.py)",
        "// Each value is annotated with how it was derived.",
        "// Review all parameters against your acquisition protocol before use.",
        "// Parameters marked TODO must be filled in manually.",
        "// ================================================================",
        "",
        "// ── Resolution & depth ────────────────────────────────────────",
        f"resolution = {res_um:<8} {res_comment}",
        f"crop_interface_out_depth = {crop_depth:<6} {depth_comment}",
        "",
        "// ── Tile stitching ────────────────────────────────────────────",
        f"stitch_overlap_fraction = {overlap}  {overlap_src}",
        "",
        "// ── Slice registration ────────────────────────────────────────",
        f"registration_slicing_interval_mm = {slicing} {slice_src}",
        f"registration_allowed_drifting_mm = {drifting}",
        "  // ↑ Z-search range; typically half the slicing interval",
        f"registration_max_translation = {max_trans_px}",
        f"  // ↑ {max_trans_comment}",
        "",
        "// ── Missing slice interpolation ───────────────────────────────",
        f"interpolate_missing_slices = {interp_val}  {interp_comment}",
        "",
        "// ── Shift outlier filtering (common-space alignment) ──────────",
        f"max_shift_mm = {max_shift:.3f}",
        "  // IQR upper bound of normal inter-slice shifts.",
        "  // Acts as a floor on the IQR detection threshold.",
        "  // Keep well below the smallest re-homing event magnitude.",
        f"common_space_max_step_mm = {max_step:.3f}",
        "  // 95th percentile of consecutive shift magnitude changes.",
        "  // Flags sudden per-step jumps that IQR alone may miss.",
    ]

    if shift_stats['has_rehoming']:
        rh = shift_stats['rehoming_rows']
        thresh = ceil_to(shift_stats['rehoming_threshold_mm'], 0.05)
        slice_ids = ", ".join(str(int(r['moving_id'])) for _, r in rh.iterrows())
        lines += [
            "",
            "// ── Re-homing correction ─────────────────────────────────────",
            "stitch_rehoming_enabled = true",
            "stitch_rehoming_use_motor = true",
            "  // Re-homing events are large FOV jumps encoded in the motor",
            "  // positions. Image-based registration is ambiguous at these",
            "  // boundaries; motor values are more reliable.",
            f"stitch_rehoming_threshold_mm = {thresh:.2f}",
            f"  // 60 % between largest normal shift "
            f"({shift_stats['normal_rows']['magnitude_mm'].max():.3f} mm)",
            f"  // and smallest re-homing event ({rh['magnitude_mm'].min():.3f} mm).",
            f"  // Boundaries at slices: {slice_ids}",
        ]
    else:
        lines += [
            "",
            "// ── Re-homing correction ─────────────────────────────────────",
            "stitch_rehoming_enabled = false  // no re-homing events detected",
        ]

    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    if out_dir.exists() and not args.overwrite:
        print(f"Output directory already exists: {out_dir}  (use -f to overwrite)",
              file=sys.stderr)
        sys.exit(1)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load and analyse shifts
    print(f"Loading shifts: {args.shifts_file}")
    df = load_shifts(args.shifts_file)
    shift_stats = analyze_shifts(df)

    # Optionally read raw data directory for acquisition metadata
    acq: dict = {}
    if args.data_dir:
        print(f"Reading acquisition metadata: {args.data_dir}")
        acq = analyze_metadata(args.data_dir, args.axial_res_um,
                               args.n_calibration_slices)
        if not acq.get('ok'):
            print(f"Warning: {acq.get('error', 'could not read metadata')}",
                  file=sys.stderr)
        elif acq.get('warnings'):
            for w in acq['warnings']:
                print(f"Warning: {w}", file=sys.stderr)

    # Build outputs
    report = build_report(shift_stats, acq, args.shifts_file)
    snippet = build_config_snippet(shift_stats, acq, args)

    sep = "-" * 62
    print()
    print(report)
    print()
    print("SUGGESTED NEXTFLOW.CONFIG PARAMETERS")
    print(sep)
    print(snippet)

    report_path = out_dir / "param_estimation_report.txt"
    config_path = out_dir / "suggested_params.config"

    report_path.write_text(
        report + "\n\n"
        + "SUGGESTED NEXTFLOW.CONFIG PARAMETERS\n"
        + sep + "\n"
        + snippet + "\n"
    )
    config_path.write_text(snippet + "\n")

    print(f"\nWrote:\n  {report_path}\n  {config_path}")


if __name__ == "__main__":
    main()
