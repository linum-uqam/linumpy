#!/usr/bin/env python3
"""
Suggest 3D reconstruction pipeline parameters from raw input files.

Analyses the motor-positions file (shifts_xy.csv) and, optionally, the raw
data directory produced by the preprocessing pipeline to automatically
estimate suitable nextflow.config parameters.

Estimable parameters
--------------------
From shifts_xy.csv:
  max_shift_mm                    -- IQR upper bound of normal inter-slice shifts
  common_space_max_step_mm        -- 95th percentile of consecutive normal shift changes
  interpolate_missing_slices      -- true when gaps (moving_id - fixed_id > 1) are found

From the raw data directory (--data_dir):
  registration_slicing_interval_mm  -- from slice_thickness in metadata.json / state.json
  stitch_overlap_fraction           -- from overlap_fraction in metadata.json / state.json
  resolution                        -- smallest standard resolution >= native lateral px size
  crop_interface_out_depth          -- depth below tissue interface; user must verify
  registration_max_translation      -- tile width in pixels at target resolution

Parameters that cannot be estimated automatically:
  crop_interface_out_depth    -- requires tissue-specific knowledge; an estimate is given
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

Cleaned data directory layout (after linum_clean_raw_data.py)
--------------------------------------------------------------
  <data_dir>/
    state.json                          # global acquisition state (unchanged)
    metadata/
      slice_z00/
        metadata.json                   # per-slice acquisition parameters (kept)
        tiles/
          tile_x##_y##_z##/
            info.txt                    # per-tile OCT scan parameters (kept)

Both layouts are detected automatically when --data_dir is provided.
"""

# Configure thread limits before numpy/scipy imports (optional; skipped if
# linumpy is not installed into the current environment)
import contextlib

with contextlib.suppress(ImportError):
    import linumpy.config.threads  # noqa: F401

import argparse
import sys
from pathlib import Path

from linumpy.diagnostics.suggest_params import (
    OCT_AXIAL_RES_UM,
    analyze_metadata,
    analyze_shifts,
    build_config_snippet,
    build_report,
    ceil_to,
    detect_rehoming,
    detect_slice_gaps,
    load_shifts,
    suggest_target_resolution,
)

__all__ = [
    "OCT_AXIAL_RES_UM",
    "analyze_metadata",
    "analyze_shifts",
    "build_config_snippet",
    "build_report",
    "ceil_to",
    "detect_rehoming",
    "detect_slice_gaps",
    "load_shifts",
    "main",
    "suggest_target_resolution",
]


# =============================================================================
# CLI
# =============================================================================


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("shifts_file", help="Motor-positions CSV file (shifts_xy.csv)")
    p.add_argument("output_dir", help="Directory for the report and suggested config snippet")
    p.add_argument(
        "--data_dir",
        default=None,
        help="Raw data directory (contains state.json and\n"
        "slice_z##/ subdirectories, or their cleaned\n"
        "equivalent with slices under metadata/). Used\n"
        "to read slice thickness, tile overlap, and tile\n"
        "dimensions. Both raw and cleaned layouts are\n"
        "detected automatically.",
    )
    p.add_argument(
        "--n_calibration_slices",
        type=int,
        default=1,
        help="Number of leading calibration slices to skip when reading per-slice metadata [%(default)s].",
    )
    p.add_argument(
        "--axial_res_um",
        type=float,
        default=OCT_AXIAL_RES_UM,
        help="OCT axial resolution in µm/pixel [%(default)s µm].\nUsed to convert tile depth (pixels) → µm.",
    )
    p.add_argument(
        "--resolution_um",
        type=float,
        default=None,
        help="Override target pipeline resolution in µm/pixel.\nDerived automatically from tile dimensions if not given.",
    )
    p.add_argument("-f", "--overwrite", action="store_true", help="Overwrite existing output directory")
    return p


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Run function."""
    parser = _build_arg_parser()
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    if out_dir.exists() and not args.overwrite:
        print(f"Output directory already exists: {out_dir}  (use -f to overwrite)", file=sys.stderr)
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
        acq = analyze_metadata(args.data_dir, args.axial_res_um, args.n_calibration_slices)
        if not acq.get("ok"):
            print(f"Warning: {acq.get('error', 'could not read metadata')}", file=sys.stderr)
        elif acq.get("warnings"):
            for w in acq["warnings"]:
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

    report_path.write_text(report + "\n\n" + "SUGGESTED NEXTFLOW.CONFIG PARAMETERS\n" + sep + "\n" + snippet + "\n")
    config_path.write_text(snippet + "\n")

    print(f"\nWrote:\n  {report_path}\n  {config_path}")


if __name__ == "__main__":
    main()
