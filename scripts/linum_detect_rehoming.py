#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read a shifts CSV produced by linum_compute_shifts_3d.py and correct encoder
glitch spikes caused by motor re-homing events.

A "re-homing event" is a large inter-slice step in XY that represents a real,
intended repositioning of the motor stage.  These must be preserved so that
consecutive slices are correctly aligned into the common space.

An "encoder glitch spike" is an apparent large step that is immediately
self-cancelled by the next step: the stage reports a big displacement but the
next measurement compensates almost exactly, so no actual re-homing occurred.
These should be zeroed out.

Detection criterion (``rehome`` method, default):
    A step at position i is treated as a glitch spike when

        |step[i] + step[i±1]| < return_fraction × |step[i]|

    i.e. the round-trip magnitude is less than ``return_fraction`` times the
    single-step magnitude (default 0.4 → adjacent step reverses > 60 %).
    Re-homing events (large step whose neighbours are small or compound in the
    same direction) satisfy |step[i] + step[i±1]| ≥ return_fraction × |step[i]|
    and are therefore left untouched.

Outputs
-------
* ``<out_shifts>`` — corrected shifts CSV (same schema as input).
* Optionally, with ``--diagnostics <dir>``:
    - ``rehoming_report.json`` — lists every glitch spike that was corrected.
    - ``rehoming_plot.png``    — per-step magnitude chart with corrections marked.
"""
# Configure thread limits before numpy/scipy imports
import linumpy._thread_config  # noqa: F401

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from linumpy.utils.shifts import filter_outlier_shifts
from linumpy.utils.io import add_overwrite_arg, assert_output_exists


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_shifts',
                   help='Shifts CSV file (e.g. shifts_xy.csv) produced by '
                        'linum_compute_shifts_3d.py.')
    p.add_argument('out_shifts',
                   help='Output corrected shifts CSV file.')
    p.add_argument('--return_fraction', type=float, default=0.4,
                   help='Round-trip fraction threshold for spike detection.\n'
                        'A step is treated as a glitch if the adjacent step '
                        'reverses more than (1 - return_fraction) of the '
                        'displacement.  Lower values are more conservative '
                        '(correct fewer spikes).  [%(default)s]')
    p.add_argument('--diagnostics', metavar='DIR', default=None,
                   help='If provided, write a JSON report and PNG plot of '
                        'corrected spikes to this directory.')
    add_overwrite_arg(p)
    return p


def _save_diagnostics(diag_dir: Path,
                      shifts_before: pd.DataFrame,
                      shifts_after: pd.DataFrame,
                      corrected_indices: list) -> None:
    """Save a JSON report and PNG plot of corrected glitch spikes."""
    diag_dir.mkdir(parents=True, exist_ok=True)

    # ----- JSON report -------------------------------------------------------
    records = []
    for idx in corrected_indices:
        row_before = shifts_before.loc[idx]
        row_after = shifts_after.loc[idx]
        records.append({
            "index": int(idx),
            "fixed_id": int(row_before["fixed_id"]),
            "moving_id": int(row_before["moving_id"]),
            "original_x_shift_mm": float(row_before["x_shift_mm"]),
            "original_y_shift_mm": float(row_before["y_shift_mm"]),
            "corrected_x_shift_mm": float(row_after["x_shift_mm"]),
            "corrected_y_shift_mm": float(row_after["y_shift_mm"]),
        })
    report = {
        "n_corrected": len(records),
        "corrected_spikes": records,
    }
    report_path = diag_dir / "rehoming_report.json"
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"  Diagnostics report: {report_path}")

    # ----- PNG plot ----------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        step_mag_before = np.sqrt(
            shifts_before["x_shift_mm"] ** 2 + shifts_before["y_shift_mm"] ** 2
        )
        step_mag_after = np.sqrt(
            shifts_after["x_shift_mm"] ** 2 + shifts_after["y_shift_mm"] ** 2
        )
        positions = np.arange(len(shifts_before))

        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

        # X
        axes[0].plot(positions, shifts_before["x_shift_mm"],
                     color="steelblue", lw=1.2, label="original")
        axes[0].plot(positions, shifts_after["x_shift_mm"],
                     color="darkorange", lw=1.2, linestyle="--", label="corrected")
        if corrected_indices:
            ci_pos = [shifts_before.index.get_loc(i) for i in corrected_indices]
            axes[0].scatter(ci_pos,
                            shifts_before.loc[corrected_indices, "x_shift_mm"],
                            color="red", zorder=5, label="corrected spike")
        axes[0].set_ylabel("x_shift_mm")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        # Y
        axes[1].plot(positions, shifts_before["y_shift_mm"],
                     color="steelblue", lw=1.2, label="original")
        axes[1].plot(positions, shifts_after["y_shift_mm"],
                     color="darkorange", lw=1.2, linestyle="--", label="corrected")
        if corrected_indices:
            axes[1].scatter(ci_pos,
                            shifts_before.loc[corrected_indices, "y_shift_mm"],
                            color="red", zorder=5, label="corrected spike")
        axes[1].set_ylabel("y_shift_mm")
        axes[1].set_xlabel("step index")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

        axes[0].set_title(
            f"Rehoming glitch correction — {len(corrected_indices)} spike(s) corrected"
        )
        fig.tight_layout()
        plot_path = diag_dir / "rehoming_plot.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"  Diagnostics plot:   {plot_path}")
    except ImportError:
        print("  matplotlib not available — skipping plot.")


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_output_exists(args.out_shifts, parser, args)

    shifts_before = pd.read_csv(args.in_shifts)
    print(f"Loaded {len(shifts_before)} pairwise shifts from {args.in_shifts}")

    shifts_after = filter_outlier_shifts(
        shifts_before,
        method="rehome",
        return_fraction=args.return_fraction,
    )

    # Identify which rows were actually modified
    diff_mask = (
        (shifts_before["x_shift_mm"] != shifts_after["x_shift_mm"])
        | (shifts_before["y_shift_mm"] != shifts_after["y_shift_mm"])
    )
    corrected_indices = list(shifts_before.index[diff_mask])
    n_corrected = len(corrected_indices)

    if n_corrected == 0:
        print("No glitch spikes detected — shifts unchanged.")
    else:
        print(f"Corrected {n_corrected} glitch spike(s):")
        for idx in corrected_indices:
            row_b = shifts_before.loc[idx]
            row_a = shifts_after.loc[idx]
            print(
                f"  step {int(row_b['fixed_id'])}→{int(row_b['moving_id'])}: "
                f"({row_b['x_shift_mm']:.3f}, {row_b['y_shift_mm']:.3f}) mm "
                f"→ ({row_a['x_shift_mm']:.3f}, {row_a['y_shift_mm']:.3f}) mm"
            )

    shifts_after.to_csv(args.out_shifts, index=False)
    print(f"Corrected shifts written to {args.out_shifts}")

    if args.diagnostics:
        _save_diagnostics(
            diag_dir=Path(args.diagnostics),
            shifts_before=shifts_before,
            shifts_after=shifts_after,
            corrected_indices=corrected_indices,
        )


if __name__ == "__main__":
    main()
