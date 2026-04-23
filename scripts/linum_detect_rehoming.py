#!/usr/bin/env python3
"""Read a shifts CSV produced by linum_compute_shifts_3d.py and detect/correct two classes of spurious inter-slice shifts.

Background
----------
The shifts file stores ``xmin_mm[fixed] - xmin_mm[moving]`` for each pair of
consecutive slices, where ``xmin_mm[i]`` is the **left-edge position of the
mosaic grid** for slice ``i``.  Two artifacts can inflate these values:

1. **Mosaic grid expansion** (``--tile_fov_mm``): The acquisition software
   adapts the mosaic size to the visible tissue.  When it adds (or removes) a
   whole tile column at the left boundary between slices, ``xmin_mm`` jumps by
   exactly ±N × tile_FOV even though the tissue did not move.  These steps are
   *persistent* (not self-cancelling) and look like valid re-homing events to
   the spike detector.  Correct them first with ``--tile_fov_mm``.

2. **Encoder glitch spikes**: An apparent large step that is immediately
   self-cancelled by the next step.  The stage reports a big displacement but
   the following measurement reverses it, so no real repositioning occurred.
   These are removed by the spike detector described below.

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
import linumpy.config.threads  # noqa: F401

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from linumpy.cli.args import add_overwrite_arg, assert_output_exists
from linumpy.io import slice_config as slice_config_io
from linumpy.stack_alignment.filter import correct_tile_offset_shifts, filter_outlier_shifts


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("in_shifts", help="Shifts CSV file (e.g. shifts_xy.csv) produced by linum_compute_shifts_3d.py.")
    p.add_argument("out_shifts", help="Output corrected shifts CSV file.")
    p.add_argument(
        "--return_fraction",
        type=float,
        default=0.4,
        help="Round-trip fraction threshold for spike detection.\n"
        "A step is treated as a glitch if the adjacent step "
        "reverses more than (1 - return_fraction) of the "
        "displacement.  Lower values are more conservative "
        "(correct fewer spikes).  [%(default)s]",
    )
    p.add_argument(
        "--max_shift_mm",
        type=float,
        default=0.5,
        help="Steps with magnitude below this threshold are not\n"
        "checked for spike patterns.  Lower this value to\n"
        "catch smaller self-cancelling glitches.  [%(default)s]",
    )
    p.add_argument(
        "--tile_fov_mm",
        type=float,
        default=None,
        help="Observed artifact step size in mm: the amount xmin_mm shifts\n"
        "spuriously at mosaic grid-expansion transitions.\n"
        "This value must be determined empirically from the\n"
        "shifts_xy.csv data — it is NOT simply tile_size_um × (1-overlap).\n"
        "To find it: look for a cluster of near-equal large steps in\n"
        "x_shift_mm (e.g. several rows all ≈ +0.875 mm).  The common\n"
        "value is the artifact step; its magnitude depends on the mosaic\n"
        "grid layout at the time of acquisition.\n"
        "When set, any step within tile_fov_tolerance of N × tile_fov_mm\n"
        "(N integer ≠ 0) is corrected by subtracting N × tile_fov_mm.\n"
        "If unsure, leave unset and inspect the --diagnostics plot.\n"
        "[%(default)s]",
    )
    p.add_argument(
        "--tile_fov_tolerance",
        type=float,
        default=0.05,
        help="Fractional tolerance for tile-FOV multiple detection.\n"
        "Default 0.05 → a 5 %% margin around each integer multiple."
        "  [%(default)s]",
    )
    p.add_argument(
        "--diagnostics",
        metavar="DIR",
        default=None,
        help="If provided, write a JSON report and PNG plot of corrected spikes to this directory.",
    )
    p.add_argument(
        "--slice_config_in",
        metavar="SLICE_CONFIG_CSV",
        default=None,
        help="Optional slice_config.csv to stamp with rehoming flags.",
    )
    p.add_argument(
        "--slice_config_out",
        metavar="SLICE_CONFIG_CSV",
        default=None,
        help="Output slice_config.csv path (requires --slice_config_in). "
        "Each transition's moving_id slice is stamped with "
        "rehomed=true/false and rehoming_reliable=0/1.",
    )
    add_overwrite_arg(p)
    return p


def _save_diagnostics(
    diag_dir: Path,
    shifts_before: pd.DataFrame,
    shifts_after: pd.DataFrame,
    corrected_indices: list,
    tile_corrected_indices: list | None = None,
) -> None:
    """Save a JSON report and PNG plot of corrected glitch spikes."""
    diag_dir.mkdir(parents=True, exist_ok=True)

    # ----- JSON report -------------------------------------------------------
    if tile_corrected_indices is None:
        tile_corrected_indices = []

    records = []
    for idx in corrected_indices:
        row_before = shifts_before.loc[idx]
        row_after = shifts_after.loc[idx]
        records.append(
            {
                "index": int(idx),
                "fixed_id": int(row_before["fixed_id"]),
                "moving_id": int(row_before["moving_id"]),
                "correction_type": "spike",
                "original_x_shift_mm": float(row_before["x_shift_mm"]),
                "original_y_shift_mm": float(row_before["y_shift_mm"]),
                "corrected_x_shift_mm": float(row_after["x_shift_mm"]),
                "corrected_y_shift_mm": float(row_after["y_shift_mm"]),
            }
        )
    for idx in tile_corrected_indices:
        row_before = shifts_before.loc[idx]
        row_after = shifts_after.loc[idx]
        records.append(
            {
                "index": int(idx),
                "fixed_id": int(row_before["fixed_id"]),
                "moving_id": int(row_before["moving_id"]),
                "correction_type": "tile_offset",
                "original_x_shift_mm": float(row_before["x_shift_mm"]),
                "original_y_shift_mm": float(row_before["y_shift_mm"]),
                "corrected_x_shift_mm": float(row_after["x_shift_mm"]),
                "corrected_y_shift_mm": float(row_after["y_shift_mm"]),
            }
        )
    records.sort(key=lambda r: r["index"])
    report = {
        "n_corrected": len(records),
        "corrected_spikes": [r for r in records if r["correction_type"] == "spike"],
        "corrected_tile_offsets": [r for r in records if r["correction_type"] == "tile_offset"],
    }
    report_path = diag_dir / "rehoming_report.json"
    with Path(report_path).open("w") as fh:
        json.dump(report, fh, indent=2)
    print(f"  Diagnostics report: {report_path}")

    # ----- PNG plot ----------------------------------------------------------
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        positions = np.arange(len(shifts_before))

        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

        # X
        axes[0].plot(positions, shifts_before["x_shift_mm"], color="steelblue", lw=1.2, label="original")
        axes[0].plot(positions, shifts_after["x_shift_mm"], color="darkorange", lw=1.2, linestyle="--", label="corrected")
        if corrected_indices:
            ci_pos = [shifts_before.index.get_loc(i) for i in corrected_indices]
            axes[0].scatter(
                ci_pos, shifts_before.loc[corrected_indices, "x_shift_mm"], color="red", zorder=5, label="spike correction"
            )
        if tile_corrected_indices:
            ti_pos = [shifts_before.index.get_loc(i) for i in tile_corrected_indices]
            axes[0].scatter(
                ti_pos,
                shifts_before.loc[tile_corrected_indices, "x_shift_mm"],
                color="magenta",
                marker="^",
                zorder=5,
                label="tile-offset correction",
            )
        axes[0].set_ylabel("x_shift_mm")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        # Y
        axes[1].plot(positions, shifts_before["y_shift_mm"], color="steelblue", lw=1.2, label="original")
        axes[1].plot(positions, shifts_after["y_shift_mm"], color="darkorange", lw=1.2, linestyle="--", label="corrected")
        if corrected_indices:
            ci_pos = [shifts_before.index.get_loc(i) for i in corrected_indices]
            axes[1].scatter(
                ci_pos, shifts_before.loc[corrected_indices, "y_shift_mm"], color="red", zorder=5, label="spike correction"
            )
        if tile_corrected_indices:
            ti_pos = [shifts_before.index.get_loc(i) for i in tile_corrected_indices]
            axes[1].scatter(
                ti_pos,
                shifts_before.loc[tile_corrected_indices, "y_shift_mm"],
                color="magenta",
                marker="^",
                zorder=5,
                label="tile-offset correction",
            )
        axes[1].set_ylabel("y_shift_mm")
        axes[1].set_xlabel("step index")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

        n_tile = len(tile_corrected_indices)
        axes[0].set_title(f"Rehoming correction — {len(corrected_indices)} spike(s), {n_tile} tile-offset(s) corrected")
        fig.tight_layout()
        plot_path = diag_dir / "rehoming_plot.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"  Diagnostics plot:   {plot_path}")
    except ImportError:
        print("  matplotlib not available — skipping plot.")


def _stamp_slice_config(
    path_in: Path,
    path_out: Path,
    shifts_after: pd.DataFrame,
    spike_indices: list,
    tile_indices: list,
) -> None:
    """Stamp per-slice rehoming flags into ``slice_config.csv``.

    A slice is ``rehomed`` when its arriving transition (``moving_id == slice``)
    was corrected by either pass (spike or tile-offset); it is
    ``rehoming_reliable=1`` when that transition's corrected motor step is
    small enough (``reliable=1`` in the shifts file), else 0.
    """
    corrected = set(spike_indices) | set(tile_indices)
    updates: dict[str, dict[str, object]] = {}
    for idx, row in shifts_after.iterrows():
        sid = slice_config_io.normalize_slice_id(int(row["moving_id"]))
        reliable = int(row["reliable"]) if "reliable" in row else 1
        updates[sid] = {
            "rehomed": idx in corrected,
            "rehoming_reliable": reliable,
        }
    slice_config_io.stamp_many(path_in, path_out, updates)
    print(f"Slice-config updates written to {path_out}")


def main() -> None:
    """Run function operation."""
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_output_exists(args.out_shifts, parser, args)

    shifts_before = pd.read_csv(args.in_shifts)
    print(f"Loaded {len(shifts_before)} pairwise shifts from {args.in_shifts}")

    # --- Pass 1: mosaic grid-expansion correction -----------------------------
    # When the acquisition adds/removes tile columns at the mosaic boundary,
    # xmin_mm shifts by N × tile_FOV (no real tissue movement).  These steps
    # are persistent and do NOT self-cancel, so the spike detector misses them.
    tile_corrected_indices = []
    shifts_after = shifts_before
    if args.tile_fov_mm is not None:
        shifts_after, tile_corrected_indices = correct_tile_offset_shifts(
            shifts_before,
            tile_fov_x_mm=args.tile_fov_mm,
            tolerance=args.tile_fov_tolerance,
        )
        if not tile_corrected_indices:
            print("No tile-FOV-multiple offsets detected.")
        else:
            print(f"Corrected {len(tile_corrected_indices)} tile-FOV offset(s):")
            for idx in tile_corrected_indices:
                row_b = shifts_before.loc[idx]
                row_a = shifts_after.loc[idx]
                print(
                    f"  step {int(row_b['fixed_id'])}→{int(row_b['moving_id'])}: "
                    f"({row_b['x_shift_mm']:.4f}, {row_b['y_shift_mm']:.4f}) mm "
                    f"→ ({row_a['x_shift_mm']:.4f}, {row_a['y_shift_mm']:.4f}) mm"
                )

    # --- Pass 2: self-cancelling spike correction ----------------------------
    shifts_intermediate = shifts_after
    shifts_after = filter_outlier_shifts(
        shifts_intermediate,
        method="rehome",
        max_shift_mm=args.max_shift_mm,
        return_fraction=args.return_fraction,
    )

    # Identify which rows were modified by the spike pass
    diff_mask = (shifts_intermediate["x_shift_mm"] != shifts_after["x_shift_mm"]) | (
        shifts_intermediate["y_shift_mm"] != shifts_after["y_shift_mm"]
    )
    corrected_indices = list(shifts_intermediate.index[diff_mask])
    n_corrected = len(corrected_indices)

    if n_corrected == 0:
        print("No self-cancelling glitch spikes detected.")
    else:
        print(f"Corrected {n_corrected} self-cancelling spike(s):")
        for idx in corrected_indices:
            row_b = shifts_intermediate.loc[idx]
            row_a = shifts_after.loc[idx]
            print(
                f"  step {int(row_b['fixed_id'])}→{int(row_b['moving_id'])}: "
                f"({row_b['x_shift_mm']:.4f}, {row_b['y_shift_mm']:.4f}) mm "
                f"→ ({row_a['x_shift_mm']:.4f}, {row_a['y_shift_mm']:.4f}) mm"
            )

    total_corrected = len(tile_corrected_indices) + n_corrected
    if total_corrected == 0:
        print("No encoder artifacts detected — shifts unchanged.")

    # Add a 'reliable' column: 0 for transitions whose *corrected* motor step
    # magnitude still exceeds max_shift_mm — meaning neither Pass 1 (tile
    # offset) nor Pass 2 (spike) was able to explain the motor step, so
    # the true XY transition is unknown. Rows that pass 1/2 successfully
    # corrected are marked reliable=1.
    # This drives linum_align_mosaics_3d_from_shifts.py --refine_unreliable,
    # which falls back to image-based registration only for reliable=0 rows.
    shifts_after = shifts_after.copy()
    shift_mag_after = np.sqrt(shifts_after["x_shift_mm"] ** 2 + shifts_after["y_shift_mm"] ** 2)
    shifts_after["reliable"] = (shift_mag_after <= args.max_shift_mm).astype(int)
    n_unreliable = int((shifts_after["reliable"] == 0).sum())
    if n_unreliable > 0:
        unreliable_ids = [
            f"{int(row['fixed_id'])}→{int(row['moving_id'])}"
            for _, row in shifts_after[shifts_after["reliable"] == 0].iterrows()
        ]
        print(f"Flagged {n_unreliable} transition(s) as unreliable (reliable=0): {', '.join(unreliable_ids)}")
    else:
        print("All transitions flagged as reliable.")

    shifts_after.to_csv(args.out_shifts, index=False)
    print(f"Corrected shifts written to {args.out_shifts}")

    if args.slice_config_out:
        if not args.slice_config_in:
            parser.error("--slice_config_out requires --slice_config_in")
        _stamp_slice_config(
            Path(args.slice_config_in),
            Path(args.slice_config_out),
            shifts_after=shifts_after,
            spike_indices=corrected_indices,
            tile_indices=tile_corrected_indices,
        )

    if args.diagnostics:
        _save_diagnostics(
            diag_dir=Path(args.diagnostics),
            shifts_before=shifts_before,
            shifts_after=shifts_after,
            corrected_indices=corrected_indices,
            tile_corrected_indices=tile_corrected_indices,
        )


if __name__ == "__main__":
    main()
