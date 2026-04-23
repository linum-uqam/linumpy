#!/usr/bin/env python3
"""
Interpolate a missing slice using information from adjacent slices.

Uses z-aware morphing (``zmorph``): an affine transform ``T`` between the
boundary planes of the two neighbours is computed, then for each output
plane at fractional depth ``alpha`` the before-boundary is warped by
``T**alpha`` and the after-boundary by ``T**(alpha - 1)`` and the two are
cross-faded.

**Hard skip on gate failure.** When zmorph's 2D boundary registration fails
any quality gate, no interpolated zarr is produced: the slot is left as a
genuine gap rather than filled with a blended (and therefore fabricated)
volume. A manifest fragment and diagnostics JSON are still emitted so the
failure is visible in ``slice_config_final.csv`` and the final report. The
``average`` / ``weighted`` methods remain available as explicit,
user-requested baselines.

Only a SINGLE missing slice can be reconstructed — two or more consecutive
gaps carry insufficient information.

See ``docs/SLICE_INTERPOLATION_FEATURE.md`` for the physical model and
parameter-tuning guidance.

Example usage:
    linum_interpolate_missing_slice.py slice_z00.ome.zarr slice_z02.ome.zarr \\
        slice_z01_interpolated.ome.zarr

    # Finalise mode: merge per-slice manifest fragments into slice_config.csv
    linum_interpolate_missing_slice.py --finalise \\
        --slice_config_in  slice_config.csv \\
        --slice_config_out slice_config_final.csv \\
        --fragments        interpolate_missing_slice/
"""

import linumpy.config.threads  # noqa: F401
from linumpy.config.threads import configure_all_libraries

import argparse
import json
from pathlib import Path

import dask.array as da
import matplotlib.pyplot as plt
import numpy as np

from linumpy.cli.args import add_overwrite_arg, assert_output_exists
from linumpy.io import slice_config as slice_config_io
from linumpy.io.zarr import read_omezarr, save_omezarr
from linumpy.mosaic.interpolation import (
    interpolate_average,
    interpolate_weighted,
    interpolate_z_morph,
)

configure_all_libraries()


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("slice_before", nargs="?", help="Path to the slice BEFORE the missing slice (*.ome.zarr)")
    p.add_argument("slice_after", nargs="?", help="Path to the slice AFTER the missing slice (*.ome.zarr)")
    p.add_argument("output", nargs="?", help="Output path for the interpolated slice (*.ome.zarr)")
    p.add_argument(
        "--method",
        choices=["zmorph", "average", "weighted"],
        default="zmorph",
        help="Interpolation method:\n"
        "  zmorph   - z-aware morphing (respects serial-section geometry; recommended)\n"
        "  average  - Simple average of adjacent slices\n"
        "  weighted - Weighted average with distance falloff\n"
        "[default: %(default)s]",
    )
    p.add_argument(
        "--blend_method",
        choices=["linear", "gaussian"],
        default="gaussian",
        help="Blending method for combining warped slices:\n"
        "  linear - Equal 50/50 blend (may show edges)\n"
        "  gaussian - Feathered blend using distance transform (recommended)\n"
        "[default: %(default)s]",
    )
    p.add_argument(
        "--registration_metric",
        choices=["MSE", "CC", "MI"],
        default="MSE",
        help="Metric for registration [default: %(default)s]",
    )
    p.add_argument(
        "--max_iterations", type=int, default=1000, help="Maximum iterations for registration [default: %(default)s]"
    )
    p.add_argument(
        "--overlap_search_window",
        type=int,
        default=5,
        help="Number of z-planes to search at each volume boundary\n"
        "when selecting the registration reference pair automatically.\n"
        "[default: %(default)s]",
    )
    p.add_argument(
        "--min_overlap_correlation",
        type=float,
        default=0.3,
        help="Minimum normalized cross-correlation required between the\n"
        "boundary planes to proceed with registration. Below this\n"
        "threshold zmorph emits no output (hard skip).\n"
        "[default: %(default)s]",
    )
    p.add_argument(
        "--reference_slab_size",
        type=int,
        default=3,
        help="Number of z-planes averaged around the boundary reference\n"
        "plane when running the 2D registration. Larger slabs are\n"
        "more robust to per-plane noise. [default: %(default)s]",
    )
    p.add_argument(
        "--min_foreground_fraction",
        type=float,
        default=0.1,
        help="Minimum fraction of foreground pixels required for a\n"
        "candidate boundary plane to be considered. [default: %(default)s]",
    )
    p.add_argument(
        "--min_ncc_improvement",
        type=float,
        default=0.05,
        help="Minimum improvement in boundary NCC required after 2D\n"
        "registration to accept the transform. Below this zmorph emits\n"
        "no output (hard skip). [default: %(default)s]",
    )
    p.add_argument(
        "--diagnostics",
        type=str,
        default=None,
        help="Path to write a JSON diagnostics file (plane selection,\n"
        "pre/post NCC, half-transform verification, fallback reason).",
    )
    p.add_argument(
        "--manifest_entry",
        type=str,
        default=None,
        help="Path to write a single-line CSV manifest entry for this\ninterpolated slice (aggregated downstream).",
    )
    p.add_argument(
        "--slice_id",
        type=str,
        default=None,
        help="Slice id string (e.g. '02') to record in the manifest entry.",
    )

    # Finalise / slice_config merge mode
    finalise_group = p.add_argument_group(
        "Finalise Mode",
        "Merge per-slice interpolation manifest fragments into slice_config.csv. "
        "When --finalise is set, the positional slice arguments are ignored and "
        "the script instead consumes --slice_config_in + --fragments and writes "
        "--slice_config_out.",
    )
    finalise_group.add_argument(
        "--finalise",
        action="store_true",
        help="Run in finalise mode: merge fragments into slice_config.csv.",
    )
    finalise_group.add_argument(
        "--slice_config_in",
        type=str,
        default=None,
        help="Input slice_config.csv (finalise mode).",
    )
    finalise_group.add_argument(
        "--slice_config_out",
        type=str,
        default=None,
        help="Output slice_config.csv (finalise mode).",
    )
    finalise_group.add_argument(
        "--fragments",
        type=str,
        default=None,
        help="Directory containing per-slice manifest fragment CSVs (finalise mode). "
        "Fragments are discovered by glob; empty dir results in slice_config copied "
        "unchanged.",
    )

    # Preview/debug options
    preview_group = p.add_argument_group("Preview Options", "Generate visual previews for quality checking")
    preview_group.add_argument(
        "--preview",
        type=str,
        default=None,
        help=(
            "Path to save a preview image (PNG) showing slice before, slice\n"
            "after, and the interpolated result. Useful for verifying\n"
            "interpolation quality."
        ),
    )
    preview_group.add_argument(
        "--preview_slice", type=int, default=None, help="Z-index to use for preview. Default: middle slice."
    )
    preview_group.add_argument("--preview_dpi", type=int, default=150, help="DPI for preview image [default: %(default)s]")

    add_overwrite_arg(p)
    return p


def generate_preview(
    vol_before,
    vol_after,
    interpolated,
    output_path,
    preview_slice=None,
    dpi=150,
    failure_reason: str | None = None,
):
    """
    Generate a preview image showing the interpolation results.

    When *interpolated* is ``None`` the preview degrades gracefully to a
    before/after pair with a red banner explaining why no output was
    produced. This keeps visual QA working even for hard-skipped slices.

    Parameters
    ----------
    vol_before, vol_after : np.ndarray
        Neighbouring volumes.
    interpolated : np.ndarray | None
        Interpolated result, or ``None`` for a hard-skipped slice.
    output_path : str or Path
        Path to save the preview image.
    preview_slice : int, optional
        Z-index to use for preview. Default: middle slice.
    dpi : int
        DPI for the output image.
    failure_reason : str, optional
        Text shown in the banner when *interpolated* is ``None``.
    """
    if preview_slice is None:
        preview_slice = vol_before.shape[0] // 2
    preview_slice = max(0, min(preview_slice, vol_before.shape[0] - 1))

    def normalize_for_display(img):
        img = img.astype(np.float32)
        p1, p99 = np.percentile(img[img > 0], [1, 99]) if np.any(img > 0) else (0, 1)
        if p99 > p1:
            img = (img - p1) / (p99 - p1)
        return np.clip(img, 0, 1)

    before_slice = normalize_for_display(vol_before[preview_slice])
    after_slice = normalize_for_display(vol_after[preview_slice])

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    axes[0].imshow(before_slice, cmap="gray")
    axes[0].set_title("Slice Before (input)")
    axes[0].axis("off")

    axes[1].imshow(after_slice, cmap="gray")
    axes[1].set_title("Slice After (input)")
    axes[1].axis("off")

    if interpolated is None:
        axes[2].set_facecolor("#3a0a0a")
        axes[2].text(
            0.5,
            0.55,
            "INTERPOLATION FAILED",
            color="white",
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold",
            transform=axes[2].transAxes,
        )
        if failure_reason:
            axes[2].text(
                0.5,
                0.35,
                f"reason: {failure_reason}",
                color="#ffd0d0",
                ha="center",
                va="center",
                fontsize=10,
                transform=axes[2].transAxes,
            )
        axes[2].set_xticks([])
        axes[2].set_yticks([])

        # XZ view still useful for context — skip the missing middle slab.
        y_mid = vol_before.shape[1] // 2
        xz_before = normalize_for_display(vol_before[:, y_mid, :])
        xz_after = normalize_for_display(vol_after[:, y_mid, :])
        gap = np.full(
            (max(1, min(vol_before.shape[0], vol_after.shape[0]) // 2), xz_before.shape[1]),
            0.15,
            dtype=np.float32,
        )
        xz_combined = np.vstack([xz_before, gap, xz_after])
        axes[3].imshow(xz_combined, cmap="gray", aspect="auto")
        axes[3].set_title("XZ View: Before | [skipped] | After")
        axes[3].axis("off")
    else:
        interp_slice = normalize_for_display(interpolated[preview_slice])
        axes[2].imshow(interp_slice, cmap="gray")
        axes[2].set_title("Interpolated (output)")
        axes[2].axis("off")

        y_mid = vol_before.shape[1] // 2
        xz_before = normalize_for_display(vol_before[:, y_mid, :])
        xz_interp = normalize_for_display(interpolated[:, y_mid, :])
        xz_after = normalize_for_display(vol_after[:, y_mid, :])
        xz_combined = np.vstack([xz_before, xz_interp, xz_after])
        axes[3].imshow(xz_combined, cmap="gray", aspect="auto")
        axes[3].set_title("XZ View: Before | Interp | After")
        axes[3].axhline(y=xz_before.shape[0], color="cyan", linestyle="--", linewidth=0.5)
        axes[3].axhline(y=xz_before.shape[0] + xz_interp.shape[0], color="cyan", linestyle="--", linewidth=0.5)
        axes[3].axis("off")

    title = (
        f"Slice Interpolation Preview (z={preview_slice}) — FAILED"
        if interpolated is None
        else f"Slice Interpolation Preview (z={preview_slice})"
    )
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Preview saved to: {output_path}")


_FRAGMENT_COLUMN_MAP = {
    "method_used": "interpolation_method_used",
    "fallback_reason": "interpolation_fallback_reason",
}


def _finalise(args) -> None:
    """Merge per-slice manifest fragments into a single slice_config.csv.

    Each fragment represents one attempt, successful or not:

    * ``interpolation_failed != true`` → slice was reconstructed. Stamps
      ``interpolated=true`` plus method / fallback_reason.
    * ``interpolation_failed == true`` → slice was hard-skipped. Stamps
      ``interpolated=false``, ``interpolation_failed=true``, and
      ``interpolation_fallback_reason=<reason>`` so the final report can
      surface it.
    """
    if args.slice_config_in is None or args.slice_config_out is None or args.fragments is None:
        raise SystemExit("--finalise requires --slice_config_in, --slice_config_out, and --fragments")

    fragments_dir = Path(args.fragments)
    fragment_paths: list[Path] = []
    if fragments_dir.is_dir():
        fragment_paths = sorted(fragments_dir.glob("*.csv"))
    elif fragments_dir.exists():
        fragment_paths = [fragments_dir]
    else:
        print(f"  Fragments path does not exist: {fragments_dir} — copying slice_config unchanged.")

    updates: dict[str, dict[str, object]] = {}
    interpolated_ids: set[str] = set()
    failed_ids: set[str] = set()
    import csv

    for frag_path in fragment_paths:
        with frag_path.open() as fh:
            reader = csv.DictReader(fh)
            for raw in reader:
                sid = slice_config_io.normalize_slice_id(raw.get("slice_id", ""))
                if not sid:
                    continue
                failed_raw = (raw.get("interpolation_failed") or "").strip().lower()
                failed = failed_raw in ("true", "1", "yes", "y", "t")
                entry: dict[str, object] = {
                    "interpolated": not failed,
                    "interpolation_failed": failed,
                }
                for col, val in raw.items():
                    if col in ("slice_id", "interpolation_failed", None) or val is None:
                        continue
                    target = _FRAGMENT_COLUMN_MAP.get(col)
                    if target:
                        entry[target] = val
                if failed:
                    failed_ids.add(sid)
                    entry["interpolation_method_used"] = ""
                else:
                    interpolated_ids.add(sid)
                updates[sid] = entry

    slice_config_io.stamp_many(args.slice_config_in, args.slice_config_out, updates)
    print(
        f"Finalise: merged {len(fragment_paths)} fragment(s), "
        f"{len(interpolated_ids)} interpolated, {len(failed_ids)} failed → {args.slice_config_out}"
    )


def main():
    p = _build_arg_parser()
    args = p.parse_args()

    if args.finalise:
        _finalise(args)
        return

    if not (args.slice_before and args.slice_after and args.output):
        p.error("slice_before, slice_after, and output are required unless --finalise is set")

    slice_before_path = Path(args.slice_before)
    slice_after_path = Path(args.slice_after)
    output_path = Path(args.output)

    if not slice_before_path.exists():
        p.error(f"Slice before not found: {slice_before_path}")
    if not slice_after_path.exists():
        p.error(f"Slice after not found: {slice_after_path}")

    assert_output_exists(output_path, p, args)

    print(f"Loading slice before: {slice_before_path}")
    vol_before, res_before = read_omezarr(slice_before_path)
    vol_before = np.array(vol_before)

    print(f"Loading slice after: {slice_after_path}")
    vol_after, res_after = read_omezarr(slice_after_path)
    vol_after = np.array(vol_after)

    # Handle shape mismatches
    if vol_before.shape != vol_after.shape:
        print(f"Shape mismatch detected: {vol_before.shape} vs {vol_after.shape}")

        # Handle z-dimension mismatch by truncating to minimum
        min_z = min(vol_before.shape[0], vol_after.shape[0])
        if vol_before.shape[0] != vol_after.shape[0]:
            print(f"  Truncating z-dimension to minimum: {min_z}")
            vol_before = vol_before[:min_z]
            vol_after = vol_after[:min_z]

        # Handle X/Y dimension mismatch by using maximum and zero-padding
        if vol_before.shape[1:] != vol_after.shape[1:]:
            max_x = max(vol_before.shape[1], vol_after.shape[1])
            max_y = max(vol_before.shape[2], vol_after.shape[2])
            print(f"  Adjusting X/Y dimensions to: ({max_x}, {max_y})")

            # Pad vol_before if needed
            if vol_before.shape[1] < max_x or vol_before.shape[2] < max_y:
                padded = np.zeros((min_z, max_x, max_y), dtype=vol_before.dtype)
                padded[:, : vol_before.shape[1], : vol_before.shape[2]] = vol_before
                vol_before = padded

            # Pad vol_after if needed
            if vol_after.shape[1] < max_x or vol_after.shape[2] < max_y:
                padded = np.zeros((min_z, max_x, max_y), dtype=vol_after.dtype)
                padded[:, : vol_after.shape[1], : vol_after.shape[2]] = vol_after
                vol_after = padded

        print(f"  Adjusted shapes: {vol_before.shape}")

    # Validate resolutions match
    if res_before != res_after:
        print(f"Warning: Resolution mismatch: {res_before} vs {res_after}")

    print(f"Volume shape: {vol_before.shape}")
    print(f"Resolution: {res_before}")
    print(f"Method: {args.method}")

    diagnostics: dict = {}

    if args.method == "zmorph":
        print("Performing z-morph interpolation...")
        interpolated, diagnostics = interpolate_z_morph(
            vol_before,
            vol_after,
            metric=args.registration_metric,
            max_iterations=args.max_iterations,
            blend_method=args.blend_method,
            overlap_search_window=args.overlap_search_window,
            min_overlap_correlation=args.min_overlap_correlation,
            reference_slab_size=args.reference_slab_size,
            min_foreground_fraction=args.min_foreground_fraction,
            min_ncc_improvement=args.min_ncc_improvement,
        )
    elif args.method == "average":
        print("Performing simple average interpolation...")
        interpolated = interpolate_average(vol_before, vol_after)
        diagnostics = {"method": "average", "method_used": "average", "fallback_reason": None}
    elif args.method == "weighted":
        print("Performing weighted average interpolation...")
        interpolated = interpolate_weighted(vol_before, vol_after)
        diagnostics = {"method": "weighted", "method_used": "weighted", "fallback_reason": None}
    else:
        p.error(f"Unknown method: {args.method}")

    failed = interpolated is None or diagnostics.get("interpolation_failed") is True

    if failed:
        print(
            f"  [interpolation] zmorph could not produce a reliable output "
            f"(reason={diagnostics.get('fallback_reason')}); emitting no zarr, "
            f"the slice will be treated as a gap downstream."
        )

    # Preview is still useful on failure (shows the before/after pair and the
    # skipped middle slab), so emit it regardless.
    if args.preview is not None:
        print("Generating preview...")
        generate_preview(
            vol_before,
            vol_after,
            interpolated,
            output_path=args.preview,
            preview_slice=args.preview_slice,
            dpi=args.preview_dpi,
            failure_reason=diagnostics.get("fallback_reason") if failed else None,
        )

    if not failed:
        final_result = interpolated
        original_dtype = vol_before.dtype
        if np.issubdtype(original_dtype, np.integer):
            final_result = np.clip(final_result, 0, np.iinfo(original_dtype).max)
            final_result = final_result.astype(original_dtype)

        print(f"Saving interpolated slice to: {output_path}")
        save_omezarr(da.from_array(final_result), str(output_path), res_before)
    else:
        print("Skipping zarr output — no fabricated data will enter the reconstruction.")

    if args.diagnostics is not None:
        diagnostics["slice_id"] = args.slice_id
        diagnostics["slice_before_path"] = str(slice_before_path)
        diagnostics["slice_after_path"] = str(slice_after_path)
        diagnostics["output_path"] = None if failed else str(output_path)
        diagnostics["interpolation_failed"] = bool(failed)
        diagnostics_path = Path(args.diagnostics)
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        with diagnostics_path.open("w") as fh:
            json.dump(diagnostics, fh, indent=2, default=_json_default)
        print(f"Diagnostics saved to: {diagnostics_path}")

    # Manifest records pipeline-relevant flags only — raw metrics live in the
    # per-slice diagnostics JSON. The `interpolation_failed` column is what
    # finalise_interpolation uses to decide whether to stamp the slice as
    # successfully interpolated or as a hard-skip.
    if args.manifest_entry is not None:
        manifest_path = Path(args.manifest_entry)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "slice_id": args.slice_id or "",
            "method": diagnostics.get("method", args.method),
            "method_used": "" if failed else (diagnostics.get("method_used") or args.method),
            "fallback_reason": diagnostics.get("fallback_reason") or "",
            "interpolation_failed": "true" if failed else "false",
            "output_path": "" if failed else str(output_path),
        }
        header = ",".join(row.keys())
        values = ",".join(_csv_escape(v) for v in row.values())
        with manifest_path.open("w") as fh:
            fh.write(header + "\n")
            fh.write(values + "\n")
        print(f"Manifest entry saved to: {manifest_path}")

    print("Done!")


def _json_default(obj):
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _csv_escape(value) -> str:
    s = "" if value is None else str(value)
    if "," in s or '"' in s or "\n" in s:
        s = '"' + s.replace('"', '""') + '"'
    return s


if __name__ == "__main__":
    main()
