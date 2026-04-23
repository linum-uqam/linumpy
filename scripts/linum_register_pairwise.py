#!/usr/bin/env python3
"""
Simplified pairwise registration for motor-position-based reconstruction.

This script performs two tasks:
1. Z-MATCHING: Find the optimal Z-overlap between consecutive slices
2. REFINEMENT: Compute small rotation and sub-pixel corrections

The XY alignment is handled by motor positions (shifts_xy.csv), so this script
only computes small corrections, not large translations.

Output:
- transform.tfm: SimpleITK transform file (rotation + small translation)
- offsets.txt: Z-index correspondence [fixed_z, moving_z]
- metrics.json: Registration quality metrics
"""

import linumpy.config.threads  # noqa: F401

import argparse
import logging
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from linumpy.cli.args import add_overwrite_arg
from linumpy.io.zarr import read_omezarr
from linumpy.metrics import collect_pairwise_registration_metrics
from linumpy.registration.transforms import (
    centre_of_mass_offset,
    create_transform,
    find_best_z,
    gradient_magnitude_alignment,
    register_refinement,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("in_fixed", help="Fixed volume (.ome.zarr) - bottom slice")
    p.add_argument("in_moving", help="Moving volume (.ome.zarr) - top slice")
    p.add_argument("out_directory", help="Output directory")

    # Z-matching
    z_group = p.add_argument_group("Z-matching")
    z_group.add_argument(
        "--slicing_interval_mm", type=float, default=0.200, help="Physical slice thickness in mm [%(default)s]"
    )
    z_group.add_argument(
        "--search_range_mm", type=float, default=0.100, help="Search range around expected Z in mm [%(default)s]"
    )
    z_group.add_argument("--moving_z_index", type=int, default=0, help="Z-index in moving volume to align [%(default)s]")

    # Refinement
    ref_group = p.add_argument_group("Refinement")
    ref_group.add_argument(
        "--enable_rotation",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Enable rotation correction. Use --no-enable_rotation to disable. [%(default)s]",
    )
    # Legacy alias retained for backward-compatibility with the Nextflow pipeline
    # (workflows/reconst_3d/soct_3d_reconst.nf still emits --no_rotation).
    ref_group.add_argument(
        "--no_rotation",
        dest="enable_rotation",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    ref_group.add_argument(
        "--max_rotation_deg", type=float, default=5.0, help="Maximum rotation correction in degrees [%(default)s]"
    )
    ref_group.add_argument(
        "--max_translation_px", type=float, default=20.0, help="Maximum translation refinement in pixels [%(default)s]"
    )
    ref_group.add_argument(
        "--initial_alignment",
        choices=["none", "com", "gradient", "both"],
        default="both",
        help="Initial alignment method before refinement:\n"
        "  none     - no initial alignment\n"
        "  com      - centre of mass alignment\n"
        "  gradient - gradient magnitude phase correlation\n"
        "  both     - try gradient first, fall back to com [%(default)s]",
    )

    # Output
    p.add_argument("--out_transform", default="transform.tfm")
    p.add_argument("--out_offsets", default="offsets.txt")
    p.add_argument("--screenshot", default=None, help="Save debug screenshot")

    add_overwrite_arg(p)
    return p


def normalize(image):
    """Normalize image to [0, 1] using percentile clipping."""
    valid = image > 0
    if not np.any(valid):
        return np.zeros_like(image, dtype=np.float32)

    pmin = np.percentile(image[valid], 5)
    pmax = np.percentile(image[valid], 95)

    if pmax <= pmin:
        return np.zeros_like(image, dtype=np.float32)

    norm = (image.astype(np.float32) - pmin) / (pmax - pmin)
    return np.clip(norm, 0, 1)


def main():
    p = _build_arg_parser()
    args = p.parse_args()

    # Load volumes
    logger.info(f"Loading fixed: {args.in_fixed}")
    fixed_vol, res = read_omezarr(args.in_fixed)

    logger.info(f"Loading moving: {args.in_moving}")
    moving_vol, _ = read_omezarr(args.in_moving)

    # Create output directory
    out_dir = Path(args.out_directory)
    if out_dir.exists() and not args.f:
        p.error(f"Output directory exists: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Get moving slice
    moving_slice = np.array(moving_vol[args.moving_z_index])
    moving_norm = normalize(moving_slice)

    # Calculate expected Z position
    # The moving slice (top of moving volume) should match near the BOTTOM of fixed volume
    # expected_z is where in fixed_vol we expect to find a match for moving_slice

    # NOTE: read_omezarr returns resolution in millimeters (OME-NGFF standard)
    res_z_mm = res[0] if len(res) >= 1 else 0.010  # mm (default 10 µm)

    logger.info(f"Resolution from metadata: {res}")
    logger.info(f"Using Z resolution: {res_z_mm} mm ({res_z_mm * 1000:.2f} µm)")

    # Calculate interval in voxels: slicing_interval_mm / res_z_mm
    interval_vox = round(args.slicing_interval_mm / res_z_mm)
    search_vox = round(args.search_range_mm / res_z_mm)

    # The overlap region is at the bottom of fixed volume
    # The match should be near: fixed_vol.shape[0] - interval_vox + moving_z_index
    fixed_nz = fixed_vol.shape[0]
    expected_z = fixed_nz - interval_vox + args.moving_z_index

    logger.info(f"Fixed volume: {fixed_nz} slices")
    logger.info(f"Interval: {args.slicing_interval_mm} mm = {interval_vox} voxels")
    logger.info(f"Search range: {args.search_range_mm} mm = {search_vox} voxels")
    logger.info(f"Expected Z (before clamp): {expected_z}")

    # Ensure expected_z is within bounds
    expected_z = max(0, min(fixed_nz - 1, expected_z))

    logger.info(f"Searching for match near z={expected_z} in fixed volume (search ±{search_vox})")

    # Find best Z match
    best_z, z_correlation = find_best_z(fixed_vol, moving_slice, expected_z, search_vox)

    logger.info(f"Best Z match: {best_z} (expected: {expected_z}, correlation: {z_correlation:.4f})")

    # Warn if z-match deviates significantly from expected
    z_deviation = abs(best_z - expected_z)
    if z_deviation > search_vox // 2:
        logger.warning(f"Z-match deviation is large ({z_deviation} voxels) - may indicate alignment issues")

    # Get fixed slice at best Z
    fixed_slice = np.array(fixed_vol[best_z])
    fixed_norm = normalize(fixed_slice)

    # Compute initial alignment offset
    initial_offset = None
    if args.initial_alignment != "none":
        if args.initial_alignment in ("gradient", "both"):
            dy, dx = gradient_magnitude_alignment(fixed_norm, moving_norm)
            mag = np.sqrt(dy**2 + dx**2)
            if mag > 1.0:
                initial_offset = (dy, dx)
                logger.info(f"Gradient magnitude initial offset: dy={dy:.1f}, dx={dx:.1f}")

        if initial_offset is None and args.initial_alignment in ("com", "both"):
            dy, dx = centre_of_mass_offset(fixed_norm, moving_norm)
            mag = np.sqrt(dy**2 + dx**2)
            if mag > 1.0:
                initial_offset = (dy, dx)
                logger.info(f"Centre of mass initial offset: dy={dy:.1f}, dx={dx:.1f}")

        if initial_offset is None:
            logger.info("No significant initial offset detected, starting from identity")

    # Compute refinement
    logger.info(f"Computing refinement (rotation={args.enable_rotation})...")
    tx, ty, angle_deg, metric = register_refinement(
        fixed_norm,
        moving_norm,
        enable_rotation=args.enable_rotation,
        max_rotation_deg=args.max_rotation_deg,
        max_translation_px=args.max_translation_px,
        initial_offset=initial_offset,
    )

    logger.info(f"Refinement: tx={tx:.2f}px, ty={ty:.2f}px, rot={angle_deg:.3f}°")

    # Create and save transform
    center = [fixed_slice.shape[1] / 2.0, fixed_slice.shape[0] / 2.0]
    transform = create_transform(tx, ty, angle_deg, center)
    sitk.WriteTransform(transform, str(out_dir / args.out_transform))

    # Save offsets
    np.savetxt(str(out_dir / args.out_offsets), np.array([best_z, args.moving_z_index]), fmt="%d")

    # Detect interpolated neighbours. Registrations where either volume is a
    # synthetic (interpolated) slice produce unreliable rotation/translation
    # because one side of the pair is a blend of non-overlapping tissue. We
    # still run the registration (so a .tfm exists), but force the metrics
    # into the "error" status so the downstream stacking gate
    # (skip_error_status in linum_stack_slices_motor.py) discards the
    # transform and falls back to motor-only positioning for that slice.
    fixed_is_interpolated = "_interpolated" in Path(args.in_fixed).name
    moving_is_interpolated = "_interpolated" in Path(args.in_moving).name
    touches_interpolated = fixed_is_interpolated or moving_is_interpolated
    if touches_interpolated:
        logger.warning(
            "Registration involves an interpolated slice "
            f"(fixed={fixed_is_interpolated}, moving={moving_is_interpolated}); "
            "marking transform as unreliable."
        )

    # Collect metrics using standard collector
    collect_pairwise_registration_metrics(
        registration_error=float(metric) if metric != float("inf") else 0.0,
        tx=float(tx),
        ty=float(ty),
        rotation_deg=float(angle_deg),
        best_z_index=int(best_z),
        expected_z_index=int(expected_z),
        output_path=str(out_dir),
        fixed_path=args.in_fixed,
        moving_path=args.in_moving,
        z_correlation=float(z_correlation),
        params={
            "slicing_interval_mm": args.slicing_interval_mm,
            "search_range_mm": args.search_range_mm,
            "enable_rotation": args.enable_rotation,
            "max_rotation_deg": args.max_rotation_deg,
            "max_translation_px": args.max_translation_px,
            "z_correlation": float(z_correlation),
            "z_deviation": int(z_deviation),
            "fixed_is_interpolated": bool(fixed_is_interpolated),
            "moving_is_interpolated": bool(moving_is_interpolated),
        },
    )

    if touches_interpolated:
        # Re-save the metrics JSON with a forced error status so
        # stack_slices_motor discards this transform via skip_error_status.
        import json

        metrics_file = out_dir / "pairwise_registration_metrics.json"
        if metrics_file.exists():
            with metrics_file.open() as f:
                data = json.load(f)
            data["overall_status"] = "error"
            data.setdefault("errors", []).append("One or both inputs are an interpolated slice; transform is synthetic.")
            if "registration_confidence" in data.get("metrics", {}):
                data["metrics"]["registration_confidence"]["value"] = 0.0
                data["metrics"]["registration_confidence"]["status"] = "error"
            with metrics_file.open("w") as f:
                json.dump(data, f, indent=2)

    logger.info(f"Results saved to {out_dir}")

    # Screenshot
    if args.screenshot:
        import matplotlib.pyplot as plt

        # Apply transform for visualization
        moving_sitk = sitk.GetImageFromArray(moving_norm.astype(np.float32))
        transform_2d = sitk.Euler2DTransform()
        transform_2d.SetCenter(center)
        transform_2d.SetAngle(np.radians(angle_deg))
        transform_2d.SetTranslation([tx, ty])

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(moving_sitk)
        resampler.SetTransform(transform_2d)
        resampler.SetInterpolator(sitk.sitkLinear)
        registered = sitk.GetArrayFromImage(resampler.Execute(moving_sitk))

        _fig, axes = plt.subplots(2, 2, figsize=(12, 12))

        axes[0, 0].imshow(fixed_norm, cmap="gray")
        axes[0, 0].set_title(f"Fixed (z={best_z})")

        axes[0, 1].imshow(moving_norm, cmap="gray")
        axes[0, 1].set_title(f"Moving (z={args.moving_z_index})")

        axes[1, 0].imshow(registered, cmap="gray")
        axes[1, 0].set_title(f"Registered (tx={tx:.1f}, ty={ty:.1f}, rot={angle_deg:.2f}°)")

        # Overlay
        overlay = np.zeros((*fixed_norm.shape, 3))
        overlay[:, :, 0] = fixed_norm  # Red
        overlay[:, :, 1] = registered  # Green
        axes[1, 1].imshow(overlay)
        axes[1, 1].set_title("Overlay (fixed=red, registered=green)")

        for ax in axes.flat:
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(args.screenshot, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Screenshot saved to {args.screenshot}")


if __name__ == "__main__":
    main()
