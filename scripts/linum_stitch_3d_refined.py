#!/usr/bin/env python3
"""
Stitch a 3D mosaic grid with registration-refined blending.

This script uses the Lefebvre et al. (2017) motor displacement model to
compute tile positions.  Neighbor tile phase-correlation is used to fit a
full 2x2 affine transform that accounts for:
  - scan-to-stage rotation (θ)
  - non-perpendicularity of the motor X/Y axes (φ)
  - effective overlap fractions (Ox, Oy)

This corrects the systematic tile-position drift that occurs when the
motor axes are not perfectly perpendicular, which is visible as
misalignment at the mosaic edges.

Registration-based sub-pixel refinements can additionally improve
blending quality at tile boundaries.
"""

# Configure thread limits before numpy/scipy imports
import linumpy.config.threads  # noqa: F401

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from linumpy.io.zarr import read_omezarr
from linumpy.mosaic.grid import add_volume_to_mosaic
from linumpy.mosaic.motor import (
    apply_blend_shift_refinement,
    compute_affine_output_shape,
    compute_affine_positions,
    compute_registration_refinements,
    estimate_affine_from_pairs,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input_volume", help="Full path to a 3D mosaic grid volume (.ome.zarr)")
    p.add_argument("output_volume", help="Output stitched mosaic filename (.ome.zarr)")

    p.add_argument("--overlap_fraction", type=float, default=0.2, help="Expected tile overlap fraction (0-1). [%(default)s]")
    p.add_argument(
        "--blending_method",
        type=str,
        default="diffusion",
        choices=["none", "average", "diffusion"],
        help="Blending method for overlap regions. [%(default)s]",
    )
    p.add_argument(
        "--refinement_mode",
        type=str,
        default="blend_shift",
        choices=["none", "blend_shift", "full_shift"],
        help="How to apply registration refinements:\n"
        "  none: Pure motor positions, no refinement\n"
        "  blend_shift: Shift blending weights (recommended)\n"
        "  full_shift: Apply sub-pixel shifts to tiles [%(default)s]",
    )
    p.add_argument(
        "--max_refinement_px",
        type=float,
        default=10.0,
        help="Maximum allowed refinement shift in pixels. [%(default)s]\n"
        "Larger shifts are clamped to prevent bad registrations.",
    )
    p.add_argument(
        "--input_transform",
        type=str,
        default=None,
        help="Pre-computed 2x2 affine transform (.npy) for tile positioning.\n"
        "If not provided, the transform is estimated from neighbor\n"
        "tile correlation within the slice.",
    )
    p.add_argument(
        "--output_refinements", type=str, default=None, help="Output JSON file to save computed refinements for analysis."
    )
    p.add_argument("--overwrite", "-f", action="store_true", help="Overwrite output if it exists.")
    return p


def stitch_with_refinements(
    volume, tile_shape, positions, blending_method, refinement_mode, refinements, output_shape, overlap_fraction=0.2
):
    """
    Stitch tiles using pre-computed positions with optional registration refinements.
    """
    tile_height, tile_width = tile_shape[1], tile_shape[2]
    nx = volume.shape[1] // tile_height
    ny = volume.shape[2] // tile_width

    # Offset positions so the minimum is at (0, 0)
    # (off-diagonal terms can produce negative coordinates)
    min_row = min(p[0] for p in positions)
    min_col = min(p[1] for p in positions)
    if min_row < 0 or min_col < 0:
        positions = [(p[0] - min_row, p[1] - min_col) for p in positions]

    # Initialize output array
    output = np.zeros(output_shape, dtype=np.float32)

    for i in range(nx):
        for j in range(ny):
            # Extract tile
            r_start = i * tile_height
            r_end = (i + 1) * tile_height
            c_start = j * tile_width
            c_end = (j + 1) * tile_width

            tile = volume[:, r_start:r_end, c_start:c_end].copy()

            if np.any(tile < 0):
                tile = tile - tile.min()

            # Get position from motor positions
            pos = list(positions[i * ny + j])

            # Apply refinements if requested
            if refinement_mode == "blend_shift":
                # Collect refinements for this tile from its neighbors
                tile_refinements = []

                # From horizontal neighbor to the left
                if j > 0 and (i, j - 1) in refinements.get("horizontal", {}):
                    ref = refinements["horizontal"][(i, j - 1)]
                    tile_refinements.append({"dy": -ref["dy"], "dx": -ref["dx"]})

                # From horizontal neighbor to the right
                if (i, j) in refinements.get("horizontal", {}):
                    ref = refinements["horizontal"][(i, j)]
                    tile_refinements.append(ref)

                # From vertical neighbor above
                if i > 0 and (i - 1, j) in refinements.get("vertical", {}):
                    ref = refinements["vertical"][(i - 1, j)]
                    tile_refinements.append({"dy": -ref["dy"], "dx": -ref["dx"]})

                # From vertical neighbor below
                if (i, j) in refinements.get("vertical", {}):
                    ref = refinements["vertical"][(i, j)]
                    tile_refinements.append(ref)

                tile = apply_blend_shift_refinement(tile, tile_refinements, overlap_fraction)

            elif refinement_mode == "full_shift":
                # Apply average refinement as position offset (sub-pixel)
                # This is more aggressive - shifts the entire tile
                tile_refinements = []

                if j > 0 and (i, j - 1) in refinements.get("horizontal", {}):
                    tile_refinements.append(refinements["horizontal"][(i, j - 1)])
                if (i, j) in refinements.get("horizontal", {}):
                    tile_refinements.append(refinements["horizontal"][(i, j)])
                if i > 0 and (i - 1, j) in refinements.get("vertical", {}):
                    tile_refinements.append(refinements["vertical"][(i - 1, j)])
                if (i, j) in refinements.get("vertical", {}):
                    tile_refinements.append(refinements["vertical"][(i, j)])

                if tile_refinements:
                    avg_dy = np.mean([r["dy"] for r in tile_refinements]) / 2
                    avg_dx = np.mean([r["dx"] for r in tile_refinements]) / 2
                    pos[0] += avg_dy
                    pos[1] += avg_dx

            # Add tile to mosaic
            add_volume_to_mosaic(tile, pos, output, blendingMethod=blending_method)

    return output


def main():
    p = _build_arg_parser()
    args = p.parse_args()

    input_file = Path(args.input_volume)
    output_file = Path(args.output_volume)

    if output_file.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output_file}. Use -f to overwrite.")

    # Load volume
    logger.info(f"Loading mosaic grid: {input_file}")
    vol_dask, resolution = read_omezarr(str(input_file), level=0)
    if not hasattr(vol_dask, "chunks") or vol_dask.chunks is None:
        raise ValueError(
            f"Input mosaic {input_file} has no chunk metadata; tile shape "
            "cannot be determined. Regenerate the zarr with linumpy's OME-Zarr "
            "writer or pass --tile_shape explicitly."
        )
    tile_shape = vol_dask.chunks
    volume = np.array(vol_dask[:])

    logger.info(f"Volume shape: {volume.shape}")
    logger.info(f"Tile shape: {tile_shape}")
    logger.info(f"Overlap fraction: {args.overlap_fraction}")
    logger.info(f"Refinement mode: {args.refinement_mode}")

    nx = volume.shape[1] // tile_shape[1]
    ny = volume.shape[2] // tile_shape[2]
    logger.info(f"Grid: {nx} x {ny} tiles")

    # Correlate neighboring tiles (needed for affine estimation and blend refinement)
    logger.info("Computing neighbor tile correlations...")
    refinements = compute_registration_refinements(volume, tile_shape, nx, ny, args.overlap_fraction, args.max_refinement_px)

    stats = refinements["stats"]
    logger.info(f"  Total tile pairs: {stats['total_pairs']}")
    logger.info(f"  Valid registrations: {stats['valid_pairs']}")
    logger.info(f"  Clamped (large shifts): {stats['clamped_pairs']}")
    logger.info(f"  Mean refinement: {stats['mean_refinement']:.2f} px")
    logger.info(f"  Max refinement: {stats['max_refinement']:.2f} px")

    # Estimate or load the 2x2 affine displacement model
    if args.input_transform:
        transform = np.load(args.input_transform)
        logger.info(f"Loaded pre-computed transform from {args.input_transform}")
        from linumpy.mosaic.motor import _extract_displacement_params

        diagnostics = _extract_displacement_params(transform, tile_shape, args.overlap_fraction)
        diagnostics["fallback"] = False
        diagnostics["n_pairs"] = stats["valid_pairs"]
        diagnostics["lstsq_residual"] = 0.0
    else:
        transform, diagnostics = estimate_affine_from_pairs(refinements["pairs"], tile_shape, args.overlap_fraction)

    logger.info("Displacement model (Lefebvre et al. 2017):")
    logger.info(f"  Transform: [[{transform[0, 0]:.2f}, {transform[0, 1]:.2f}],")
    logger.info(f"              [{transform[1, 0]:.2f}, {transform[1, 1]:.2f}]]")
    if not diagnostics.get("fallback", False):
        logger.info(f"  Scan-to-stage rotation (θ): {diagnostics['theta_deg']:.3f}°")
        logger.info(f"  Non-perpendicularity (φ):   {diagnostics['phi_deg']:.3f}°")
        logger.info(f"  Effective overlap Ox:        {diagnostics['Ox_fraction']:.4f} (expected {args.overlap_fraction:.4f})")
        logger.info(f"  Effective overlap Oy:        {diagnostics['Oy_fraction']:.4f} (expected {args.overlap_fraction:.4f})")
        logger.info(f"  Off-diagonal terms:          {diagnostics['off_diagonal_px']} px/tile")

    # Compute tile positions from affine transform
    positions = compute_affine_positions(nx, ny, transform)

    # Compute output shape from affine positions (accounts for off-diagonal terms)
    output_shape = compute_affine_output_shape(nx, ny, tile_shape, transform)

    # Save refinements + affine diagnostics
    if args.output_refinements:
        json_refinements = {
            "horizontal": {f"{k[0]},{k[1]}": v for k, v in refinements["horizontal"].items()},
            "vertical": {f"{k[0]},{k[1]}": v for k, v in refinements["vertical"].items()},
            "stats": refinements["stats"],
            "displacement_model": diagnostics,
            "parameters": {
                "overlap_fraction": args.overlap_fraction,
                "max_refinement_px": args.max_refinement_px,
                "refinement_mode": args.refinement_mode,
                "input_transform": args.input_transform,
            },
        }
        with Path(args.output_refinements).open("w") as f:
            json.dump(json_refinements, f, indent=2)
        logger.info(f"Refinements saved to: {args.output_refinements}")

    logger.info(f"Output shape: {output_shape}")

    # Stitch with affine positions
    logger.info(f"Stitching with {args.blending_method} blending...")
    output = stitch_with_refinements(
        volume,
        tile_shape,
        positions,
        args.blending_method,
        args.refinement_mode,
        refinements,
        output_shape,
        args.overlap_fraction,
    )

    # Save output
    logger.info(f"Saving to: {output_file}")
    import dask.array as da

    from linumpy.io.zarr import save_omezarr

    save_omezarr(da.from_array(output), str(output_file), resolution, n_levels=3)

    # Collect metrics
    from linumpy.metrics import PipelineMetrics

    metrics = PipelineMetrics("stitch_3d_refined", str(output_file.parent))
    metrics.add_info("input_volume", str(input_file), "Input mosaic grid path")
    metrics.add_info("output_volume", str(output_file), "Output stitched volume path")
    metrics.add_info("input_shape", list(volume.shape), "Input mosaic shape")
    metrics.add_info("output_shape", list(output_shape), "Output stitched shape")
    metrics.add_info("num_tiles", nx * ny, "Number of tiles stitched")
    metrics.add_info("resolution", [float(r) for r in resolution], "Output resolution (mm)")
    metrics.add_info("blending_method", args.blending_method, "Blending method used")
    metrics.add_info("refinement_mode", args.refinement_mode, "Refinement strategy")

    metrics.add_metric("total_pairs", stats["total_pairs"], description="Total tile pairs evaluated")
    metrics.add_metric(
        "valid_pairs", stats["valid_pairs"], description="Successfully registered tile pairs", threshold_name="correlation"
    )
    metrics.add_metric("clamped_pairs", stats["clamped_pairs"], description="Pairs with clamped large shifts")
    metrics.add_metric("mean_refinement", stats["mean_refinement"], unit="px", description="Mean refinement shift in pixels")
    metrics.add_metric("max_refinement", stats["max_refinement"], unit="px", description="Max refinement shift in pixels")

    if not diagnostics.get("fallback", False):
        metrics.add_metric("theta_deg", diagnostics["theta_deg"], unit="deg", description="Scan-to-stage rotation")
        metrics.add_metric("phi_deg", diagnostics["phi_deg"], unit="deg", description="Non-perpendicularity angle")
        metrics.add_metric("Ox_fraction", diagnostics["Ox_fraction"], description="Effective overlap fraction (X)")
        metrics.add_metric("Oy_fraction", diagnostics["Oy_fraction"], description="Effective overlap fraction (Y)")

    overlap_reduction = 1.0 - (np.prod(output_shape) / np.prod(volume.shape))
    metrics.add_metric("overlap_reduction", float(overlap_reduction), description="Fraction of pixels removed by stitching")

    metrics.save(f"{output_file.stem}_metrics.json")
    metrics.log_issues()

    logger.info("Done!")


if __name__ == "__main__":
    main()
