#!/usr/bin/env python3
"""Per-slice affine diagnostic for the refined stitching step.

For each mosaic_grid_*.ome.zarr in a directory, load only the central Z
plane and run the same two calls that ``linum_stitch_3d_refined.py`` makes
before writing any output:

    refinements = compute_registration_refinements(...)
    transform, diagnostics = estimate_affine_from_pairs(
        refinements["pairs"], tile_shape, overlap_fraction
    )

Emit one CSV row per slice with the fitted 2x2 affine, the Lefebvre
displacement-model parameters (theta, phi, Ox, Oy), and the raw
number-of-pairs / residual statistics.  Also emit one per-slice JSON
capturing the full refinements["pairs"] list so further analysis can be
done offline without re-reading the (very large) mosaic grids.

The script does **not** write any stitched output and does not touch the
nextflow output directories of the reconstruction pipeline, so it is
safe to run alongside a production run and does not invalidate any
downstream manual-alignment baseline.
"""

# Configure thread limits before numpy/scipy imports
import linumpy.config.threads  # noqa: F401

import argparse
import csv
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

from linumpy.io import slice_config as slice_config_io
from linumpy.io.zarr import read_omezarr
from linumpy.mosaic.motor import compute_registration_refinements, estimate_affine_from_pairs

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_SLICE_RE = re.compile(r"z(\d+)")


class _NumpyEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument(
        "input_dir",
        help="Directory containing mosaic_grid_*z??.ome.zarr files (the preproc output)",
    )
    p.add_argument(
        "output_csv",
        help="Path to the per-slice affine diagnostics CSV to write",
    )
    p.add_argument(
        "--overlap_fraction",
        type=float,
        default=0.2,
        help="Expected tile overlap fraction (must match the acquisition). [%(default)s]",
    )
    p.add_argument(
        "--max_refinement_px",
        type=float,
        default=1e9,
        help="Clamp threshold for stored refinements. Does not affect the absolute\n"
        "displacements fed into the affine LS fit. [%(default)s]",
    )
    p.add_argument(
        "--slice_config",
        type=str,
        default=None,
        help="Optional slice_config.csv; slices with use=false are skipped.",
    )
    p.add_argument(
        "--json_dir",
        type=str,
        default=None,
        help="Optional directory to write one per-slice refinements JSON (including\n"
        "the full pair list).  Not created if the argument is not provided.",
    )
    p.add_argument(
        "--pattern",
        type=str,
        default="mosaic_grid*_z*.ome.zarr",
        help="Glob pattern used to discover input volumes. [%(default)s]",
    )
    p.add_argument(
        "--skip_first",
        type=int,
        default=0,
        help="Skip this many leading slices from the sorted discovery order. [%(default)s]",
    )
    return p


def _extract_slice_id(path: Path) -> str:
    match = _SLICE_RE.search(path.name)
    return match.group(1) if match else "unknown"


def _serialize_pairs(pairs: list[dict]) -> list[dict]:
    return [
        {
            "row_delta": int(p["row_delta"]),
            "col_delta": int(p["col_delta"]),
            "measured_dy": float(p["measured_dy"]),
            "measured_dx": float(p["measured_dx"]),
        }
        for p in pairs
    ]


def _analyze_slice(
    zarr_path: Path,
    slice_id: str,
    overlap_fraction: float,
    max_refinement_px: float,
    json_dir: Path | None,
) -> dict[str, object]:
    vol, _resolution = read_omezarr(str(zarr_path), level=0)

    tile_shape = tuple(vol.chunks)
    if len(tile_shape) != 3:
        raise ValueError(f"Expected 3D mosaic grid, got chunks {tile_shape} for {zarr_path}")

    nx = vol.shape[1] // tile_shape[1]
    ny = vol.shape[2] // tile_shape[2]
    if nx == 0 or ny == 0:
        raise ValueError(
            f"Mosaic grid {zarr_path} has fewer than one full tile per axis (shape={vol.shape}, chunks={tile_shape})"
        )

    z_mid_full = vol.shape[0] // 2
    logger.info(
        "slice %s: shape=%s tile=%s grid=%dx%d reading z_mid=%d",
        slice_id,
        tuple(vol.shape),
        tile_shape,
        nx,
        ny,
        z_mid_full,
    )

    z_plane = np.asarray(vol[z_mid_full : z_mid_full + 1])

    refinements = compute_registration_refinements(
        z_plane,
        tile_shape,
        nx,
        ny,
        overlap_fraction,
        max_refinement_px=max_refinement_px,
    )

    pairs = refinements["pairs"]
    if not pairs:
        logger.warning("slice %s: no valid tile pairs produced by phase correlation", slice_id)
    transform, diagnostics = estimate_affine_from_pairs(pairs, tile_shape, overlap_fraction)

    step_y = float(np.sqrt(transform[0, 0] ** 2 + transform[1, 0] ** 2))
    step_x = float(np.sqrt(transform[0, 1] ** 2 + transform[1, 1] ** 2))

    stats = refinements["stats"]
    row: dict[str, object] = {
        "slice_id": slice_id,
        "nx": int(nx),
        "ny": int(ny),
        "tile_h": int(tile_shape[1]),
        "tile_w": int(tile_shape[2]),
        "n_total_pairs": int(stats["total_pairs"]),
        "n_valid_pairs": int(stats["valid_pairs"]),
        "n_clamped_pairs": int(stats["clamped_pairs"]),
        "mean_refinement_px": float(stats["mean_refinement"]),
        "max_refinement_px": float(stats["max_refinement"]),
        "A00": float(transform[0, 0]),
        "A01": float(transform[0, 1]),
        "A10": float(transform[1, 0]),
        "A11": float(transform[1, 1]),
        "step_y_px": step_y,
        "step_x_px": step_x,
        "theta_deg": float(diagnostics.get("theta_deg", float("nan"))),
        "phi_deg": float(diagnostics.get("phi_deg", float("nan"))),
        "Ox_fraction": float(diagnostics.get("Ox_fraction", float("nan"))),
        "Oy_fraction": float(diagnostics.get("Oy_fraction", float("nan"))),
        "expected_overlap": float(diagnostics.get("expected_overlap", overlap_fraction)),
        "lstsq_residual": float(diagnostics.get("lstsq_residual", float("nan"))),
        "fallback": bool(diagnostics.get("fallback", False)),
    }

    if json_dir is not None:
        json_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "slice_id": slice_id,
            "tile_shape": list(tile_shape),
            "grid_shape": [int(nx), int(ny)],
            "overlap_fraction": overlap_fraction,
            "transform": transform.tolist(),
            "displacement_model": diagnostics,
            "stats": stats,
            "pairs": _serialize_pairs(pairs),
        }
        (json_dir / f"slice_z{slice_id}_affine.json").write_text(json.dumps(payload, indent=2, cls=_NumpyEncoder))

    return row


def main() -> int:
    """Run function."""
    parser = _build_arg_parser()
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        parser.error(f"Input directory does not exist: {input_dir}")

    zarr_paths = sorted(input_dir.glob(args.pattern))
    if args.skip_first > 0:
        zarr_paths = zarr_paths[args.skip_first :]
    if not zarr_paths:
        parser.error(f"No mosaic grids matching {args.pattern!r} in {input_dir}")

    used_slices: set[str] | None = None
    if args.slice_config is not None:
        slice_config_path = Path(args.slice_config)
        if not slice_config_path.exists():
            parser.error(f"slice_config.csv not found: {slice_config_path}")
        used_slices = slice_config_io.filter_slices_to_use(slice_config_path)
        logger.info("slice_config: %d slices marked as use=true", len(used_slices))

    json_dir = Path(args.json_dir) if args.json_dir else None

    fieldnames = [
        "slice_id",
        "nx",
        "ny",
        "tile_h",
        "tile_w",
        "n_total_pairs",
        "n_valid_pairs",
        "n_clamped_pairs",
        "mean_refinement_px",
        "max_refinement_px",
        "A00",
        "A01",
        "A10",
        "A11",
        "step_y_px",
        "step_x_px",
        "theta_deg",
        "phi_deg",
        "Ox_fraction",
        "Oy_fraction",
        "expected_overlap",
        "lstsq_residual",
        "fallback",
    ]

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for zarr_path in zarr_paths:
            slice_id = _extract_slice_id(zarr_path)
            if used_slices is not None and slice_id not in used_slices:
                logger.info("slice %s: skipped (slice_config use=false)", slice_id)
                continue
            try:
                row = _analyze_slice(
                    zarr_path,
                    slice_id,
                    overlap_fraction=args.overlap_fraction,
                    max_refinement_px=args.max_refinement_px,
                    json_dir=json_dir,
                )
            except Exception:
                logger.exception("slice %s: analysis failed for %s", slice_id, zarr_path)
                continue
            writer.writerow(row)
            fh.flush()
            logger.info(
                "slice %s: step_y=%.3f step_x=%.3f theta=%+.3f deg phi=%+.3f deg Ox=%.4f Oy=%.4f (%d valid / %d pairs)",
                slice_id,
                row["step_y_px"],
                row["step_x_px"],
                row["theta_deg"],
                row["phi_deg"],
                row["Ox_fraction"],
                row["Oy_fraction"],
                row["n_valid_pairs"],
                row["n_total_pairs"],
            )

    logger.info("wrote %s", output_csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
