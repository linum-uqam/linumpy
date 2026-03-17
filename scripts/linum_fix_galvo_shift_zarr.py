#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fix galvo shift artefacts in assembled mosaic OME-Zarr files.

When the raw ``.bin`` files are no longer available, this script provides a
way to detect and correct galvo mirror artefacts directly from the assembled
OME-Zarr mosaic grid.

The galvo return region creates a dark band at a fixed position in each OCT
tile.  In an *unfixed* mosaic (false-negative detection during the pipeline),
this band remains inside each tile's data and produces repeating dark vertical
stripes in the XY view of the mosaic.

**How it works**

Each OME-Zarr chunk corresponds exactly to one OCT tile (the zarr chunk shape
equals the tile size used during assembly).  Detection therefore works by
sampling a few representative chunks, computing their average-intensity
projection, and calling the same dark-band detector used for raw tiles.

The fix per chunk uses a circular roll (``np.roll``) identical to the raw-tile
fix, moving the dark band to the end of the tile's A-line range.  Those edge
pixels are then linearly interpolated from the adjoining valid columns.

``--mode undo`` reverses a previously applied fix by rolling each chunk in the
opposite direction.  Use this when the pipeline incorrectly applied a galvo fix
(false-positive detection).

Examples
--------
Detect only (dry-run, no files written)::

    linum_fix_galvo_shift_zarr.py mosaic_grid_3d_z47.ome.zarr fixed_z47.ome.zarr \\
        --detect_only

Auto-detect and fix::

    linum_fix_galvo_shift_zarr.py mosaic_grid_3d_z47.ome.zarr fixed_z47.ome.zarr

Manually specify band position (skip auto-detection)::

    linum_fix_galvo_shift_zarr.py mosaic_grid_3d_z47.ome.zarr fixed_z47.ome.zarr \\
        --band_start 440 --band_width 40

Undo an incorrectly applied fix (shift value from pipeline log or slice_config)::

    linum_fix_galvo_shift_zarr.py mosaic_grid_3d_z50.ome.zarr fixed_z50.ome.zarr \\
        --mode undo --shift 60

Update slice_config.csv after fixing::

    linum_fix_galvo_shift_zarr.py mosaic_grid_3d_z47.ome.zarr fixed_z47.ome.zarr \\
        --update_config path/to/slice_config.csv --slice_id 47
"""
# Configure thread limits before numpy/scipy imports
import linumpy._thread_config  # noqa: F401

import argparse
import csv
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from linumpy.io.zarr import OmeZarrWriter
from linumpy.preproc.xyzcorr import detect_galvo_band_in_tile
from linumpy.utils.io import add_overwrite_arg, assert_output_exists


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input_zarr",
                   help="Input mosaic grid OME-Zarr file (*.ome.zarr).")
    p.add_argument("output_zarr",
                   help="Output corrected OME-Zarr file path.")

    mode_group = p.add_argument_group("Operation mode")
    mode_group.add_argument("--detect_only", action="store_true",
                            help="Only detect and print band info; do not write output.")
    mode_group.add_argument("--mode", choices=["fix", "undo"], default="fix",
                            help="'fix': apply galvo fix (default).\n"
                                 "'undo': reverse a previously applied fix.")

    detect_group = p.add_argument_group(
        "Band detection overrides",
        "Override auto-detection with manual values.")
    detect_group.add_argument("--band_start", type=int, default=None,
                              help="Start position of dark band within a tile (pixels). "
                                   "Overrides auto-detection.")
    detect_group.add_argument("--band_width", type=int, default=None,
                              help="Width of dark band (pixels). "
                                   "Overrides auto-detection.")
    detect_group.add_argument("--shift", type=int, default=None,
                              help="Explicit roll shift for --mode undo. "
                                   "Equals the shift that was applied during pipeline creation.")
    detect_group.add_argument("--detection_level", type=int, default=1,
                              help="Pyramid level used for auto-detection (0=full res). "
                                   "Default: 1 (2× downsampled for speed).")
    detect_group.add_argument("--min_confidence", type=float, default=0.2,
                              help="Minimum detection confidence to proceed with fix "
                                   "in auto mode (default: 0.2).")

    config_group = p.add_argument_group("Slice config update")
    config_group.add_argument("--update_config", metavar="SLICE_CONFIG_CSV",
                              help="Path to slice_config.csv to update after fixing.")
    config_group.add_argument("--slice_id", type=int, default=None,
                              help="Slice ID to update in slice_config.csv "
                                   "(required with --update_config).")

    p.add_argument("-v", "--verbose", action="store_true",
                   help="Print per-chunk detection results.")
    add_overwrite_arg(p)
    return p


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def _open_level(zarr_root: Path, level: int):
    """Open a specific pyramid level from an OME-Zarr, returning (zarr_array, res)."""
    import zarr
    from ome_zarr.io import parse_url
    from ome_zarr.reader import Reader, Multiscales

    location = parse_url(str(zarr_root))
    if location is None:
        raise FileNotFoundError(f"Cannot open as OME-Zarr: {zarr_root}")
    reader = Reader(location)
    nodes = list(reader())
    image_node = nodes[0]
    multiscale = next(s for s in image_node.specs if isinstance(s, Multiscales))

    # Clamp to available levels
    actual_level = min(level, len(multiscale.datasets) - 1)
    arr = zarr.open_array(zarr_root / multiscale.datasets[actual_level], mode='r')

    coord_transforms = image_node.metadata["coordinateTransformations"][0]
    res = [1.0] * len(arr.shape)
    for tr in coord_transforms:
        if tr['type'] == 'scale':
            res = tr['scale']
            break

    return arr, res, actual_level, multiscale


def _auto_detect(zarr_root: Path, detection_level: int, verbose: bool = False):
    """Sample representative chunks and return (band_start, band_width, confidence).

    band_start and band_width are expressed in level-0 (full-resolution) pixels.
    """
    det_arr, _, actual_level, _ = _open_level(zarr_root, detection_level)
    scale_factor = 2 ** actual_level  # ratio between detection level and level 0

    chunk_x = det_arr.chunks[1]
    chunk_y = det_arr.chunks[2]
    n_cx = det_arr.shape[1] // chunk_x
    n_cy = det_arr.shape[2] // chunk_y

    # Sample from the central region (more likely to contain tissue).
    cx_lo = max(0, n_cx // 4)
    cx_hi = max(cx_lo, min(n_cx - 1, 3 * n_cx // 4))
    cy_mid = n_cy // 2

    n_samples = min(6, cx_hi - cx_lo + 1)
    cx_indices = list(dict.fromkeys(
        np.linspace(cx_lo, cx_hi, n_samples, dtype=int).tolist()
    ))

    detections = []
    for cx in cx_indices:
        xs = cx * chunk_x
        xe = xs + chunk_x
        ys = cy_mid * chunk_y
        ye = ys + chunk_y

        chunk = np.asarray(det_arr[:, xs:xe, ys:ye], dtype=np.float32)
        if float(chunk.mean()) < 5.0:
            if verbose:
                print(f"  Chunk ({cx}, {cy_mid}): skipped (low signal mean={chunk.mean():.1f})")
            continue

        tile_aip = chunk.mean(axis=0)  # (chunk_x, chunk_y)
        bs, bw, conf = detect_galvo_band_in_tile(tile_aip)
        detections.append((bs, bw, conf))

        if verbose:
            print(f"  Chunk ({cx}, {cy_mid}): band_start={bs:4d}px, "
                  f"band_width={bw:3d}px, confidence={conf:.3f}")

    if not detections:
        return 0, 0, 0.0

    best_conf = float(np.max([d[2] for d in detections]))
    med_start = float(np.median([d[0] for d in detections]))
    med_width = float(np.median([d[1] for d in detections]))

    # Penalise inconsistency across chunks.
    if len(detections) > 1:
        starts = np.array([d[0] for d in detections])
        n_consistent = int(np.sum(np.abs(starts - med_start) <= max(chunk_x * 0.05, 5)))
        consistency = n_consistent / len(detections)
        best_conf *= consistency ** 0.5

    # Scale back to level-0 pixels.
    band_start_l0 = int(round(med_start * scale_factor))
    band_width_l0 = int(round(med_width * scale_factor))

    return band_start_l0, band_width_l0, best_conf


# ---------------------------------------------------------------------------
# Fix / undo
# ---------------------------------------------------------------------------

def _apply_fix(zarr_root: Path, output_path: Path,
               band_start: int, band_width: int,
               mode: str, undo_shift: int,
               verbose: bool = False):
    """Write a corrected zarr by rolling each level-0 chunk along the A-line axis.

    Parameters
    ----------
    zarr_root : Path
    output_path : Path
    band_start : int
        Position of the dark band start within a tile (fix mode only).
    band_width : int
        Width of the dark band in pixels (fix mode only).
    mode : str
        'fix' or 'undo'.
    undo_shift : int or None
        The original shift to reverse (undo mode only).
    """
    arr, res, _, _ = _open_level(zarr_root, level=0)
    shape = arr.shape          # (nz, nx_mosaic, ny_mosaic)
    chunk_x = arr.chunks[1]   # OCT tile width in X
    chunk_y = arr.chunks[2]   # OCT tile width in Y
    dtype = arr.dtype

    n_cx = shape[1] // chunk_x
    n_cy = shape[2] // chunk_y

    # Shift to apply per chunk along axis=1.
    # Fix: roll so the dark band lands at the END of the tile (positions
    #      [chunk_x - band_width, chunk_x)), then interpolate.
    # Undo: roll in the opposite direction by the original shift value.
    if mode == 'fix':
        roll_shift = chunk_x - band_start - band_width
    else:  # undo
        roll_shift = -undo_shift

    print(f"Roll shift per tile chunk: {roll_shift:+d} px  (mode={mode})")

    writer = OmeZarrWriter(
        str(output_path),
        shape=shape,
        chunk_shape=(shape[0], chunk_x, chunk_y),
        dtype=dtype,
        overwrite=True,
    )

    for kx in tqdm(range(n_cx), desc="Tile columns (axis 1)"):
        xs = kx * chunk_x
        xe = xs + chunk_x

        # Pre-load the leftmost column of the next tile for interpolation.
        if mode == 'fix' and kx < n_cx - 1:
            right_col_full = np.asarray(arr[:, xe:xe + 1, :], dtype=np.float32)
        else:
            right_col_full = None

        for ky in range(n_cy):
            ys = ky * chunk_y
            ye = ys + chunk_y

            chunk = np.asarray(arr[:, xs:xe, ys:ye], dtype=np.float32)
            fixed = np.roll(chunk, roll_shift, axis=1)

            if mode == 'fix' and band_width > 0:
                # After rolling, the dark band sits at columns
                # [chunk_x - band_width, chunk_x).
                # Replace with linear interpolation between the last valid
                # column inside this tile and the first valid column of the
                # next tile.
                left_col = fixed[:, chunk_x - band_width - 1, :]  # (nz, ny_chunk)

                if right_col_full is not None:
                    right_col = right_col_full[:, 0, ys:ye]   # (nz, ny_chunk)
                else:
                    right_col = left_col  # last tile: repeat edge

                for i in range(band_width):
                    alpha = (i + 1.0) / (band_width + 1.0)
                    fixed[:, chunk_x - band_width + i, :] = (
                        (1.0 - alpha) * left_col + alpha * right_col
                    )

            writer[0:shape[0], xs:xe, ys:ye] = fixed.astype(dtype)

    print("Regenerating OME-Zarr pyramid levels ...")
    writer.finalize(res, n_levels=5)


# ---------------------------------------------------------------------------
# Slice-config update
# ---------------------------------------------------------------------------

def _update_slice_config(config_path: Path, slice_id: int,
                         confidence: float, fix_applied: bool, mode: str):
    """Update galvo_confidence and galvo_fix columns for one slice."""
    rows = []
    fieldnames = None
    with open(config_path, newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Empty or malformed CSV: {config_path}")
        fieldnames = list(reader.fieldnames)
        for row in reader:
            rows.append(dict(row))

    updated = False
    for row in rows:
        if int(row['slice_id']) == slice_id:
            if 'galvo_confidence' in row:
                row['galvo_confidence'] = f"{confidence:.3f}"
            if 'galvo_fix' in row:
                row['galvo_fix'] = 'true' if fix_applied else 'false'
            if 'notes' in row:
                existing = row.get('notes', '')
                tag = f"zarr_retrofix_{mode}"
                row['notes'] = f"{existing}; {tag}".strip('; ') if existing else tag
            updated = True
            break

    if not updated:
        print(f"  Warning: slice_id {slice_id:02d} not found in {config_path}")
        return

    with open(config_path, 'w', newline='') as f:
        writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
        writer_csv.writeheader()
        writer_csv.writerows(rows)

    print(f"Updated {config_path}  →  slice {slice_id:02d}: "
          f"galvo_fix={'true' if fix_applied else 'false'}, "
          f"confidence={confidence:.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    input_path = Path(args.input_zarr).resolve()
    if not input_path.exists():
        parser.error(f"Input not found: {input_path}")

    output_path = Path(args.output_zarr).resolve()
    if not args.detect_only:
        assert_output_exists(output_path, parser, args)

    # ------------------------------------------------------------------
    # Step 1 – determine band / shift parameters
    # ------------------------------------------------------------------
    band_start, band_width, confidence = 0, 0, 0.0
    undo_shift = args.shift

    if args.mode == 'fix':
        if args.band_start is not None and args.band_width is not None:
            band_start = args.band_start
            band_width = args.band_width
            confidence = 1.0
            print(f"[manual] band_start={band_start}px, band_width={band_width}px")
        else:
            print(f"Auto-detecting galvo band "
                  f"(pyramid level {args.detection_level}) ...")
            band_start, band_width, confidence = _auto_detect(
                input_path, args.detection_level, verbose=args.verbose)

            print(f"\nDetection result (scaled to level-0 pixels):")
            print(f"  band_start   = {band_start} px")
            print(f"  band_width   = {band_width} px")
            print(f"  confidence   = {confidence:.3f}")

            if confidence < args.min_confidence:
                print(f"\nConfidence {confidence:.3f} is below threshold "
                      f"{args.min_confidence}.")
                if not args.detect_only:
                    print("No fix applied.  "
                          "Use --band_start / --band_width to override detection, "
                          "or lower --min_confidence.")
                    return
            else:
                print(f"  → band detected; fix will be applied.")

    elif args.mode == 'undo':
        if undo_shift is None:
            parser.error("--shift N is required for --mode undo  "
                         "(provide the shift value that was applied during pipeline creation).")
        confidence = 1.0
        print(f"[undo] will reverse roll shift={undo_shift}px per tile chunk")

    # ------------------------------------------------------------------
    # Step 2 – open level-0 array to report tile metadata
    # ------------------------------------------------------------------
    arr, res, _, _ = _open_level(input_path, level=0)
    chunk_x = arr.chunks[1]
    chunk_y = arr.chunks[2]
    n_cx = arr.shape[1] // chunk_x
    n_cy = arr.shape[2] // chunk_y

    print(f"\nMosaic info (level 0):")
    print(f"  shape        = {arr.shape}  (Z, X, Y)")
    print(f"  tile chunks  = ({chunk_x}, {chunk_y}) px in (X, Y)")
    print(f"  tile grid    = {n_cx} × {n_cy} tiles")
    if args.mode == 'fix':
        computed_roll = chunk_x - band_start - band_width
        print(f"  roll shift   = chunk_x - band_start - band_width "
              f"= {chunk_x} - {band_start} - {band_width} = {computed_roll} px")

    if args.detect_only:
        print("\n--detect_only: no output written.")
        return

    # ------------------------------------------------------------------
    # Step 3 – apply fix / undo and write output zarr
    # ------------------------------------------------------------------
    print(f"\nWriting corrected zarr → {output_path}")
    _apply_fix(
        zarr_root=input_path,
        output_path=output_path,
        band_start=band_start,
        band_width=band_width,
        mode=args.mode,
        undo_shift=undo_shift,
        verbose=args.verbose,
    )
    print(f"Corrected zarr written: {output_path}")

    # ------------------------------------------------------------------
    # Step 4 – optionally update slice_config.csv
    # ------------------------------------------------------------------
    if args.update_config:
        if args.slice_id is None:
            print("Warning: --update_config given without --slice_id; "
                  "skipping config update.")
        else:
            config_path = Path(args.update_config)
            if not config_path.exists():
                print(f"Warning: {config_path} not found; skipping update.")
            else:
                fix_applied = (args.mode == 'fix'
                               and confidence >= args.min_confidence)
                _update_slice_config(config_path, args.slice_id,
                                     confidence, fix_applied, args.mode)

    print("\nDone.")


if __name__ == "__main__":
    main()
