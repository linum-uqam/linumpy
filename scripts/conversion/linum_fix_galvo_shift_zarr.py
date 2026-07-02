#!/usr/bin/env python3
r"""
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
fix (``linumpy.geometry.galvo.fix_galvo_shift``), moving the dark galvo-return
band to the end of the tile's A-line range.  No interpolation is performed --
the galvo-return columns are valid data once rolled to the right edge, and the
downstream pipeline crops them away.  Pass ``--use_gpu`` to run the per-chunk
roll on a CuPy device through ``linumpy.gpu.corrections.fix_galvo_shift``.

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

Undo an incorrectly applied fix (shift value from the pipeline log)::

    linum_fix_galvo_shift_zarr.py mosaic_grid_3d_z50.ome.zarr fixed_z50.ome.zarr \\
        --mode undo --shift 60

Update slice_config.csv after fixing::

    linum_fix_galvo_shift_zarr.py mosaic_grid_3d_z47.ome.zarr fixed_z47.ome.zarr \\
        --update_config path/to/slice_config.csv --slice_id 47
"""

# Configure thread limits before numpy/scipy imports
import linumpy.config.threads  # noqa: F401

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from linumpy.cli.args import add_overwrite_arg, assert_output_exists
from linumpy.geometry.galvo import (
    aggregate_band_detections,
    decide_tile_shift,
    detect_galvo_band_in_tile,
    detect_galvo_shift,
    fix_galvo_shift,
)
from linumpy.io import slice_config as slice_config_io
from linumpy.io.zarr import OmeZarrWriter, read_omezarr


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input_zarr", help="Input mosaic grid OME-Zarr file (*.ome.zarr).")
    p.add_argument("output_zarr", help="Output corrected OME-Zarr file path.")

    mode_group = p.add_argument_group("Operation mode")
    mode_group.add_argument("--detect_only", action="store_true", help="Only detect and print band info; do not write output.")
    mode_group.add_argument(
        "--mode",
        choices=["fix", "undo"],
        default="fix",
        help="'fix': apply galvo fix (default).\n'undo': reverse a previously applied fix.",
    )

    detect_group = p.add_argument_group("Band detection overrides", "Override auto-detection with manual values.")
    detect_group.add_argument(
        "--n_extra",
        type=int,
        default=40,
        help="Number of galvo-return pixels (the ``n_extra`` field from acquisition metadata, "
        "typically 40). Enables the gradient-pair detector used by the pipeline; "
        "strongly recommended for reliable detection. "
        "Note: the assembled mosaic tile width is already cropped to ``n_alines`` (the trailing "
        "n_extra guard columns are stripped during pre-processing), but when the galvo fix was "
        "missed the dark band still sits inside the kept tile range and is still ~n_extra pixels "
        "wide -- so this value remains the correct one to pass.",
    )
    detect_group.add_argument(
        "--band_start",
        type=int,
        default=None,
        help="Start position of dark band within a tile (pixels). Fully overrides auto-detection.",
    )
    detect_group.add_argument(
        "--band_width", type=int, default=None, help="Width of dark band (pixels). Fully overrides auto-detection."
    )
    detect_group.add_argument(
        "--band_offset",
        type=int,
        default=0,
        help="Shift detected band_start by ±N pixels to fine-tune without re-running detection [%(default)s].",
    )
    detect_group.add_argument(
        "--shift",
        type=int,
        default=None,
        help="Explicit roll shift for --mode undo. Equals the shift that was applied during pipeline creation.",
    )
    detect_group.add_argument(
        "--min_confidence",
        type=float,
        default=0.5,
        help="Minimum detection confidence to proceed with fix in auto mode [%(default)s].",
    )
    detect_group.add_argument(
        "--min_tile_signal",
        type=float,
        default=5.0,
        help="Mean intensity threshold below which a tile is treated as background and "
        "left UNCHANGED (no roll applied) [%(default)s]. Detection on dark tiles is "
        "unreliable and produces visible displacement artefacts in the output.",
    )
    detect_group.add_argument(
        "--skip_tiles",
        default="",
        help="Semicolon-separated list of 'kx,ky' tile coordinates to leave UNCHANGED "
        "(no roll). Use to manually patch a small set of tiles where the "
        "auto-detected shift wraps tissue across the tile boundary. "
        "Example: '13,4;13,8;3,3'.",
    )

    config_group = p.add_argument_group("Slice config update")
    config_group.add_argument(
        "--update_config", metavar="SLICE_CONFIG_CSV", help="Path to slice_config.csv to update after fixing."
    )
    config_group.add_argument(
        "--slice_id", type=int, default=None, help="Slice ID to update in slice_config.csv (required with --update_config)."
    )

    preview_group = p.add_argument_group("Preview")
    preview_group.add_argument(
        "--preview",
        metavar="OUT_PNG",
        help="Save a before/after comparison PNG after fixing. Uses the same 3-panel XY/XZ/YZ layout as the pipeline preview.",
    )
    preview_group.add_argument("--cmap", default="magma", help="Colormap for the preview [%(default)s].")

    scan_group = p.add_argument_group(
        "Band-start scan",
        "Sweep band_start over a range to visually find the correct value. "
        "Generates a contact-sheet PNG -- no fix is applied. "
        "Requires --band_width.",
    )
    scan_group.add_argument("--scan", metavar="OUT_PNG", help="Output PNG for the band-start contact sheet.")
    scan_group.add_argument(
        "--scan_range",
        nargs=3,
        type=int,
        metavar=("START", "STOP", "STEP"),
        default=None,
        help="Range of band_start values to try, in pixels. E.g. --scan_range 50 250 10",
    )

    p.add_argument("-v", "--verbose", action="store_true", help="Print per-chunk detection results.")
    p.add_argument(
        "--use_gpu",
        action="store_true",
        help="Run the per-strip roll on a CuPy device via linumpy.gpu.corrections.fix_galvo_shift. "
        "Detection always runs on CPU; only useful when zarr I/O is not the bottleneck.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of threads pipelining read/roll/write across tile columns [%(default)s]. "
        "Each worker holds one tile-column strip in memory (~chunk_x * ny * nz bytes).",
    )
    add_overwrite_arg(p)
    return p


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------


def _generate_comparison_preview(
    before_path: Path,
    after_path: Path,
    out_png: Path,
    cmap: str = "magma",
    band_start: int | None = None,
    band_width: int | None = None,
    chunk_x: int | None = None,
) -> None:
    """Save a side-by-side before/after comparison PNG.

    Layout mirrors the pipeline's ``linum_screenshot_omezarr.py`` output:
    three panels (XY, XZ, YZ) repeated for before (top row) and after
    (bottom row).  A shared colour scale derived from the *after* volume
    is used so the dark band in the before image is clearly visible.

    Parameters
    ----------
    before_path, after_path : Path
        OME-Zarr directories to compare.
    out_png : Path
        Output PNG file path.
    cmap : str
        Matplotlib colourmap.
    band_start : int or None
        Start column of the galvo band in pixels (optional overlay).
    band_width : int or None
        Width of the galvo band in pixels (optional overlay).
    chunk_x : int or None
        Tile chunk width in pixels (optional overlay).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _read_panels(zarr_path: Path) -> tuple:
        arr, _ = read_omezarr(zarr_path)
        vol = np.asarray(arr, dtype=np.float32)
        # Pick the Z slice with the highest mean signal so tissue is always visible.
        z_means = vol.mean(axis=(1, 2))
        z = int(np.argmax(z_means))
        x = vol.shape[1] // 2
        y = vol.shape[2] // 2
        print(
            f"  XY panel: using Z={z} (peak mean={z_means[z]:.1f}, "
            f"mid={vol.shape[0] // 2} has mean={z_means[vol.shape[0] // 2]:.1f})"
        )
        xy = np.array(vol[z, :, :]).T  # leftmost: what the pipeline shows
        xz = np.array(vol[:, x, :])[::-1, ::-1]
        yz = np.array(vol[:, :, y])[::-1]
        return xy, xz, yz

    print("Reading before zarr for preview ...")
    before_panels = _read_panels(before_path)
    print("Reading after zarr for preview ...")
    after_panels = _read_panels(after_path)

    # Shared colour limits from the after volume (cleaner signal).
    all_after = np.concatenate([p.ravel() for p in after_panels])
    vmin = float(np.percentile(all_after, 0.1))
    vmax = float(np.percentile(all_after, 99.9))

    titles_top = ["BEFORE  -  XY", "BEFORE  -  XZ", "BEFORE  -  YZ"]
    titles_bot = ["AFTER   -  XY", "AFTER   -  XZ", "AFTER   -  YZ"]
    width_ratios = [p.shape[1] for p in before_panels]

    fig, axes = plt.subplots(2, 3, gridspec_kw={"width_ratios": width_ratios, "hspace": 0.05, "wspace": 0.02})
    fig.set_size_inches(24, 18)
    fig.set_dpi(200)
    fig.patch.set_facecolor("black")

    for col, (bpanel, apanel, ttop, tbot) in enumerate(zip(before_panels, after_panels, titles_top, titles_bot, strict=False)):
        for row, (panel, title) in enumerate([(bpanel, ttop), (apanel, tbot)]):
            ax = axes[row, col]
            ax.imshow(panel, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_title(title, color="white", fontsize=11, pad=3)
            ax.set_axis_off()

    # Annotate detected band position on the XY panels with vertical lines,
    # repeated at every tile chunk so the pattern is visible across the mosaic.
    if band_start is not None and band_width is not None and chunk_x is not None:
        xy_w = before_panels[0].shape[1]  # total mosaic X width in zarr pixels
        n_tiles = xy_w // chunk_x

        for k in range(n_tiles):
            # BEFORE row: original band position
            x0_before = band_start + k * chunk_x
            x1_before = x0_before + band_width
            axes[0, 0].axvline(x0_before, color="cyan", linewidth=0.6, linestyle="--", alpha=0.8)
            axes[0, 0].axvline(x1_before, color="deepskyblue", linewidth=0.6, linestyle=":", alpha=0.8)
            # AFTER row: residual band now at right edge of each tile
            x0_after = (k + 1) * chunk_x - band_width
            x1_after = (k + 1) * chunk_x
            axes[1, 0].axvline(x0_after, color="cyan", linewidth=0.6, linestyle="--", alpha=0.8)
            axes[1, 0].axvline(x1_after, color="deepskyblue", linewidth=0.6, linestyle=":", alpha=0.8)

        # Scale bar annotation (bottom-left of BEFORE XY panel).
        fig_w_px = 24 * 200  # fig_width_in * dpi
        total_ratio = sum(width_ratios)
        xy_subplot_px = fig_w_px * width_ratios[0] / total_ratio
        zarr_px_per_preview_px = xy_w / xy_subplot_px
        note = (
            f"band [{band_start}:{band_start + band_width}] per tile  "
            f"| scale ≈ {zarr_px_per_preview_px:.1f} zarr px / preview px  "
            f"| 1 visible px ≈ {zarr_px_per_preview_px:.0f} zarr px"
        )
        axes[0, 0].text(
            0.01,
            0.01,
            note,
            transform=axes[0, 0].transAxes,
            color="cyan",
            fontsize=7,
            va="bottom",
            bbox={"facecolor": "black", "alpha": 0.5, "pad": 2},
        )
        print(f"\nPreview scale: {zarr_px_per_preview_px:.1f} zarr px per preview px in the XY panel.")
        print(
            f"  → If the band line appears N px off, use "
            f"--band_offset ±{zarr_px_per_preview_px:.0f}*N  "
            f"(e.g. 3 px off → --band_offset ±{3 * zarr_px_per_preview_px:.0f})"
        )

    fig.savefig(str(out_png), bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print(f"Preview saved → {out_png}")


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------


def _auto_detect(zarr_root: Path, n_extra: int | None = None, verbose: bool = False) -> tuple:
    """Sample representative chunks and return ``(band_start, band_width, confidence)``.

    All values are in pixels.  Mosaic-grid OME-Zarrs are written without a
    pyramid, so detection always runs at full resolution.

    When *n_extra* is provided the same gradient-pair detector used by the
    pipeline (``detect_galvo_shift``) is applied to each chunk AIP -- this is
    much more robust than the threshold-based fallback.  Without *n_extra* the
    simpler ``detect_galvo_band_in_tile`` is used.

    Parameters
    ----------
    zarr_root : Path
        Path to the OME-Zarr root directory.
    n_extra : int or None
        Number of galvo-return pixels from acquisition metadata (the ``n_extra``
        field in info.txt / Nextflow config).  Strongly recommended.
    verbose : bool
        If True, print per-chunk detection details.
    """
    arr, _ = read_omezarr(zarr_root)

    chunk_x = arr.chunks[1]
    chunk_y = arr.chunks[2]
    n_cx = arr.shape[1] // chunk_x
    n_cy = arr.shape[2] // chunk_y

    # Sample a spread of chunks from the central region (more likely tissue).
    cx_lo = max(0, n_cx // 4)
    cx_hi = max(cx_lo, min(n_cx - 1, 3 * n_cx // 4))
    cy_mid = n_cy // 2

    n_samples = min(8, cx_hi - cx_lo + 1)
    cx_indices = list(dict.fromkeys(np.linspace(cx_lo, cx_hi, n_samples, dtype=int).tolist()))

    detections = []
    for cx in cx_indices:
        xs = cx * chunk_x
        xe = xs + chunk_x
        ys = cy_mid * chunk_y
        ye = ys + chunk_y

        chunk = np.asarray(arr[:, xs:xe, ys:ye], dtype=np.float32)
        if float(chunk.mean()) < 5.0:
            if verbose:
                print(f"  Chunk ({cx}, {cy_mid}): skipped (low signal mean={chunk.mean():.1f})")
            continue

        tile_aip = chunk.mean(axis=0)  # (chunk_x, chunk_y)

        if n_extra:
            # Use the proven gradient-pair detector from the pipeline.
            # detect_galvo_shift returns (shift, confidence); the band sits at
            # band_start = chunk_x - shift - n_extra after the implied roll.
            shift, conf = detect_galvo_shift(tile_aip, n_pixel_return=n_extra)
            bs = chunk_x - shift - n_extra
            bw = n_extra
        else:
            # Fallback: threshold-based detector (less reliable)
            bs, bw, conf = detect_galvo_band_in_tile(tile_aip)

        if verbose:
            print(
                f"  Chunk ({cx:3d}, {cy_mid}): "
                f"band_start={bs:4d}px  band_width={bw:3d}px  "
                f"confidence={conf:.3f}" + ("  [gradient-pair]" if n_extra else "  [threshold fallback]")
            )

        detections.append((bs, bw, conf))

    return aggregate_band_detections(detections, chunk_x, verbose=verbose)


# ---------------------------------------------------------------------------
# Band-start scan (contact sheet)
# ---------------------------------------------------------------------------


def _scan_band_start(
    zarr_root: Path,
    band_width: int,
    scan_start: int,
    scan_stop: int,
    scan_step: int,
    out_png: Path,
    cmap: str = "magma",
) -> None:
    """Sweep *band_start* over a range and save a contact-sheet PNG.

    A representative tile (average of several mid-mosaic tiles) is rolled
    for each candidate value so you can visually identify the correct
    ``band_start`` without running the full fix.

    Parameters
    ----------
    zarr_root : Path
        Path to the OME-Zarr root directory to scan.
    band_width : int
        Width of the dark band in pixels (typically ``n_extra``).
    scan_start, scan_stop, scan_step : int
        Range in pixels (Python-style: *scan_stop* is exclusive).
    out_png : Path
        Output contact-sheet PNG.
    cmap : str
        Matplotlib colourmap name.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arr, _ = read_omezarr(zarr_root)
    chunk_x = arr.chunks[1]
    chunk_y = arr.chunks[2]
    n_cx = arr.shape[1] // chunk_x
    n_cy = arr.shape[2] // chunk_y

    bw = max(1, band_width)
    step = max(1, scan_step)

    # Sample a spread of central tiles.
    cx_lo = max(0, n_cx // 4)
    cx_hi = min(n_cx - 1, 3 * n_cx // 4)
    cy_mid = n_cy // 2
    n_samples = min(5, cx_hi - cx_lo + 1)
    cx_indices = list(dict.fromkeys(np.linspace(cx_lo, cx_hi, n_samples, dtype=int).tolist()))

    tiles = []
    for cx in cx_indices:
        chunk = np.asarray(
            arr[:, cx * chunk_x : (cx + 1) * chunk_x, cy_mid * chunk_y : (cy_mid + 1) * chunk_y], dtype=np.float32
        )
        if float(chunk.mean()) > 5.0:
            tiles.append(chunk.mean(axis=0))  # (chunk_x, chunk_y) AIP

    if not tiles:
        print("  No tiles with sufficient signal found -- cannot generate scan.")
        return

    avg_tile = np.mean(np.stack(tiles, axis=0), axis=0)  # representative XY view

    vmin = float(np.percentile(avg_tile, 0.5))
    vmax = float(np.percentile(avg_tile, 99.5))

    candidates = list(range(scan_start, scan_stop, step))
    n_cand = len(candidates)
    n_cols = min(8, n_cand + 1)
    n_rows = (n_cand + 1 + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 4))
    fig.patch.set_facecolor("black")
    axes_flat = np.array(axes).flatten()

    # First panel: original (no roll applied).
    axes_flat[0].imshow(avg_tile.T, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", origin="lower")
    axes_flat[0].set_title("ORIGINAL", color="white", fontsize=8)
    axes_flat[0].set_axis_off()

    for i, bs in enumerate(candidates):
        roll = chunk_x - bs - bw
        fixed = np.roll(avg_tile, roll, axis=0)
        ax = axes_flat[i + 1]
        ax.imshow(fixed.T, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", origin="lower")
        ax.set_title(f"bs={bs}  r={roll}", color="white", fontsize=7)
        ax.set_axis_off()

    for j in range(n_cand + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        f"band_start scan  |  band_width={band_width}px",
        color="white",
        fontsize=10,
    )
    plt.tight_layout()
    fig.savefig(str(out_png), bbox_inches="tight", facecolor="black", dpi=150)
    plt.close(fig)

    print(f"Scan contact sheet saved → {out_png}")
    print(f"  {n_cand} candidates in range [{scan_start}:{scan_stop}:{scan_step}]px")
    print("  Title format: bs=<band_start>  r=<roll_amount>  (px)")


# ---------------------------------------------------------------------------
# Fix / undo
# ---------------------------------------------------------------------------


def _parse_skip_tiles(spec: str) -> frozenset[tuple[int, int]]:
    """Parse a 'kx,ky;kx,ky' string into a set of tile coords."""
    if not spec:
        return frozenset()
    out: set[tuple[int, int]] = set()
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        kx_str, ky_str = chunk.split(",")
        out.add((int(kx_str), int(ky_str)))
    return frozenset(out)


def _apply_fix(
    zarr_root: Path,
    output_path: Path,
    band_start: int,
    band_width: int,
    mode: str,
    undo_shift: int,
    overwrite: bool = True,
    use_gpu: bool = False,
    workers: int = 4,
    n_extra: int | None = None,
    min_confidence: float = 0.5,
    min_tile_signal: float = 5.0,
    skip_tiles: frozenset[tuple[int, int]] = frozenset(),
    _verbose: bool = False,
) -> None:
    """Write a corrected OME-Zarr, processing each chunk individually.

    **fix mode**: The galvo desynchronisation means A-lines are out of order within
    each tile chunk.  A single circular roll by ``chunk_x - band_start - band_width``
    positions reorders them correctly, moving the dark galvo-return band to the right
    edge of the tile and placing the two valid sweep segments in the correct order.

    **undo mode**: reverses a galvo fix that was incorrectly applied during
    mosaic creation by rolling each chunk back by ``-undo_shift``.

    Parameters
    ----------
    zarr_root : Path
        Path to the input OME-Zarr root directory.
    output_path : Path
        Path for the corrected output OME-Zarr.
    band_start : int
        Start column of the dark band within a tile chunk (fix mode).
    band_width : int
        Width of the dark band in pixels (fix mode).
    mode : str
        ``'fix'`` or ``'undo'``.
    undo_shift : int
        The roll shift that was applied by the pipeline (undo mode).
    overwrite : bool
        Overwrite *output_path* if it already exists.
    use_gpu : bool
        Run the per-strip roll on a CuPy device via ``linumpy.gpu.corrections``.
    workers : int
        Number of threads pipelining read → roll → write across tile columns.
        Each worker holds one tile-column strip in memory.
    n_extra : int or None
        If set, per-tile detection uses the gradient-pair detector
        (``detect_galvo_shift``) with this guard width. Same value as the
        global detection step.
    min_confidence : float
        Confidence threshold for accepting a per-tile shift. Tiles below this
        threshold fall back to the global ``band_start`` / ``band_width``.
    min_tile_signal : float
        Tiles whose mean intensity is below this value are treated as
        background and left unchanged (no roll). Detection on dark tiles is
        unreliable and produces visible displacement artefacts.
    skip_tiles : frozenset of (kx, ky)
        Tile coordinates manually flagged to be left unchanged (no roll),
        in addition to background tiles. Use to patch a small set of tiles
        where auto-detection wraps tissue across the tile boundary.
    """
    arr, res = read_omezarr(zarr_root)
    shape = arr.shape  # (nz, nx_mosaic, ny_mosaic)
    chunk_x = arr.chunks[1]  # OCT tile width in X (A-line axis)
    chunk_y = arr.chunks[2]  # OCT tile height in Y (B-scan axis)
    dtype = arr.dtype

    n_cx = shape[1] // chunk_x
    n_cy = shape[2] // chunk_y

    if mode == "fix":
        if not 0 <= band_start < chunk_x or band_width <= 0 or band_start + band_width > chunk_x:
            raise ValueError(
                f"Band [{band_start}:{band_start + band_width}] does not fit inside a tile of width {chunk_x}px. "
                "Check --band_start / --band_width or detection inputs."
            )
        band_end = band_start + band_width
        default_shift = chunk_x - band_start - band_width
        print(
            f"Per-tile galvo fix: fallback shift +{default_shift} px "
            f"(global band [{band_start}:{band_end}]) for tiles below "
            f"min_confidence={min_confidence:.2f} in {n_cx}x{n_cy} grid."
        )
    else:
        default_shift = -int(undo_shift)
        print(f"Rolling each tile chunk by {default_shift:+d} px to reverse applied galvo fix")

    # CPU roll helper (used for per-tile sub-blocks). The GPU roll helper is
    # only meaningful in undo mode where the whole strip shares one shift; for
    # fix mode, per-tile detection means many small rolls and the CPU path is
    # the right choice.
    def _roll_cpu(block: np.ndarray, shift: int) -> np.ndarray:
        return fix_galvo_shift(block, shift=shift, axis=1)

    if use_gpu and mode != "fix":
        from linumpy.gpu.corrections import fix_galvo_shift as _fix_galvo_shift_gpu

        def _roll_strip(strip: np.ndarray) -> np.ndarray:
            return _fix_galvo_shift_gpu(strip, default_shift, axis=1, use_gpu=True)
    else:

        def _roll_strip(strip: np.ndarray) -> np.ndarray:
            return fix_galvo_shift(strip, shift=default_shift, axis=1)

    writer = OmeZarrWriter(
        output_path,
        shape=shape,
        chunk_shape=(shape[0], chunk_x, chunk_y),
        dtype=dtype,
        overwrite=overwrite,
    )

    nz = shape[0]

    # Per-tile-column accounting (thread-safe: each kx writes its own slot).
    n_per_tile = np.zeros(n_cx, dtype=np.int32)
    n_fallback = np.zeros(n_cx, dtype=np.int32)
    n_skipped = np.zeros(n_cx, dtype=np.int32)
    shifts_used: list[list[int]] = [[] for _ in range(n_cx)]

    def _process_column(kx: int) -> None:
        xs = kx * chunk_x
        xe = xs + chunk_x
        strip = arr[:, xs:xe, :]

        if mode != "fix":
            writer[0:nz, xs:xe, :] = _roll_strip(strip)
            return

        # Per-tile detect+roll: AIP once, then n_cy small rolls.
        aip_strip = strip.mean(axis=0)  # (chunk_x, ny_total), float
        out = np.empty_like(strip)
        for ky in range(n_cy):
            ys = ky * chunk_y
            ye = ys + chunk_y
            tile_aip = aip_strip[:, ys:ye]
            # Manual override: skip tiles the user has flagged as
            # producing wrap artefacts.
            if (kx, ky) in skip_tiles:
                out[:, :, ys:ye] = strip[:, :, ys:ye]
                n_skipped[kx] += 1
                continue
            # Background tiles: leave content untouched. Detection on noise
            # produces spurious shifts that visibly displace tile content.
            if float(tile_aip.mean()) < min_tile_signal:
                out[:, :, ys:ye] = strip[:, :, ys:ye]
                n_skipped[kx] += 1
                continue
            sh, _cf, used = decide_tile_shift(tile_aip, default_shift, min_confidence, n_extra)
            out[:, :, ys:ye] = _roll_cpu(strip[:, :, ys:ye], sh)
            shifts_used[kx].append(sh)
            if used:
                n_per_tile[kx] += 1
            else:
                n_fallback[kx] += 1
        writer[0:nz, xs:xe, :] = out

    n_workers = max(1, int(workers))
    if n_workers == 1:
        for kx in tqdm(range(n_cx), desc="Tile columns"):
            _process_column(kx)
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            list(
                tqdm(
                    pool.map(_process_column, range(n_cx)),
                    total=n_cx,
                    desc=f"Tile columns ({n_workers} workers)",
                )
            )

    if mode == "fix":
        total = int(n_per_tile.sum() + n_fallback.sum() + n_skipped.sum())
        all_shifts = [s for col in shifts_used for s in col]
        if all_shifts:
            uniq, counts = np.unique(np.asarray(all_shifts), return_counts=True)
            order = np.argsort(-counts)
            top = ", ".join(f"{int(uniq[i]):+d}px x{int(counts[i])}" for i in order[:5])
            print(
                f"Per-tile detection: {int(n_per_tile.sum())}/{total} per-tile, "
                f"{int(n_fallback.sum())} fallback (+{default_shift}px), "
                f"{int(n_skipped.sum())} skipped (background, mean<{min_tile_signal:.1f}). "
                f"Shift histogram (top {min(5, len(uniq))}): {top}"
            )

    # Mosaic grids are written without a pyramid (single level), so we
    # finalize with no extra levels too.
    writer.finalize(res, n_levels=0)


# ---------------------------------------------------------------------------
# Slice-config update
# ---------------------------------------------------------------------------


def _update_slice_config(config_path: Path, slice_id: int, confidence: float, fix_applied: bool, mode: str) -> None:
    """Stamp ``galvo_confidence`` / ``galvo_fix`` / ``notes`` for one slice."""
    rows = slice_config_io.read(config_path)
    sid = slice_config_io.normalize_slice_id(slice_id)
    if sid not in rows:
        print(f"  Warning: slice_id {sid} not found in {config_path}")
        return

    row = rows[sid]
    row["galvo_confidence"] = f"{confidence:.3f}"
    row["galvo_fix"] = "true" if fix_applied else "false"
    tag = f"zarr_retrofix_{mode}"
    existing_notes = row.get("notes", "")
    row["notes"] = f"{existing_notes}; {tag}".strip("; ") if existing_notes else tag

    slice_config_io.write(config_path, rows.values())

    print(
        f"Updated {config_path}  →  slice {sid}: galvo_fix={'true' if fix_applied else 'false'}, confidence={confidence:.3f}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run function."""
    parser = _build_arg_parser()
    args = parser.parse_args()

    input_path = Path(args.input_zarr).resolve()
    if not input_path.exists():
        parser.error(f"Input not found: {input_path}")

    output_path = Path(args.output_zarr).resolve()
    if not args.detect_only:
        assert_output_exists(output_path, parser, args)

    # ------------------------------------------------------------------
    # Step 0 - band-start scan (optional, exits early without writing fix)
    # ------------------------------------------------------------------
    if args.scan:
        if args.scan_range is None:
            parser.error("--scan requires --scan_range START STOP STEP.")
        if args.band_width is None:
            parser.error("--scan requires --band_width.")
        print(
            f"Band-start scan: range [{args.scan_range[0]}:{args.scan_range[1]}:{args.scan_range[2]}]px "
            f"band_width={args.band_width}px ..."
        )
        _scan_band_start(
            input_path,
            band_width=args.band_width,
            scan_start=args.scan_range[0],
            scan_stop=args.scan_range[1],
            scan_step=args.scan_range[2],
            out_png=Path(args.scan),
            cmap=args.cmap,
        )
        return

    # ------------------------------------------------------------------
    # Step 1 - determine band / shift parameters
    # ------------------------------------------------------------------
    band_start, band_width, confidence = 0, 0, 0.0
    undo_shift = args.shift

    if args.mode == "fix":
        if args.band_start is not None and args.band_width is not None:
            band_start = args.band_start + args.band_offset
            band_width = args.band_width
            confidence = 1.0
            print(f"[manual] band_start={band_start}px (offset applied: {args.band_offset:+d}px), band_width={band_width}px")
        else:
            detector = "gradient-pair" if args.n_extra else "threshold fallback"
            print(
                f"Auto-detecting galvo band using {detector} detector"
                + (f", n_extra={args.n_extra}px" if args.n_extra else "")
                + " ..."
            )
            band_start, band_width, confidence = _auto_detect(input_path, n_extra=args.n_extra, verbose=args.verbose)

            band_start += args.band_offset

            print("\nDetection result (pixels):")
            print(f"  band_start   = {band_start} px" + (f"  (offset: {args.band_offset:+d}px)" if args.band_offset else ""))
            print(f"  band_width   = {band_width} px")
            print(f"  confidence   = {confidence:.3f}")

            if confidence < args.min_confidence:
                print(f"\nConfidence {confidence:.3f} is below threshold {args.min_confidence}.")
                if not args.detect_only:
                    print(
                        "No fix applied.\n"
                        "  → Provide --n_extra (galvo return pixels from acquisition "
                        "metadata) for more reliable detection, or\n"
                        "  → Use --band_start / --band_width to set position manually, or\n"
                        "  → Lower --min_confidence."
                    )
                    return
            else:
                print("  → band detected; fix will be applied.")

    elif args.mode == "undo":
        if undo_shift is None:
            parser.error(
                "--shift N is required for --mode undo  (provide the shift value that was applied during pipeline creation)."
            )
        confidence = 1.0
        print(f"[undo] will reverse roll shift={undo_shift}px per tile chunk")

    # ------------------------------------------------------------------
    # Step 2 - open array to report tile metadata
    # ------------------------------------------------------------------
    arr, _res = read_omezarr(input_path)
    chunk_x = arr.chunks[1]
    chunk_y = arr.chunks[2]
    n_cx = arr.shape[1] // chunk_x
    n_cy = arr.shape[2] // chunk_y

    print("\nMosaic info:")
    print(f"  shape        = {arr.shape}  (Z, Y, X)")
    print(f"  tile chunks  = ({chunk_x}, {chunk_y}) px in (X, Y)")
    print(f"  tile grid    = {n_cx} x {n_cy} tiles")
    if args.mode == "fix":
        print(f"  band columns = [{band_start}:{band_start + band_width}] px (within each tile chunk of width {chunk_x})")

    if args.detect_only:
        print("\n--detect_only: no output written.")
        return

    # ------------------------------------------------------------------
    # Step 3 - apply fix / undo and write output zarr
    # ------------------------------------------------------------------
    print(f"\nWriting corrected zarr → {output_path}")
    _apply_fix(
        zarr_root=input_path,
        output_path=output_path,
        band_start=band_start,
        band_width=band_width,
        mode=args.mode,
        undo_shift=undo_shift,
        overwrite=args.overwrite,
        use_gpu=args.use_gpu,
        workers=args.workers,
        n_extra=args.n_extra,
        min_confidence=args.min_confidence,
        min_tile_signal=args.min_tile_signal,
        skip_tiles=_parse_skip_tiles(args.skip_tiles),
        _verbose=args.verbose,
    )
    print(f"Corrected zarr written: {output_path}")

    # ------------------------------------------------------------------
    # Step 4 - optionally generate before/after comparison preview
    # ------------------------------------------------------------------
    if args.preview:
        preview_path = Path(args.preview)
        _generate_comparison_preview(
            input_path,
            output_path,
            preview_path,
            cmap=args.cmap,
            band_start=band_start if args.mode == "fix" else None,
            band_width=band_width if args.mode == "fix" else None,
            chunk_x=chunk_x,
        )

    # ------------------------------------------------------------------
    # Step 5 - optionally update slice_config.csv
    # ------------------------------------------------------------------
    if args.update_config:
        if args.slice_id is None:
            print("Warning: --update_config given without --slice_id; skipping config update.")
        else:
            config_path = Path(args.update_config)
            if not config_path.exists():
                print(f"Warning: {config_path} not found; skipping update.")
            else:
                fix_applied = args.mode == "fix" and confidence >= args.min_confidence
                _update_slice_config(config_path, args.slice_id, confidence, fix_applied, args.mode)

    print("\nDone.")


if __name__ == "__main__":
    main()
