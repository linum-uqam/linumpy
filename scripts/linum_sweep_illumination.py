#!/usr/bin/env python3
r"""Parameter sweep tool for illumination-correction tuning.

Runs the BaSiC-based illumination correction for every combination of the
supplied parameters and writes a labelled PNG for each configuration, so the
effect of each choice is immediately visible without re-running the full
pipeline.

Sweep mode (default)
--------------------
Corrects only the selected Z-slice (``--z_slice``) for each config.  The full
Z stack is still used to *fit* the model (pooled-tile approach), but only one
plane is written, making each config fast.

AIP mode (``--aip``)
--------------------
In addition to the single-slice preview, the correction is applied to the
full volume and a grid of Average Intensity Projections (AIPs) is saved.
AIPs are computed in slabs of ``--aip_slab_size`` Z-planes, giving one image
per slab per config.  For a volume with Z=55 and slab_size=5 this yields 11
AIP columns showing how the correction quality changes with depth.

Output files
------------
For each config the following PNGs are written to ``output_dir``:

- ``c{N}_p{pmax}_{df}_s{samples}_i{iters}_z{z}.png``
  Four panels: RAW slice | CORRECTED slice | flatfield | darkfield (if used).
- ``c{N}_..._aips.png`` (only with ``--aip``)
  Two-row grid: top row = RAW slab AIPs, bottom row = CORRECTED slab AIPs.

A ``sweep_summary.csv`` with fit diagnostics is written to ``output_dir``.

Example
-------
::

    linum_sweep_illumination.py mosaic_grid_z25_focal_fix.ome.zarr sweep/ \\
        --percentile_max none,99.0,99.5,99.9 \\
        --use_darkfield true,false \\
        --darkfield_percentile 2,5,10 \\
        --fit_max_samples 2000,8000 \\
        --max_iterations 500 \\
        --aip
"""

import linumpy.config.threads  # noqa: F401

import argparse
import csv
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from basicpy import BaSiC
from scipy.ndimage import gaussian_filter
from tqdm.auto import tqdm

from linumpy.io.zarr import read_omezarr

# ---------------------------------------------------------------------------
# Tile helpers (mirror of linum_fix_illumination_3d, kept local to avoid
# importing a script as a module)
# ---------------------------------------------------------------------------


def _split_into_tiles(plane: np.ndarray, tile_shape: tuple[int, int]) -> np.ndarray:
    ty, tx = tile_shape
    ny, nx = plane.shape[0] // ty, plane.shape[1] // tx
    tiles = np.empty((ny * nx, ty, tx), dtype=plane.dtype)
    for i in range(ny):
        for j in range(nx):
            tiles[i * nx + j] = plane[i * ty : (i + 1) * ty, j * tx : (j + 1) * tx]
    return tiles


def _assemble_from_tiles(tiles: np.ndarray, plane_shape: tuple[int, int], tile_shape: tuple[int, int]) -> np.ndarray:
    ty, tx = tile_shape
    ny, nx = plane_shape[0] // ty, plane_shape[1] // tx
    out = np.zeros(plane_shape, dtype=tiles.dtype)
    for i in range(ny):
        for j in range(nx):
            out[i * ty : (i + 1) * ty, j * tx : (j + 1) * tx] = tiles[i * nx + j]
    return out


# ---------------------------------------------------------------------------
# Core correction for one parameter configuration
# ---------------------------------------------------------------------------


def run_one_config(
    vol: np.ndarray,
    tile_shape: tuple[int, int],
    *,
    percentile_max: float | None,
    use_darkfield: bool,
    darkfield_percentile: float,
    fit_max_samples: int,
    max_iterations: int,
    smoothness_flatfield: float | None,
    working_size: int | None,
    apply_z: list[int],
    per_z_fit: bool = False,
    darkfield_smooth_sigma: float = 0.0,
    darkfield_z_window: int = 0,
    flatfield_smooth_sigma: float = 0.0,
    n_workers: int = 1,
) -> tuple[dict[int, np.ndarray], np.ndarray, np.ndarray | None, dict]:
    """Fit BaSiC and apply the model to ``apply_z``.

    When ``per_z_fit=False`` (default) fits on the pooled Z-tile stack.
    When ``per_z_fit=True`` fits a separate model per plane in ``apply_z``.

    Returns
    -------
    corrected : dict mapping z-index -> corrected (float32, clipped >= 0)
    flatfield : (ty, tx) float32 array (representative, from first successful fit)
    darkfield : (ty, tx) float32 array or None
    stats     : dict with fit diagnostics
    """
    n_axial = vol.shape[0]
    plane_shape: tuple[int, int] = (vol.shape[1], vol.shape[2])
    tiles_per_plane = (plane_shape[0] // tile_shape[0]) * (plane_shape[1] // tile_shape[1])

    def _fit_basic_on(pool_arr: np.ndarray) -> np.ndarray | None:
        basic_kwargs: dict = {"get_darkfield": False, "max_iterations": max_iterations}
        if smoothness_flatfield is not None:
            basic_kwargs["smoothness_flatfield"] = smoothness_flatfield
        if working_size is not None:
            basic_kwargs["working_size"] = working_size
        # (flatfield_smooth_sigma applied after fit, see below)
        opt = BaSiC(**basic_kwargs)
        opt.fit(pool_arr)
        ff = np.asarray(opt.flatfield, dtype=np.float32)
        if np.isnan(ff).any() or ff.max() <= 0:
            return None
        if flatfield_smooth_sigma > 0:
            ff = gaussian_filter(ff, sigma=flatfield_smooth_sigma).astype(np.float32)
        return ff

    def _apply_ff(tiles: np.ndarray, ff: np.ndarray, df: np.ndarray | None) -> np.ndarray:
        x = tiles.astype(np.float32, copy=False)
        if df is not None:
            x = x - df[None]
        return x / ff[None]

    def _prep(tiles_arr: np.ndarray, df: np.ndarray | None) -> np.ndarray:
        if percentile_max is not None:
            tiles_arr = np.clip(tiles_arr, None, float(np.percentile(tiles_arr, percentile_max)))
        if df is not None:
            tiles_arr = np.clip(tiles_arr - df[None], 0.0, None)
        return tiles_arr

    corrected: dict[int, np.ndarray] = {}

    if not per_z_fit:
        # ── Global fit ───────────────────────────────────────────────────────
        fit_max_eff = max(fit_max_samples, tiles_per_plane)
        n_planes = min(n_axial, max(1, fit_max_eff // tiles_per_plane))
        fit_z = np.arange(n_axial) if n_planes >= n_axial else np.linspace(0, n_axial - 1, n_planes, dtype=int)

        pool_parts = [_split_into_tiles(vol[int(z)], tile_shape) for z in fit_z]
        pool = np.concatenate(pool_parts, axis=0)
        del pool_parts

        keep = np.mean(pool != 0, axis=(1, 2)) > 0.5
        pool = pool[keep]
        n_fit = pool.shape[0]
        if n_fit == 0:
            msg = "No non-empty tiles in fit pool."
            raise RuntimeError(msg)

        darkfield: np.ndarray | None = (
            np.percentile(pool, darkfield_percentile, axis=0).astype(np.float32) if use_darkfield else None
        )
        if darkfield is not None and darkfield_smooth_sigma > 0:
            darkfield = gaussian_filter(darkfield, sigma=darkfield_smooth_sigma).astype(np.float32)
        pool = _prep(pool, darkfield)

        flatfield = _fit_basic_on(pool)
        del pool
        if flatfield is None:
            msg = f"BaSiC flatfield diverged: fit on {n_fit} tiles."
            raise RuntimeError(msg)

        for z in tqdm(apply_z, desc="  Applying", leave=False):
            plane = vol[z]
            tiles = _split_into_tiles(plane, tile_shape)
            empty_mask = np.all(tiles == 0, axis=(1, 2))
            c = _apply_ff(tiles, flatfield, darkfield)
            if empty_mask.any():
                c[empty_mask] = 0.0
            c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
            corrected[z] = np.clip(_assemble_from_tiles(c, plane_shape, tile_shape), 0.0, None)

    else:
        # ── Per-Z fit ────────────────────────────────────────────────────────
        min_tiles = max(4, tiles_per_plane // 4)
        flatfield = None
        darkfield = None
        n_fit = 0

        def _process_plane(z: int) -> tuple[int, np.ndarray, np.ndarray | None, np.ndarray | None, int]:
            plane = vol[z]
            tiles = _split_into_tiles(plane, tile_shape)
            keep = np.mean(tiles != 0, axis=(1, 2)) > 0.5
            tiles_fit = tiles[keep].astype(np.float32)
            fallback = np.clip(_assemble_from_tiles(tiles.astype(np.float32), plane_shape, tile_shape), 0.0, None)

            if tiles_fit.shape[0] < min_tiles:
                return z, fallback, None, None, 0

            if use_darkfield:
                if darkfield_z_window != 0:
                    # -1 = all planes; >0 = z±window neighbours
                    z_range = (
                        range(n_axial)
                        if darkfield_z_window == -1
                        else range(max(0, z - darkfield_z_window), min(n_axial, z + darkfield_z_window + 1))
                    )
                    df_parts = []
                    for zz in z_range:
                        t = _split_into_tiles(vol[zz], tile_shape)
                        ok = np.mean(t != 0, axis=(1, 2)) > 0.5
                        df_parts.append(t[ok].astype(np.float32))
                    df_pool = np.concatenate(df_parts, axis=0)
                    df_z: np.ndarray | None = np.percentile(df_pool, darkfield_percentile, axis=0).astype(np.float32)
                    del df_pool
                else:
                    df_z = np.percentile(tiles_fit, darkfield_percentile, axis=0).astype(np.float32)
                if darkfield_smooth_sigma > 0:
                    df_z = gaussian_filter(df_z, sigma=darkfield_smooth_sigma).astype(np.float32)
            else:
                df_z = None
            ff_z = _fit_basic_on(_prep(tiles_fit, df_z))
            if ff_z is None:
                return z, fallback, None, None, 0

            empty_mask = np.all(tiles == 0, axis=(1, 2))
            c = _apply_ff(tiles.astype(np.float32), ff_z, df_z)
            if empty_mask.any():
                c[empty_mask] = 0.0
            c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
            return z, np.clip(_assemble_from_tiles(c, plane_shape, tile_shape), 0.0, None), ff_z, df_z, tiles_fit.shape[0]

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            plane_futures = [executor.submit(_process_plane, z) for z in apply_z]
            for fut in tqdm(as_completed(plane_futures), total=len(apply_z), desc="  Per-Z fit+apply", leave=False):
                z_idx, plane_result, ff_z, df_z, n = fut.result()
                corrected[z_idx] = plane_result
                if ff_z is not None and flatfield is None:
                    flatfield = ff_z
                    darkfield = df_z
                    n_fit = n

        if flatfield is None:
            msg = "Per-Z fit: no plane had enough tiles for a successful BaSiC fit."
            raise RuntimeError(msg)

    df = darkfield  # alias for stats
    stats: dict = {
        "n_fit_tiles": n_fit,
        "ff_min": float(flatfield.min()),
        "ff_max": float(flatfield.max()),
        "ff_mean": float(flatfield.mean()),
        "df_min": float(df.min()) if df is not None else None,
        "df_max": float(df.max()) if df is not None else None,
        "df_mean": float(df.mean()) if df is not None else None,
        "smoothness_flatfield": smoothness_flatfield,
        "working_size": working_size,
    }
    return corrected, flatfield, darkfield, stats


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------


def _display_vmax(arr: np.ndarray, p: float = 99.9) -> float:
    pos = arr[arr > 0]
    return float(np.percentile(pos, p)) if pos.size > 0 else 1.0


def _save_slice_comparison(
    raw_plane: np.ndarray,
    corr_plane: np.ndarray,
    flatfield: np.ndarray,
    darkfield: np.ndarray | None,
    out_path: str,
    title: str,
) -> None:
    """Save RAW | CORRECTED | flatfield [| darkfield] in one figure."""
    ncols = 4 if darkfield is not None else 3
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 6), facecolor="black")
    fig.suptitle(title, color="white", fontsize=8)

    for ax in axes:
        ax.set_axis_off()
        ax.set_facecolor("black")

    axes[0].imshow(raw_plane, cmap="magma", vmin=0, vmax=_display_vmax(raw_plane))
    axes[0].set_title("RAW", color="white", fontsize=9)

    axes[1].imshow(corr_plane, cmap="magma", vmin=0, vmax=_display_vmax(corr_plane))
    axes[1].set_title("CORRECTED", color="white", fontsize=9)

    axes[2].imshow(flatfield, cmap="viridis")
    axes[2].set_title(f"Flatfield [{flatfield.min():.3f}, {flatfield.max():.3f}]", color="white", fontsize=9)

    if darkfield is not None:
        axes[3].imshow(darkfield, cmap="inferno")
        axes[3].set_title(f"Darkfield [{darkfield.min():.2f}, {darkfield.max():.2f}]", color="white", fontsize=9)

    plt.tight_layout()
    fig.savefig(out_path, facecolor="black", dpi=150)
    plt.close(fig)


def _save_aip_grid(
    raw_vol: np.ndarray,
    corr_vol: np.ndarray,
    slab_size: int,
    out_path: str,
    title: str,
) -> None:
    """Save a two-row grid: top = RAW slab AIPs, bottom = CORRECTED slab AIPs."""
    n_z = corr_vol.shape[0]
    starts = list(range(0, n_z, slab_size))
    n_slabs = len(starts)

    fig, axes = plt.subplots(2, n_slabs, figsize=(max(n_slabs * 3, 6), 8), facecolor="black")
    fig.suptitle(title, color="white", fontsize=8)

    # ensure 2-D indexing for single-slab edge case
    axes = np.asarray(axes).reshape(2, n_slabs)

    for col, z0 in enumerate(starts):
        z1 = min(z0 + slab_size, n_z)
        raw_aip = raw_vol[z0:z1].mean(axis=0)
        corr_aip = corr_vol[z0:z1].mean(axis=0)

        for row, (aip, label) in enumerate([(raw_aip, "RAW"), (corr_aip, "CORR")]):
            axes[row, col].imshow(aip, cmap="magma", vmin=0, vmax=_display_vmax(aip))
            axes[row, col].set_title(f"{label} z{z0}-{z1 - 1}", color="white", fontsize=7)
            axes[row, col].set_axis_off()
            axes[row, col].set_facecolor("black")

    plt.tight_layout()
    fig.savefig(out_path, facecolor="black", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Argument parsing helpers
# ---------------------------------------------------------------------------


def _parse_float_none_list(s: str) -> list[float | None]:
    result: list[float | None] = []
    for tok in s.split(","):
        tok = tok.strip()
        result.append(None if tok.lower() == "none" else float(tok))
    return result


def _parse_bool_list(s: str) -> list[bool]:
    result = []
    for tok in s.split(","):
        tok = tok.strip().lower()
        if tok in ("true", "1", "yes"):
            result.append(True)
        elif tok in ("false", "0", "no"):
            result.append(False)
        else:
            msg = f"Cannot parse bool value: {tok!r}. Use true/false."
            raise ValueError(msg)
    return result


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input_zarr", help="Input mosaic-grid OME-Zarr (focal-fix or raw).")
    p.add_argument("output_dir", help="Directory where sweep results are written (created if absent).")

    # sweep axes - comma-separated lists -> cartesian product
    p.add_argument(
        "--percentile_max",
        default="99.9",
        help="Comma-sep upper-clip percentile values applied to the fit pool\n"
        "('none' = no clip).  E.g. 'none,99.0,99.5,99.9'  [%(default)s]",
    )
    p.add_argument(
        "--use_darkfield",
        default="true,false",
        help="Comma-sep: true and/or false.  E.g. 'true,false'  [%(default)s]",
    )
    p.add_argument(
        "--darkfield_percentile",
        default="5",
        help="Comma-sep per-pixel percentile for darkfield estimation\n"
        "(only used when use_darkfield=true).  E.g. '2,5,10'  [%(default)s]",
    )
    p.add_argument(
        "--fit_max_samples",
        default="2000",
        help="Comma-sep max tile samples drawn for the BaSiC fit.  [%(default)s]",
    )
    p.add_argument(
        "--max_iterations",
        default="500",
        help="Comma-sep BaSiC iteration counts.  [%(default)s]",
    )
    p.add_argument(
        "--smoothness_flatfield",
        default="none",
        help="Comma-sep BaSiC regularization strength for the flatfield.\n"
        "Higher = smoother flatfield (less tile-edge noise but may miss\n"
        "real spatial variation). 'none' lets BaSiC auto-select (~0.1).\n"
        "E.g. 'none,0.01,0.05,0.1,0.5'  [%(default)s]",
    )
    p.add_argument(
        "--working_size",
        default="none",
        help="Comma-sep internal BaSiC resize dimension (pixels).\n"
        "Smaller = faster but less spatial detail in the flatfield.\n"
        "'none' keeps BaSiC default (128). Try '64,128'.  [%(default)s]",
    )
    p.add_argument(
        "--per_z_fit",
        default="false",
        help="Comma-sep: true and/or false.  When true, fits a separate BaSiC\n"
        "model per Z plane instead of a single global model.\n"
        "E.g. 'true,false'  [%(default)s]",
    )
    p.add_argument(
        "--darkfield_smooth_sigma",
        default="none",
        help="Comma-sep Gaussian sigma(s) for spatially smoothing the estimated darkfield.\n"
        "Reduces pixel-level noise in the per-pixel percentile estimate.\n"
        "'none' or 0 disables smoothing. E.g. 'none,1.5,3.0'  [%(default)s]",
    )
    p.add_argument(
        "--darkfield_z_window",
        default="0",
        help="Comma-sep: number of neighbouring Z planes to include in the darkfield pool\n"
        "(per_z_fit=true only). 0 = current plane only. 1 = z+-1 (3x tiles). 'all' = every\n"
        "Z plane (~55x tiles; physically valid since darkfield is depth-independent).\n"
        "E.g. '0,1,all'  [%(default)s]",
    )
    p.add_argument(
        "--flatfield_smooth_sigma",
        default="none",
        help="Comma-sep Gaussian sigma(s) for smoothing the BaSiC flatfield after fitting.\n"
        "Suppresses residual high-frequency noise in the fitted flatfield.\n"
        "'none' or 0 disables. E.g. 'none,1.0,2.0'  [%(default)s]",
    )
    p.add_argument(
        "--tile_fov_mm",
        type=float,
        default=0.0,
        help="Acquisition tile field-of-view in mm.  When > 0, overrides the\n"
        "chunk-derived tile size (same as pipeline param tile_fov_mm).  [%(default)s]",
    )
    p.add_argument(
        "--n_workers",
        type=int,
        default=1,
        help="Parallel workers for per-Z fitting (per_z_fit=true only).\n"
        "Each worker runs an independent BaSiC fit; set to number of available\n"
        "CPU cores for maximum throughput. Has no effect for global fits.  [%(default)s]",
    )

    # output control
    p.add_argument(
        "--z_slice",
        type=int,
        help="Z index for the single-slice preview. Default: centre of volume.",
    )
    p.add_argument(
        "--aip",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Also apply correction to the full volume and save a slab-AIP\n"
        "grid showing depth-wise correction quality.  [%(default)s]",
    )
    p.add_argument(
        "--aip_slab_size",
        type=int,
        default=5,
        help="Number of Z-planes per AIP slab.  For Z=30, slab_size=5 → 6 projections.  [%(default)s]",
    )
    p.add_argument(
        "--level",
        type=int,
        default=0,
        help="OME-Zarr pyramid level to load (0 = full resolution).  [%(default)s]",
    )
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the illumination-parameter sweep."""
    p = _build_arg_parser()
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load volume once ────────────────────────────────────────────────────
    print(f"Loading {args.input_zarr} (level {args.level})…")
    vol_lazy, resolution = read_omezarr(Path(args.input_zarr), level=args.level)
    if args.tile_fov_mm > 0:
        pixel_size_mm = float(resolution[1])
        tile_px = round(args.tile_fov_mm / pixel_size_mm)
        tile_shape: tuple[int, int] = (tile_px, tile_px)
        print(f"tile_fov_mm={args.tile_fov_mm}: tile_size_px={tile_px} (pixel_size={pixel_size_mm:.4f}mm/px)")
    else:
        tile_shape = (int(vol_lazy.chunks[1]), int(vol_lazy.chunks[2]))
    print(f"  shape={vol_lazy.shape}  tile={tile_shape}  dtype={vol_lazy.dtype}")
    print("  Reading into RAM…")
    vol: np.ndarray = np.asarray(vol_lazy).astype(np.float32)

    n_axial = vol.shape[0]
    z_slice = args.z_slice if args.z_slice is not None else n_axial // 2
    print(f"  Z preview slice: {z_slice}  (use --z_slice N to override)")

    # ── build sweep grid ────────────────────────────────────────────────────
    try:
        p_maxes = _parse_float_none_list(args.percentile_max)
        use_darks = _parse_bool_list(args.use_darkfield)
        df_percs = [float(x) for x in args.darkfield_percentile.split(",")]
        fit_samps = [int(x) for x in args.fit_max_samples.split(",")]
        max_iters = [int(x) for x in args.max_iterations.split(",")]
        smooth_ffs = _parse_float_none_list(args.smoothness_flatfield)
        working_sizes = _parse_float_none_list(args.working_size)
        per_z_fits = _parse_bool_list(args.per_z_fit)
        df_smooth_sigmas = _parse_float_none_list(args.darkfield_smooth_sigma)
        df_z_windows = [(-1 if x.strip().lower() == "all" else int(x)) for x in args.darkfield_z_window.split(",")]
        ff_smooth_sigmas = _parse_float_none_list(args.flatfield_smooth_sigma)
    except ValueError as exc:
        p.error(str(exc))

    raw_grid = list(
        itertools.product(
            p_maxes,
            use_darks,
            df_percs,
            fit_samps,
            max_iters,
            smooth_ffs,
            working_sizes,
            per_z_fits,
            df_smooth_sigmas,
            df_z_windows,
            ff_smooth_sigmas,
        )
    )

    # de-duplicate: when use_darkfield=False, df params are irrelevant;
    # when per_z_fit=False, df_z_window is irrelevant
    seen: set[tuple] = set()
    configs: list[tuple] = []
    for c in raw_grid:
        pmax, use_dark, df_p, samp, iters, smooth_ff, ws, per_z, df_sig, df_zw, ff_sig = c
        key = (
            pmax,
            use_dark,
            df_p if use_dark else "N/A",
            samp,
            iters,
            smooth_ff,
            ws,
            per_z,
            df_sig if use_dark else "N/A",
            df_zw if (use_dark and per_z) else "N/A",
            ff_sig,
        )
        if key not in seen:
            seen.add(key)
            configs.append(c)

    n_slabs = (n_axial + args.aip_slab_size - 1) // args.aip_slab_size
    print(f"\nSweep: {len(configs)} unique configurations")
    if args.aip:
        print(f"AIP mode ON: {n_slabs} slabs x {args.aip_slab_size} Z-planes")

    summary_rows: list[dict] = []

    for idx, (pmax, use_dark, df_p, samp, iters, smooth_ff, ws, per_z, df_sig, df_zw, ff_sig) in enumerate(configs, start=1):
        pmax_s = f"p{pmax:.1f}" if pmax is not None else "pNone"
        df_s = f"_df{df_p:.0f}" if use_dark else "_nodf"
        sm_s = f"_sm{smooth_ff:.3f}" if smooth_ff is not None else ""
        ws_s = f"_ws{int(ws)}" if ws is not None else ""
        pz_s = "_perz" if per_z else "_global"
        dfsig_s = f"_dfsig{df_sig:.2f}" if (df_sig is not None and df_sig > 0) else ""
        dfzw_raw = "all" if df_zw == -1 else str(df_zw)
        dfzw_s = f"_dfzw{dfzw_raw}" if (per_z and use_dark and df_zw != 0) else ""
        ffsig_s = f"_ffsig{ff_sig:.2f}" if (ff_sig is not None and ff_sig > 0) else ""
        label = f"c{idx:03d}_{pmax_s}{df_s}_s{samp}_i{iters}{sm_s}{ws_s}{pz_s}{dfsig_s}{dfzw_s}{ffsig_s}"
        desc = (
            f"pmax={pmax}  dark={use_dark}  dfp={df_p if use_dark else '-'}  "
            f"samples={samp}  iters={iters}  smooth={smooth_ff}  ws={ws}  per_z_fit={per_z}  "
            f"df_smooth_sigma={df_sig}  df_z_window={df_zw if (per_z and use_dark) else '-'}  ff_smooth_sigma={ff_sig}"
        )
        print(f"\n[{idx}/{len(configs)}] {label}")
        print(f"  {desc}")

        apply_z = list(range(n_axial)) if args.aip else [z_slice]

        try:
            corrected, flatfield, darkfield, stats = run_one_config(
                vol,
                tile_shape,
                percentile_max=pmax,
                use_darkfield=use_dark,
                darkfield_percentile=df_p,
                fit_max_samples=samp,
                max_iterations=iters,
                smoothness_flatfield=smooth_ff,
                working_size=int(ws) if ws is not None else None,
                apply_z=apply_z,
                per_z_fit=per_z,
                darkfield_smooth_sigma=df_sig if df_sig is not None else 0.0,
                darkfield_z_window=df_zw,
                flatfield_smooth_sigma=ff_sig if ff_sig is not None else 0.0,
                n_workers=args.n_workers,
            )
        except Exception as exc:
            print(f"  FAILED: {exc}")
            summary_rows.append(
                {
                    "config": label,
                    "status": "FAILED",
                    "error": str(exc),
                    "pmax": pmax,
                    "use_darkfield": use_dark,
                    "df_percentile": df_p,
                    "fit_max_samples": samp,
                    "max_iterations": iters,
                    "per_z_fit": per_z,
                    "darkfield_smooth_sigma": df_sig,
                    "darkfield_z_window": df_zw,
                    "flatfield_smooth_sigma": ff_sig,
                    "n_fit_tiles": "",
                    "ff_min": "",
                    "ff_max": "",
                    "df_min": "",
                    "df_max": "",
                    "slice_nonzero": "",
                    "slice_max": "",
                }
            )
            continue

        # ── single-slice preview ─────────────────────────────────────────────
        ff_range = f"ff=[{stats['ff_min']:.3f},{stats['ff_max']:.3f}]"
        df_range = f"  df=[{stats['df_min']:.2f},{stats['df_max']:.2f}]" if use_dark else ""
        title = f"{label}  z={z_slice}  {ff_range}{df_range}  fit_tiles={stats['n_fit_tiles']}"

        slice_path = str(out_dir / f"{label}_z{z_slice:03d}.png")
        _save_slice_comparison(vol[z_slice], corrected[z_slice], flatfield, darkfield, slice_path, title=title)
        print(f"  → {Path(slice_path).name}")

        # ── AIP grid ─────────────────────────────────────────────────────────
        if args.aip:
            corr_stack = np.stack([corrected[z] for z in range(n_axial)], axis=0)
            aip_path = str(out_dir / f"{label}_aips.png")
            _save_aip_grid(
                vol,
                corr_stack,
                args.aip_slab_size,
                aip_path,
                title=f"{label}  slab={args.aip_slab_size}px  ({n_axial} Z-planes → {n_slabs} projections)",
            )
            print(f"  → {Path(aip_path).name}")

        summary_rows.append(
            {
                "config": label,
                "status": "OK",
                "error": "",
                "pmax": pmax,
                "use_darkfield": use_dark,
                "df_percentile": df_p,
                "fit_max_samples": samp,
                "max_iterations": iters,
                "smoothness_flatfield": smooth_ff,
                "working_size": ws,
                "per_z_fit": per_z,
                "darkfield_smooth_sigma": df_sig,
                "darkfield_z_window": df_zw,
                "flatfield_smooth_sigma": ff_sig,
                "n_fit_tiles": stats["n_fit_tiles"],
                "ff_min": round(stats["ff_min"], 4),
                "ff_max": round(stats["ff_max"], 4),
                "df_min": round(stats["df_min"], 4) if stats["df_min"] is not None else "",
                "df_max": round(stats["df_max"], 4) if stats["df_max"] is not None else "",
                "slice_nonzero": round(float(np.mean(corrected[z_slice] > 0)), 4),
                "slice_max": round(float(corrected[z_slice].max()), 2),
            }
        )

    # ── summary CSV ──────────────────────────────────────────────────────────
    csv_path = out_dir / "sweep_summary.csv"
    if summary_rows:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    n_ok = sum(1 for r in summary_rows if r["status"] == "OK")
    n_fail = sum(1 for r in summary_rows if r["status"] == "FAILED")
    print(f"\nDone.  {n_ok} OK, {n_fail} failed.  Summary → {csv_path}")


if __name__ == "__main__":
    main()
