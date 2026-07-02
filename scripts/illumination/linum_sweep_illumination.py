#!/usr/bin/env python3
r"""Parameter sweep tool for illumination-correction tuning.

Runs the linum-basic illumination correction for every combination of the
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
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from linumpy.intensity.sweep import (
    build_sweep_grid,
    parse_bool_list,
    parse_float_none_list,
    run_one_config,
)
from linumpy.io.zarr import read_omezarr

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

    # Shared vmax for raw/corrected so the correction effect is visible
    shared_vmax = max(_display_vmax(raw_plane), _display_vmax(corr_plane))

    axes[0].imshow(raw_plane, cmap="magma", vmin=0, vmax=shared_vmax)
    axes[0].set_title("RAW", color="white", fontsize=9)

    axes[1].imshow(corr_plane, cmap="magma", vmin=0, vmax=shared_vmax)
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

        # Shared vmax so raw/corrected are on the same scale and differences are visible
        shared_vmax = max(_display_vmax(raw_aip), _display_vmax(corr_aip))

        for row, (aip, label) in enumerate([(raw_aip, "RAW"), (corr_aip, "CORR")]):
            axes[row, col].imshow(aip, cmap="magma", vmin=0, vmax=shared_vmax)
            axes[row, col].set_title(f"{label} z{z0}-{z1 - 1}", color="white", fontsize=7)
            axes[row, col].set_axis_off()
            axes[row, col].set_facecolor("black")

    plt.tight_layout()
    fig.savefig(out_path, facecolor="black", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Argument parsing helpers
# ---------------------------------------------------------------------------


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
    p.add_argument(
        "--use_gpu",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Use linum-basic torch backend on a CUDA/MPS device when available. [%(default)s]",
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
        p_maxes = parse_float_none_list(args.percentile_max)
        use_darks = parse_bool_list(args.use_darkfield)
        df_percs = [float(x) for x in args.darkfield_percentile.split(",")]
        fit_samps = [int(x) for x in args.fit_max_samples.split(",")]
        max_iters = [int(x) for x in args.max_iterations.split(",")]
        smooth_ffs = parse_float_none_list(args.smoothness_flatfield)
        working_sizes = parse_float_none_list(args.working_size)
        per_z_fits = parse_bool_list(args.per_z_fit)
        df_smooth_sigmas = parse_float_none_list(args.darkfield_smooth_sigma)
        df_z_windows = [(-1 if x.strip().lower() == "all" else int(x)) for x in args.darkfield_z_window.split(",")]
        ff_smooth_sigmas = parse_float_none_list(args.flatfield_smooth_sigma)
    except ValueError as exc:
        p.error(str(exc))

    configs = build_sweep_grid(
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
                preview_z=z_slice,
                per_z_fit=per_z,
                darkfield_smooth_sigma=df_sig if df_sig is not None else 0.0,
                darkfield_z_window=df_zw,
                flatfield_smooth_sigma=ff_sig if ff_sig is not None else 0.0,
                use_gpu=args.use_gpu,
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
