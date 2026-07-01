#!/usr/bin/env python3
"""Fix lateral illumination inhomogeneities of a 3D mosaic grid with linum-basic.

This is an alternative to ``linum_fix_illumination_3d.py`` that uses the
``linum-basic`` reimplementation of the BaSiC shading-correction algorithm
instead of BaSiCPy. It operates on a mosaic-grid OME-Zarr (Z, Y, X) whose tiles
are laid out on a regular grid, fits a BaSiC flat-/dark-field model and writes
the corrected volume back as OME-Zarr.

Two fitting strategies are available:

* ``--no-per_z_fit`` (default): a single flat-/dark-field is obtained by
  averaging per-plane fits across axial (Z) planes, then applied uniformly to
  every plane. This avoids per-plane jitter that produces tile-period banding.
* ``--per_z_fit``: a separate flat-/dark-field is fit for each axial plane,
  capturing depth-dependent illumination variation from focal curvature.

GPU acceleration is used through the PyTorch backend of linum-basic when
``--use_gpu`` is set and a CUDA device is available.

When ``--diagnostics_dir`` is provided the script writes:

* a ``*_metrics.json`` quality report in the current working directory,
* a ``<diagnostics_dir>/<stem>_diagnostic.png`` figure with the estimated
  flat-field and raw-vs-corrected mosaic comparison,
* a ``<diagnostics_dir>/<stem>_aip.png`` average-intensity projection preview.
"""

import linumpy.config.threads  # noqa: F401

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from linumpy.cli.args import add_processes_arg
from linumpy.io.zarr import read_omezarr, save_omezarr
from linumpy.metrics import PipelineMetrics

if TYPE_CHECKING:
    from linum_basic import MosaicFit, MosaicGrid


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input_zarr", help="Full path to the input mosaic-grid zarr file.")
    p.add_argument("output_zarr", help="Full path to the output zarr file.")
    p.add_argument(
        "--max_iterations",
        type=int,
        default=10,
        help="Maximum number of outer reweighting iterations for BaSiC. [%(default)s]",
    )
    p.add_argument(
        "--smoothness_flatfield",
        type=float,
        default=None,
        help="Flatfield DCT regularization weight (linum-basic ``l_s``). Higher =\n"
        "smoother flatfield with less spatial detail. When omitted, linum-basic\n"
        "derives it automatically from the data. [auto]",
    )
    p.add_argument(
        "--use_darkfield",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Estimate an additive darkfield (per-pixel offset) in addition to\nthe multiplicative flatfield. [%(default)s]",
    )
    p.add_argument(
        "--per_z_fit",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Fit a separate flat-/dark-field per axial (Z) plane. When disabled\n"
        "(default) per-plane fits are averaged into a single global field. [%(default)s]",
    )
    p.add_argument(
        "--fit_max_samples",
        type=int,
        default=2000,
        help="Upper bound on the number of tile samples used to fit BaSiC. Axial\n"
        "planes are sub-sampled uniformly so the pooled tile count stays below\n"
        "this bound. [%(default)s]",
    )
    p.add_argument(
        "--tile_fov_mm",
        type=float,
        default=0.0,
        help="Acquisition tile field-of-view in millimetres. When > 0 the tile\n"
        "size is computed as round(tile_fov_mm / pixel_size_mm) instead of using\n"
        "the zarr chunk size. [%(default)s]",
    )
    p.add_argument(
        "--n_extra_rows",
        type=int,
        default=0,
        help="Number of galvo-return rows at the top of each tile to exclude from\n"
        "the fit and pass through uncorrected. [%(default)s]",
    )
    p.add_argument(
        "--n_levels",
        type=int,
        default=5,
        help="Number of pyramid levels in the output OME-Zarr. [%(default)s]",
    )
    p.add_argument(
        "--use_gpu",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Use the PyTorch backend on a CUDA device when available. [%(default)s]",
    )
    p.add_argument(
        "--darkfield_percentile",
        type=float,
        default=5.0,
        help="Accepted for Nextflow-pipeline compatibility. linum-basic estimates\n"
        "the darkfield internally and does not use this value. [%(default)s]",
    )
    p.add_argument(
        "--percentile_max",
        type=float,
        default=None,
        help="Accepted for Nextflow-pipeline compatibility. Not used by the\nlinum-basic backend. [%(default)s]",
    )
    p.add_argument(
        "--diagnostics_dir",
        type=Path,
        default=None,
        help="Directory to write diagnostic figures (.png) alongside the metrics\n"
        "JSON. When omitted no figures are written but the metrics JSON is\n"
        "still saved to the current working directory. [%(default)s]",
    )
    p.add_argument(
        "--slice_id",
        type=str,
        default=None,
        help="Slice identifier string used as a prefix for diagnostic output\n"
        "files. Inferred from the output path when omitted. [%(default)s]",
    )
    add_processes_arg(p)
    return p


def main() -> None:
    """Fit and apply a linum-basic illumination correction to a mosaic grid."""
    from linumpy.config.threads import configure_all_libraries

    configure_all_libraries()

    p = _build_arg_parser()
    args = p.parse_args()

    from linum_basic.fit import apply_fit, fit_mosaic
    from linum_basic.mosaic import MosaicGrid

    input_zarr = Path(args.input_zarr)
    output_zarr = Path(args.output_zarr)

    vol, resolution = read_omezarr(input_zarr, level=0)
    array = np.asarray(vol)
    if np.iscomplexobj(array):
        array = np.abs(array)
    array = array.astype(np.float32, copy=False)

    plane_shape = array.shape[1:]
    if args.tile_fov_mm > 0:
        pixel_size_mm = float(resolution[1])
        tile_px = round(args.tile_fov_mm / pixel_size_mm)
        tile_shape: tuple[int, int] = (tile_px, tile_px)
        print(f"tile_fov_mm={args.tile_fov_mm}: tile_size_px={tile_px} (pixel_size={pixel_size_mm}mm/px)")
    else:
        tile_shape = (int(vol.chunks[-2]), int(vol.chunks[-1]))

    # Crop array to tile-divisible dimensions.  The modernisation branch of
    # linum-basic enforces that H and W are exact multiples of tile height/width.
    # Edge pixels that don't form a complete tile are preserved in the output
    # unchanged (they retain their raw values).
    th, tw = tile_shape
    h, w = array.shape[1], array.shape[2]
    h_crop = (h // th) * th
    w_crop = (w // tw) * tw
    if h_crop != h or w_crop != w:
        print(f"Cropping mosaic from ({h},{w}) to ({h_crop},{w_crop}) to match tile grid (tile={tile_shape}).")
    array_crop = array[:, :h_crop, :w_crop]

    mosaic = MosaicGrid(array=array_crop, tile_shape=tile_shape)
    print(f"Mosaic grid: {mosaic.n_z} planes, {mosaic.n_rows}x{mosaic.n_cols} tiles of {tile_shape} (plane {plane_shape}).")

    # Sub-sample axial planes so the pooled tile count stays within budget.
    tiles_per_plane = mosaic.n_rows * mosaic.n_cols
    if tiles_per_plane == 0:
        msg = f"Tile shape {tile_shape} does not fit in plane shape {plane_shape}."
        raise ValueError(msg)
    n_planes_for_fit = min(mosaic.n_z, max(1, args.fit_max_samples // tiles_per_plane))
    if n_planes_for_fit >= mosaic.n_z:
        z_indices: list[int] = list(range(mosaic.n_z))
    else:
        z_indices = np.linspace(0, mosaic.n_z - 1, n_planes_for_fit, dtype=int).tolist()

    basic_kwargs: dict[str, object] = {
        "estimate_darkfield": args.use_darkfield,
        "max_reweighting_iterations": args.max_iterations,
        "backend": "torch" if args.use_gpu else "numpy",
        "verbose": False,
    }
    if args.smoothness_flatfield is not None:
        basic_kwargs["l_s"] = args.smoothness_flatfield

    field_mode = "per-z" if args.per_z_fit else "global"
    print(
        f"Fitting BaSiC (field_mode={field_mode}, darkfield={args.use_darkfield}) "
        f"on {len(z_indices)} / {mosaic.n_z} axial planes."
    )
    fit = fit_mosaic(
        mosaic,
        z_indices=z_indices,
        field_mode=field_mode,
        basic_kwargs=basic_kwargs,
        n_extra_rows=args.n_extra_rows,
        n_workers=args.n_processes,
        verbose=True,
    )

    corrected_crop = apply_fit(mosaic, fit, n_extra_rows=args.n_extra_rows)

    # Embed the corrected crop back into the full (possibly larger) array so that
    # edge pixels at the boundary are preserved as-is.
    corrected = array.copy()
    corrected[:, :h_crop, :w_crop] = corrected_crop

    out_min = float(corrected.min())
    out_max = float(corrected.max())
    nonzero_frac = float(np.mean(corrected != 0))
    print(f"Corrected volume stats: min={out_min:.4g} max={out_max:.4g} nonzero_frac={nonzero_frac:.4f}")
    if nonzero_frac < 0.01 or out_max <= 0:
        msg = (
            f"Illumination correction collapsed the volume "
            f"(nonzero_frac={nonzero_frac:.4f}, max={out_max:.4g}). Refusing to write all-zero output."
        )
        raise RuntimeError(msg)
    if out_min < 0:
        print(f"Minimum value in the output volume is {out_min}. Clipping at 0.")
        corrected = np.clip(corrected, 0.0, None)

    # ------------------------------------------------------------------
    # Quality metrics
    # ------------------------------------------------------------------
    seam_pairs = mosaic.seam_pairs()
    eval_metrics: dict[str, float] = {}
    if seam_pairs:
        from linum_basic.metrics import evaluate_correction_volume

        eval_metrics = evaluate_correction_volume(mosaic, fit, metrics=("seam", "curvature"))
        print(
            f"Seam metrics: L1={eval_metrics['seam_l1']:.4f}  "
            f"1-Pearson={eval_metrics['seam_1minus_pearson']:.4f}  "
            f"curvature={eval_metrics['seam_curvature']:.4f}"
        )

    slice_id = args.slice_id or output_zarr.stem
    metrics = PipelineMetrics("fix_illumination_basic", str(output_zarr.parent))
    metrics.add_info("input_volume", str(input_zarr), "Input mosaic grid path")
    metrics.add_info("output_volume", str(output_zarr), "Output corrected volume path")
    metrics.add_info("input_shape", list(array.shape), "Input array shape (Z,Y,X)")
    metrics.add_info("mosaic_grid", f"{mosaic.n_rows}x{mosaic.n_cols}", "Tile grid layout")
    metrics.add_info("tile_shape", list(tile_shape), "Tile size in pixels (H,W)")
    metrics.add_info("n_planes_fit", len(z_indices), "Number of axial planes used for fitting")
    metrics.add_info("field_mode", field_mode, "Flat-field fitting mode")
    metrics.add_info("use_darkfield", args.use_darkfield, "Whether a dark-field was estimated")
    if eval_metrics:
        metrics.add_metric(
            "seam_l1",
            eval_metrics["seam_l1"],
            description="Mean per-seam relative L1 intensity mismatch after correction (lower is better)",
            custom_thresholds={"warning": 0.12, "error": 0.25},
        )
        metrics.add_metric(
            "seam_pearson",
            1.0 - eval_metrics["seam_1minus_pearson"],
            description="Mean Pearson correlation across seam pairs after correction (higher is better)",
            custom_thresholds={"warning": 0.7, "error": 0.5, "higher_is_better": True},
        )
        metrics.add_metric(
            "seam_curvature",
            eval_metrics["seam_curvature"],
            description="Flatfield focal-curvature metric (lower is better)",
            custom_thresholds={"warning": 0.15, "error": 0.30},
        )
    metrics.save(f"{slice_id}_metrics.json")

    # ------------------------------------------------------------------
    # Diagnostic figures
    # ------------------------------------------------------------------
    if args.diagnostics_dir is not None:
        args.diagnostics_dir.mkdir(parents=True, exist_ok=True)
        _write_diagnostics(
            mosaic=mosaic,
            fit=fit,
            corrected_crop=corrected_crop,
            corrected_full=corrected,
            resolution=resolution,
            seam_pairs=seam_pairs,
            eval_metrics=eval_metrics,
            diagnostics_dir=args.diagnostics_dir,
            slice_id=slice_id,
        )

    save_omezarr(corrected, output_zarr, voxel_size=resolution, chunks=vol.chunks, n_levels=args.n_levels)
    print(f"Saved corrected mosaic grid to {output_zarr}")


def _write_diagnostics(
    *,
    mosaic: MosaicGrid,
    fit: MosaicFit,
    corrected_crop: np.ndarray,
    corrected_full: np.ndarray,
    resolution: tuple | None,
    seam_pairs: list,
    eval_metrics: dict,
    diagnostics_dir: Path,
    slice_id: str,
) -> None:
    """Write diagnostic PNG figures for the illumination correction step."""
    try:
        from linum_basic import viz
    except ImportError:
        print("linum_basic.viz not available — skipping diagnostic figures.")
        return

    try:
        from linum_basic.metrics import seam_l1 as _seam_l1
    except ImportError:
        return

    pixel_size_mm = float(resolution[1]) if resolution is not None else None

    # --- flat-field for display ------------------------------------------
    ff_display: np.ndarray = fit.flatfields if fit.flatfields.ndim == 2 else fit.flatfields.mean(axis=0)
    df_display: np.ndarray | None = None
    if fit.darkfields is not None and fit.darkfields.ndim >= 2:
        df_candidate = fit.darkfields if fit.darkfields.ndim == 2 else fit.darkfields.mean(axis=0)
        if float(df_candidate.max()) > 1e-6:
            df_display = df_candidate

    # --- representative z-level -----------------------------------------
    z_idx_mid = len(fit.z_indices) // 2
    z_mid = fit.z_indices[z_idx_mid]

    raw_tiles = mosaic.iter_tiles(z_mid)
    if fit.field_mode == "global":
        ff_z: np.ndarray = fit.flatfields
        df_z: np.ndarray = fit.darkfields
    else:
        ff_z = fit.flatfields[z_idx_mid]
        df_z = fit.darkfields[z_idx_mid]

    corrected_tiles = (raw_tiles.astype(np.float32) - df_z[np.newaxis]) / (ff_z[np.newaxis] + 1e-6)

    # Raw seam score for the representative plane
    raw_seam = _seam_l1(raw_tiles, seam_pairs) if seam_pairs else 0.0
    corr_seam = eval_metrics.get("seam_l1", _seam_l1(corrected_tiles, seam_pairs) if seam_pairs else 0.0)

    # Center tile as representative
    center_tile_idx = len(raw_tiles) // 2
    raw_tile = raw_tiles[center_tile_idx]
    corr_tile = corrected_tiles[center_tile_idx]

    # Full mosaic planes for the representative z
    raw_mosaic_plane = mosaic.array[z_mid]
    corr_mosaic_plane = corrected_crop[z_mid]

    try:
        fig_diag = viz.figure_apply_correction(
            flatfield=ff_display,
            raw_mosaic=raw_mosaic_plane,
            corrected_mosaic=corr_mosaic_plane,
            raw_tile=raw_tile,
            corrected_tile=corr_tile,
            seam_raw=raw_seam,
            seam_corrected=corr_seam,
            darkfield=df_display,
            title=f"Illumination correction — {slice_id}",
            pixel_size_mm=pixel_size_mm,
        )
        diag_path = diagnostics_dir / f"{slice_id}_diagnostic.png"
        viz.save_figure(fig_diag, str(diag_path))
        print(f"Saved illumination diagnostic figure to {diag_path}")
    except Exception as exc:
        print(f"Warning: could not save diagnostic figure: {exc}")

    try:
        fig_aip = viz.aip_preview(
            corrected_full,
            pixel_size_mm=pixel_size_mm,
            title=f"Corrected AIP — {slice_id}",
        )
        aip_path = diagnostics_dir / f"{slice_id}_aip.png"
        viz.save_figure(fig_aip, str(aip_path))
        print(f"Saved AIP preview to {aip_path}")
    except Exception as exc:
        print(f"Warning: could not save AIP preview: {exc}")


if __name__ == "__main__":
    main()
