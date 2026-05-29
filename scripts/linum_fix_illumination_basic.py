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
"""

import linumpy.config.threads  # noqa: F401

import argparse
from pathlib import Path

import numpy as np

from linumpy.cli.args import add_processes_arg
from linumpy.io.zarr import read_omezarr, save_omezarr


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

    mosaic = MosaicGrid(array=array, tile_shape=tile_shape)
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
        verbose=True,
    )

    corrected = apply_fit(mosaic, fit, n_extra_rows=args.n_extra_rows)

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

    save_omezarr(corrected, output_zarr, voxel_size=resolution, chunks=vol.chunks, n_levels=args.n_levels)
    print(f"Saved corrected mosaic grid to {output_zarr}")


if __name__ == "__main__":
    main()
