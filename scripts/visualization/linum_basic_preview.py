#!/usr/bin/env python3
"""Generate a 2D preview PNG from a 3D OME-Zarr volume.

Produces an average-intensity projection (AIP) of a mosaic-grid or stitched
OME-Zarr volume along a chosen axis and saves it as a PNG using the shared
linum-basic visualization theme (viridis colormap, physical scale bar). Useful
for quickly inspecting the result of illumination correction or stitching in a
processing pipeline, e.g. a per-slice ``z27``-style screenshot.
"""

import linumpy.config.threads  # noqa: F401

import argparse
from pathlib import Path

import numpy as np

from linumpy.io.zarr import read_omezarr


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input_zarr", help="Full path to the input OME-Zarr volume.")
    p.add_argument("output_png", help="Full path to the output PNG file.")
    p.add_argument(
        "--axis",
        type=int,
        default=0,
        help="Axis along which to project the volume (0=Z, 1=Y, 2=X). [%(default)s]",
    )
    p.add_argument(
        "--percentile",
        type=float,
        default=99.5,
        help="Upper percentile used to set the display range (clips bright outliers). [%(default)s]",
    )
    p.add_argument(
        "--cmap",
        type=str,
        default="viridis",
        help="Matplotlib colormap name for the preview. [%(default)s]",
    )
    p.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional title drawn on the preview. [input file name]",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Output PNG resolution in dots per inch. [%(default)s]",
    )
    return p


def main() -> None:
    """Render an average-intensity projection of an OME-Zarr volume to PNG."""
    from linumpy.config.threads import configure_all_libraries

    configure_all_libraries()

    p = _build_arg_parser()
    args = p.parse_args()

    from linum_basic import viz

    input_zarr = Path(args.input_zarr)
    output_png = Path(args.output_png)

    vol, resolution = read_omezarr(input_zarr, level=0)
    volume = np.asarray(vol)
    if np.iscomplexobj(volume):
        volume = np.abs(volume)
    volume = volume.astype(np.float32, copy=False)

    if volume.ndim != 3:
        msg = f"Expected a 3D volume, got shape {volume.shape}."
        raise ValueError(msg)

    # In-plane pixel size: the scale of the last axis not collapsed by the projection.
    in_plane_axis = 2 if args.axis != 2 else 1
    pixel_size_mm = float(resolution[in_plane_axis])

    title = args.title if args.title is not None else input_zarr.name
    fig = viz.aip_preview(
        volume,
        axis=args.axis,
        pixel_size_mm=pixel_size_mm,
        cmap=args.cmap,
        title=title,
        percentile=args.percentile,
    )
    viz.save_figure(fig, output_png, dpi=args.dpi)
    print(f"Saved preview to {output_png}")


if __name__ == "__main__":
    main()
