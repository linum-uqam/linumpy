#!/usr/bin/env python3
"""
Generate orthogonal view screenshots from an OME-Zarr volume with Z-slice index annotations.

Creates a figure showing coronal and sagittal views with Z-slice index numbers
marked on the side, making it easy to identify which input slice corresponds
to which horizontal band in the reconstruction.
"""
# Configure thread limits before numpy/scipy imports
import linumpy._thread_config  # noqa: F401

import argparse
from pathlib import Path
from linumpy.io.zarr import read_omezarr
from linumpy.utils.visualization import save_annotated_views, estimate_n_slices_from_zarr, add_z_slice_labels  # noqa: F401


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("in_zarr",
                   help="Full path to a zarr file.")
    p.add_argument("out_figure",
                   help="Full path to the output figure")
    p.add_argument('--x_slice', type=int,
                   help='Slice index along the second axis (X/rows) for ZY view.')
    p.add_argument('--y_slice', type=int,
                   help='Slice index along the last axis (Y/columns) for ZX view.')
    p.add_argument('--n_slices', type=int,
                   help='Number of input slices (auto-detected from OME-Zarr metadata if not specified).')
    p.add_argument('--slice_ids', type=str,
                   help='Comma-separated list of actual slice IDs (e.g., "05,12,18"). '
                        'If provided, these will be shown instead of sequential numbers.')
    p.add_argument('--font_size', type=int, default=7,
                   help='Font size for slice labels (default: 7)')
    p.add_argument('--label_every', type=int, default=1,
                   help='Label every Nth slice (default: 1, label all)')
    p.add_argument('--show_lines', action='store_true',
                   help='Draw horizontal lines at slice boundaries')
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Validate input path
    in_path = Path(args.in_zarr)
    if not in_path.exists():
        parser.error(f"Input file not found: {args.in_zarr}")

    # Resolve symlinks (common in Nextflow work directories)
    in_path = in_path.resolve()

    image, _ = read_omezarr(str(in_path))

    # Determine number of input slices
    n_input_slices = args.n_slices if (args.n_slices is not None and args.n_slices > 0) else None

    # Parse slice_ids if provided
    slice_ids = None
    if args.slice_ids:
        slice_ids = [s.strip() for s in args.slice_ids.split(',')]
        if n_input_slices is None:
            n_input_slices = len(slice_ids)

    save_annotated_views(image, args.out_figure,
                         n_input_slices=n_input_slices,
                         x_slice=args.x_slice,
                         y_slice=args.y_slice,
                         font_size=args.font_size,
                         label_every=args.label_every,
                         show_lines=args.show_lines,
                         slice_ids=slice_ids,
                         zarr_path=str(in_path))


if __name__ == '__main__':
    main()
