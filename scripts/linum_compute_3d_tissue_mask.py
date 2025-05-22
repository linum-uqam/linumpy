#!/usr/bin/env python3

"""Computes the 3D tissue mask of a given slice."""

import argparse
from pathlib import Path
import SimpleITK as sitk
import numpy as np
import zarr
import dask.array as da
from linumpy.preproc import xyzcorr
from linumpy.io.zarr import read_omezarr, save_zarr


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )

    p.add_argument("input", help="A single 3D slice (.ome.zarr /.zarr) to process")
    p.add_argument("output", help="Output 3D tissue mask (.ome.zarr /.zarr)")
    p.add_argument(
        "--s_xy",
        type=int,
        default=3,
        help="Lateral smoothing kernel size (default = %(default)s)",
    )
    p.add_argument(
        "--s_z",
        type=int,
        default=3,
        help="Axial smoothing kernel size (default = %(default)s)",
    )
    p.add_argument(
        "--morpho_size",
        type=int,
        default=5,
        help="Morphological filter kernel size (default = %(default)s)",
    )
    p.add_argument(
        "--use_log",
        action="store_true",
        help="Use log of intensity to compute the interface ? (default=%(default)s)",
    )
    p.add_argument(
        "--thickness",
        type=float,
        default=300.0,
        help="Maximum mask thickness (in micron). (default = %(default)s, -1 will use the full mask under the interface)",
    )
    p.add_argument(
        "-d",
        "--depth",
        default=0.0,
        type=float,
        help="Depth shift below the water/tissue interface to keep in microns. (default = %(default)s)",
    )
    return p


def main():
    # Parse arguments
    p = _build_arg_parser()
    args = p.parse_args()

    # Loading and smoothing the slice
    volc, res = read_omezarr(args.input, level=0)
    resolution = res[0] * 1000  # Convert to microns

    # Reorient the volume so that axis 0 (depth) becomes the last axis
    vol = np.moveaxis(volc, 0, -1)

    # Computing AIP
    aip = np.mean(vol, axis=2)

    # Computing water-tissue interface
    interface = xyzcorr.findTissueInterface(
        vol, s_xy=args.s_xy, s_z=args.s_z, useLog=args.use_log
    )
    interface[aip == 0] = 0
    interface += int(np.ceil(args.s_z / 2.0))  # Compensate axial smoothing

    # Shift the interface by a predefined depth, to move away from the water/tissue interface
    interface += int(args.depth / float(resolution))

    # Filter out small structures in this map
    if args.morpho_size > 0:
        interface_itk = sitk.GetImageFromArray(interface)
        dilated_interface = sitk.GrayscaleDilate(
            interface_itk, [args.morpho_size, args.morpho_size, 1]
        )
        interface_without_holes = sitk.ReconstructionByErosion(
            dilated_interface, interface_itk
        )
        eroded_interface = sitk.GrayscaleErode(
            interface_without_holes, [args.morpho_size, args.morpho_size, 1]
        )
        interface_without_peaks = sitk.ReconstructionByDilation(
            eroded_interface, interface_without_holes
        )
        interface = sitk.GetArrayFromImage(interface_without_peaks)

    # Computing the tissue mask under the interface
    mask_interface = xyzcorr.maskUnderInterface(vol, interface, returnMask=True)

    # Limiting thickness of the mask
    nx, ny, nz = vol.shape
    if args.thickness > 0:
        bottom = ((interface * resolution + args.thickness) / resolution).astype(int)
        bottom[bottom > nz] = nz
    else:  # Use the full thickness under the interface
        bottom = np.ones_like(interface, dtype=int) * nz
    _, _, zz = np.meshgrid(
        list(range(nx)), list(range(ny)), list(range(nz)), indexing="ij"
    )
    bottom_3d = np.tile(np.reshape(bottom, (nx, ny, 1)), (1, 1, nz))
    mask_interface[zz > bottom_3d] = False

    # Flip the mask back to original orientation (depth as axis 0)
    mask_interface = np.moveaxis(mask_interface, -1, 0)

    # Saving output
    output_file = Path(args.output)
    output_file.parent.mkdir(exist_ok=True, parents=True)
    temp_store = zarr.TempStore(suffix=".zarr")
    mosaic = zarr.open(
        temp_store,
        mode="w",
        shape=mask_interface.shape,
        dtype=np.float32,
        chunks=volc.chunks[1:3],
    )
    mosaic[:] = mask_interface.astype(np.float32)
    out_dask = da.from_zarr(mosaic)
    save_zarr(out_dask, output_file, res)


if __name__ == "__main__":
    main()
