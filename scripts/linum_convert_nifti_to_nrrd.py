#!/usr/bin/env python3

"""Convert a nifti volume into a nrrd volume."""

# Configure thread limits before numpy/scipy imports
import linumpy.config.threads  # noqa: F401

import argparse
from pathlib import Path

import nibabel as nib
import nrrd
import numpy as np


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input", type=Path, help="Full path to a 3D .nii file")
    p.add_argument("output", type=Path, help="Full path to the .nrrd file")
    p.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize the data (default=%(default)s)",
    )
    return p


def main() -> None:
    """Run the NIfTI-to-NRRD conversion script."""
    # Parse arguments
    p = _build_arg_parser()
    args = p.parse_args()

    img = nib.load(args.input)
    assert isinstance(img, nib.Nifti1Image)

    # Load the data
    # Neuroglancer doesn't support float64
    vol = img.get_fdata(dtype=np.float32)

    # Normalize the data
    if args.normalize:
        vol -= vol.min()
        vol /= vol.max()

    # Invert the x and z axis
    vol = np.moveaxis(vol, (0, 1, 2), (2, 1, 0))

    nrrd.write(args.output, vol)


if __name__ == "__main__":
    main()
