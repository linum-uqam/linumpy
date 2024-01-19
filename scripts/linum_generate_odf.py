#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generate an image intensity derived 3D ODF using steerable filters."""

import argparse
import logging
import os
from pathlib import Path
import imageio
import nibabel as nib
import numpy as np

from linumpy.utils import __multiple_shell_sampling as sampling
from linumpy.filters import steerable_oriented_energy_3d

EPILOG = """

Usage Example
-------------
python scripts/linum_generate_odf.py ~/data/tmp/test_angio_64px.tif 9 60 ~/data/tmp/test_angio_64px_3dodf

References
----------
- [1] Freeman, W. T., & Adelson, E. H. (1991). The design and use of steerable filters.
      IEEE Transactions on Pattern Analysis and Machine Intelligence, 13(9), 891‑906. https://doi.org/10.1109/34.93808
- [2] Derpanis, K. G., & Gryn, J. M. (2005). Three-dimensional nth derivative of Gaussian separable steerable filters.
      IEEE International Conference on Image Processing 2005, III‑553. https://doi.org/10.1109/ICIP.2005.1530451
"""

SUPPORTED_FORMATS = [".tif", ".tiff", ".nii", ".nii.gz"]


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
        epilog=EPILOG)

    parser.add_argument('input_filename',
                        help=f"Input volume ({SUPPORTED_FORMATS})")

    parser.add_argument("size", type=int,
                        help="Steerable Gaussian Filter Size")

    parser.add_argument('nb_samples', type=int,
                        help="Number of orientation samples.")

    parser.add_argument('out_basename',
                        help='Output basename (don\'t include extension)')

    parser.add_argument('-v', action='store_true', dest='verbose',
                        help='If set, produces verbose output.')

    return parser


def main():
    """Generate an image intensity derived 3D ODF using steerable filters."""
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Preparation
    out_basename = Path(args.out_basename)
    directory = out_basename.parent
    filename = out_basename.name
    extension = "".join(out_basename.suffixes)
    directory.mkdir(exist_ok=True, parents=True)
    out_basename, _ = os.path.splitext(args.out_basename)
    out_filename = {"vol": directory / (filename.strip(extension) + "_vol.nii"),
                    "odf": directory / (filename.strip(extension) + ".nii"),
                    "bval": directory / (filename.strip(extension) + ".bval"),
                    "bvec": directory / (filename.strip(extension) + ".bvec")}

    logging.basicConfig(level=logging.INFO)
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    # The algorithm starts here
    # Load the input volume
    extension = "".join(Path(args.input_filename).suffixes)
    assert extension in SUPPORTED_FORMATS, f"Input volume extension must be: {SUPPORTED_FORMATS}"
    if extension in [".nii", ".nii.gz"]:
        vol = nib.load(args.input_filename).get_data()
    else:
        vol = imageio.volread(args.input_filename)

    # Generate the orientation sampling directions
    nb_points_per_shell = args.nb_samples
    bvecs = sampling.multiple_shell(1, args.nb_samples, np.array([[1.0]]), verbose=0)

    # Apply the steerable gaussians
    odf = steerable_oriented_energy_3d(vol, bvecs, size=args.size)

    # Adding a "b0" signal
    output = np.zeros((*vol.shape, nb_points_per_shell + 1), dtype=odf.dtype)
    output[..., 0] = vol
    output[..., 1::] = odf
    bvecs = [np.zeros((3,)), *bvecs]
    shell_idx = [0, *[1] * nb_points_per_shell]
    bvals = [0.0, 1.0e3]  # scilpy expects shells with minimum bval difference of 40. So choosing artificial high values

    # Save the output
    nib.save(nib.Nifti1Image(vol, np.eye(4)), out_filename["vol"])
    nib.save(nib.Nifti1Image(output, np.eye(4)), out_filename["odf"])

    # Save the corresponding bvals and bvecs
    sampling.save_gradient_sampling_fsl(bvecs, shell_idx, bvals, out_filename['bval'], out_filename['bvec'])


if __name__ == "__main__":
    main()
