#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Reorient a volume to RAS+ using control points in pixel coordinates (ex: from Fiji).

 The control points are positioned on the anterior, posterior, superior, and inferior sides of the brain.
 The script will estimate the main axis of the volume and reorient it to RAS+. Currently, the script only
 performs 90° rotations and flips."""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input_volume",
                   help="Full path to the input volume (.nii or .nii.gz)")
    p.add_argument("output_volume",
                   help="Full path to the output volume (.nii or .nii.gz)")
    p.add_argument("--pos_anterior", nargs=3, type=int, required=True,
                   help="Position of the anterior control point in pixel.")
    p.add_argument("--pos_posterior", nargs=3, type=int, required=True,
                   help="Position of the posterior control point in pixel.")
    p.add_argument("--pos_superior", nargs=3, type=int, required=True,
                   help="Position of the superior control point in pixel.")
    p.add_argument("--pos_inferior", nargs=3, type=int, required=True,
                   help="Position of the inferior control point in pixel.")

    return p


def main():
    # Parse arguments
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Parameters
    input_volume = Path(args.input_volume)
    output_volume = Path(args.output_volume)

    # Control points position in pixel (ex: from Fiji)
    pos_anterior = args.pos_anterior
    pos_posterior = args.pos_posterior
    pos_superior = args.pos_superior
    pos_inferior = args.pos_inferior

    # Estimate the main axis of the volume
    vector_pa = np.array(pos_anterior) - np.array(pos_posterior)
    vector_is = np.array(pos_superior) - np.array(pos_inferior)
    axis_pa = np.argmax(np.abs(vector_pa))
    axis_pa_sign = np.sign(vector_pa[axis_pa])
    axis_is = np.argmax(np.abs(vector_is))
    axis_is_sign = np.sign(vector_is[axis_is])

    vector_pa = np.zeros(3)
    vector_pa[axis_pa] = axis_pa_sign
    vector_is = np.zeros(3)
    vector_is[axis_is] = axis_is_sign
    vector_lr = np.cross(vector_pa, vector_is)

    axis_lr = np.argmax(np.abs(vector_lr))
    axis_lr_sign = np.sign(vector_lr[axis_lr])

    # Get the axis code
    axcodes = [""] * 3
    axcodes[axis_pa] = "A" if axis_pa_sign > 0 else "P"
    axcodes[axis_is] = "S" if axis_is_sign > 0 else "I"
    axcodes[axis_lr] = "L" if axis_lr_sign > 0 else "R"

    # Get the nibabel orientation transformation from axcodes to RAS+
    transformation = nib.orientations.axcodes2ornt(axcodes)

    # Apply the transformation to the volume
    img = nib.load(input_volume)
    vol = img.get_fdata(dtype=np.float32)
    vol_ras = nib.orientations.apply_orientation(vol, transformation)

    # Get the resolution of the original volume
    resolutions = img.header["pixdim"][1:4]
    new_resolutions = []
    new_resolutions.append(resolutions[axis_lr])
    new_resolutions.append(resolutions[axis_pa])
    new_resolutions.append(resolutions[axis_is])

    # Save the new volume
    affine = np.eye(4)
    affine[0, 0] = new_resolutions[0]
    affine[1, 1] = new_resolutions[1]
    affine[2, 2] = new_resolutions[2]
    img_ras = nib.Nifti1Image(vol_ras, affine)
    nib.save(img_ras, output_volume)


if __name__ == "__main__":
    main()
