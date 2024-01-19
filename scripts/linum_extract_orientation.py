#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import logging

import numpy as np
import nibabel as nib
from linumpy import preproc, io, filters

"""
Preprocesses a 3D nifti image and computes the local 3D orientation
"""


def _build_arg_parser():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--output_name', '-o',
                           default='ROI', help="Name of output file")
    argparser.add_argument('--output_dir', '-d',
                           default='orientation',
                           help='Path of output directory')
    argparser.add_argument('--input', '-i',
                           required=True, help='Name of input file')
    argparser.add_argument('--extract', dest='extract', action='store_true',
                           help='If set, extracts ROI from input volume')
    argparser.add_argument('--no-extract', dest='extract',
                           action='store_false',
                           help='If set, no extraction is done')
    argparser.set_defaults(extract=False)
    return argparser


def main():
    # Arguments
    argparser = _build_arg_parser()
    args = argparser.parse_args()

    # Valider le nom du fichier
    input_path = args.input
    if not(os.path.exists(input_path)):
        logging.error("Input path doesn't exist")
    elif input_path.endswith('.nii') or input_path.endswith('.nii.gz'):

        # Creer repertoire
        output_name = args.output_name
        output_dir = os.path.abspath(args.output_dir)
        if not(os.path.exists(output_dir)):
            os.makedirs(output_dir)

        # Importer un volume en format nii
        volume, affine = io.import_vol(input_path)

        # Pretraitement
        mask, volume = preproc.preprocess(volume)

        # Orientation 3d avec matrice hessian
        amp, orientation = filters.riesz_orientation_hessian_3d(volume)

        # Moduler orientation avec l'intensité du volume
        orientation = preproc.modulation(orientation, volume)

        # Orientation 3d en format nifti
        orientation_nii = nib.Nifti1Image(orientation, affine)
        nib.save(orientation_nii,
                 output_dir + '/' + output_name + '_orientation.nii')

        # Intensité du vol en format nifti
        vol_nii = nib.Nifti1Image(volume, affine)
        nib.save(vol_nii, output_dir + '/' + output_name + '_intensite.nii')

        # Masque du vol en format nifti
        mask_nii = nib.Nifti1Image(mask.astype(np.uint8),
                                   affine.astype(np.uint8))
        mask_nii.set_data_dtype(np.uint8)
        nib.save(mask_nii, output_dir + '/' + output_name + '_mask.nii')

    else:
        logging.error("le fichier doit être en format nifti (.nii)")


if (__name__ == "__main__"):
    main()
