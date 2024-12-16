#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import SimpleITK as sitk
from pathlib import Path

def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input_folder",
                   help="Full path to a folder containing TIFF images")
    p.add_argument("output_folder",
                   help="Full path to the output folder which will contain the nifti (.nii.gz) images")
    return p

def main():
    # Parse arguments
    p = _build_arg_parser()
    args = p.parse_args()

    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)

    output_folder.mkdir(exist_ok=True, parents=True)

    # List the TIFF files in the input folder
    tiff_files = [file for file in input_folder.glob('*') if file.suffix in [".tif", ".tiff",".TIF",".TIFF"] ]
    for tiff_file in tiff_files:
        
        # Create the output path
        output_nifti_file = output_folder / Path(tiff_file.stem + '.nii.gz')

        # Load the TIFF image
        image = sitk.ReadImage(tiff_file)

        # Save the image as a nifti file
        sitk.WriteImage(image, output_nifti_file)

if __name__ == "__main__":
    main()

