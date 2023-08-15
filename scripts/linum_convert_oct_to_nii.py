#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Convert OCT raw binary data to nifti
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import re
from tqdm import tqdm  

from linumpy.microscope.oct import OCT

def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input_directory",
                   help="Input OCT directory. This should contrain the directories for each tiles")
    p.add_argument("output_directory",
                   help="Output directory containing nifty files")
    p.add_argument("output_extension", nargs='?', default='.nii',
                   help="output extension : .nii or .nii.gz (default=%(default)s)")
    return p

def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    input_directory = Path(args.input_directory)
    
    # Detect the folders containing the .bin for every tile
    folder_pattern = '*'  # Pattern to match all folders/directories 
    matched_folders = [folder for folder in input_directory.rglob(folder_pattern) if folder.is_dir()]
    directory_name = re.compile(r".*_z(?P<z>\d+).*")

    for folder in tqdm(matched_folders, desc="Converting"): 

        # Prepare the output directory
        b = directory_name.match(folder.name)
        output = Path(args.output_directory) / ("slice_" + b.group("z")) / (folder.name + args.output_extension)

        assert args.output_extension in [".nii", ".nii.gz"], "The output file must be a nifti file."
        output.absolute()
        output.parent.mkdir(exist_ok=True, parents=True)

        # Load the oct data
        oct = OCT()
        vol = oct.load_image(str(folder))

        # Swap axes to have XYZ instead of ZXY
        vol = np.moveaxis(vol, (0, 1, 2), (2, 0, 1))

        # Prepare the affine matrix
        res_x_um = oct.info['width'] / oct.info['nx']
        res_y_um = oct.info['height'] / oct.info['ny']
        res_z_um = 3.5  # TODO: add the axial resolution to the oct scan info file.
        affine = np.eye(4)
        affine[0, 0] = res_x_um
        affine[1, 1] = res_y_um
        affine[2, 2] = res_z_um
        
        # Save the output file
        nifti_image = nib.Nifti1Image(vol, affine)
        nifti_image.header.set_xyzt_units(xyz="micron")
        nib.save(nifti_image, str(output))

if __name__ == "__main__":
    main()
