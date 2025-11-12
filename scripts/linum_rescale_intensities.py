#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rescale intensities of nifti image. Works as mi-brain level window.
"""
import nibabel as nib
import numpy as np
import argparse


def _build_arg_parser():
    p = argparse.ArgumentParser(description='__doc__',
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_nifti',
                   help='Input nifti image.')
    p.add_argument('out_nifti',
                   help='Output nifti image.')
    p.add_argument('center_window', type=float,
                   help='Center of the image level window.')
    p.add_argument('width_window', type=float,
                   help='Width of the image intensity levels.')
    p.add_argument('--target_range', nargs=2, default=(0, 1), type=float,
                   help='Target range for image intensities. [%(default)s]')
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    image = nib.load(args.in_nifti)
    data = image.get_fdata()

    min_intensity = args.center_window - args.width_window / 2.0
    max_intensity = args.center_window + args.width_window / 2.0

    data = (data - min_intensity) / (max_intensity - min_intensity)
    data = np.clip(data, 0, 1)
    data = data * (args.target_range[1] - args.target_range[0]) + args.target_range[0]
    nib.save(nib.Nifti1Image(data.astype(np.float32), image.affine), args.out_nifti)


if __name__ == '__main__':
    main()
