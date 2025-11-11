#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estimate the transform aligning `in_moving` volume over `in_fixed` volume.
Finds the best shift along the stacking direction and the best 2D transform.
The output is a directory containing a transform (.mat) and a z offset (.txt)
for aligning the moving slice over the fixed slice.
"""
import argparse
import numpy as np
from linumpy.io.zarr import read_omezarr
from linumpy.utils.io import assert_output_exists, add_overwrite_arg
from linumpy.stitching.registration import register_2d_images_sitk, apply_transform
import os
import matplotlib.pyplot as plt


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_fixed',
                   help='Input stack in .ome.zarr format.')
    p.add_argument('in_moving',
                   help='Moving image in .ome.zarr format.')
    p.add_argument('out_directory',
                   help='Output directory containing transform (.mat) and offsets (.txt) for aligning\n'
                   ' the moving slice over the previous slice.')
    p.add_argument('--out_transform', default='transform.mat',
                   help='Output transform [%(default)s].')
    p.add_argument('--out_offsets', default='offsets.txt',
                   help='Output offsets along stacking axis in fixed and moving volumes [%(default)s].')

    p.add_argument('--slicing_interval', type=float, default=0.200,
                   help='Interval between slices in mm. [%(default)s]')
    p.add_argument('--allowed_drifting', type=float, default=0.050,
                   help='Allowing error in mm on slice position along the z axis.')
    p.add_argument('--moving_slice_index', type=int, default=0,
                   help='Index of the top slice (first clean slice) to use for registration.')
    p.add_argument('--transform', choices=['euler', 'affine', 'translation'], default='affine',
                   help='Registration method to use. [%(default)s]')
    p.add_argument('--metric', choices=['MSE', 'CC', 'AntsCC', 'MI'], default='MSE',
                   help='Registration metric to use. [%(default)s]')
    p.add_argument('--max_iterations', type=int, default=10000,
                   help='Maximum number of iterations. [%(default)s]')
    p.add_argument('--grad_mag_tol', type=float, default=1e-12,
                   help='Gradient magnitude tolerance for registration. [%(default)s]')
    p.add_argument('--screenshot', default=None,
                   help='Path to save a screenshot of the fixed and moving images for debugging.')
    add_overwrite_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    fixed_vol, res = read_omezarr(args.in_fixed)
    moving_vol, res = read_omezarr(args.in_moving)

    assert_output_exists(args.out_directory, parser, args)

    # create output directory
    os.makedirs(args.out_directory)

    moving_image = moving_vol[args.moving_slice_index]

    moving_image -= np.percentile(moving_image[moving_image > 0], 0.5)
    moving_image /= np.percentile(moving_image, 99.5)
    moving_image = np.clip(moving_image, 0, 1)

    # the index in fixed image which is expected to match the moving image
    interval_vox = int(np.ceil(args.slicing_interval / res[0]))  # in voxels
    allowed_drifting_vox = int(np.ceil(args.allowed_drifting / res[0]))
    expected_corresponding_index = interval_vox + args.moving_slice_index

    candidate_indices = np.arange(max(0, expected_corresponding_index - allowed_drifting_vox),
                                  min(expected_corresponding_index + allowed_drifting_vox + 1, fixed_vol.shape[0]))

    errors = []
    transforms = []
    for i in candidate_indices:
        fixed_image = fixed_vol[i]

        fixed_image -= np.percentile(fixed_image[fixed_image > 0], 0.5)
        fixed_image /= np.percentile(fixed_image, 99.5)
        fixed_image = np.clip(fixed_image, 0, 1)

        # align the slices
        transform, _, error = register_2d_images_sitk(
            fixed_image, moving_image, metric=args.metric,
            method=args.transform, max_iterations=args.max_iterations,
            grad_mag_tol=args.grad_mag_tol, return_3d_transform=True,
            verbose=False)
        errors.append(error)
        transforms.append(transform)

    best_fit_index = np.argmin(errors)
    best_candidate_index = candidate_indices[best_fit_index]
    best_transform = transforms[best_fit_index]

    best_transform.WriteTransform(os.path.join(args.out_directory, args.out_transform))
    np.savetxt(os.path.join(args.out_directory, args.out_offsets),
               np.array([best_candidate_index, args.moving_slice_index]), fmt='%d')

    out = apply_transform(moving_vol, best_transform)
    out_vol = np.zeros((best_candidate_index + moving_vol.shape[0] - args.moving_slice_index, *fixed_vol.shape[1:]),
                       dtype=fixed_vol.dtype)
    out_vol[:best_candidate_index] = fixed_vol[:best_candidate_index]
    out_vol[best_candidate_index:] = out[args.moving_slice_index:]

    if args.screenshot is not None:
        # Save the images for debugging
        fig, ax = plt.subplots(2, 2, figsize=(10, 6))
        ax[0, 0].imshow(fixed_image, cmap='gray')
        ax[0, 0].set_title(f'Slice fixed {best_candidate_index}')
        ax[0, 1].imshow(moving_image, cmap='gray')
        ax[0, 1].set_title(f'Slice moving {args.moving_slice_index}')
        ax[1, 0].imshow(out_vol[:, fixed_vol.shape[1] // 2, :], cmap='gray')
        ax[1, 0].set_title('Aligned slices (second axis)')
        ax[1, 1].imshow(out_vol[:, :, fixed_vol.shape[2] // 2], cmap='gray')
        ax[1, 1].set_title('Aligned slices (third axis)')
        fig.tight_layout()
        fig.savefig(args.screenshot, dpi=400)
        plt.close(fig)


if __name__ == '__main__':
    main()
