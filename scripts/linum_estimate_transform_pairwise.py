#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
import argparse
import numpy as np
from linumpy.io.zarr import read_omezarr
from linumpy.stitching.registration import register_2d_images_sitk, apply_transform

import matplotlib.pyplot as plt


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_stack',
                   help='Input stack in .ome.zarr format.')
    p.add_argument('in_offsets',
                   help='Input offsets file (.npy) with z offsets for each slice.')
    p.add_argument('moving_index', type=int,
                   help='Index of the moving slice.')
    p.add_argument('out_transform',
                   help='Output transform to align the moving slice over the previous slice (.mat).')
    p.add_argument('--first_slice_index', type=int, default=0,
                   help='Index of the first slice in the stack. [%(default)s]')
    p.add_argument('--slicing_interval', type=float, default=0.200,
                   help='Interval between slices in mm. [%(default)s]')
    p.add_argument('--factor_extra', type=float, default=1.1,
                   help='Factor by which to increase the slicing interval. [%(default)s]')
    p.add_argument('--method', choices=['euler', 'affine', 'translation'], default='euler',
                   help='Registration method to use. [%(default)s]')
    p.add_argument('--metric', choices=['MSE', 'CC', 'AntsCC', 'MI'], default='MSE',
                   help='Registration metric to use. [%(default)s]')
    p.add_argument('--max_iterations', type=int, default=10000,
                   help='Maximum number of iterations. [%(default)s]')
    p.add_argument('--grad_mag_tol', type=float, default=1e-12,
                   help='Gradient magnitude tolerance for registration. [%(default)s]')
    p.add_argument('--screenshot', default=None,
                   help='Path to save a screenshot of the fixed and moving images for debugging.')
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    vol, res = read_omezarr(args.in_stack)

    interval_vox = int(np.ceil(args.slicing_interval / res[0]))  # in voxels
    n_overlap_vox = int(interval_vox * (args.factor_extra - 1.0))  # in voxels

    z_offsets_source = np.load(args.in_offsets)
    n_slices = len(z_offsets_source)

    moving_slice_idx = args.moving_index - args.first_slice_index
    # fixed index is the slice before the moving slice
    fixed_slice_idx = moving_slice_idx - 1

    if fixed_slice_idx < 0 or fixed_slice_idx >= n_slices - 1:
        parser.error(f"Fixed index {args.fixed_index} is out of bounds for the number of slices {n_slices}.")

    fixed_vol = vol[z_offsets_source[fixed_slice_idx]:z_offsets_source[moving_slice_idx]]

    out_depth = len(fixed_vol) - n_overlap_vox
    moving_vol = vol[z_offsets_source[moving_slice_idx]:z_offsets_source[moving_slice_idx] + out_depth]

    # modify these lines to use top and bottom slices
    # allow for keeping overlap
    fixed_image = fixed_vol[-n_overlap_vox:].mean(axis=0)
    moving_image = moving_vol[:n_overlap_vox].mean(axis=0)

    # image intensity normalization
    fixed_image -= np.percentile(fixed_image[fixed_image > 0], 0.5)
    fixed_image /= np.percentile(fixed_image, 99.5)
    fixed_image = np.clip(fixed_image, 0, 1)

    # moving_image = moving_vol.mean(axis=0)
    moving_image -= np.percentile(moving_image[moving_image > 0], 0.5)
    moving_image /= np.percentile(moving_image, 99.5)
    moving_image = np.clip(moving_image, 0, 1)

    transform, stop_condition = register_2d_images_sitk(fixed_image, moving_image,
                                                        metric=args.metric,
                                                        method=args.method,
                                                        max_iterations=args.max_iterations,
                                                        grad_mag_tol=args.grad_mag_tol,
                                                        return_3d_transform=True)
    print(f"Stop condition: {stop_condition}")
    transform.WriteTransform(args.out_transform)

    out = apply_transform(moving_vol, transform)
    out_vol = np.zeros((2*out_depth, *fixed_vol.shape[1:]),
                       dtype=fixed_vol.dtype)
    out_vol[:out_depth] = fixed_vol[:out_depth]
    out_vol[out_depth:] = out

    if args.screenshot is not None:
        # Save the images for debugging
        fig, ax = plt.subplots(2, 2, figsize=(10, 8))
        ax[0, 0].imshow(fixed_image, cmap='gray')
        ax[0, 0].set_title(f'Slice {fixed_slice_idx} (fixed)')
        ax[0, 1].imshow(moving_image, cmap='gray')
        ax[0, 1].set_title(f'Slice {moving_slice_idx} (moving)')
        ax[1, 0].imshow(out_vol[:, fixed_vol.shape[1] // 2, :], cmap='gray')
        ax[1, 0].set_title('Aligned slices (second axis)')
        ax[1, 1].imshow(out_vol[:, :, fixed_vol.shape[2] // 2], cmap='gray')
        ax[1, 1].set_title('Aligned slices (third axis)')
        fig.savefig(args.screenshot, dpi=400)
        plt.close(fig)


if __name__ == '__main__':
    main()
