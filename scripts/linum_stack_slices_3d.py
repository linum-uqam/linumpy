#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stack 3D mosaics on top of each other in a single 3D volume using the
transforms from `linum_estimate_transform_pairwise.py`. Expects all 3D
mosaics to be in the same space (same dimensions for last two axes).
"""
import argparse
import re
from pathlib import Path
import numpy as np
from linumpy.io.zarr import read_omezarr, OmeZarrWriter
from linumpy.stitching.registration import apply_transform
from linumpy.utils.mosaic_grid import getDiffusionBlendingWeights
from skimage.filters import threshold_otsu
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import os

import SimpleITK as sitk


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_mosaics_dir',
                   help='Input mosaics directory in .ome.zarr format.')
    p.add_argument('in_transforms_dir',
                   help='Input transforms directory. Each subdirectory should have the\n'
                   'same name as the corresponding mosaic file (without the .ome.zarr\n'
                   'extension) and contain a .mat transform file and .txt offsets file.')
    p.add_argument('out_stack',
                   help='Output stack in .ome.zarr format.')
    p.add_argument('--normalize', action='store_true',
                   help='Normalize slices during reconstruction.')
    p.add_argument('--blend', action='store_true',
                   help='Use diffusion method for blending consecutive slices.')
    p.add_argument('--overlap', type=int,
                   help='Number of overlapping voxels to keep from bottom of previous mosaic.\n'
                        'By default keeps all.')
    return p


def get_input(mosaics_dir, transforms_dir, parser):
    # get all .ome.zarr files in in_mosaics_dir
    in_mosaics_dir = Path(mosaics_dir)
    in_transforms_dir = Path(transforms_dir)
    mosaics_files = [p for p in in_mosaics_dir.glob('*.ome.zarr')]
    pattern = r".*z(\d+)_.*"
    slice_ids = []
    for f in mosaics_files:
        foo = re.match(pattern, f.name)
        slice_id = int(foo.groups()[0])
        slice_ids.append(slice_id)

    transforms = []
    offsets = []
    mosaics_sorted = []
    slice_ids_argsort = np.argsort(slice_ids)
    first_mosaic = mosaics_files[slice_ids_argsort[0]]
    for arg_idx in slice_ids_argsort[1:]:
        f = mosaics_files[arg_idx]
        current_transform_dirname, ext = os.path.splitext(f.name)
        while not ext == '':  # remove all trailing extensions
            current_transform_dirname, ext = os.path.splitext(current_transform_dirname)
        current_transform_dir = in_transforms_dir / current_transform_dirname

        if not os.path.exists(current_transform_dir):
            parser.error(f'Transform {current_transform_dir} not found.')

        current_mat_file = list(current_transform_dir.glob('*.mat'))
        current_txt_file = list(current_transform_dir.glob('*.txt'))
        if len(current_mat_file) != 1:
            parser.error(f'Found {len(current_mat_file)} .mat file under {current_transform_dir.as_posix()}')
        current_mat_file = current_mat_file[0]
        if len(current_txt_file) > 1:
            parser.error(f'Found {len(current_txt_file)} .txt file under {current_transform_dir.as_posix()}')
        current_txt_file = current_txt_file[0]
        mosaics_sorted.append(f)
        transforms.append(sitk.ReadTransform(current_mat_file))
        offsets.append(np.loadtxt(current_txt_file))
    return first_mosaic, mosaics_sorted, transforms, np.array(offsets, dtype=int)


def get_agarose_mask(vol):
    reference = np.mean(vol, axis=0)
    reference_smooth = gaussian_filter(reference, sigma=1.0)
    threshold = threshold_otsu(reference_smooth[reference > 0])

    # voxels in mask are expected to be agarose voxels
    agarose_mask = np.logical_and(reference_smooth < threshold, reference > 0)
    return agarose_mask


def normalize(vol, percentile_max=99.9):
    # voxels in mask are expected to be agarose voxels
    agarose_mask = get_agarose_mask(vol)

    pmax = np.percentile(vol, percentile_max, axis=(1, 2))
    vol = np.clip(vol, None, pmax[:, None, None])

    background_thresholds = []
    for curr_slice in vol:
        agarose = curr_slice[agarose_mask]
        bg_median = np.median(agarose)
        background_thresholds.append(bg_median)

    background_thresholds = np.array(background_thresholds)
    vol = np.clip(vol, background_thresholds[:, None, None], None)

    # rescale
    vol = vol - np.min(vol, axis=(1, 2), keepdims=True)
    vmax = np.max(vol, axis=(1, 2))
    vol[vmax > 0] = vol[vmax > 0] / vmax[:, None, None]
    return vol


def get_tissue_mask(vol):
    vol_smooth = gaussian_filter(vol, sigma=(0.0, 1.0, 1.0))
    mask = vol_smooth > np.percentile(vol_smooth, 10)

    return mask


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    first_mosaic, mosaics_sorted, transforms, offsets =\
        get_input(args.in_mosaics_dir, args.in_transforms_dir, parser)

    vol, res = read_omezarr(first_mosaic)
    _, nr, nc = vol.shape

    last_vol, _ = read_omezarr(mosaics_sorted[-1])

    fixed_offsets = offsets[:, 0] - offsets[:, 1]
    nz = np.sum(fixed_offsets) + last_vol.shape[0]  # because we add the last volume as a whole
    output_shape = (nz, nr, nc)

    output_vol = OmeZarrWriter(args.out_stack, output_shape, vol.chunks, dtype=vol.dtype)

    if args.normalize:
        vol = normalize(vol)
        if args.overlap is not None:
            vol = vol[:fixed_offsets[0]+args.overlap]
    output_vol[:vol.shape[0]] = vol[:]

    # fixed_offsets[0] is where the next moving slice will start
    stack_offset = fixed_offsets[0]

    # assemble volume
    for i in tqdm(range(len(mosaics_sorted)), desc='Apply transforms to volume'):
        vol, res = read_omezarr(mosaics_sorted[i])
        composite_transform = sitk.CompositeTransform(transforms[i::-1])
        register_vol = apply_transform(vol, composite_transform)

        # crop the volume at next fixed offset + overlap
        if i < len(mosaics_sorted) - 1:
            next_fixed_offset = fixed_offsets[i + 1]
            if args.overlap is not None:
                register_vol = register_vol[:next_fixed_offset+args.overlap]
        else:
            next_fixed_offset = register_vol.shape[0]

        if args.normalize:
            register_vol = normalize(register_vol)

        if args.blend:
            blending_mask_fixed = get_tissue_mask(output_vol[stack_offset:stack_offset+register_vol.shape[0]])
            blending_mask_moving = get_tissue_mask(register_vol)

            alphas = getDiffusionBlendingWeights(blending_mask_fixed, blending_mask_moving, factor=2)
        else:
            alphas = 1

        output_vol[stack_offset:stack_offset+register_vol.shape[0]] =\
            (1-alphas)*output_vol[stack_offset:stack_offset+register_vol.shape[0]]+(alphas)*register_vol[:]
        stack_offset += next_fixed_offset

    output_vol.finalize(res)


if __name__ == "__main__":
    main()
