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


def normalize(vol, percentile_min=0.0, percentile_max=99.9):
    pmin = np.percentile(vol, percentile_min, axis=(1, 2))
    pmax = np.percentile(vol, percentile_max, axis=(1, 2))
    divisor = pmax - pmin
    vol = (vol - pmin[:, None, None])
    vol[divisor > 0] = vol[divisor > 0] / np.reshape(divisor[divisor > 0], (-1, 1, 1))
    vol = np.clip(vol, 0, 1)
    return vol


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    first_mosaic, mosaics_sorted, transforms, offsets =\
        get_input(args.in_mosaics_dir, args.in_transforms_dir, parser)

    vol, res = read_omezarr(first_mosaic)
    _, nr, nc = vol.shape
    print(vol.shape[0])

    last_vol, _ = read_omezarr(mosaics_sorted[-1])

    fixed_offsets = offsets[:, 0]
    moving_offsets = offsets[:, 1]
    nz = np.sum(fixed_offsets - moving_offsets) + last_vol.shape[0]  # because we add the last volume as a whole
    output_shape = (nz, nr, nc)

    output_vol = OmeZarrWriter(args.out_stack, output_shape, vol.chunks, dtype=vol.dtype)

    # fixed_offsets[0] is where the next moving slice will start
    stack_offset = fixed_offsets[0]
    if args.normalize:
        vol = normalize(vol)
    output_vol[:stack_offset] = vol[:stack_offset]

    # assemble volume
    for i in tqdm(range(len(mosaics_sorted)), desc='Apply transforms to volume'):
        vol, res = read_omezarr(mosaics_sorted[i])
        print(vol.shape[0])
        composite_transform = sitk.CompositeTransform(transforms[i::-1])
        register_vol = apply_transform(vol, composite_transform)

        if args.normalize:
            register_vol = normalize(register_vol)

        current_moving_offset = moving_offsets[i]
        if i < len(mosaics_sorted) - 1:
            next_fixed_offset = fixed_offsets[i + 1]
        else:
            next_fixed_offset = vol.shape[0] - current_moving_offset

        print('Output vol indices:', stack_offset, stack_offset+next_fixed_offset)
        print('Register vol indices:', current_moving_offset, current_moving_offset+next_fixed_offset)
        output_vol[stack_offset:stack_offset+next_fixed_offset] =\
            register_vol[current_moving_offset:current_moving_offset+next_fixed_offset]
        stack_offset += next_fixed_offset - current_moving_offset

    output_vol.finalize(res)


if __name__ == "__main__":
    main()
