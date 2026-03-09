#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Extract one or more pyramid levels from an OME-Zarr volume as NIfTI files.

NIfTI files are saved next to the input .ome.zarr directory, named
    <zarr_stem>_level<N>_<resolution>.nii.gz

Example
-------
# List available levels:
linum_extract_pyramid_levels.py /data/3d_volume.ome.zarr --list

# Extract levels 0 and 2:
linum_extract_pyramid_levels.py /data/3d_volume.ome.zarr 0 2
"""

# Configure thread limits before numpy/scipy imports
import linumpy._thread_config  # noqa: F401

import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import zarr
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader, Multiscales

from linumpy.io.zarr import read_omezarr


def _get_pyramid_info(zarr_path: Path) -> list[dict]:
    """Return metadata for every pyramid level without loading data."""
    reader = Reader(parse_url(str(zarr_path)))
    nodes = list(reader())
    image_node = nodes[0]

    multiscale = None
    for spec in image_node.specs:
        if isinstance(spec, Multiscales):
            multiscale = spec
            break

    coord_transforms_list = image_node.metadata["coordinateTransformations"]
    n_levels = len(coord_transforms_list)

    levels = []
    for i in range(n_levels):
        scale = None
        for tr in coord_transforms_list[i]:
            if tr["type"] == "scale":
                scale = tr["scale"]
                break

        dataset_path = multiscale.datasets[i]
        arr = zarr.open_array(zarr_path / dataset_path, mode="r")
        levels.append({"index": i, "shape": arr.shape, "scale_mm": scale})

    return levels


def _resolution_tag(scale_mm: list[float]) -> str:
    """Build a compact resolution tag, e.g. '10um' or '10x10x15um' (z,y,x → x,y,z)."""
    um = [s * 1000 for s in scale_mm]
    spatial = um[-3:]  # last three axes: z, y, x
    if len({round(v, 3) for v in spatial}) == 1:
        return f"{round(spatial[0])}um"
    return "x".join(str(round(v, 1)) for v in spatial) + "um"


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument("input",
                   help="Path to an OME-Zarr pyramid directory (.ome.zarr)")
    p.add_argument("levels", nargs="*", type=int,
                   help="Pyramid level index/indices to extract (0 = finest). "
                        "Required unless --list is given.")
    p.add_argument("--list", action="store_true",
                   help="Print available pyramid levels and exit")
    return p


def main():
    p = _build_arg_parser()
    args = p.parse_args()

    zarr_path = Path(args.input)
    if not zarr_path.exists():
        p.error(f"Input not found: {zarr_path}")

    levels_info = _get_pyramid_info(zarr_path)

    if args.list:
        print(f"Pyramid levels in {zarr_path.name}:")
        for lv in levels_info:
            um = [round(s * 1000, 2) for s in lv["scale_mm"]]
            tag = _resolution_tag(lv["scale_mm"])
            print(f"  Level {lv['index']:2d}  shape {lv['shape']}  "
                  f"resolution {um} µm  ({tag})")
        return

    if not args.levels:
        p.error("Specify at least one level index, or use --list to see available levels.")

    n_available = len(levels_info)
    # Strip both .ome.zarr and bare .zarr suffixes
    stem = zarr_path.name
    for suffix in (".ome.zarr", ".zarr"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    output_dir = zarr_path.parent

    for level in args.levels:
        if level < 0 or level >= n_available:
            print(f"WARNING: Level {level} out of range (0–{n_available - 1}), skipping.")
            continue

        lv = levels_info[level]
        tag = _resolution_tag(lv["scale_mm"])
        out_path = output_dir / f"{stem}_level{level}_{tag}.nii"

        print(f"Extracting level {level} ({tag})  shape {lv['shape']} → {out_path.name}")

        vol, scale_mm = read_omezarr(str(zarr_path), level=level)
        data = np.asarray(vol, dtype=np.float32)

        # NIfTI spacing is in mm; OME-Zarr scale is already in mm.
        # SimpleITK spacing order is (x, y, z); scale_mm is (z, y, x) in OME-Zarr.
        spacing = (float(scale_mm[-1]), float(scale_mm[-2]), float(scale_mm[-3]))

        img = sitk.GetImageFromArray(data)
        img.SetSpacing(spacing)
        sitk.WriteImage(img, str(out_path))
        print(f"  Saved: {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
