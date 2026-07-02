#!/usr/bin/env python3
from pathlib import Path
from typing import Any, cast

import zarr

from linumpy.io.test_data import get_data
from linumpy.microscope.oct import OCT


def _assert_omezarr_zyx_contract(zarr_path: Path, tile_dir: Path) -> None:
    """Output mosaic zarr follows (Z, Y, X) shape and resolution ordering."""
    oct = OCT(tile_dir)
    vol = oct.load_image(crop=True, fix_galvo_shift=False, fix_camera_shift=False)
    tile_shape_zyx = vol.shape
    tile_res_zyx = oct.resolution

    root = zarr.open_group(zarr_path, mode="r")
    mosaic = root["0"]
    assert len(mosaic.shape) == 3
    assert mosaic.shape[0] == tile_shape_zyx[0]

    ome_meta = cast(dict[str, Any], root.attrs["ome"])
    multiscales = cast(list[dict[str, Any]], ome_meta["multiscales"])
    axes = [axis["name"] for axis in multiscales[0]["axes"]]
    assert axes == ["z", "y", "x"]
    scale = multiscales[0]["datasets"][0]["coordinateTransformations"][0]["scale"]
    assert len(scale) == 3
    assert scale[0] == tile_res_zyx[0]
    assert scale[1] == tile_res_zyx[1]
    assert scale[2] == tile_res_zyx[2]


def test_help(script_runner):
    ret = script_runner.run(["linum-create-mosaic-grid-3d", "--help"])
    assert ret.success


def test_execution_from_directory(script_runner, tmp_path):
    input = get_data("raw_tiles")
    output = tmp_path / "output.ome.zarr"
    ret = script_runner.run(
        [
            "linum-create-mosaic-grid-3d",
            output,
            "--from_root_directory",
            input,
            "-z",
            0,
            "-r",
            -1,
            "--no-preprocess",
        ]
    )
    assert ret.success
    _assert_omezarr_zyx_contract(output, Path(input) / "tile_x00_y00_z00")


def test_execution_from_list(script_runner, tmp_path):
    input = get_data("raw_tiles")
    input_path = Path(input)
    output = tmp_path / "output.ome.zarr"
    ret = script_runner.run(
        [
            "linum-create-mosaic-grid-3d",
            output,
            "--from_tiles_list",
            str(input_path / "tile_x00_y00_z01"),
            str(input_path / "tile_x01_y00_z01"),
            str(input_path / "tile_x00_y01_z01"),
            str(input_path / "tile_x01_y01_z01"),
            "-r",
            -1,
            "--no-preprocess",
        ]
    )
    assert ret.success
    _assert_omezarr_zyx_contract(output, input_path / "tile_x00_y00_z01")
