"""Utilities for loading test data."""

from pathlib import Path
from typing import Any

import dask.array as da
import nibabel as nib
import numpy as np
from skimage.data import cells3d

from linumpy import LINUMPY_HOME
from linumpy.io.zarr import save_omezarr


def get_data(name: str) -> Any:
    """Return test data for the given key name."""
    data = {
        "mosaic_3d_omezarr": _get_mosaic_3d_omezarr,
        "mosaic_3d_nifti": _get_mosaic_3d_nifti,
        "raw_tiles": _get_raw_tiles,
        "aip": _get_aip,
    }
    if name not in data:
        raise ValueError(f"Unknown key for data: {name}")
    return data[name]()


def _create_linumpy_home_if_not_exists() -> None:
    Path(LINUMPY_HOME).mkdir(parents=True, exist_ok=True)


def _get_mosaic_3d_nifti() -> Path:
    _create_linumpy_home_if_not_exists()
    filename = Path(LINUMPY_HOME) / "mosaic_3d.nii.gz"
    if not filename.exists():
        # create test data
        data = np.mean(cells3d(), axis=1)  # (60, 256, 256)

        img = nib.Nifti1Image(data, np.eye(4))
        img.header.set_xyzt_units(xyz="micron")
        nib.save(img, filename)
    return filename


def _get_mosaic_3d_omezarr() -> Path:
    _create_linumpy_home_if_not_exists()
    filename = Path(LINUMPY_HOME) / "mosaic_3d.ome.zarr"
    if not filename.exists():
        # create test data
        data = np.mean(cells3d(), axis=1)  # (60, 256, 256)
        data = data[:5, :, :]

        dask_array = da.from_array(data)
        save_omezarr(dask_array, filename, chunks=(5, 32, 32), n_levels=5, overwrite=False)
    return filename


def _get_aip() -> Path:
    _create_linumpy_home_if_not_exists()
    filename = Path(LINUMPY_HOME) / "aip.ome.zarr"
    if not filename.exists():
        # create test data
        data = np.mean(cells3d(), axis=(0, 1))  # (256, 256)

        dask_array = da.from_array(data)
        save_omezarr(
            dask_array,
            filename,
            voxel_size=(0.001, 0.001),
            chunks=(32, 32),
            n_levels=3,
            overwrite=False,
        )
    return filename


def _get_scan_info(
    nx: int,
    ny: int,
    top_z: int,
    bottom_z: int,
    width_mm: float,
    height_mm: float,
    x_pos_mm: float,
    y_pos_mm: float,
    z_pos_mm: float,
) -> str:
    focus_z = int((top_z + bottom_z) / 2)
    scan_info = "Scan info\n"
    scan_info += f"nx: {nx}\n"
    scan_info += f"ny: {ny}\n"
    scan_info += "n_repeat: 1\n"
    scan_info += f"width: {width_mm}\n"
    scan_info += f"height: {height_mm}\n"
    scan_info += "n_extra: 0\n"
    scan_info += "line_rate: 80\n"
    scan_info += "exposure: 23\n"
    scan_info += "alinerepeat: 1\n"
    scan_info += f"top_z: {top_z}\n"
    scan_info += f"bottom_z: {bottom_z}\n"
    scan_info += f"focus_z: {focus_z}\n"
    scan_info += f"stage_x_pos_mm: {x_pos_mm}\n"
    scan_info += f"stage_y_pos_mm: {y_pos_mm}\n"
    scan_info += f"stage_z_pos_mm: {z_pos_mm}\n"
    return scan_info


def _get_raw_tiles() -> Path:
    _create_linumpy_home_if_not_exists()
    folder = Path(LINUMPY_HOME) / "raw_tiles"
    bounds_xy = [(0, 140), ((256 - 140), 256)]
    bounds_z = [(0, 35), ((60 - 35), 60)]
    if not folder.exists():
        folder.mkdir(parents=True)
        data = np.mean(cells3d(), axis=1)  # (order z, y, x)
        for zi, (zmin, zmax) in enumerate(bounds_z):
            for yi, (ymin, ymax) in enumerate(bounds_xy):
                for xi, (xmin, xmax) in enumerate(bounds_xy):
                    tile_folder = folder / f"tile_x0{xi}_y0{yi}_z0{zi}"
                    tile_folder.mkdir(parents=True)
                    tile_xyz = data[zmin:zmax, ymin:ymax, xmin:xmax]
                    tile_xyz = tile_xyz[:, ::-1, ::-1]
                    tile_xyz[:, 0, 0] = 2.0 * np.max(tile_xyz)
                    tile_xyz.astype(np.float32).reshape(-1, order="F").tofile(tile_folder / "image_00000.bin")
                    nx = width = xmax - xmin
                    ny = height = ymax - ymin
                    top_z = zmin
                    bottom_z = zmax - 1
                    stage_x_pos = xmin
                    stage_y_pos = ymin
                    stage_z_pos = zmin
                    info = _get_scan_info(
                        nx,
                        ny,
                        top_z,
                        bottom_z,
                        width,
                        height,
                        stage_x_pos,
                        stage_y_pos,
                        stage_z_pos,
                    )
                    with (tile_folder / "info.txt").open("w") as f:
                        f.writelines(info)
    return folder
