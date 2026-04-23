"""
Mosaic grid resampling utilities.

Consolidated from linum_resample_mosaic_grid.py.
"""

from typing import Any

import numpy as np


def resolution_is_mm(source_res: tuple | list) -> bool:
    """Heuristic: source resolution in mm if all components < 1, otherwise µm.

    Used across the pipeline to accept either unit in OME-Zarr metadata or
    CLI arguments without breaking legacy data. Pixel sizes below 1 µm would
    imply sub-nanometre voxels, so the heuristic is safe for all realistic
    acquisitions.
    """
    return float(source_res[0]) < 1.0


def resample_mosaic_grid(
    vol: Any,
    source_res: tuple | list,
    target_res_um: float,
    n_levels: int = 5,
    out_path: str | None = None,
) -> np.ndarray | None:
    """Resample a mosaic grid volume to a target isotropic resolution.

    Processes tiles individually to avoid loading the entire mosaic into memory.
    Uses anti-aliasing and 1st-order interpolation.

    Parameters
    ----------
    vol : dask array or zarr array
        Mosaic grid volume with chunk structure (each chunk = one tile).
        Shape: (Z, nx*tile_h, ny*tile_w)
    source_res : tuple
        Source resolution (res_z, res_y, res_x) in whatever unit.
    target_res_um : float
        Target isotropic resolution in microns.
    n_levels : int
        Number of pyramid levels in output.
    out_path : str or None
        If provided, save the result to this OME-Zarr path.

    Returns
    -------
    np.ndarray or None
        Resampled array if out_path is None, else None (writes to file).
    """
    from skimage.transform import rescale

    tile_shape = vol.chunks if hasattr(vol, "chunks") else None
    if tile_shape is None:
        raise ValueError("vol must have a 'chunks' attribute (dask or zarr array)")

    # Convert target resolution to same unit as source_res
    target_res = target_res_um / 1000.0 if resolution_is_mm(source_res) else float(target_res_um)

    scaling_factor = np.asarray(source_res) / target_res
    tile_00 = np.array(vol[: tile_shape[0], : tile_shape[1], : tile_shape[2]])
    out_tile_00 = rescale(tile_00, scaling_factor, order=1, preserve_range=True, anti_aliasing=True)
    out_tile_shape = out_tile_00.shape

    nx = vol.shape[1] // tile_shape[1]
    ny = vol.shape[2] // tile_shape[2]
    out_shape = (out_tile_shape[0], nx * out_tile_shape[1], ny * out_tile_shape[2])

    if out_path is not None:
        import itertools

        from linumpy.io.zarr import OmeZarrWriter

        out_zarr = OmeZarrWriter(out_path, out_shape, out_tile_shape, dtype=vol.dtype, overwrite=True)
        out_zarr[: out_tile_shape[0], : out_tile_shape[1], : out_tile_shape[2]] = out_tile_00
        for i, j in itertools.product(range(nx), range(ny)):
            if i == 0 and j == 0:
                continue  # already written
            current_vol = np.array(
                vol[:, i * tile_shape[1] : (i + 1) * tile_shape[1], j * tile_shape[2] : (j + 1) * tile_shape[2]]
            )
            out_zarr[
                :, i * out_tile_shape[1] : (i + 1) * out_tile_shape[1], j * out_tile_shape[2] : (j + 1) * out_tile_shape[2]
            ] = rescale(current_vol, scaling_factor, order=1, preserve_range=True, anti_aliasing=True)

        out_res = [target_res] * 3 if resolution_is_mm(source_res) else [target_res_um] * 3
        out_zarr.finalize(out_res, n_levels)
        return None
    else:
        import itertools

        result = np.zeros(out_shape, dtype=np.float32)
        result[: out_tile_shape[0], : out_tile_shape[1], : out_tile_shape[2]] = out_tile_00
        for i, j in itertools.product(range(nx), range(ny)):
            if i == 0 and j == 0:
                continue
            current_vol = np.array(
                vol[:, i * tile_shape[1] : (i + 1) * tile_shape[1], j * tile_shape[2] : (j + 1) * tile_shape[2]]
            )
            result[
                :, i * out_tile_shape[1] : (i + 1) * out_tile_shape[1], j * out_tile_shape[2] : (j + 1) * out_tile_shape[2]
            ] = rescale(current_vol, scaling_factor, order=1, preserve_range=True, anti_aliasing=True)
        return result
