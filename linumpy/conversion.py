import os
import shutil
from typing import List, Tuple, Any, Callable, Union

import dask.array as da
import numpy as np
import zarr
from numpy import ndarray
from ome_zarr.dask_utils import resize as dask_resize
from ome_zarr.io import parse_url
from ome_zarr.scale import Scaler, ArrayLike
from ome_zarr.writer import write_image
from skimage.transform import resize
from pathlib import Path

base_key = "reconstructed_frame"

""" 
    This file contains functions for converting between different file formats.
"""


class CustomScaler(Scaler):
    """
    A custom scaler that can downsample 3D images for OME-Zarr Conversion
    """

    def __init__(self, downscale=2, method="nearest"):
        super().__init__(downscale=downscale, method=method)

    def nearest(self, base: np.ndarray) -> List[np.ndarray]:
        """
        Downsample using :func:`skimage.transform.resize`.
        """
        return self._by_plane(base, self.__nearest)

    def __nearest(self, plane: ArrayLike, sizeY: int, sizeX: int) -> np.ndarray:
        """Apply the 2-dimensional transformation."""
        if isinstance(plane, da.Array):

            def _resize(
                    image: ArrayLike, output_shape: Tuple, **kwargs: Any
            ) -> ArrayLike:
                return dask_resize(image, output_shape, **kwargs)

        else:
            _resize = resize

        return _resize(
            plane,
            output_shape=(
                plane.shape[0] // self.downscale, plane.shape[1] // self.downscale, plane.shape[2] // self.downscale),
            order=1,
            preserve_range=True,
            anti_aliasing=True,
        ).astype(plane.dtype)

    def _by_plane(
            self,
            base: np.ndarray,
            func: Callable[[np.ndarray, int, int], np.ndarray],
    ) -> list[Union[ndarray, ndarray, None]]:
        """Loop over 3 of the 5 dimensions and apply the func transform."""

        rv = [base]
        for i in range(self.max_layer):
            stack_to_scale = rv[-1]
            shape_5d = (*(1,) * (5 - stack_to_scale.ndim), *stack_to_scale.shape)
            T, C, Z, Y, X = shape_5d

            # If our data is already 2D, simply resize and add to pyramid
            if stack_to_scale.ndim == 2:
                rv.append(func(stack_to_scale, Y, X))
                continue

            # stack_dims is any dims over 3D
            new_stack = None
            for t in range(T):
                for c in range(C):
                    plane = stack_to_scale[:]
                    out = func(plane, Y, X)
                    # first iteration of loop creates the new nd stack
                    if new_stack is None:
                        new_stack = np.zeros(
                            (out.shape[0], out.shape[1], out.shape[2]),
                            dtype=base.dtype,
                        )
                    # insert resized plane into the stack at correct indices
                    new_stack[:] = out
            rv.append(new_stack)
        return rv


def create_transformation_dict(scales, levels):
    """
    Create a dictionary with the transformation information for 3D images.

    :param scales: The scale of the image, in z y x order.
    :param levels: The number of levels in the pyramid.
    :return:
    """
    coord_transforms = []
    for i in range(levels):
        transform_dict = [{
            "type": "scale",
            "scale": [scales[0] * (2 ** i), scales[1] * (2 ** i), scales[2] * (2 ** i)]
        }]
        coord_transforms.append(transform_dict)
    return coord_transforms


def generate_axes_dict():
    """
    Generate the axes dictionary for the zarr file.

    :return: The axes dictionary
    """
    axes = [
        {"name": "x", "type": "space", "unit": "micrometer"},
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "z", "type": "space", "unit": "micrometer"}
    ]
    return axes
def save_zarr(data, zarr_file, scales=(6.5, 6.5, 6.5), chunks=(32, 32, 32), n_levels=5, overwrite=True):
    """
    Save a numpy array to a zarr file.
    :param data: The data to save.
    :param zarr_file: The zarr file to save to.
    :param scales: The resolution of the image, in z y x order.
    :param chunks: The chunk size to use.
    :return:
    """
    print("Saving ome-zarr...")
    directory = Path(zarr_file)
    if directory.exists():
        if overwrite:
            shutil.rmtree(directory)
        else:
            raise Exception("Directory already exists, and overwrite is False!")

    directory.mkdir(exist_ok=overwrite, parents=True)
    store = parse_url(zarr_file, mode="w").store

    root = zarr.group(store=store)
    write_image(image=data, group=root, axes=generate_axes_dict(),
                coordinate_transformations=create_transformation_dict(scales, n_levels),
                storage_options=dict(chunks=chunks), scaler=CustomScaler(downscale=2, method="nearest"))
    print("Done!")
