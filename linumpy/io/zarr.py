import shutil
from pathlib import Path
import numpy as np

import dask.array as da
import zarr
from ome_zarr.dask_utils import resize as dask_resize
from ome_zarr.io import parse_url
from ome_zarr.scale import Scaler
from ome_zarr.writer import write_image
from skimage.transform import resize

"""
    This file contains functions for working with zarr files
"""


class CustomScaler(Scaler):
    """
    Custom ome_zarr.scale.Scaler class for handling downscaling up to 3D.

    Only `resize_image` method is implemented. Interpolation is ALWAYS done
    using 1-st order (linear) interpolation.
    """
    def resize_image(self, image):
        """
        Resize a numpy array OR a dask array to a smaller array (not pyramid).

        This method is the only method from the Scaler class called from
        `ome_zarr.writer.write_image`.
        """
        if isinstance(image, da.Array):

            def _resize(image, out_shape, **kwargs):
                return dask_resize(image, out_shape, **kwargs)

        else:
            _resize = resize

        # downsample in X, Y, and Z.
        new_shape = list(image.shape)
        new_shape[-1] = image.shape[-1] // self.downscale
        new_shape[-2] = image.shape[-2] // self.downscale
        if len(new_shape) > 2:
            new_shape[-3] = image.shape[-3] // self.downscale
        out_shape = tuple(new_shape)

        dtype = image.dtype
        if np.iscomplexobj(image):
            image = _resize(
                image.real.astype(float),
                out_shape,
                order=1,
                mode="reflect",
                anti_aliasing=False,
            ) + 1j * _resize(
                image.imag.astype(float),
                out_shape,
                order=1,
                mode="reflect",
                anti_aliasing=False,
            )
        else:
            image = _resize(
                image.astype(float),
                out_shape,
                order=1,
                mode="reflect",
                anti_aliasing=False,
            )
        return image.astype(dtype)

    def linear(self, base):
        """
        Downsample using :func:`skimage.transform.resize`
        with linear interpolation.
        """
        pyramid = [base]
        max_axes_resize = min(len(base.shape), 3)
        level = self.max_layer
        while level > 0 and np.all(np.asarray(pyramid[-1].shape[:-max_axes_resize])
                                   >= self.downscale):
            pyramid.append(self.resize_image(pyramid[-1]))
            level -= 1
        return pyramid

    def _by_plane(self, base, func):
        # This method is called by base class when interpolation methods (e.g. nearest)
        # are called directly. Because `write_image` never call these methods, we don't
        # need to implement it here. We raise an error to make sure the CustomScaler class
        # is not used for this purpose.
        raise NotImplementedError("_by_plane method not implemented for CustomScaler")


def create_transformation_dict(nlevels, voxel_size, ndims=3):
    """
    Create a dictionary with the transformation information for
    images up to 4 dimensions.

    :type nlevels: int
    :param nlevels: The number of levels in the pyramid.
    :type voxel_size: tuple
    :param voxel_size: The voxel size of the dataset.
    :type ndims: int
    :param ndims: The number of dimensions of the dataset.
    :type coord_transforms: list of Dict
    :return coord_transforms: List of coordinate transformations
    """
    def _get_scale(level, ndims):
        scale_def = [1.0,
                     (voxel_size[0]*2.0**level),
                     (voxel_size[1]*2.0**level),
                     (voxel_size[2]*2.0**level)]
        offset = len(scale_def) - ndims
        return scale_def[offset:]

    coord_transforms = []
    for i in range(nlevels):
        transform_dict = [{
            "type": "scale",
            "scale": _get_scale(i, ndims)
        }]
        coord_transforms.append(transform_dict)
    return coord_transforms


def generate_axes_dict(ndims=3):
    """
    Generate the axes dictionary for up to 4 dimensions.

    Dimensions are returned in order (c, z, y, x).

    :type ndims: int
    :param ndims: Number of dimensions.

    :type axes: list of Dict
    :return axes: The axes dictionary.
    """
    axes = [
        {"name": "c", "type": "channel"},
        {"name": "z", "type": "space", "unit": "millimeter"},
        {"name": "y", "type": "space", "unit": "millimeter"},
        {"name": "x", "type": "space", "unit": "millimeter"}
    ]
    offset = len(axes) - ndims
    return axes[offset:]


def create_directory(store_path, overwrite=False):
    directory = Path(store_path)
    if directory.exists():
        if overwrite:
            shutil.rmtree(directory)
        else:
            raise FileExistsError('Directory {} already exists. '
                                  'Set overwrite=True to overwrite.'
                                  .format(directory.as_posix()))
    directory.mkdir(parents=True)
    return directory


def save_omezarr(data, store_path, voxel_size=(1e-3, 1e-3, 1e-3),
                 chunks=(128, 128, 128), n_levels=5, overwrite=True):
    """
    Save numpy array to disk in zarr format following OME-NGFF file specifications.
    Expected ordering for axes in `data` and `scales` is `(c, z, y, x)`. Does not
    support saving for multi-channel 2D images with axes (c, y, x).

    :type data: numpy or dask array
    :param data: numpy or dask array to save as zarr.
    :type store_path: str
    :param store_path: The path of the output zarr group.
    :type voxel_size: tuple of n `float`, with n the number of dimensions.
    :param voxel_size: Voxel size in mm.
    :type chunks: tuple of n `int`, with n the number of dimensions.
    :param chunks: Chunk size on disk.
    :type n_levels: int
    :param n_levels: Number of levels in Gaussian pyramid.
    :type overwrite: bool
    :param overwrite: Overwrite `store_path` if it already exists.

    :type zarr_group: zarr.hierarchy.group
    :return zarr_group: Resulting zarr group saved to disk.
    """
    # pyramidal decomposition (ome_zarr.scale.Scaler) keywords
    pyramid_kw = {"max_layer": n_levels,
                  "method": "linear",
                  "downscale": 2}
    ndims = len(data.shape)

    # metadata describes the downsampling method used for generating
    # multiscale data representation (see also type in write_image)
    metadata = {"method": "ome_zarr.scale.Scaler",
                "version": "0.5",
                "args": pyramid_kw}

    # axes and coordinate transformations
    axes = generate_axes_dict(ndims)
    coordinate_transformations = create_transformation_dict(n_levels+1, voxel_size=voxel_size, ndims=ndims)

    # create directory for zarr storage
    create_directory(store_path, overwrite)
    store = parse_url(store_path, mode='w').store
    zarr_group = zarr.group(store=store)

    # the base transformation is applied to all levels of the pyramid
    # and describes the original voxel size of the dataset
    base_coord_transformation = [
        {"type": "scale", "scale": [1, 1, 1]}
    ]
    write_image(data, zarr_group, storage_options=dict(chunks=chunks),
                scaler=CustomScaler(**pyramid_kw),
                axes=axes, coordinate_transformations=coordinate_transformations,
                compute=True, metadata=metadata, type="linear",
                coordinateTransformations=base_coord_transformation)

    # return zarr group containing saved data
    return zarr_group


def read_omezarr(zarr_path, level=0):
    """
    Read omezarr image at `zarr_path` and loads image data for `level` level
    in the pyramid. Also returns voxel size for chosen level.

    :type zarr_path: str
    :param zarr_path: Path of OME-zarr file to load.
    :type level: int >= 0
    :param level: The level of the pyramid to load (0 is full resolution data).

    :type vol: zarr.array
    :return vol: Requested zarr array.
    :type res: tuple (3,)
    :return res: Voxel size of zarr array.
    """
    omezarr = zarr.open(zarr_path, mode='r')
    if "multiscales" not in omezarr.attrs:
        raise ValueError(f'Missing "multiscales" field for file {zarr_path}.')
    multiscales_attrs = omezarr.attrs["multiscales"][0]

    # res = omezarr.attrs["multiscales"][0]["coordinateTransformations"][0]["scale"]
    resolution = np.ones(len(multiscales_attrs["axes"]),)
    if "coordinateTransformations" in multiscales_attrs:
        base_coord_transform = multiscales_attrs["coordinateTransformations"]
        for transform in base_coord_transform:
            if "scale" in transform:
                resolution *= np.asarray(transform["scale"])[-3:]

    vol_header = multiscales_attrs['datasets'][level]
    if "coordinateTransformations" in vol_header:
        level_transform = vol_header["coordinateTransformations"]
        for transform in level_transform:
            if "scale" in transform:
                resolution *= np.asarray(transform["scale"])[-3:]
    else:
        raise ValueError(f'Mandatory "coordinateTransformations" field missing for level {level}.')

    if "path" in vol_header:
        vol = omezarr[vol_header["path"]]
    else:
        raise ValueError(f'Mandatory "path" field missing for level {level}.')

    return vol, resolution
