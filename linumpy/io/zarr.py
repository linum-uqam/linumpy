import os
import shutil
import tempfile
from importlib.metadata import version
from pathlib import Path

import dask.array as da
import numpy as np
import zarr
from ome_zarr.dask_utils import resize as da_resize
from ome_zarr.format import CurrentFormat
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader, Multiscales
from ome_zarr.scale import Scaler
from ome_zarr.writer import write_image, write_multiscales_metadata
from skimage.transform import resize

"""
    This file contains functions for working with zarr files
"""


def create_tempstore(dir=None, suffix=None):
    """
    Create a zarr store inside a temporary directory.

    :type dir: str
    :param dir: Directory inside which to create the temporary directory.
    :type suffix: str
    :param suffix: Suffix of temporary directory.
    :type zarr_store: zarr.storage.LocalStore
    :return zarr_store: Temporary ZarrStore.
    """
    tempdir = Path(tempfile.TemporaryDirectory(dir=dir, suffix=suffix).name)
    zarr_store = zarr.storage.LocalStore(tempdir)
    return zarr_store


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
                return da_resize(image, out_shape, **kwargs)

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

    def _get_scale(i):
        scale = np.zeros(ndims)
        scale[:-len(voxel_size) - 1:-1] = np.asarray(voxel_size)[::-1] * 2.0 ** i
        return scale.tolist()

    coord_transforms = []
    for i in range(nlevels):
        transform_dict = [{
            "type": "scale",
            "scale": _get_scale(i)
        }]
        coord_transforms.append(transform_dict)
    return coord_transforms


def generate_axes_dict(ndims=3, unit="millimeter"):
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
        {"name": "z", "type": "space", "unit": unit},
        {"name": "y", "type": "space", "unit": unit},
        {"name": "x", "type": "space", "unit": unit}
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


def validate_n_levels(n_levels, shape, downscale_factor=2):
    """
    Validate n_levels such that it does not go beyond the volume shape.

    :type n_levels: int
    :param n_levels: Requested number of levels
    :type shape: tuple of int
    :param shape: Shape of volume to save
    :type downscale_factor: int
    :param downscale_factor: The downscale factor

    :type adjusted_n_levels: int
    :return adjusted_n_levels: Adjusted n_levels such that we don't exceed volume shape.
    """
    def logn(arr, n):
        return np.log2(arr) / np.log2(n)

    adjusted_n_levels = min(*logn(shape, downscale_factor).astype(int), n_levels)
    if n_levels > adjusted_n_levels:
        print(f'WARNING: Requested n_levels {n_levels} too high for image dimensions: {shape}.\n'
              f'Setting to {adjusted_n_levels}.')
    return int(adjusted_n_levels)


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
    n_levels = validate_n_levels(n_levels, data.shape)

    # pyramidal decomposition (ome_zarr.scale.Scaler) keywords
    pyramid_kw = {"max_layer": int(n_levels),
                  "method": "linear",
                  "downscale": 2}

    ome_zarr_version = version("ome-zarr")
    metadata = {
        "method": "ome_zarr.scale.Scaler",
        "version": ome_zarr_version,
        "args": pyramid_kw
    }

    # # axes and coordinate transformations
    ndims = len(data.shape)
    axes = generate_axes_dict(ndims)
    coordinate_transformations = create_transformation_dict(n_levels + 1, voxel_size=voxel_size, ndims=ndims)

    # create directory for zarr storage
    create_directory(store_path, overwrite)
    store = parse_url(store_path, mode='w').store
    zarr_group = zarr.group(store=store)

    write_image(data, zarr_group, axes=axes,
                scaler=CustomScaler(**pyramid_kw),
                storage_options=dict(chunks=chunks),
                coordinate_transformations=coordinate_transformations,
                compute=True, metadata=metadata)

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
    # read the image data
    reader = Reader(parse_url(zarr_path))
    # nodes may include images, labels etc
    nodes = list(reader())

    # first node will be the image pixel data
    image_node = nodes[0]

    # By default omezarr will return dask array. this can be achieved with:
    #    vol = image_node.data[level]
    # However here we will prefer loading a zarr array directly and let
    # the user convert to dask by themselves in their code.
    multiscale = None
    for spec in image_node.specs:
        if isinstance(spec, Multiscales):
            multiscale = spec
    vol = zarr.open_array(Path(zarr_path) / multiscale.datasets[level], mode='r')

    coordTransforms = image_node.metadata["coordinateTransformations"][level]
    scale = [1] * len(vol.shape)
    for tr in coordTransforms:
        if tr['type'] == 'scale':
            scale = tr['scale']
            break

    return vol, scale


class OmeZarrWriter:
    fmt: CurrentFormat
    shape: tuple
    downscale_factor: int
    root: zarr.Group
    axes: list
    zarray: zarr.Array

    def __init__(self, store_path: str | Path, shape: tuple, chunk_shape: tuple, dtype: np.dtype, overwrite: bool,
                 downscale_factor: int = 2, unit: str = 'millimeter'):
        """
        Class for writing ome-zarr files to disk in a pyramidal format.

        :type store_path: str or Path
        :param store_path: Path to the output zarr group.
        :type shape: tuple of n `int`, with n the number of dimensions.
        :param shape: Shape of the dataset.
        :type chunk_shape: tuple of n `int`, with n the number of dimensions.
        :param chunk_shape: Chunk size on disk.
        :type dtype: np.dtype
        :param dtype: Data type of the dataset.
        :type overwrite: bool
        :param overwrite: Overwrite `store_path` if it already exists.
        :type downscale_factor: int
        :param downscale_factor: Downscale factor between levels in the pyramid.
        :type unit: str
        :param unit: Unit of the spatial dimensions.
        Notes
        -----
        * Expected ordering for axes in `shape` and `chunk_shape` is `(c,
            z, y, x)`.
        """
        self.fmt = CurrentFormat()
        self.shape = shape
        self.downscale_factor = downscale_factor

        if os.path.exists(store_path):
            if overwrite:
                shutil.rmtree(store_path)
            else:
                raise ValueError(f"Overwrite set to False and {store_path} non-empty.")

        store = parse_url(store_path, mode="w", fmt=self.fmt).store
        self.root = zarr.group(store=store)

        shape = [int(v) for v in shape]
        chunk_shape = [int(v) for v in chunk_shape]

        # create empty array at root of pyramid
        # This is the array we will fill on-the-fly
        self.axes = generate_axes_dict(len(shape), unit=unit)
        self.zarray = self.root.require_array(
            "0",
            shape=shape,
            exact=True,
            chunks=chunk_shape,
            dtype=dtype,
            chunk_key_encoding=self.fmt.chunk_key_encoding,
            dimension_names=[axis["name"] for axis in self.axes],  # omit for v0.4
        )

    def _downsample_pyramid_on_disk(self, parent, paths):
        """
        Takes a high-resolution Zarr array at paths[0] in the zarr group
        and down-samples it by a given factor for each of the other paths
        """
        group_path = str(parent.store_path)
        img_path = parent.store_path / parent.path
        image_path = os.path.join(group_path, parent.path)
        print("downsample_pyramid_on_disk", image_path)
        for count, path in enumerate(paths[1:]):
            target_path = os.path.join(image_path, path)
            if os.path.exists(target_path):
                print("path exists: %s" % target_path)
                continue
            # open previous resolution from disk via dask...
            path_to_array = os.path.join(image_path, paths[count])
            dask_image = da.from_zarr(path_to_array)

            # resize in X and Y
            dims = list(dask_image.shape)
            dims[-1] = dims[-1] // self.downscale_factor
            dims[-2] = dims[-2] // self.downscale_factor
            if len(dims) > 2:
                dims[-3] = dask_image.shape[-3] // self.downscale_factor
            output = da_resize(
                dask_image, tuple(dims), preserve_range=True, anti_aliasing=False
            )

            options = {}
            if self.fmt.zarr_format == 2:
                options["dimension_separator"] = "/"
            else:
                options["chunk_key_encoding"] = self.fmt.chunk_key_encoding
                options["dimension_names"] = [axis["name"] for axis in self.axes]
            # write to disk
            da.to_zarr(
                arr=output, url=img_path, component=path,
                zarr_format=self.fmt.zarr_format, **options
            )

    def __setitem__(self, index, data):
        self.zarray[index] = data

    def __getitem__(self, index):
        return self.zarray[index]

    @property
    def ndim(self):
        return len(self.shape)

    def finalize(self, res, n_levels=5):
        n_levels = validate_n_levels(n_levels, self.shape, self.downscale_factor)
        paths = [f"{i}" for i in range(n_levels + 1)]
        self._downsample_pyramid_on_disk(self.root, paths)
        transformations = create_transformation_dict(n_levels + 1, res, len(self.shape))
        datasets = []
        for p, t in zip(paths, transformations):
            datasets.append({"path": p, "coordinateTransformations": t})

        pyramid_kw = {"max_layer": n_levels,
                      "method": "linear",
                      "downscale": self.downscale_factor}

        ome_zarr_version = version("ome-zarr")
        metadata = {
            "method": "ome_zarr.scale.Scaler",
            "version": ome_zarr_version,
            "args": pyramid_kw
        }

        write_multiscales_metadata(self.root, datasets, axes=self.axes, metadata=metadata)
