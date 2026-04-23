"""Zarr and OME-Zarr I/O utilities."""

# Configure dask thread pool based on environment variables
from linumpy.config.threads import configure_dask

import shutil
import tempfile
from importlib.metadata import version
from pathlib import Path
from typing import Any

import dask.array as da
import numpy as np
import numpy.typing as npt
import zarr
import zarr.storage
from ome_zarr.dask_utils import resize as da_resize
from ome_zarr.format import CurrentFormat
from ome_zarr.io import parse_url
from ome_zarr.reader import Multiscales, Reader
from ome_zarr.scale import Scaler
from ome_zarr.writer import write_image, write_multiscales_metadata
from skimage.transform import resize

configure_dask()

"""
    This file contains functions for working with zarr files
"""


def create_tempstore(dir: str | None = None, suffix: str | None = None) -> zarr.storage.LocalStore:
    """
    Create a zarr store inside a temporary directory.

    Parameters
    ----------
    dir : str, optional
        Directory inside which to create the temporary directory.
    suffix : str, optional
        Suffix of temporary directory.

    Returns
    -------
    zarr.storage.LocalStore
        Temporary ZarrStore.
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

    def resize_image(self, image: np.ndarray | da.Array) -> np.ndarray | da.Array:
        """
        Resize a numpy array OR a dask array to a smaller array (not pyramid).

        This method is the only method from the Scaler class called from
        `ome_zarr.writer.write_image`.
        """
        if isinstance(image, da.Array):

            def _resize(image: da.Array, out_shape: tuple[int, ...], **kwargs: Any) -> da.Array:
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
                image.real.astype(np.float64),  # ty: ignore[invalid-argument-type]  # _resize is a union type; correct branch is selected at runtime
                out_shape,
                order=1,
                mode="reflect",
                anti_aliasing=False,
            ) + 1j * _resize(
                image.imag.astype(np.float64),  # ty: ignore[invalid-argument-type]  # _resize is a union type; correct branch is selected at runtime
                out_shape,
                order=1,
                mode="reflect",
                anti_aliasing=False,
            )
        else:
            image = _resize(
                image.astype(np.float64),  # ty: ignore[invalid-argument-type]  # _resize is a union type; correct branch is selected at runtime
                out_shape,
                order=1,
                mode="reflect",
                anti_aliasing=False,
            )
        return image.astype(dtype)

    def linear(self, base: np.ndarray | da.Array) -> list[np.ndarray | da.Array]:
        """
        Downsample using :func:`skimage.transform.resize`.

        with linear interpolation.
        """
        pyramid = [base]
        max_axes_resize = min(len(base.shape), 3)
        level = self.max_layer
        while level > 0 and np.all(np.asarray(pyramid[-1].shape[:-max_axes_resize]) >= self.downscale):
            pyramid.append(self.resize_image(pyramid[-1]))
            level -= 1
        return pyramid

    def _by_plane(self, base: np.ndarray | da.Array, func: Any) -> np.ndarray:
        # This method is called by base class when interpolation methods (e.g. nearest)
        # are called directly. Because `write_image` never call these methods, we don't
        # need to implement it here. We raise an error to make sure the CustomScaler class
        # is not used for this purpose.
        raise NotImplementedError("_by_plane method not implemented for CustomScaler")


def create_transformation_dict(nlevels: int, voxel_size: tuple[float | int, ...] | list[float | int] | npt.NDArray[Any], ndims: int = 3) -> list[list[dict]]:
    """
    Create a dictionary with the transformation information for images up to 4 dimensions.

    Parameters
    ----------
    nlevels : int
        The number of levels in the pyramid.
    voxel_size : tuple or list or ndarray
        The voxel size of the dataset.
    ndims : int, optional
        The number of dimensions of the dataset, by default 3.

    Returns
    -------
    list[list[dict]]
        List of coordinate transformations.
    """

    def _get_scale(i: int) -> list[float]:
        scale = np.zeros(ndims)
        scale[: -len(voxel_size) - 1 : -1] = np.asarray(voxel_size)[::-1] * 2.0**i
        return scale.tolist()

    coord_transforms = []
    for i in range(nlevels):
        transform_dict = [{"type": "scale", "scale": _get_scale(i)}]
        coord_transforms.append(transform_dict)
    return coord_transforms


def generate_axes_dict(ndims: int = 3, unit: str = "millimeter") -> list[dict]:
    """
    Generate the axes dictionary for up to 4 dimensions.

    Dimensions are returned in order (c, z, y, x).

    Parameters
    ----------
    ndims : int, optional
        Number of dimensions, by default 3.
    unit : str, optional
        Spatial unit, by default 'millimeter'.

    Returns
    -------
    list[dict]
        The axes dictionary.
    """
    axes = [
        {"name": "c", "type": "channel"},
        {"name": "z", "type": "space", "unit": unit},
        {"name": "y", "type": "space", "unit": unit},
        {"name": "x", "type": "space", "unit": unit},
    ]
    offset = len(axes) - ndims
    return axes[offset:]


def create_directory(store_path: Path, overwrite: bool = False) -> Path:
    """Create a directory at the given store path, optionally overwriting an existing one."""
    directory = Path(store_path)
    # Check for symlink first: is_symlink() is True even for dangling symlinks,
    # while exists() follows the link and returns False for dangling ones.
    # shutil.rmtree raises OSError on symlinks, so handle them separately.
    if directory.is_symlink():
        if overwrite:
            directory.unlink()  # Remove the symlink only; target is NOT deleted
        else:
            raise FileExistsError(f"Path {directory.as_posix()} already exists as a symlink. Set overwrite=True to overwrite.")
    elif directory.exists():
        if overwrite:
            shutil.rmtree(directory)
        else:
            raise FileExistsError(f"Directory {directory.as_posix()} already exists. Set overwrite=True to overwrite.")
    directory.mkdir(parents=True)
    return directory


def validate_n_levels(n_levels: int, shape: tuple[int, ...], downscale_factor: int = 2) -> int:
    """
    Validate n_levels such that it does not go beyond the volume shape.

    Parameters
    ----------
    n_levels : int
        Requested number of levels.
    shape : tuple[int, ...]
        Shape of volume to save.
    downscale_factor : int, optional
        The downscale factor, by default 2.

    Returns
    -------
    int
        Adjusted n_levels such that we don't exceed volume shape.
    """

    def logn(arr: np.ndarray, n: int) -> np.ndarray:
        return np.log2(arr) / np.log2(n)

    adjusted_n_levels = min(*logn(np.array(shape), downscale_factor).astype(int), n_levels)
    if n_levels > adjusted_n_levels:
        print(
            f"WARNING: Requested n_levels {n_levels} too high for image dimensions: {shape}.\nSetting to {adjusted_n_levels}."
        )
    return int(adjusted_n_levels)


def save_omezarr(data: np.ndarray | da.Array, store_path: Path, voxel_size: tuple[float | int, ...] | list[float | int] | npt.NDArray[Any] = (1e-3, 1e-3, 1e-3), chunks: tuple[int, ...] = (128, 128, 128), n_levels: int = 5, overwrite: bool = True) -> zarr.Group:
    """
    Save numpy array to disk in zarr format following OME-NGFF file specifications.

    Expected ordering for axes in `data` and `scales` is `(c, z, y, x)`. Does not
    support saving for multi-channel 2D images with axes (c, y, x).

    Parameters
    ----------
    data : ndarray or dask array
        Numpy or dask array to save as zarr.
    store_path : Path
        The path of the output zarr group.
    voxel_size : tuple of float, optional
        Voxel size in mm, by default (1e-3, 1e-3, 1e-3).
    chunks : tuple of int, optional
        Chunk size on disk, by default (128, 128, 128).
    n_levels : int, optional
        Number of levels in Gaussian pyramid, by default 5.
    overwrite : bool, optional
        Overwrite `store_path` if it already exists, by default True.

    Returns
    -------
    zarr.Group
        Resulting zarr group saved to disk.
    """
    n_levels = validate_n_levels(n_levels, data.shape)

    # pyramidal decomposition (ome_zarr.scale.Scaler) keywords
    pyramid_kw = {"max_layer": int(n_levels), "method": "linear", "downscale": 2}

    ome_zarr_version = version("ome-zarr")
    metadata = {"method": "ome_zarr.scale.Scaler", "version": ome_zarr_version, "args": pyramid_kw}

    # # axes and coordinate transformations
    ndims = len(data.shape)
    axes = generate_axes_dict(ndims)
    coordinate_transformations = create_transformation_dict(n_levels + 1, voxel_size=voxel_size, ndims=ndims)

    # create directory for zarr storage
    create_directory(store_path, overwrite)
    _loc = parse_url(store_path, mode="w")
    assert _loc is not None
    store = _loc.store
    zarr_group = zarr.group(store=store)

    write_image(
        data,
        zarr_group,
        axes=axes,
        scaler=CustomScaler(
            max_layer=int(pyramid_kw["max_layer"]),
            method=str(pyramid_kw["method"]),
            downscale=int(pyramid_kw["downscale"]),
        ),
        storage_options={"chunks": chunks},
        coordinate_transformations=coordinate_transformations,
        compute=True,
        metadata=metadata,
    )

    # return zarr group containing saved data
    return zarr_group


def read_omezarr(zarr_path: Path, level: int = 0) -> tuple[zarr.Array, list[float]]:
    """
    Read omezarr image at `zarr_path` and load image data for `level` in the pyramid.

    Also returns voxel size for chosen level.

    Parameters
    ----------
    zarr_path : Path
        Path of OME-zarr file to load.
    level : int, optional
        The level of the pyramid to load (0 is full resolution data), by default 0.

    Returns
    -------
    zarr.Array
        Requested zarr array.
    list[float]
        Voxel size of zarr array.
    """
    # read the image data
    _zarr_loc = parse_url(zarr_path)
    assert _zarr_loc is not None
    reader = Reader(_zarr_loc)
    # nodes may include images, labels etc
    nodes = list(reader())

    # first node will be the image pixel data
    image_node = nodes[0]

    # By default omezarr will return dask array; here we load a zarr array directly
    # and let the user convert to dask in their own code.
    multiscale = None
    for spec in image_node.specs:
        if isinstance(spec, Multiscales):
            multiscale = spec
    assert multiscale is not None
    vol = zarr.open_array(Path(zarr_path) / multiscale.datasets[level], mode="r")

    coord_transforms = image_node.metadata["coordinateTransformations"][level]
    scale = [1] * len(vol.shape)
    for tr in coord_transforms:
        if tr["type"] == "scale":
            scale = tr["scale"]
            break

    return vol, [float(s) for s in scale]


class OmeZarrWriter:
    """Write data to OME-Zarr format with multi-resolution pyramid support."""

    fmt: CurrentFormat
    shape: tuple
    _shape_list: list
    downscale_factor: int
    root: zarr.Group
    axes: list
    zarray: zarr.Array

    def __init__(
        self,
        store_path: Path,
        shape: tuple,
        chunk_shape: tuple,
        shards: tuple | None = None,
        dtype: npt.DTypeLike = np.float32,
        overwrite: bool = True,
        downscale_factor: int = 2,
        unit: str = "millimeter",
    ) -> None:
        """
        Class for writing ome-zarr files to disk in a pyramidal format.

        Parameters
        ----------
        store_path : Path
            Path to the output zarr group.
        shape : tuple[int, ...]
            Shape of the dataset.
        chunk_shape : tuple[int, ...]
            Chunk size on disk.
        shards : tuple[int, ...], optional
            Dimension of shards. None for no sharding.
        dtype : np.dtype, optional
            Data type of the dataset, by default np.float32.
        overwrite : bool, optional
            Overwrite `store_path` if it already exists, by default True.
        downscale_factor : int, optional
            Downscale factor between levels in the pyramid, by default 2.
        unit : str, optional
            Unit of the spatial dimensions, by default 'millimeter'.

        Notes
        -----
        Expected ordering for axes in `shape` and `chunk_shape` is `(c,
        z, y, x)`.
        """
        self.fmt = CurrentFormat()
        self.shape = shape
        self.downscale_factor = downscale_factor

        if Path(store_path).exists() or Path(store_path).is_symlink():
            if overwrite:
                if Path(store_path).is_symlink():
                    Path(store_path).unlink()
                else:
                    shutil.rmtree(store_path)
            else:
                raise ValueError(f"Overwrite set to False and {store_path} non-empty.")

        _loc = parse_url(store_path, mode="w", fmt=self.fmt)
        assert _loc is not None
        store = _loc.store
        self.root = zarr.group(store=store)

        shape_list = [int(v) for v in shape]
        chunk_shape_list = [int(v) for v in chunk_shape]

        # create empty array at root of pyramid
        # This is the array we will fill on-the-fly
        self.axes = generate_axes_dict(len(shape_list), unit=unit)
        self.zarray = self.root.require_array(
            "0",
            shape=shape_list,
            exact=True,
            chunks=chunk_shape_list,
            shards=shards,
            dtype=dtype,
            chunk_key_encoding=self.fmt.chunk_key_encoding,
            dimension_names=[axis["name"] for axis in self.axes],  # omit for v0.4
        )

    def _downsample_pyramid_on_disk(self, parent: zarr.Group, paths: list[str]) -> None:
        """
        Take a high-resolution Zarr array at paths[0] in the zarr group.

        and down-samples it by a given factor for each of the other paths.
        """
        group_path = str(parent.store_path)
        img_path = parent.store_path / parent.path
        image_path = str(Path(group_path) / parent.path)
        print("downsample_pyramid_on_disk", image_path)
        for count, path in enumerate(paths[1:]):
            target_path = str(Path(image_path) / path)
            if Path(target_path).exists():
                print(f"path exists: {target_path}")
                continue
            # open previous resolution from disk via dask...
            path_to_array = str(Path(image_path) / paths[count])
            dask_image = da.from_zarr(path_to_array)

            # resize in X and Y
            dims = list(dask_image.shape)
            dims[-1] = dims[-1] // self.downscale_factor
            dims[-2] = dims[-2] // self.downscale_factor
            if len(dims) > 2:
                dims[-3] = dask_image.shape[-3] // self.downscale_factor
            output = da_resize(dask_image, tuple(dims), preserve_range=True, anti_aliasing=False)

            options = {}
            if self.fmt.zarr_format == 2:
                options["dimension_separator"] = "/"
            else:
                options["chunk_key_encoding"] = self.fmt.chunk_key_encoding
                options["dimension_names"] = [axis["name"] for axis in self.axes]
            # write to disk
            da.to_zarr(arr=output, url=img_path, component=path, zarr_format=self.fmt.zarr_format, **options)

    def __setitem__(self, index: Any, data: np.ndarray) -> None:
        """Write data to the underlying zarr array."""
        self.zarray[index] = data

    def __getitem__(self, index: Any) -> Any:
        """Read data from the underlying zarr array."""
        return self.zarray[index]

    @property
    def ndim(self) -> int:
        """Return the number of dimensions."""
        return len(self.shape)

    @property
    def dtype(self) -> npt.DTypeLike:
        """Return the data type of the underlying zarr array."""
        return self.zarray.dtype

    def finalize(self, res: list[float], n_levels: int | None = 5, **_kwargs: Any) -> None:  # type: ignore[override]
        """
        Finalize the OME-Zarr with traditional power-of-2 pyramid levels.

        Parameters
        ----------
        res : list of float
            Resolution in mm for each axis (e.g., [0.01, 0.01, 0.01] for 10 µm isotropic)
        n_levels : int
            Number of pyramid levels (default: 5). Each level is 2x downsampled.
        """
        if n_levels is None:
            n_levels = 5
        n_levels = validate_n_levels(n_levels, self.shape, self.downscale_factor)
        paths = [f"{i}" for i in range(n_levels + 1)]
        self._downsample_pyramid_on_disk(self.root, paths)
        transformations = create_transformation_dict(n_levels + 1, res, len(self.shape))
        datasets = []
        for p, t in zip(paths, transformations, strict=False):
            datasets.append({"path": p, "coordinateTransformations": t})

        pyramid_kw = {"max_layer": n_levels, "method": "linear", "downscale": self.downscale_factor}

        ome_zarr_version = version("ome-zarr")
        metadata = {"method": "ome_zarr.scale.Scaler", "version": ome_zarr_version, "args": pyramid_kw}

        write_multiscales_metadata(self.root, datasets, axes=self.axes, metadata=metadata)


class AnalysisOmeZarrWriter(OmeZarrWriter):
    """
    OmeZarrWriter subclass that supports custom analysis-friendly resolution pyramids.

    This class extends OmeZarrWriter to create pyramid levels at specific target
    resolutions (e.g., 10, 25, 50, 100 µm) instead of traditional power-of-2
    downsampling. This is useful for creating output volumes optimized for
    downstream analysis at specific scales.

    Example
    -------
    >>> writer = AnalysisOmeZarrWriter("output.ome.zarr", shape, chunks, dtype=np.float32)
    >>> writer[:] = data  # Write data at full resolution
    >>> writer.finalize(base_res, [10, 25, 50, 100])

    Notes
    -----
    - Use `finalize()` for traditional power-of-2 pyramids (inherited from OmeZarrWriter)
    - Use `finalize_with_resolutions()` for custom analysis-friendly resolutions
    """

    def _downsample_to_resolution(self, parent: zarr.Group, source_path: str, target_path: str, target_shape: list[int]) -> None:
        """Downsample from source_path to target_path with specific target shape."""
        group_path = str(parent.store_path)
        # Remove file:// prefix if present (from zarr URL format)
        if group_path.startswith("file://"):
            group_path = group_path[7:]
        img_path = parent.store_path / parent.path
        image_path = str(Path(group_path) / parent.path)

        full_target_path = str(Path(image_path) / target_path)
        if Path(full_target_path).exists():
            print(f"Path exists: {full_target_path}")
            return

        # Open source from disk via dask
        path_to_array = str(Path(image_path) / source_path)
        dask_image = da.from_zarr(path_to_array)

        output = da_resize(dask_image, tuple(target_shape), preserve_range=True, anti_aliasing=True)

        options = {}
        if self.fmt.zarr_format == 2:
            options["dimension_separator"] = "/"
        else:
            options["chunk_key_encoding"] = self.fmt.chunk_key_encoding
            options["dimension_names"] = [axis["name"] for axis in self.axes]

        da.to_zarr(arr=output, url=img_path, component=target_path, zarr_format=self.fmt.zarr_format, **options)

    def finalize_analysis(self, res: list[float], target_resolutions_um: tuple[float, ...] = (10, 25, 50, 100), n_levels: int | None = None, make_isotropic: bool = True) -> None:
        """
        Finalize the OME-Zarr with pyramid levels.

        Parameters
        ----------
        res : list of float
            Base resolution in mm (e.g., [0.01, 0.01, 0.01] for 10 µm isotropic,
            or [0.0015, 0.01, 0.01] for anisotropic z=1.5µm, xy=10µm)
        target_resolutions_um : list of float, optional
            Target resolutions in microns (default: [10, 25, 50, 100]).
            Ignored if n_levels is specified.
        n_levels : int, optional
            If specified, uses traditional power-of-2 downsampling instead of
            custom resolutions (backward compatible with OmeZarrWriter).
        make_isotropic : bool, optional
            If True (default), resamples anisotropic data to produce isotropic
            voxels at each target resolution. Each dimension is scaled independently
            to achieve the target resolution.
            If False, preserves the original aspect ratio by scaling all dimensions
            uniformly based on the finest (smallest) base resolution.

        Notes
        -----
        By default, creates pyramid levels at specific analysis-friendly
        resolutions (e.g., 10, 25, 50, 100 µm). If n_levels is provided,
        falls back to traditional power-of-2 downsampling for backward
        compatibility.

        Examples
        --------
        For anisotropic data with base resolution [1.5, 10, 10] µm targeting 25 µm:

        With make_isotropic=True (default):
            - Scale factors: [16.67, 2.5, 2.5] (per-dimension)
            - Output: isotropic 25 µm voxels
            - Shape aspect ratio changes

        With make_isotropic=False:
            - Scale factor: 16.67 (uniform, based on finest dimension 1.5 µm)
            - Output: anisotropic [25, 167, 167] µm voxels
            - Shape aspect ratio preserved
        """
        # Backward compatibility: if n_levels is specified, use parent's power-of-2 method
        if n_levels is not None:
            super().finalize(res, n_levels)
            return
        # Convert base resolution to microns
        base_res_um = [r * 1000 for r in res]  # mm to µm
        min_base_res_um = min(base_res_um)

        # Filter target resolutions to only include those >= finest base resolution
        valid_targets = sorted([t for t in target_resolutions_um if t >= min_base_res_um])

        if not valid_targets:
            print(f"WARNING: No valid target resolutions. Base resolution is {base_res_um} µm")
            # Fall back to parent's power-of-2 finalize with no levels
            super().finalize(res, n_levels=0)
            return

        # Inform user about anisotropic handling
        is_anisotropic = max(base_res_um) / min_base_res_um > 1.1  # 10% threshold
        if is_anisotropic:
            if make_isotropic:
                print("Creating ISOTROPIC pyramid from anisotropic data")
                print(f"  Base resolution: {base_res_um} µm (anisotropic)")
                print(f"  Output: isotropic voxels at {valid_targets} µm")
            else:
                print("Creating pyramid PRESERVING ASPECT RATIO")
                print(f"  Base resolution: {base_res_um} µm (anisotropic)")
                print(f"  Scaling uniformly based on finest dimension ({min_base_res_um} µm)")
        else:
            print(f"Creating pyramid with target resolutions: {valid_targets} µm")

        paths = []
        resolutions = []

        # Get the path to level 0 (base resolution) for downsampling source
        # Remove file:// prefix if present (from zarr URL format)
        group_path = str(self.root.store_path)
        if group_path.startswith("file://"):
            group_path = group_path[7:]  # Remove "file://" prefix

        for i, target_um in enumerate(valid_targets):
            path = f"{i}"
            paths.append(path)

            if make_isotropic:
                # Per-dimension scaling to achieve isotropic target resolution
                # Each dimension scales independently to reach the target resolution
                scale_factors = [target_um / base_res_um_d for base_res_um_d in base_res_um]
            else:
                # Uniform scaling to preserve aspect ratio
                # All dimensions scale by the same factor based on finest resolution
                uniform_scale = target_um / min_base_res_um
                scale_factors = [uniform_scale] * len(base_res_um)

            # Calculate target shape using scale factors
            target_shape = [max(1, int(s / sf)) for s, sf in zip(self.shape, scale_factors, strict=False)]

            # Calculate target resolution per-dimension
            target_res_mm = [r * sf for r, sf in zip(res, scale_factors, strict=False)]
            resolutions.append(target_res_mm)

            # Display resolution info
            target_res_um = [r * 1000 for r in target_res_mm]
            if make_isotropic or not is_anisotropic:
                print(f"  Level {i}: {target_um} µm -> shape {target_shape}")
            else:
                print(f"  Level {i}: {target_res_um} µm -> shape {target_shape}")

            if i == 0:
                # For level 0, we need to replace the base resolution data
                # First downsample to a temp location, then replace
                temp_path = f"_temp_{i}"
                self._downsample_to_resolution(self.root, "0", temp_path, target_shape)

                # Remove original level 0 and rename temp
                original_path = str(Path(group_path) / self.root.path / "0")
                temp_full_path = str(Path(group_path) / self.root.path / temp_path)

                if Path(original_path).exists():
                    shutil.rmtree(original_path)
                shutil.move(temp_full_path, original_path)
            else:
                # For other levels, downsample from the new level 0
                self._downsample_to_resolution(self.root, "0", path, target_shape)

        # Create transformation metadata
        datasets = []
        for path, res_mm in zip(paths, resolutions, strict=False):
            transforms = [{"type": "scale", "scale": res_mm}]
            datasets.append({"path": path, "coordinateTransformations": transforms})

        ome_zarr_version = version("ome-zarr")
        metadata = {
            "method": "custom_resolution_pyramid",
            "version": ome_zarr_version,
            "args": {"target_resolutions_um": valid_targets, "make_isotropic": make_isotropic},
        }

        write_multiscales_metadata(self.root, datasets, axes=self.axes, metadata=metadata)
