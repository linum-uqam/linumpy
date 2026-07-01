"""OME-Zarr I/O helpers for linumpy."""

# Configure dask thread pool based on environment variables
from linumpy.config.threads import configure_dask

import atexit
import shutil
import tempfile
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Any

import dask.array as da
import numpy as np
import zarr
import zarr.storage
from ome_zarr.dask_utils import resize as da_resize
from ome_zarr.format import CurrentFormat
from ome_zarr.io import parse_url
from ome_zarr.reader import Multiscales, Reader
from ome_zarr.scale import Methods
from ome_zarr.writer import write_image, write_multiscales_metadata

configure_dask()


def create_tempstore(dir: str | None = None, suffix: str | None = None) -> zarr.storage.LocalStore:
    """
    Create a zarr store inside a temporary directory.

    Defaults to creating the temp directory inside the current working
    directory (typically a Nextflow task scratch dir) rather than ``/tmp``,
    so leaked temp zarrs are reclaimed when the work dir is cleaned and do
    not pile up on small ``/tmp`` partitions.

    :type dir: str
    :param dir: Directory inside which to create the temporary directory.
                Defaults to the current working directory.
    :type suffix: str
    :param suffix: Suffix of temporary directory.
    :type zarr_store: zarr.storage.LocalStore
    :return zarr_store: Temporary ZarrStore.
    """
    parent = dir if dir is not None else "."
    tempdir = Path(tempfile.mkdtemp(dir=parent, suffix=suffix))
    # Register cleanup explicitly. mkdtemp does not auto-clean, but atexit
    # is more reliable than TemporaryDirectory's finalizer when zarr keeps
    # handles open until interpreter shutdown.
    atexit.register(shutil.rmtree, tempdir, ignore_errors=True)
    zarr_store = zarr.storage.LocalStore(tempdir)
    return zarr_store


def create_transformation_dict(nlevels: int, voxel_size: Sequence, ndims: int = 3) -> list:
    """Create a list of coordinate transformation dicts for OME-Zarr pyramid levels.

    Supports images up to 4 dimensions.

    :type nlevels: int
    :param nlevels: The number of levels in the pyramid.
    :type voxel_size: tuple
    :param voxel_size: The voxel size of the dataset.
    :type ndims: int
    :param ndims: The number of dimensions of the dataset.
    :type coord_transforms: list of Dict
    :return coord_transforms: List of coordinate transformations
    """

    def _get_scale(i: int) -> list:
        scale = np.zeros(ndims)
        scale[: -len(voxel_size) - 1 : -1] = np.asarray(voxel_size)[::-1] * 2.0**i
        return scale.tolist()

    coord_transforms = []
    for i in range(nlevels):
        transform_dict = [{"type": "scale", "scale": _get_scale(i)}]
        coord_transforms.append(transform_dict)
    return coord_transforms


def generate_axes_dict(ndims: int = 3, unit: str = "millimeter") -> list:
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
        {"name": "x", "type": "space", "unit": unit},
    ]
    offset = len(axes) - ndims
    return axes[offset:]


def create_directory(store_path: Path, overwrite: bool = False) -> Path:
    """Create directory at *store_path*, optionally removing an existing one."""
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


def validate_n_levels(n_levels: int, shape: Sequence, downscale_factor: int = 2) -> int:
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

    def logn(arr: np.ndarray | Sequence, n: int) -> np.ndarray:
        return np.log2(arr) / np.log2(n)

    adjusted_n_levels = min(*logn(shape, downscale_factor).astype(int), n_levels)
    if n_levels > adjusted_n_levels:
        print(
            f"WARNING: Requested n_levels {n_levels} too high for image dimensions: {shape}.\nSetting to {adjusted_n_levels}."
        )
    return int(adjusted_n_levels)


def save_omezarr(
    data: np.ndarray | da.Array,
    store_path: Path,
    voxel_size: tuple | Sequence = (1e-3, 1e-3, 1e-3),
    chunks: tuple | Sequence = (128, 128, 128),
    n_levels: int = 5,
    overwrite: bool = True,
    shards: tuple | Sequence | None = None,
) -> zarr.Group:
    """Save array to disk in OME-NGFF zarr format.

    Expected ordering for axes in `data` and `scales` is `(c, z, y, x)`.

    :type data: numpy or dask array
    :param data: numpy or dask array to save as zarr.
    :type store_path: str
    :param store_path: The path of the output zarr group.
    :type voxel_size: tuple of n `float`, with n the number of dimensions.
    :param voxel_size: Voxel size in mm.
    :type chunks: tuple of n `int`, with n the number of dimensions.
    :param chunks: Chunk size on disk.
    :type n_levels: int
    :param n_levels: Number of pyramid levels (downsamples by 2 along all spatial
        axes including Z).
    :type overwrite: bool
    :param overwrite: Overwrite `store_path` if it already exists.
    :type shards: tuple of n `int` or None
    :param shards: If provided, group multiple chunks together into a single
        shard on disk (zarr v3 native sharding). Must be a multiple of
        ``chunks`` along every dimension. Sharding reduces the number of files
        on disk dramatically for large pyramids while keeping per-chunk random
        access; useful for stacked volumes whose chunk size is far smaller than
        the per-axis extent. ``None`` disables sharding.

    :type zarr_group: zarr.hierarchy.group
    :return zarr_group: Resulting zarr group saved to disk.
    """
    n_levels = validate_n_levels(n_levels, data.shape)

    # axes and coordinate transformations (c, z, y, x order)
    ndims = len(data.shape)
    axes = generate_axes_dict(ndims)
    coordinate_transformations = create_transformation_dict(n_levels + 1, voxel_size=voxel_size, ndims=ndims)

    # ome-zarr's default ``scale_factors`` (e.g. ``(2, 4, 8, 16)``) applies
    # downsampling only to spatial axes *except z*. linumpy datasets are
    # acquired with isotropic-ish voxel sizes, so we want true 3D downsampling
    # at every level. Build the per-level dict explicitly.
    spatial_axes = [a["name"] for a in axes if a.get("type") == "space"]
    scale_factors: list[dict[str, int]] = [dict.fromkeys(spatial_axes, 2**i) for i in range(1, n_levels + 1)]

    # storage_options: one dict per dataset (level0 + n_levels). Each level
    # gets the same chunk shape; sharding (when requested) likewise applies
    # to every level.
    storage_options: list[dict[str, Any]] = []
    for _ in range(n_levels + 1):
        opts: dict[str, Any] = {"chunks": tuple(chunks)}
        if shards is not None:
            opts["shards"] = tuple(shards)
        storage_options.append(opts)

    pyramid_kw = {"max_layer": int(n_levels), "method": Methods.RESIZE.value, "downscale": 2}
    ome_zarr_version = version("ome-zarr")
    metadata = {"method": "ome_zarr.writer.write_image", "version": ome_zarr_version, "args": pyramid_kw}

    create_directory(store_path, overwrite)
    _loc = parse_url(store_path, mode="w")
    assert _loc is not None
    store = _loc.store
    zarr_group = zarr.group(store=store)

    write_image(
        data,
        zarr_group,
        scale_factors=scale_factors,
        method=Methods.RESIZE,
        axes=axes,
        storage_options=storage_options,
        coordinate_transformations=coordinate_transformations,
        compute=True,
        metadata=metadata,
    )

    return zarr_group


def resolve_omezarr_level_path(zarr_path: Path, level: int = 0) -> Path:
    """Return the on-disk path of the array for *level* of an OME-Zarr group.

    Resolves the multiscale dataset path advertised by the OME-Zarr metadata
    (``zarr_path / multiscale.datasets[level]``) without opening the array.
    This is the backend-agnostic way to locate the actual chunked store on
    disk — independent of any particular ``zarr.storage`` implementation.

    Parameters
    ----------
    zarr_path
        Path to the OME-Zarr group.
    level
        Pyramid level (0 = full resolution).

    Returns
    -------
    Path
        On-disk path of the level's zarr array.
    """
    _zarr_loc = parse_url(zarr_path)
    assert _zarr_loc is not None
    reader = Reader(_zarr_loc)
    nodes = list(reader())
    image_node = nodes[0]

    multiscale = None
    for spec in image_node.specs:
        if isinstance(spec, Multiscales):
            multiscale = spec
    assert multiscale is not None, "No Multiscales spec found in zarr file"
    return Path(zarr_path) / multiscale.datasets[level]


def read_omezarr(zarr_path: Path, level: int = 0) -> tuple:
    """Read OME-Zarr image at *zarr_path* and return the array and voxel size.

    Loads image data for *level* in the pyramid.

    :type zarr_path: str
    :param zarr_path: Path of OME-zarr file to load.
    :type level: int >= 0
    :param level: The level of the pyramid to load (0 is full resolution data).

    :type vol: zarr.array
    :return vol: Requested zarr array.
    :type res: tuple (3,)
    :return res: Voxel size of zarr array.
    """
    _zarr_loc = parse_url(zarr_path)
    assert _zarr_loc is not None
    reader = Reader(_zarr_loc)
    nodes = list(reader())
    image_node = nodes[0]

    multiscale = None
    for spec in image_node.specs:
        if isinstance(spec, Multiscales):
            multiscale = spec
    assert multiscale is not None, "No Multiscales spec found in zarr file"
    vol = zarr.open_array(Path(zarr_path) / multiscale.datasets[level], mode="r")

    coord_transforms = image_node.metadata["coordinateTransformations"][level]
    scale = [1] * len(vol.shape)
    for tr in coord_transforms:
        if tr["type"] == "scale":
            scale = tr["scale"]
            break

    return vol, scale


def read_omezarr_array(
    zarr_path: Path,
    level: int = 0,
    *,
    use_gpu: bool = False,
) -> tuple[Any, list[float]]:
    """Read an OME-Zarr image and materialise it into a single in-memory array.

    Unlike :func:`read_omezarr` (which returns a lazy ``zarr.Array``), this helper
    returns a fully-loaded array sized to fit in host or device memory, plus the
    voxel size for *level*.

    When ``use_gpu=True`` the underlying multiscale array is loaded via
    :func:`linumpy.gpu.zarr_io.read_zarr_to_gpu`, which selects the fastest path
    available at runtime (kvikio / GPUDirect Storage when native mode is on and
    the array is uncompressed; otherwise ``zarr.config.enable_gpu``). The
    returned array is a ``cupy.ndarray``.

    When ``use_gpu=False`` (default) the array is returned as a contiguous
    ``numpy.ndarray`` — keeps existing CPU-only pipelines working unchanged.

    Parameters
    ----------
    zarr_path
        Path to the OME-Zarr group.
    level
        Pyramid level to load (0 = full resolution).
    use_gpu
        If True and CuPy is available, return a device-resident ``cupy.ndarray``
        loaded through the GPU dispatcher.

    Returns
    -------
    array : numpy.ndarray or cupy.ndarray
        Fully-materialised volume.
    scale : list of float
        Voxel size for the requested pyramid level.
    """
    vol, scale = read_omezarr(zarr_path, level=level)
    if use_gpu:
        # Resolve the on-disk path of the actual array (level subdirectory) so
        # the GPU dispatcher can hit kvikio's raw-bytes fast path when possible.
        from linumpy.gpu.zarr_io import read_zarr_to_gpu

        array_path = resolve_omezarr_level_path(zarr_path, level=level)
        return read_zarr_to_gpu(array_path), scale
    return np.asarray(vol[:]), scale


class OmeZarrWriter:
    """Write OME-Zarr files to disk in a pyramidal format, chunk by chunk."""

    fmt: CurrentFormat
    shape: Sequence
    downscale_factor: int
    root: zarr.Group
    axes: list
    zarray: zarr.Array

    def __init__(
        self,
        store_path: Path,
        shape: tuple | Sequence,
        chunk_shape: tuple | Sequence,
        shards: tuple | None = None,
        dtype: type | np.dtype = np.float32,
        overwrite: bool = True,
        downscale_factor: int = 2,
        unit: str = "millimeter",
    ) -> None:
        """
        Class for writing ome-zarr files to disk in a pyramidal format.

        :type store_path: str or Path
        :param store_path: Path to the output zarr group.
        :type shape: tuple of n `int`, with n the number of dimensions.
        :param shape: Shape of the dataset.
        :type chunk_shape: tuple of n `int`, with n the number of dimensions.
        :param chunk_shape: Chunk size on disk.
        :type shards: tuple of `int`
        :param shards: Dimension of shards. `None` for no sharding.
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

        store_path = Path(store_path)
        if store_path.exists() or store_path.is_symlink():
            if overwrite:
                if store_path.is_symlink():
                    store_path.unlink()
                else:
                    shutil.rmtree(store_path)
            else:
                raise ValueError(f"Overwrite set to False and {store_path} non-empty.")

        _store_loc = parse_url(store_path, mode="w", fmt=self.fmt)
        assert _store_loc is not None
        store = _store_loc.store
        self.root = zarr.group(store=store)

        shape = tuple(int(v) for v in shape)
        chunk_shape = tuple(int(v) for v in chunk_shape)

        # create empty array at root of pyramid
        # This is the array we will fill on-the-fly
        self.axes = generate_axes_dict(len(shape), unit=unit)
        self.zarray = self.root.require_array(
            "0",
            shape=shape,
            exact=True,
            chunks=chunk_shape,
            shards=shards,
            dtype=dtype,
            chunk_key_encoding=self.fmt.chunk_key_encoding,
            dimension_names=[axis["name"] for axis in self.axes],  # omit for v0.4
        )

    def _downsample_pyramid_on_disk(self, parent: zarr.Group, paths: list) -> None:
        """Downsample the high-resolution array at *paths[0]* to fill each remaining level."""
        group_path = str(parent.store_path)
        img_path = parent.store_path / parent.path
        image_path = Path(group_path) / parent.path
        print("downsample_pyramid_on_disk", image_path)
        for count, path in enumerate(paths[1:]):
            target_path = image_path / path
            if target_path.exists():
                print(f"path exists: {target_path}")
                continue
            # open previous resolution from disk via dask...
            path_to_array = image_path / paths[count]
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

    def __setitem__(self, index: Any, data: Any) -> None:
        """Write *data* at *index* into the underlying zarr array."""
        self.zarray[index] = data

    def __getitem__(self, index: Any) -> Any:
        """Read a slice from the underlying zarr array."""
        return self.zarray[index]

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self.shape)

    @property
    def dtype(self) -> np.dtype:
        """Data type of the underlying zarr array."""
        return self.zarray.dtype

    def finalize(self, res: list | tuple, n_levels: int = 5) -> None:
        """
        Finalize the OME-Zarr with traditional power-of-2 pyramid levels.

        Parameters
        ----------
        res : list of float
            Resolution in mm for each axis (e.g., [0.01, 0.01, 0.01] for 10 µm isotropic)
        n_levels : int
            Number of pyramid levels (default: 5). Each level is 2x downsampled.
        """
        n_levels = validate_n_levels(n_levels, self.shape, self.downscale_factor)
        paths = [f"{i}" for i in range(n_levels + 1)]
        self._downsample_pyramid_on_disk(self.root, paths)
        transformations = create_transformation_dict(n_levels + 1, res, len(self.shape))
        datasets = []
        for p, t in zip(paths, transformations, strict=False):
            datasets.append({"path": p, "coordinateTransformations": t})

        pyramid_kw = {
            "max_layer": n_levels,
            "method": "resize",
            "downscale": self.downscale_factor,
        }

        ome_zarr_version = version("ome-zarr")
        metadata = {
            "method": "linumpy.io.zarr.OmeZarrWriter._downsample_pyramid_on_disk",
            "version": ome_zarr_version,
            "args": pyramid_kw,
        }

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
    >>> writer.finalize(base_res, target_resolutions_um=[10, 25, 50, 100])

    Notes
    -----
    - Use `finalize()` for traditional power-of-2 pyramids (inherited from OmeZarrWriter)
    - Use `finalize_with_resolutions()` for custom analysis-friendly resolutions
    """

    def _downsample_to_resolution(
        self, parent: zarr.Group, source_path: str, target_path: str, target_shape: tuple | list
    ) -> None:
        """Downsample from *source_path* to *target_path* with a specific target shape."""
        group_path = str(parent.store_path)
        # Remove file:// prefix if present (from zarr URL format)
        if group_path.startswith("file://"):
            group_path = group_path[7:]
        img_path = parent.store_path / parent.path
        image_path = Path(group_path) / parent.path

        full_target_path = image_path / target_path
        if full_target_path.exists():
            print(f"Path exists: {full_target_path}")
            return

        # Open source from disk via dask
        path_to_array = image_path / source_path
        dask_image = da.from_zarr(path_to_array)

        output = da_resize(dask_image, tuple(target_shape), preserve_range=True, anti_aliasing=True)

        options = {}
        if self.fmt.zarr_format == 2:
            options["dimension_separator"] = "/"
        else:
            options["chunk_key_encoding"] = self.fmt.chunk_key_encoding
            options["dimension_names"] = [axis["name"] for axis in self.axes]

        da.to_zarr(arr=output, url=img_path, component=target_path, zarr_format=self.fmt.zarr_format, **options)

    def finalize(
        self,
        res: list | tuple,
        n_levels: int | None = None,
        *,
        target_resolutions_um: tuple | Sequence = (10, 25, 50, 100),
        make_isotropic: bool = True,
    ) -> None:
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
                original_path = Path(group_path) / self.root.path / "0"
                temp_full_path = Path(group_path) / self.root.path / temp_path

                if original_path.exists():
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
