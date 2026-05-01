"""High-level zarr → GPU loading with automatic backend selection.

Public entry points:

* :func:`read_zarr_to_gpu` — load an entire zarr array onto the GPU using the
  fastest available path (kvikio / GDS, falling back to ``zarr.config.enable_gpu``).
* :func:`gpu_zarr_context` — context manager that flips zarr into GPU mode for
  its duration, so subsequent ``zarr.open_array(...)`` calls return arrays whose
  slicing materialises directly into ``cupy.ndarray``. Use this for tile-by-tile
  / per-slab access patterns where loading the whole volume at once is wasteful.

Selection order (when ``prefer='auto'``):

1. **kvikio (GPUDirect Storage, native mode)** — chunks DMA'd directly from
   NVMe into GPU memory. Requires ``kvikio`` installed, GDS in native mode,
   and an uncompressed zarr v2/v3.
2. **zarr.config.enable_gpu()** — host I/O with on-host decode then a single
   H→D copy. Works for any zarr (compressed or not). The fallback when GDS
   is unavailable, in compat mode, or the array is compressed.

Backend implementations live in their own modules:

* :mod:`linumpy.gpu.kvikio_zarr` — kvikio / GDS reader.

Reference numbers on a 16 GiB float32 zarr v3 (RTX A6000, ext4, GDS native)
warm cache: kvikio ~9.9 GiB/s, zarr-gpu ~7.1 GiB/s.
"""

from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Iterator

Backend = Literal["auto", "kvikio", "zarr-gpu"]


def _kvikio_native_mode_available() -> bool:
    """Return True iff kvikio is importable and not stuck in compat mode.

    kvikio always succeeds at import; the relevant question is whether
    ``cufile.json`` (or env vars) allow native GDS. The API moved in kvikio
    25.x → 26.x:

    * Older: ``kvikio.defaults.compat_mode()`` → ``CompatMode`` enum
      (``OFF``/``ON``/``AUTO``).
    * 26.04+: ``kvikio.defaults.is_compat_mode_preferred()`` → ``bool``
      (``True`` means kvikio will use the POSIX bounce-buffer path).
    """
    try:
        import kvikio  # noqa: F401
        from kvikio import defaults
    except ImportError:
        return False
    # Newer API first.
    is_compat_pref = getattr(defaults, "is_compat_mode_preferred", None)
    if callable(is_compat_pref):
        try:
            return not bool(is_compat_pref())
        except Exception:  # pragma: no cover - hardware-dependent
            return False
    # Legacy enum API.
    compat_mode = getattr(defaults, "compat_mode", None)
    if callable(compat_mode):
        try:
            mode = compat_mode()
        except Exception:  # pragma: no cover - older kvikio variants
            return False
        name = getattr(mode, "name", str(mode)).upper()
        return name in {"OFF", "AUTO"}
    return False


def _array_is_kvikio_compatible(array_path: Path) -> bool:
    """Return True iff the on-disk array meets kvikio's raw-bytes constraints."""
    from linumpy.gpu.kvikio_zarr import _load_array_spec

    try:
        _load_array_spec(array_path)
    except NotImplementedError, FileNotFoundError, ValueError:
        return False
    return True


def read_zarr_via_zarr_gpu(array_path: str | Path) -> Any:
    """Load a zarr array onto the GPU using ``zarr.config.enable_gpu()``.

    Host I/O with on-host decode, then a single H→D copy. Works for any zarr
    array (including compressed) and is the recommended fallback when GDS is
    unavailable or stuck in compat mode.

    Parameters
    ----------
    array_path
        Path to the zarr array directory.

    Returns
    -------
    cupy.ndarray
        Device-resident array.
    """
    try:
        import cupy
        import zarr
    except ImportError as exc:  # pragma: no cover - hardware-dependent
        raise RuntimeError("cupy + zarr are required for the zarr-gpu fallback path.") from exc

    with zarr.config.enable_gpu():
        z = zarr.open_array(str(array_path), mode="r")
        dev = z[:]
    cupy.cuda.Stream.null.synchronize()
    return dev


def read_zarr_to_gpu(array_path: str | Path, *, prefer: Backend = "auto") -> Any:
    """Load a zarr array onto the GPU using the fastest available path.

    Selection order (when ``prefer='auto'``):

    1. kvikio / GDS — only if kvikio is in native or auto mode AND the array
       is uncompressed v2/v3.
    2. ``zarr.config.enable_gpu()`` — works for any zarr.

    Parameters
    ----------
    array_path
        Path to the zarr array directory.
    prefer
        ``'auto'`` (default), ``'kvikio'``, or ``'zarr-gpu'``. Forcing a path
        will raise if that path is unavailable for this array.

    Returns
    -------
    cupy.ndarray
        Device-resident array of shape and dtype matching the zarr metadata.
    """
    path = Path(array_path)

    if prefer == "kvikio":
        from linumpy.gpu.kvikio_zarr import read_zarr_via_kvikio

        return read_zarr_via_kvikio(path)
    if prefer == "zarr-gpu":
        return read_zarr_via_zarr_gpu(path)
    if prefer != "auto":
        raise ValueError(f"unknown prefer={prefer!r}; expected 'auto', 'kvikio', or 'zarr-gpu'")

    if _kvikio_native_mode_available() and _array_is_kvikio_compatible(path):
        from linumpy.gpu.kvikio_zarr import read_zarr_via_kvikio

        try:
            return read_zarr_via_kvikio(path)
        except RuntimeError, OSError, NotImplementedError:
            pass

    return read_zarr_via_zarr_gpu(path)


@contextmanager
def gpu_zarr_context() -> Iterator[None]:
    """Context manager that puts zarr into GPU mode for arbitrary slice reads.

    Inside this context, any subsequent ``zarr.open_array(...)`` returns an
    array whose slicing produces ``cupy.ndarray`` results — chunks are decoded
    on host then transferred to device on each ``vol[slice]`` operation. This
    is the right pattern for tile-by-tile or per-slab work where loading the
    full volume at once is wasteful.

    Outside this context, zarr falls back to its normal numpy-backed mode.

    Examples
    --------
    >>> from linumpy.gpu.zarr_io import gpu_zarr_context
    >>> from linumpy.io.zarr import read_omezarr
    >>> with gpu_zarr_context():
    ...     vol, _ = read_omezarr(path, level=0)
    ...     for tile_region in regions:
    ...         tile_gpu = vol[tile_region]  # already cupy

    Raises
    ------
    RuntimeError
        If zarr is unavailable.
    """
    try:
        import zarr
    except ImportError as exc:  # pragma: no cover - zarr is a hard dep
        raise RuntimeError("zarr is required for gpu_zarr_context().") from exc

    with zarr.config.enable_gpu():
        yield
