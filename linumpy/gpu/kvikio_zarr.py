"""kvikio / GPUDirect Storage reader for uncompressed zarr arrays.

This module implements the :func:`read_zarr_via_kvikio` backend used by
:mod:`linumpy.gpu.zarr_io`. It reads zarr v2 *or* zarr v3 chunks directly
into GPU memory using ``kvikio.CuFile``. The version is auto-detected from
the on-disk metadata file (``zarr.json`` for v3, ``.zarray`` for v2).

Most callers should use :func:`linumpy.gpu.zarr_io.read_zarr_to_gpu`, which
dispatches to this backend only when GDS native mode is available and the
array is uncompressed; otherwise it falls back to ``zarr.config.enable_gpu``.

Supported on-disk formats
-------------------------

* zarr v3: ``codecs=[{"name": "bytes"}]`` only (or empty) — raw little/big-
  endian bytes. Any compression codec (blosc, gzip, zstd, ...) is rejected
  because GDS reads bytes verbatim and on-device decompression would require
  nvCOMP.
* zarr v2: ``compressor=None``, ``filters=None``, ``order='C'``.

Notes
-----
* Requires ``kvikio`` and ``cupy``. Both are imported lazily so the rest of
  linumpy is unaffected if they are not installed.
* For the GDS fast path the source filesystem must support GDS natively
  (ext4 on local NVMe, IOMMU disabled or in passthrough, and
  ``properties.use_compat_mode=false`` in ``/etc/cufile.json``). Otherwise
  kvikio falls back to a posix bounce-buffer path with no speed-up.
"""

from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


def _require_kvikio() -> tuple[Any, Any]:
    """Import kvikio + cupy lazily and return (kvikio, cupy)."""
    try:
        import cupy
        import kvikio
    except ImportError as exc:  # pragma: no cover - hardware-dependent
        raise RuntimeError(
            "kvikio + cupy are required for the GDS prototype. Install with:\n"
            "    pip install kvikio-cu13 cupy-cuda13x  # or matching your CUDA"
        ) from exc
    return kvikio, cupy


def _parse_v3_dtype(dt: Any) -> np.dtype:
    """Map a zarr v3 ``data_type`` field to a numpy dtype."""
    if isinstance(dt, str):
        return np.dtype(dt)
    raise NotImplementedError(f"unsupported v3 data_type: {dt!r}")


def _v3_chunk_path(array_path: Path, idx: tuple[int, ...], encoding: dict) -> Path:
    """Build the on-disk chunk path for a zarr v3 array."""
    name = encoding.get("name", "default")
    sep = encoding.get("configuration", {}).get("separator", "/")
    parts = sep.join(str(i) for i in idx)
    if name == "default":
        return array_path / "c" / parts if sep == "/" else array_path / f"c{sep}{parts}"
    if name == "v2":
        return array_path / parts
    raise NotImplementedError(f"unsupported chunk_key_encoding: {name!r}")


def _v2_chunk_path(array_path: Path, idx: tuple[int, ...], dim_separator: str) -> Path:
    return array_path / dim_separator.join(str(i) for i in idx)


class _ArraySpec:
    """Resolved, format-agnostic view of a zarr array on disk."""

    __slots__ = ("_v2_dim_sep", "_v3_encoding", "chunks", "dtype", "fill_value", "format", "path", "shape")

    def __init__(
        self,
        *,
        path: Path,
        shape: tuple[int, ...],
        chunks: tuple[int, ...],
        dtype: np.dtype,
        fill_value: Any,
        format: int,
        v3_encoding: dict | None = None,
        v2_dim_sep: str | None = None,
    ) -> None:
        self.path = path
        self.shape = shape
        self.chunks = chunks
        self.dtype = dtype
        self.fill_value = fill_value
        self.format = format
        self._v3_encoding = v3_encoding
        self._v2_dim_sep = v2_dim_sep

    def chunk_path(self, idx: tuple[int, ...]) -> Path:
        if self.format == 3:
            assert self._v3_encoding is not None
            return _v3_chunk_path(self.path, idx, self._v3_encoding)
        assert self._v2_dim_sep is not None
        return _v2_chunk_path(self.path, idx, self._v2_dim_sep)


def _load_array_spec(array_path: Path) -> _ArraySpec:
    """Inspect ``array_path`` and return a normalized spec for v2 or v3."""
    v3_meta = array_path / "zarr.json"
    v2_meta = array_path / ".zarray"

    if v3_meta.exists():
        meta = json.loads(v3_meta.read_text())
        if meta.get("zarr_format") != 3:
            raise ValueError(f"{array_path}: zarr.json has zarr_format != 3")
        if meta.get("node_type") != "array":
            raise ValueError(f"{array_path}: zarr.json is not an array node")
        codecs = meta.get("codecs", [])
        non_bytes = [c for c in codecs if c.get("name") != "bytes"]
        if non_bytes:
            raise NotImplementedError(
                f"{array_path}: codecs={[c.get('name') for c in codecs]!r}; this prototype "
                "requires raw bytes only (no compression). On-device decompression needs nvCOMP."
            )
        host_endian = "big" if sys.byteorder == "big" else "little"
        for c in codecs:
            endian = c.get("configuration", {}).get("endian", "little")
            if endian != host_endian:
                raise NotImplementedError(f"{array_path}: endian={endian!r} differs from host")
        chunk_grid = meta.get("chunk_grid", {})
        if chunk_grid.get("name") != "regular":
            raise NotImplementedError(f"{array_path}: chunk_grid={chunk_grid.get('name')!r}")
        return _ArraySpec(
            path=array_path,
            shape=tuple(meta["shape"]),
            chunks=tuple(chunk_grid["configuration"]["chunk_shape"]),
            dtype=_parse_v3_dtype(meta["data_type"]),
            fill_value=meta.get("fill_value", 0),
            format=3,
            v3_encoding=meta.get("chunk_key_encoding", {"name": "default", "configuration": {"separator": "/"}}),
        )

    if v2_meta.exists():
        meta = json.loads(v2_meta.read_text())
        if meta.get("zarr_format") != 2:
            raise ValueError(f"{array_path}: .zarray has zarr_format != 2")
        if meta.get("compressor") is not None:
            raise NotImplementedError(
                f"{array_path}: compressor={meta['compressor']!r}; this prototype "
                "requires uncompressed chunks. On-device decompression needs nvCOMP."
            )
        if meta.get("order", "C") != "C":
            raise NotImplementedError(f"{array_path}: order={meta['order']!r} unsupported")
        if meta.get("filters"):
            raise NotImplementedError(f"{array_path}: filters unsupported in prototype")
        return _ArraySpec(
            path=array_path,
            shape=tuple(meta["shape"]),
            chunks=tuple(meta["chunks"]),
            dtype=np.dtype(meta["dtype"]),
            fill_value=meta.get("fill_value", 0),
            format=2,
            v2_dim_sep=meta.get("dimension_separator", "."),
        )

    raise FileNotFoundError(f"{array_path}: no zarr.json (v3) or .zarray (v2) found")


def _iter_chunk_indices(shape: Iterable[int], chunks: Iterable[int]) -> Iterator[tuple[int, ...]]:
    n = [(s + c - 1) // c for s, c in zip(shape, chunks, strict=True)]
    yield from product(*[range(k) for k in n])


def read_zarr_via_kvikio(array_path: str | Path) -> Any:
    """Load a full uncompressed zarr (v2 or v3) array into a CuPy array via GDS.

    Parameters
    ----------
    array_path
        Path to the zarr array directory (containing ``zarr.json`` for v3 or
        ``.zarray`` for v2).

    Returns
    -------
    cupy.ndarray
        Device-resident array of shape and dtype matching the zarr metadata.
    """
    kvikio, cupy = _require_kvikio()
    spec = _load_array_spec(Path(array_path))

    out = cupy.full(spec.shape, spec.fill_value, dtype=spec.dtype)
    chunk_nbytes_full = int(np.prod(spec.chunks) * spec.dtype.itemsize)
    scratch = cupy.empty(spec.chunks, dtype=spec.dtype)

    for idx in _iter_chunk_indices(spec.shape, spec.chunks):
        cf_path = spec.chunk_path(idx)
        if not cf_path.exists():
            continue  # zarr fill-value semantics

        slices: list[slice] = []
        edge_shape: list[int] = []
        for k, c, s in zip(idx, spec.chunks, spec.shape, strict=True):
            start = k * c
            stop = min(start + c, s)
            slices.append(slice(start, stop))
            edge_shape.append(stop - start)
        edge_shape_t = tuple(edge_shape)

        # kvikio.CuFile.pread requires a contiguous device buffer; a slice
        # into ``out`` is generally not contiguous, so we read into a
        # chunk-shaped scratch and copy the valid region into ``out``.
        with kvikio.CuFile(str(cf_path), "r") as f:
            f.pread(scratch, chunk_nbytes_full).get()
        if edge_shape_t == spec.chunks:
            out[tuple(slices)] = scratch
        else:
            sub = tuple(slice(0, e) for e in edge_shape_t)
            out[tuple(slices)] = scratch[sub]

    return out
