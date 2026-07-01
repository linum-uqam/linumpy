"""GPU-accelerated zstd codec for zarr-python, backed by nvCOMP.

This is a vendored stand-in for upstream zarr-python PR #2863 (`zarr.codecs.gpu.NvcompZstdCodec`),
which is approved but not yet released. The codec implementation matches that PR closely so we
can swap to the upstream class once it lands without changing call sites.

Why this exists
---------------
With ``zarr.config.enable_gpu()``, zarr-python (≤3.2) reads zstd-compressed chunks by:

1. ``GDSStore`` (or any store) yields a CPU buffer.
2. The default ``zarr.codecs.zstd.ZstdCodec`` decodes on the CPU (numcodecs).
3. The decoded chunk is copied H→D into a ``cupy`` buffer.

That CPU decode + H→D copy is the bottleneck for compressed tiles. This module replaces step 2:
chunks are uploaded to the GPU and decoded with ``nvidia.nvcomp.Codec("Zstd")``, so the result
already lives on device.

Usage
-----
The codec is registered under the name ``"zstd"`` (same as the default), so any existing zarr
file with zstd-compressed chunks will use it once registered. Registration happens lazily from
:mod:`linumpy.gpu.zarr_io` whenever a GPU read path is taken, so CPU-only workflows are
unaffected.

References
----------
* https://github.com/zarr-developers/zarr-python/pull/2863
* https://docs.nvidia.com/cuda/nvcomp/
"""

import asyncio
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np
from zarr.abc.codec import BytesBytesCodec
from zarr.core.common import JSON, parse_named_configuration

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Self

    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer import Buffer


def _parse_zstd_level(data: JSON) -> int:
    if isinstance(data, int):
        if data >= 23:
            raise ValueError(f"Value must be less than or equal to 22. Got {data} instead.")
        return data
    raise TypeError(f"Got value with type {type(data)}, but expected an int.")


def _parse_checksum(data: JSON) -> bool:
    if isinstance(data, bool):
        return data
    raise TypeError(f"Expected bool. Got {type(data)}.")


@dataclass(frozen=True)
class NvcompZstdCodec(BytesBytesCodec):
    """zstd codec that decodes/encodes on an NVIDIA GPU via nvCOMP."""

    is_fixed_size = True

    level: int = 0
    checksum: bool = False

    def __init__(self, *, level: int = 0, checksum: bool = False) -> None:
        level_parsed = _parse_zstd_level(level)
        checksum_parsed = _parse_checksum(checksum)
        object.__setattr__(self, "level", level_parsed)
        object.__setattr__(self, "checksum", checksum_parsed)

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        """Construct codec from its serialised metadata representation."""
        _, configuration_parsed = parse_named_configuration(data, "zstd")
        cfg: dict[str, Any] = dict(configuration_parsed)  # type: ignore[arg-type]
        level = cfg.get("level", 0)
        checksum = cfg.get("checksum", False)
        if not isinstance(level, int):
            raise TypeError(f"zstd codec level must be int, got {type(level).__name__}")
        if not isinstance(checksum, bool):
            raise TypeError(f"zstd codec checksum must be bool, got {type(checksum).__name__}")
        return cls(level=level, checksum=checksum)

    def to_dict(self) -> dict[str, JSON]:
        """Return the codec's metadata dict (named ``zstd`` for on-disk compatibility)."""
        return {
            "name": "zstd",
            "configuration": {"level": self.level, "checksum": self.checksum},
        }

    @cached_property
    def _zstd_codec(self) -> Any:
        import cupy as cp
        from nvidia import nvcomp

        device = cp.cuda.Device()
        stream = cp.cuda.get_current_stream()
        return nvcomp.Codec(
            algorithm="Zstd",
            bitstream_kind=nvcomp.BitstreamKind.RAW,
            device_id=device.id,
            cuda_stream=stream.ptr,
        )

    def _convert_to_nvcomp_arrays(
        self,
        chunks_and_specs: Iterable[tuple[Buffer | None, ArraySpec]],
    ) -> tuple[list[Any], list[int]]:
        from nvidia import nvcomp

        none_indices = [i for i, (b, _) in enumerate(chunks_and_specs) if b is None]
        filtered_inputs = [b.as_array_like() for b, _ in chunks_and_specs if b is not None]
        return nvcomp.as_arrays(filtered_inputs), none_indices

    def _convert_from_nvcomp_arrays(
        self,
        arrays: Iterable[Any],
        chunks_and_specs: Iterable[tuple[Buffer | None, ArraySpec]],
    ) -> Iterable[Buffer | None]:
        import cupy as cp

        result: list[Buffer | None] = []
        for a, (_, spec) in zip(arrays, chunks_and_specs, strict=True):
            if a is None:
                result.append(None)
            else:
                a2 = cp.array(a, dtype=a.dtype, copy=False)
                if a2.dtype != np.dtype("B"):
                    a2 = a2.view(dtype=np.dtype("B"))
                result.append(spec.prototype.buffer.from_array_like(a2))
        return result

    async def decode(
        self,
        chunks_and_specs: Iterable[tuple[Buffer | None, ArraySpec]],
    ) -> Iterable[Buffer | None]:
        """Decode a batch of zstd-compressed chunks on the GPU via nvCOMP."""
        import cupy as cp

        chunks_and_specs = list(chunks_and_specs)
        filtered_inputs, none_indices = self._convert_to_nvcomp_arrays(chunks_and_specs)
        outputs = self._zstd_codec.decode(filtered_inputs) if len(filtered_inputs) > 0 else []
        event = cp.cuda.Event()
        event.record()
        await asyncio.to_thread(event.synchronize)
        outputs = list(outputs)
        for index in none_indices:
            outputs.insert(index, None)
        return self._convert_from_nvcomp_arrays(outputs, chunks_and_specs)

    async def encode(
        self,
        chunks_and_specs: Iterable[tuple[Buffer | None, ArraySpec]],
    ) -> Iterable[Buffer | None]:
        """Encode a batch of chunks with zstd on the GPU via nvCOMP."""
        import cupy as cp

        chunks_and_specs = list(chunks_and_specs)
        filtered_inputs, none_indices = self._convert_to_nvcomp_arrays(chunks_and_specs)
        outputs = self._zstd_codec.encode(filtered_inputs) if len(filtered_inputs) > 0 else []
        event = cp.cuda.Event()
        event.record()
        await asyncio.to_thread(event.synchronize)
        outputs = list(outputs)
        for index in none_indices:
            outputs.insert(index, None)
        return self._convert_from_nvcomp_arrays(outputs, chunks_and_specs)

    def compute_encoded_size(self, input_byte_length: int, chunk_spec: ArraySpec) -> int:
        """Raise NotImplementedError — encoded size is data-dependent for zstd."""
        del input_byte_length, chunk_spec
        raise NotImplementedError


_REGISTERED = False


def register_nvcomp_zstd() -> bool:
    """Register :class:`NvcompZstdCodec` under the name ``zstd``.

    Idempotent. Returns True if registration succeeded (or was already done), False if the
    required GPU dependencies (``cupy``, ``nvidia.nvcomp``) are not importable.

    Note that registration alone is not enough to make zarr use the GPU codec — zarr 3.2
    selects the concrete class via ``zarr.config["codecs.zstd"]`` and the default points at
    the CPU codec. Use :func:`gpu_zstd_config` (or rely on
    :func:`linumpy.gpu.zarr_io.gpu_zarr_context`) to flip the config for a scoped block.
    """
    global _REGISTERED
    if _REGISTERED:
        return True
    try:
        import cupy  # noqa: F401
        from nvidia import nvcomp  # noqa: F401
    except ImportError:
        return False
    from zarr.registry import register_codec

    register_codec("zstd", NvcompZstdCodec)
    _REGISTERED = True
    return True


def gpu_zstd_config() -> dict[str, str]:
    """Return the zarr config overrides that route the ``zstd`` codec through nvCOMP.

    Pass the result to ``zarr.config.set(...)`` to enable GPU-side zstd decoding for the
    duration of the ``set`` context. Caller is responsible for calling
    :func:`register_nvcomp_zstd` first (otherwise zarr cannot resolve the class name).
    """
    return {"codecs.zstd": "linumpy.gpu.nvcomp_zstd.NvcompZstdCodec"}
