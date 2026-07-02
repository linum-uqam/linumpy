#!/usr/bin/env python3
r"""Benchmark zarr → GPU loading: kvikio (GDS) vs zarr+cupy vs zarr.enable_gpu.

Three paths to get an uncompressed zarr v3 array onto the GPU are compared:

  A) ``zarr.open(...)[:]`` → ``np.asarray`` → ``cupy.asarray`` (legacy path).
  B) ``zarr.config.enable_gpu(); arr[:]`` returns a ``cupy.ndarray`` directly,
     but the codec pipeline still decodes on the CPU before the H→D copy.
  C) ``linumpy.gpu.kvikio_zarr.read_zarr_via_kvikio`` — kvikio CuFile / GDS.
  D) ``linumpy.gpu.zarr_io.read_zarr_to_gpu`` — auto-dispatch (kvikio if GDS
     native + uncompressed, else zarr-gpu).

Reference numbers on a 16 GiB float32 zarr v3 (256³ chunks) on /scratch_nvme,
RTX A6000, ext4, GDS native mode (warm cache):

* kvikio (GDS native):     ~9.9 GiB/s
* zarr.config.enable_gpu:  ~7.1 GiB/s
* zarr → numpy → cupy:     ~2.8 GiB/s

For production code, prefer :func:`linumpy.gpu.zarr_io.read_zarr_to_gpu`
which picks the best available path at runtime.

Examples
--------
Generate a 16 GiB random uncompressed zarr v3 on /scratch_nvme and bench::

    linum_benchmark_kvikio_zarr.py \
        --generate /scratch_nvme/bench.zarr \
        --shape 2048 2048 1024 --chunks 256 256 128 --dtype float32 \
        --runs 3
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import time
from pathlib import Path

import numpy as np


def _human_bytes(n: float) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    i = 0
    while n >= 1024 and i < len(units) - 1:
        n /= 1024
        i += 1
    return f"{n:.2f} {units[i]}"


def _generate_dataset(path: Path, shape: tuple[int, ...], chunks: tuple[int, ...], dtype: str) -> None:
    import zarr

    print(f"[generate] shape={shape} chunks={chunks} dtype={dtype} path={path}")
    if path.exists():
        raise SystemExit(f"refusing to overwrite existing {path}")

    # zarr v3 with raw bytes only (no compression) so the kvikio path works.
    arr = zarr.create_array(
        store=str(path),
        shape=shape,
        chunks=chunks,
        dtype=dtype,
        compressors=None,
        zarr_format=3,
        fill_value=0,
    )
    rng = np.random.default_rng(0)
    z_step = chunks[0]
    for z0 in range(0, shape[0], z_step):
        z1 = min(z0 + z_step, shape[0])
        block = rng.standard_normal((z1 - z0, *shape[1:])).astype(dtype, copy=False)
        arr[z0:z1] = block
    print(f"[generate] done: {_human_bytes(arr.nbytes)} written")


def _bench_zarr_cupy(array_path: Path) -> tuple[float, int]:
    import cupy
    import zarr

    t0 = time.perf_counter()
    z = zarr.open_array(str(array_path), mode="r")
    host = np.asarray(z[:])
    dev = cupy.asarray(host)
    cupy.cuda.Stream.null.synchronize()
    t1 = time.perf_counter()
    return t1 - t0, dev.nbytes


def _bench_zarr_gpu(array_path: Path) -> tuple[float, int]:
    """zarr.config.enable_gpu(): host I/O, host decode, final buffer is CuPy."""
    import cupy
    import zarr

    with zarr.config.enable_gpu():
        t0 = time.perf_counter()
        z = zarr.open_array(str(array_path), mode="r")
        dev = z[:]
        cupy.cuda.Stream.null.synchronize()
        t1 = time.perf_counter()
    nbytes = int(np.prod(z.shape)) * np.dtype(z.dtype).itemsize
    del dev  # keep reference alive through synchronize
    return t1 - t0, nbytes


def _bench_kvikio(array_path: Path) -> tuple[float, int]:
    import cupy

    from linumpy.gpu.kvikio_zarr import read_zarr_via_kvikio

    t0 = time.perf_counter()
    dev = read_zarr_via_kvikio(array_path)
    cupy.cuda.Stream.null.synchronize()
    t1 = time.perf_counter()
    return t1 - t0, dev.nbytes


def _bench_auto(array_path: Path) -> tuple[float, int]:
    import cupy

    from linumpy.gpu.zarr_io import read_zarr_to_gpu

    t0 = time.perf_counter()
    dev = read_zarr_to_gpu(array_path)
    cupy.cuda.Stream.null.synchronize()
    t1 = time.perf_counter()
    return t1 - t0, dev.nbytes


def _drop_caches() -> None:
    """Best-effort page-cache drop (requires root). Silently skipped otherwise."""
    with contextlib.suppress(PermissionError, FileNotFoundError, OSError):
        Path("/proc/sys/vm/drop_caches").write_text("3\n")


PATHS = {
    "zarr+cupy": _bench_zarr_cupy,
    "zarr-gpu": _bench_zarr_gpu,
    "kvikio": _bench_kvikio,
    "auto": _bench_auto,
}


def main() -> None:
    """Run the benchmark from the command line."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("array_path", type=Path, nargs="?", help="zarr array directory")
    p.add_argument("--generate", type=Path, help="create a synthetic uncompressed zarr v3 at this path")
    p.add_argument("--shape", type=int, nargs="+", default=[2048, 2048, 512])
    p.add_argument("--chunks", type=int, nargs="+", default=[256, 256, 128])
    p.add_argument("--dtype", default="float32")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--skip-cache-drop", action="store_true", help="don't try to drop page cache between runs")
    p.add_argument("--only", action="append", choices=list(PATHS), help="only benchmark these paths (repeatable)")
    args = p.parse_args()

    if args.generate is not None:
        _generate_dataset(args.generate, tuple(args.shape), tuple(args.chunks), args.dtype)
        if args.array_path is None:
            args.array_path = args.generate

    if args.array_path is None:
        p.error("array_path is required (unless --generate also gives one)")

    selected = args.only or list(PATHS)
    print(f"[bench] array={args.array_path} runs={args.runs}")
    for name in selected:
        fn = PATHS[name]
        times: list[float] = []
        nbytes = 0
        for r in range(args.runs):
            if not args.skip_cache_drop:
                _drop_caches()
            try:
                dt, nbytes = fn(args.array_path)
            except Exception as exc:
                print(f"  [{name}] run {r}: ERROR {exc!r}")
                break
            tput = nbytes / dt / (1024**3)
            times.append(dt)
            print(f"  [{name}] run {r}: {dt:.3f}s  {tput:.2f} GiB/s")
        if times:
            best = min(times)
            print(f"  [{name}] best: {best:.3f}s  {nbytes / best / (1024**3):.2f} GiB/s")


if __name__ == "__main__":
    sys.exit(main())
