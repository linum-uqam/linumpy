r"""Comprehensive N4 GPU vs SimpleITK benchmark.

Runs accuracy + timing comparisons on:
  1. A scaling sweep of synthetic phantoms.
  2. Real OCT slices from the linum-uqam pipeline.

Writes a JSON report to ``<output>/n4_gpu_benchmark.json`` and a Markdown
report (table + bullets) to ``<output>/n4_gpu_benchmark.md``.

This is the script behind the published numbers in ``docs/N4_GPU.md``.

Usage on the lab server::

    uv run python scripts/diagnostics/linum_benchmark_n4_gpu.py \\
        --output /tmp/n4_bench \\
        --live-zarr /scratch/workspace/sub-22/output/01/fix_illumination/mosaic_grid_z01_illum_fix.ome.zarr
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from linumpy.intensity.bias_field import n4_correct

# ---------------------------------------------------------------------------
# Synthetic phantom (matches test_n4_gpu_equivalency.py)
# ---------------------------------------------------------------------------


def _make_phantom(shape, bias_amp=0.5, seed=0):
    rng = np.random.default_rng(seed)
    z, y, x = shape
    zg, yg, xg = np.mgrid[0:z, 0:y, 0:x].astype(np.float32)
    cz, cy, cx = z / 2, y / 2, x / 2
    r = np.sqrt(((zg - cz) / (z / 3)) ** 2 + ((yg - cy) / (y / 3)) ** 2 + ((xg - cx) / (x / 3)) ** 2)
    truth = np.where(r < 1.0, 1.0, 0.3).astype(np.float32) + rng.normal(0.0, 0.02, size=shape).astype(np.float32)
    mask = r < 1.2
    z_norm, y_norm, x_norm = (zg - cz) / z, (yg - cy) / y, (xg - cx) / x
    bias = (
        1.0
        + bias_amp * (z_norm + 0.5 * y_norm - 0.5 * x_norm)
        + 0.5 * bias_amp * np.cos(np.pi * z_norm) * np.cos(np.pi * y_norm)
    )
    bias = np.clip(bias, 0.4, 2.5).astype(np.float32)
    return (truth * bias).astype(np.float32), bias, mask


def _bias_recovery_cv(estimated, truth, mask):
    ratio = (estimated / truth)[mask]
    return float(np.std(ratio) / np.mean(ratio))


def _residual_cv(corrected, mask_interior):
    region = corrected[mask_interior]
    return float(np.std(region) / np.mean(region))


# ---------------------------------------------------------------------------
# Run a single comparison
# ---------------------------------------------------------------------------


def _time_call(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0


def _compare(vol, mask, true_bias, *, shrink_factor, n_iter, spline_distance_mm, label):
    # Warm up GPU
    n4_correct(
        vol[:8, :64, :64], None, shrink_factor=2, n_iterations=[3], backend="gpu", spline_distance_mm=spline_distance_mm
    )

    (corr_cpu, bias_cpu), t_cpu = _time_call(
        n4_correct,
        vol,
        mask,
        shrink_factor=shrink_factor,
        n_iterations=n_iter,
        spline_distance_mm=spline_distance_mm,
        backend="cpu",
    )
    (corr_gpu, bias_gpu), t_gpu = _time_call(
        n4_correct,
        vol,
        mask,
        shrink_factor=shrink_factor,
        n_iterations=n_iter,
        spline_distance_mm=spline_distance_mm,
        backend="gpu",
    )

    record = {
        "label": label,
        "shape": list(vol.shape),
        "shrink_factor": shrink_factor,
        "n_iter": n_iter,
        "spline_distance_mm": spline_distance_mm,
        "t_cpu_s": t_cpu,
        "t_gpu_s": t_gpu,
        "speedup": t_cpu / max(t_gpu, 1e-9),
    }

    if true_bias is not None:
        m = mask if mask is not None else np.ones_like(vol, dtype=bool)
        record["cv_bias_cpu"] = _bias_recovery_cv(bias_cpu, true_bias, m)
        record["cv_bias_gpu"] = _bias_recovery_cv(bias_gpu, true_bias, m)

    if mask is not None:
        norm_cpu = bias_cpu / float(np.mean(bias_cpu[mask]))
        norm_gpu = bias_gpu / float(np.mean(bias_gpu[mask]))
        a, b = norm_cpu[mask].ravel(), norm_gpu[mask].ravel()
        record["bias_correlation"] = float(np.corrcoef(a, b)[0, 1])

        cn = corr_cpu / float(np.mean(corr_cpu[mask]))
        gn = corr_gpu / float(np.mean(corr_gpu[mask]))
        rel = np.abs(cn - gn)[mask] / max(float(np.mean(cn[mask])), 1e-6)
        record["median_corrected_rel_err"] = float(np.median(rel))
        record["p95_corrected_rel_err"] = float(np.percentile(rel, 95))

    record["mean_input"] = float(vol.mean())
    record["mean_corr_cpu"] = float(corr_cpu.mean())
    record["mean_corr_gpu"] = float(corr_gpu.mean())

    print(
        f"[{label}] shape={vol.shape} cpu={t_cpu:.2f}s gpu={t_gpu:.2f}s "
        f"speedup={record['speedup']:.2f}x"
        + (f" cv_cpu={record['cv_bias_cpu']:.3f} cv_gpu={record['cv_bias_gpu']:.3f}" if true_bias is not None else "")
        + (
            f" r={record['bias_correlation']:.3f} median_relerr={record['median_corrected_rel_err']:.3f}"
            if mask is not None
            else ""
        )
    )
    return record


# ---------------------------------------------------------------------------
# Live OCT slice
# ---------------------------------------------------------------------------


def _load_live_volume(zarr_path: Path, level: int = 0, slice_index: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Load an OME-Zarr volume (handles `.ome.zarr` directories and `.ome.zarr.zip` archives).

    If ``slice_index`` is given, returns a single Z-slice (one serial section).
    """
    import zarr

    if str(zarr_path).endswith(".zip"):
        store = zarr.storage.ZipStore(str(zarr_path), mode="r")
        # OME-Zarr-zip archives often wrap the dataset in a top-level
        # directory named after the subject (e.g. ``sub-22.ome.zarr/``).
        # Discover the inner group prefix from the archive.
        inner_prefix = ""
        try:
            root = zarr.open(store, mode="r")
        except Exception:
            import zipfile

            with zipfile.ZipFile(str(zarr_path)) as zf:
                names = zf.namelist()
            top_dirs = sorted({n.split("/", 1)[0] for n in names if "/" in n})
            inner_prefix = top_dirs[0]
            root = zarr.open(store, mode="r", path=inner_prefix)
    else:
        root = zarr.open(str(zarr_path), mode="r")
    arr = np.asarray(root[str(level)][...], dtype=np.float32)
    while arr.ndim > 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D OME-Zarr after squeeze, got shape {arr.shape}")
    if slice_index is not None:
        # Pick a single serial section: 1 along Z (synthetic stack convention).
        # The stacked volume is (Z=sections * section_thickness, Y, X).  Estimate
        # section thickness as Z // n_sections; fall back to a fixed 64-voxel slab.
        thickness = max(arr.shape[0] // 50, 32)
        z0 = slice_index * thickness
        arr = arr[z0 : z0 + thickness]
    log_v = np.log(np.maximum(arr, 1e-6))
    thr = np.percentile(log_v, 5.0)
    mask = log_v > thr
    return arr, mask


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    """Run the N4 GPU vs SimpleITK benchmark."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument(
        "--live-zarr",
        type=Path,
        default=None,
        help="OME-Zarr stacked volume (.ome.zarr or .ome.zarr.zip) for live-data benchmark.",
    )
    p.add_argument("--live-level", type=int, default=1, help="Pyramid level to load from the live OME-Zarr [%(default)s].")
    p.add_argument(
        "--live-slice-index",
        type=int,
        default=None,
        help="If set, benchmark a single serial section starting at this slice index.",
    )
    p.add_argument(
        "--max-live-shape",
        type=int,
        nargs=3,
        default=[128, 1024, 1024],
        help="Crop the live volume to at most this (Z, Y, X) for benchmarking.",
    )
    args = p.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    # ---- Synthetic scaling sweep ----
    print("\n=== Synthetic scaling sweep ===")
    sweep = [
        ((64, 128, 128), 2),
        ((128, 256, 256), 2),
        ((128, 512, 512), 2),
        ((256, 512, 512), 2),
        ((128, 1024, 1024), 4),
        ((128, 1536, 1536), 4),
    ]
    for shape, sf in sweep:
        vol, true_bias, mask = _make_phantom(shape, bias_amp=0.5)
        records.append(
            _compare(
                vol,
                mask,
                true_bias,
                shrink_factor=sf,
                n_iter=[25, 25, 25],
                spline_distance_mm=20.0,
                label=f"phantom_{shape[0]}x{shape[1]}x{shape[2]}",
            )
        )

    # ---- Live OCT volume ----
    if args.live_zarr is not None and args.live_zarr.exists():
        print(f"\n=== Live OCT volume: {args.live_zarr} (level={args.live_level}) ===")
        vol, mask = _load_live_volume(args.live_zarr, level=args.live_level, slice_index=args.live_slice_index)
        zc, yc, xc = (min(s, c) for s, c in zip(vol.shape, args.max_live_shape, strict=True))
        vol = vol[:zc, :yc, :xc].copy()
        mask = mask[:zc, :yc, :xc].copy()
        print(f"  live volume shape={vol.shape}, mask coverage={float(mask.mean()):.2%}")
        records.append(
            _compare(
                vol,
                mask,
                None,
                shrink_factor=4,
                n_iter=[40, 40, 40],
                spline_distance_mm=10.0,
                label="live_oct" + (f"_slice{args.live_slice_index}" if args.live_slice_index is not None else "_full"),
            )
        )

    # ---- Write reports ----
    json_path = args.output / "n4_gpu_benchmark.json"
    md_path = args.output / "n4_gpu_benchmark.md"
    json_path.write_text(json.dumps(records, indent=2))

    lines = ["# N4 GPU vs SimpleITK benchmark", ""]
    lines.append(
        "| Volume | shrink | iters | CPU (s) | GPU (s) | Speedup | r(bias) | median |Δ|/mean | CV bias CPU | CV bias GPU |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in records:
        shape = "x".join(str(s) for s in r["shape"])
        n_iter_str = ",".join(str(n) for n in r["n_iter"])
        lines.append(
            f"| {r['label']} ({shape}) | {r['shrink_factor']} | {n_iter_str} | "
            f"{r['t_cpu_s']:.2f} | {r['t_gpu_s']:.2f} | **{r['speedup']:.2f}x** | "
            f"{r.get('bias_correlation', float('nan')):.3f} | "
            f"{r.get('median_corrected_rel_err', float('nan')):.3f} | "
            f"{r.get('cv_bias_cpu', float('nan')):.3f} | {r.get('cv_bias_gpu', float('nan')):.3f} |"
        )
    md_path.write_text("\n".join(lines) + "\n")

    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
