#!/usr/bin/env python
"""Render a CPU vs GPU N4 visual comparison on a live OCT slab.

Loads a slab from an OME-Zarr-zip stacked volume, runs CPU SimpleITK and GPU
N4, and writes a side-by-side PNG (input | CPU corrected | GPU corrected |
|CPU - GPU|) for documentation.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr
import zarr.storage

from linumpy.intensity.bias_field import n4_correct


def _load_slab(zarr_path: Path, level: int, z0: int, dz: int):
    if str(zarr_path).endswith(".zip"):
        store = zarr.storage.ZipStore(str(zarr_path), mode="r")
        try:
            root = zarr.open(store, mode="r")
        except Exception:
            import zipfile

            with zipfile.ZipFile(str(zarr_path)) as zf:
                names = zf.namelist()
            top = min(n.split("/", 1)[0] for n in names if "/" in n)
            root = zarr.open(store, mode="r", path=top)
    else:
        root = zarr.open(str(zarr_path), mode="r")
    assert isinstance(root, zarr.Group)
    level_arr = root[str(level)]
    assert isinstance(level_arr, zarr.Array)
    arr = np.asarray(level_arr[...], dtype=np.float32)
    while arr.ndim > 3 and arr.shape[0] == 1:
        arr = arr[0]
    arr = arr[z0 : z0 + dz]
    log_v = np.log(np.maximum(arr, 1e-6))
    mask = log_v > np.percentile(log_v, 5.0)
    return arr, mask


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zarr", required=True, type=Path)
    p.add_argument("--level", type=int, default=1)
    p.add_argument("--z0", type=int, default=0)
    p.add_argument("--dz", type=int, default=64)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--shrink", type=int, default=4)
    p.add_argument("--spline-mm", type=float, default=10.0)
    args = p.parse_args()

    vol, mask = _load_slab(args.zarr, args.level, args.z0, args.dz)
    print(f"slab shape={vol.shape}  mask coverage={mask.mean():.2%}")

    print("running CPU N4 (SimpleITK)...")
    corr_cpu, bias_cpu = n4_correct(
        vol,
        mask,
        shrink_factor=args.shrink,
        n_iterations=[40, 40, 40],
        spline_distance_mm=args.spline_mm,
        backend="cpu",
    )
    print("running GPU N4...")
    # Use the GPU backend's own defaults (fewer iterations, narrower
    # FWHM): the GPU PSDB residual update is undampened compared to
    # SimpleITK's BSplineSmoothingFilter, so identical iteration counts
    # would over-fit the bias and absorb true tissue contrast.
    corr_gpu, bias_gpu = n4_correct(
        vol,
        mask,
        shrink_factor=args.shrink,
        spline_distance_mm=args.spline_mm,
        backend="gpu",
    )

    # Quantitative agreement: bias-field Pearson r and WM/GM contrast
    # preservation.  WM/GM contrast is summarised by the spread of the
    # foreground log-intensity distribution: a wider spread (larger
    # p90 - p10) means tissue contrast is preserved; a narrower spread
    # means the bias estimator absorbed it.
    bias_cpu_log = np.log(np.maximum(bias_cpu[mask], 1e-6))
    bias_gpu_log = np.log(np.maximum(bias_gpu[mask], 1e-6))
    bias_cpu_log_mean = float(bias_cpu_log.mean())
    bias_gpu_log_mean = float(bias_gpu_log.mean())
    bias_cpu_log -= bias_cpu_log_mean
    bias_gpu_log -= bias_gpu_log_mean
    pearson_r = float(np.corrcoef(bias_cpu_log, bias_gpu_log)[0, 1])

    log_in = np.log(np.maximum(vol[mask], 1e-6))
    log_cpu = np.log(np.maximum(corr_cpu[mask], 1e-6))
    log_gpu = np.log(np.maximum(corr_gpu[mask], 1e-6))

    # Restrict to true tissue (top half of input intensity) for WM/GM contrast,
    # so we are not dominated by agarose/edge voxels in the loose `mask`.
    tissue_thresh = float(np.percentile(log_in, 50))
    tissue = log_in > tissue_thresh

    def _spread(x):
        return float(np.percentile(x, 90) - np.percentile(x, 10))

    spread_in = _spread(log_in[tissue])
    spread_cpu = _spread(log_cpu[tissue])
    spread_gpu = _spread(log_gpu[tissue])
    print(f"  bias log-mean (CPU, GPU)           = {bias_cpu_log_mean:+.3f}, {bias_gpu_log_mean:+.3f}")
    print(f"  bias-field Pearson r (GPU vs CPU)  = {pearson_r:.3f}")
    print(f"  tissue log p90-p10 spread          input={spread_in:.3f}  CPU={spread_cpu:.3f}  GPU={spread_gpu:.3f}")
    print(f"  GPU/CPU tissue contrast ratio      = {spread_gpu / max(spread_cpu, 1e-6):.3f}")
    print(
        f"  tissue log medians                 input={float(np.median(log_in[tissue])):+.3f}  "
        f"CPU={float(np.median(log_cpu[tissue])):+.3f}  GPU={float(np.median(log_gpu[tissue])):+.3f}"
    )

    z_mid = vol.shape[0] // 2
    sl_in = vol[z_mid]
    sl_cpu = corr_cpu[z_mid]
    sl_gpu = corr_gpu[z_mid]
    bias_cpu_n = bias_cpu / np.mean(bias_cpu[mask])
    bias_gpu_n = bias_gpu / np.mean(bias_gpu[mask])
    diff = np.abs(bias_cpu_n - bias_gpu_n)[z_mid]

    vmax = np.percentile(np.concatenate([sl_in.ravel(), sl_cpu.ravel(), sl_gpu.ravel()]), 99.5)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, im, title in zip(
        axes,
        [sl_in, sl_cpu, sl_gpu, diff],
        ["Input", "CPU (SimpleITK)", "GPU", "|bias_CPU - bias_GPU|"],
        strict=True,
    ):
        if title.startswith("|bias"):
            h = ax.imshow(im, cmap="magma", vmin=0, vmax=max(diff.max(), 1e-6))
        else:
            h = ax.imshow(im, cmap="gray", vmin=0, vmax=vmax)
        ax.set_title(title)
        ax.axis("off")
        plt.colorbar(h, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"N4 bias-field correction -- live OCT slab (z={z_mid}, shape={vol.shape})", fontsize=12)
    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=120, bbox_inches="tight")
    print(f"wrote {args.output}")

    # Also dump full-resolution loose PNGs of the three intensity panels with
    # identical normalisation, so they can be inspected pixel-for-pixel.
    stem = args.output.with_suffix("")
    for name, panel in (("input", sl_in), ("cpu", sl_cpu), ("gpu", sl_gpu)):
        path = stem.parent / f"{stem.name}_{name}.png"
        plt.imsave(path, np.clip(panel, 0, vmax) / max(vmax, 1e-6), cmap="gray", vmin=0, vmax=1)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
