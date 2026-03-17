#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare stitching results between different methods."""

import linumpy._thread_config  # noqa: F401

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import sobel

from linumpy.io.zarr import read_omezarr


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("volume1", help="First stitched volume (.ome.zarr)")
    p.add_argument("volume2", help="Second stitched volume (.ome.zarr)")
    p.add_argument("output_dir", help="Output directory for comparison results")
    p.add_argument("--label1", type=str, default="Motor-only",
                   help="Label for first volume [%(default)s]")
    p.add_argument("--label2", type=str, default="Refined",
                   help="Label for second volume [%(default)s]")
    p.add_argument("--z_slice", type=int, default=None,
                   help="Z-slice to compare (default: middle)")
    p.add_argument("--tile_step", type=int, default=60,
                   help="Approximate tile step for seam detection [%(default)s]")
    return p


def compute_seam_sharpness(slice_data, seam_positions, width=50, direction='vertical'):
    """Compute edge sharpness at seams (lower = smoother blending)."""
    grad = sobel(slice_data, axis=1 if direction == 'vertical' else 0)
    values = []
    hw = width // 2
    for pos in seam_positions:
        if direction == 'vertical':
            start = max(0, pos - hw)
            end = min(slice_data.shape[1], pos + hw)
            region = grad[:, start:end]
        else:
            start = max(0, pos - hw)
            end = min(slice_data.shape[0], pos + hw)
            region = grad[start:end, :]
        values.append(np.mean(np.abs(region)))
    return np.mean(values) if values else 0, np.std(values) if values else 0


def main():
    p = _build_arg_parser()
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.label1}: {args.volume1}")
    vol1, _ = read_omezarr(args.volume1, level=0)
    vol1 = np.array(vol1[:])

    print(f"Loading {args.label2}: {args.volume2}")
    vol2, _ = read_omezarr(args.volume2, level=0)
    vol2 = np.array(vol2[:])

    # Crop to common shape
    min_shape = tuple(min(vol1.shape[i], vol2.shape[i]) for i in range(3))
    vol1 = vol1[:min_shape[0], :min_shape[1], :min_shape[2]]
    vol2 = vol2[:min_shape[0], :min_shape[1], :min_shape[2]]

    z_idx = args.z_slice if args.z_slice is not None else vol1.shape[0] // 2
    s1, s2 = vol1[z_idx], vol2[z_idx]

    print(f"Comparing z-slice {z_idx}")

    # Detect seam positions
    v_seams = list(range(args.tile_step, s1.shape[1] - 30, args.tile_step))
    h_seams = list(range(args.tile_step, s1.shape[0] - 30, args.tile_step))

    # Compute metrics
    m1_v, _ = compute_seam_sharpness(s1, v_seams, 50, 'vertical')
    m1_h, _ = compute_seam_sharpness(s1, h_seams, 50, 'horizontal')
    m2_v, _ = compute_seam_sharpness(s2, v_seams, 50, 'vertical')
    m2_h, _ = compute_seam_sharpness(s2, h_seams, 50, 'horizontal')

    diff = np.abs(s1.astype(float) - s2.astype(float))

    # Create comparison figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    vmin = min(s1.min(), s2.min())
    vmax = max(np.percentile(s1, 99), np.percentile(s2, 99))

    axes[0, 0].imshow(s1, cmap='gray', vmin=vmin, vmax=vmax)
    axes[0, 0].set_title(args.label1)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(s2, cmap='gray', vmin=vmin, vmax=vmax)
    axes[0, 1].set_title(args.label2)
    axes[0, 1].axis('off')

    im = axes[0, 2].imshow(diff, cmap='hot', vmin=0, vmax=np.percentile(diff, 99))
    axes[0, 2].set_title('Absolute Difference')
    axes[0, 2].axis('off')
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046)

    # Zoomed region
    cy, cx = s1.shape[0] // 2, s1.shape[1] // 2
    z_size = 100
    axes[1, 0].imshow(s1[cy-z_size:cy+z_size, cx-z_size:cx+z_size], cmap='gray', vmin=vmin, vmax=vmax)
    axes[1, 0].set_title(f'{args.label1} (zoom)')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(s2[cy-z_size:cy+z_size, cx-z_size:cx+z_size], cmap='gray', vmin=vmin, vmax=vmax)
    axes[1, 1].set_title(f'{args.label2} (zoom)')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(diff[cy-z_size:cy+z_size, cx-z_size:cx+z_size], cmap='hot')
    axes[1, 2].set_title('Difference (zoom)')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / "comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Save metrics
    metrics = {
        'z_slice': z_idx,
        'tile_step': args.tile_step,
        args.label1: {
            'vertical_seam_sharpness': float(m1_v),
            'horizontal_seam_sharpness': float(m1_h)
        },
        args.label2: {
            'vertical_seam_sharpness': float(m2_v),
            'horizontal_seam_sharpness': float(m2_h)
        },
        'difference': {
            'mean': float(np.mean(diff)),
            'max': float(np.max(diff)),
            'std': float(np.std(diff))
        }
    }

    with open(output_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    # Print summary
    print("\n" + "=" * 50)
    print("COMPARISON SUMMARY")
    print("=" * 50)
    print(f"\nSeam Sharpness (lower = smoother blending):")
    print(f"  {args.label1}:")
    print(f"    Vertical seams:   {m1_v:.4f}")
    print(f"    Horizontal seams: {m1_h:.4f}")
    print(f"  {args.label2}:")
    print(f"    Vertical seams:   {m2_v:.4f}")
    print(f"    Horizontal seams: {m2_h:.4f}")
    print(f"\nDifference ({args.label1} vs {args.label2}):")
    print(f"  Mean: {np.mean(diff):.4f}")
    print(f"  Max:  {np.max(diff):.4f}")
    print(f"\nOutputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
