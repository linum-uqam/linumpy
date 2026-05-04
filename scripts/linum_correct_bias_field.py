#!/usr/bin/env python3
"""Apply N4 bias field correction to an OME-Zarr OCT volume.

Three correction modes are supported.

* ``per_section`` -- independently correct each serial tissue section
  (removes depth-dependent attenuation per section).
* ``global`` -- correct the whole stack as one volume (removes slow
  large-scale intensity gradients).
* ``two_pass`` -- run ``per_section`` first, then ``global`` (default).

The ``--strength`` parameter (0-1) blends between the original and the
fully-corrected result:
``output = strength * corrected + (1 - strength) * input``.
"""

# Configure thread limits before numpy/scipy imports
import linumpy.config.threads  # noqa: F401

import argparse
import logging

import numpy as np

from linumpy.cli.args import add_processes_arg, parse_processes_arg
from linumpy.intensity.bias_field import (
    compute_tissue_mask,
    n4_correct,
    n4_correct_per_section,
)
from linumpy.intensity.normalization import apply_histogram_matching, apply_zprofile_smoothing
from linumpy.io.zarr import AnalysisOmeZarrWriter, read_omezarr_array

logger = logging.getLogger(__name__)

_MODES = ("per_section", "global", "two_pass")


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("in_image", help="Input OME-Zarr image.")
    p.add_argument("out_image", help="Output OME-Zarr image.")

    # Mode / strength
    p.add_argument(
        "--mode",
        choices=_MODES,
        default="two_pass",
        help="Correction mode. [%(default)s]",
    )
    p.add_argument(
        "--strength",
        type=float,
        default=1.0,
        help="Mixing weight between corrected and original (0 = no correction, 1 = full). [%(default)s]",
    )

    # Per-section options
    p.add_argument(
        "--n_serial_slices",
        type=int,
        default=1,
        help="Number of serial tissue sections stacked along Z (for per_section / two_pass). [%(default)s]",
    )
    add_processes_arg(p)

    # N4 tuning
    p.add_argument(
        "--shrink_factor",
        type=int,
        default=4,
        help="Spatial downsampling factor for the N4 fit. [%(default)s]",
    )
    p.add_argument(
        "--n_iterations",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Max N4 iterations per fitting level.  Length of list = number of fitting levels. "
            "Defaults to the backend's own choice ([50, 50, 50, 50] for cpu, [25, 25, 25] for gpu)."
        ),
    )
    p.add_argument(
        "--spline_distance_mm",
        type=float,
        default=None,
        help="Approximate B-spline knot spacing in mm.  Defaults to 2.0 for per_section, 10.0 for global.",
    )
    p.add_argument(
        "--mask_smoothing_sigma",
        type=float,
        default=2.0,
        help="Gaussian smoothing sigma for tissue mask estimation. [%(default)s]",
    )

    # Histogram-matching pre-pass (corrects inter-section intensity drift)
    p.add_argument(
        "--histogram_match",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply per-section histogram matching to a global reference distribution\n"
        "before N4 correction.  Equalises section-to-section intensity drift while\n"
        "preserving relative contrast within each section. [%(default)s]",
    )
    p.add_argument(
        "--histogram_n_bins",
        type=int,
        default=512,
        help="Number of histogram bins for matching. [%(default)s]",
    )
    p.add_argument(
        "--histogram_match_per_zplane",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Match each Z-plane independently to the global tissue distribution\n"
        "(strongest reduction of inter-slice intensity steps).  When False, the\n"
        "volume is split into --n_serial_slices chunks (legacy behaviour). [%(default)s]",
    )
    p.add_argument(
        "--tissue_threshold",
        type=float,
        default=0.0,
        help="Voxels at or below this intensity are background and left unchanged\n"
        "by histogram matching.  Use a small positive value (e.g. 0.005) to exclude\n"
        "near-zero noise. [%(default)s]",
    )
    p.add_argument(
        "--zprofile_smooth_sigma",
        type=float,
        default=0.0,
        help="After histogram matching, remove residual per-Z-plane jitter with a\n"
        "smoothed scalar gain (Gaussian sigma in Z-plane units).  0 = disabled.\n"
        "Typical: 2.0-4.0.  Eliminates the ~1-2%% inter-slice steps HM cannot\n"
        "remove while preserving the smooth depth attenuation profile. [%(default)s]",
    )

    # Background masking (zero out agarose)
    p.add_argument(
        "--zero_outside_mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Zero out voxels outside the tissue mask in the final output\n(removes agarose halo). [%(default)s]",
    )

    # Output options
    p.add_argument(
        "--save_bias_field",
        metavar="PATH",
        default=None,
        help="Save recovered bias field to this path.",
    )
    p.add_argument(
        "--pyramid_resolutions",
        type=float,
        nargs="+",
        default=[10, 25, 50, 100],
        help="Target resolutions for pyramid levels in microns. [%(default)s]",
    )
    p.add_argument(
        "--make_isotropic",
        action="store_true",
        default=True,
        help="Resample to isotropic voxels. [%(default)s]",
    )
    p.add_argument("--no_isotropic", dest="make_isotropic", action="store_false")
    p.add_argument(
        "--n_levels",
        type=int,
        default=None,
        help="Use fixed pyramid levels instead of pyramid_resolutions.",
    )
    p.add_argument("--verbose", action="store_true", help="Enable INFO-level logging.")
    p.add_argument(
        "--backend",
        type=str,
        default="cpu",
        choices=("cpu", "gpu", "auto"),
        help=(
            "N4 backend.  'cpu' uses SimpleITK; 'gpu' uses the CuPy/NumPy port "
            "in linumpy.gpu.n4; 'auto' picks gpu when CUDA is available. [%(default)s]"
        ),
    )
    return p


def _save(arr: np.ndarray, path: str, res: list, args: argparse.Namespace) -> None:
    """Save a volume to OME-Zarr using resolution-based or fixed pyramid levels."""
    from pathlib import Path

    writer = AnalysisOmeZarrWriter(Path(path), arr.shape, chunk_shape=(128, 128, 128), dtype=np.float32)
    writer[:] = arr
    writer.finalize(
        res,
        n_levels=args.n_levels,
        target_resolutions_um=args.pyramid_resolutions,
        make_isotropic=args.make_isotropic,
    )


def main() -> None:
    """Run function."""
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING)

    n_processes = parse_processes_arg(args.n_processes)

    # Resolve GPU usage from --backend choice for non-N4 stages. We resolve
    # this BEFORE reading so we can stream the volume directly into device
    # memory through the GDS / zarr-gpu fast path when the GPU is in play.
    if args.backend == "gpu":
        use_gpu_pre = True
    elif args.backend == "auto":
        from linumpy.gpu import GPU_AVAILABLE

        use_gpu_pre = GPU_AVAILABLE
    else:
        use_gpu_pre = False

    # Load volume — onto GPU directly when use_gpu_pre, else host.
    vol, res = read_omezarr_array(args.in_image, level=0, use_gpu=use_gpu_pre)
    # copy=False avoids doubling GPU memory when the array is already float32
    # (the common case for mosaic grids).
    vol = vol.astype(np.float32, copy=False)
    logger.info("Loaded volume %s from %s (gpu=%s)", vol.shape, args.in_image, use_gpu_pre)

    # Tissue mask (per serial section)
    mask = compute_tissue_mask(
        vol,
        smoothing_sigma=args.mask_smoothing_sigma,
        n_serial_slices=args.n_serial_slices,
        use_gpu=use_gpu_pre,
    )
    logger.info("Tissue mask: %d/%d voxels", int(mask.sum()), mask.size)

    # Histogram-matching pre-pass: equalise inter-section intensity drift
    if args.histogram_match:
        hm_n_serial = None if args.histogram_match_per_zplane else args.n_serial_slices
        logger.info(
            "Histogram matching (n_serial_slices=%s, n_bins=%d, threshold=%g)\u2026",
            "per_zplane" if hm_n_serial is None else hm_n_serial,
            args.histogram_n_bins,
            args.tissue_threshold,
        )
        vol = apply_histogram_matching(
            vol,
            n_serial_slices=hm_n_serial,
            n_bins=args.histogram_n_bins,
            tissue_threshold=args.tissue_threshold,
            use_gpu=use_gpu_pre,
        ).astype(np.float32)

    # Z-profile smoothing: remove residual per-Z jitter that HM cannot fully fix
    if args.zprofile_smooth_sigma > 0:
        logger.info("Z-profile gain smoothing (sigma=%g)\u2026", args.zprofile_smooth_sigma)
        vol = apply_zprofile_smoothing(vol, mask, sigma=args.zprofile_smooth_sigma).astype(np.float32)

    # Resolve spline distance defaults
    per_section_spline = args.spline_distance_mm if args.spline_distance_mm is not None else 2.0
    global_spline = args.spline_distance_mm if args.spline_distance_mm is not None else 10.0

    n4_kwargs = {
        "shrink_factor": args.shrink_factor,
        "n_iterations": args.n_iterations,
        "voxel_size_mm": (float(res[0]), float(res[1]), float(res[2])),
        "backend": args.backend,
    }

    # Correction passes
    bias_field_combined: np.ndarray | None = None

    if args.mode in ("per_section", "two_pass"):
        logger.info(
            "Running per-section N4 (n_serial_slices=%d, n_processes=%d)…",
            args.n_serial_slices,
            n_processes,
        )
        vol_ps, bias_ps = n4_correct_per_section(
            vol,
            n_serial_slices=args.n_serial_slices,
            mask=mask,
            n_processes=n_processes,
            spline_distance_mm=per_section_spline,
            **n4_kwargs,
        )
        bias_field_combined = bias_ps
        working_vol = vol_ps
        # Drop the per-section aliases so the global pass below does not have
        # to keep two extra full-size float32 volumes alive (~72 GB on a
        # 36 GB mosaic). bias_field_combined / working_vol still hold them.
        del vol_ps, bias_ps
    else:
        working_vol = vol

    if args.mode in ("global", "two_pass"):
        logger.info("Running global N4…")
        working_vol, bias_global = n4_correct(
            working_vol,
            mask,
            spline_distance_mm=global_spline,
            **n4_kwargs,
        )
        if bias_field_combined is not None:
            # Combine in place to avoid a third 36 GB allocation during the
            # multiply, then release bias_global immediately.
            bias_field_combined *= bias_global
            del bias_global
        else:
            bias_field_combined = bias_global

    corrected = working_vol

    # Strength blend
    if args.strength < 1.0:
        logger.info("Blending: strength=%.3f", args.strength)
        corrected = args.strength * corrected + (1.0 - args.strength) * vol

    corrected = corrected.astype(np.float32)

    # Zero out non-tissue voxels (suppress agarose)
    if args.zero_outside_mask:
        logger.info("Zeroing voxels outside tissue mask\u2026")
        corrected = np.where(mask, corrected, 0.0).astype(np.float32)

    # Save output
    _save(corrected, args.out_image, res, args)
    logger.info("Saved corrected volume to %s", args.out_image)

    # Optionally save bias field
    if args.save_bias_field is not None and bias_field_combined is not None:
        _save(bias_field_combined, args.save_bias_field, res, args)
        logger.info("Saved bias field to %s", args.save_bias_field)


if __name__ == "__main__":
    main()
