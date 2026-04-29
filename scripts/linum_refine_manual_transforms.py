#!/usr/bin/env python3
"""
Refine a single manually-corrected pairwise slice transform with image-based registration.

For the given fixed/moving zarr pair:
1. Loads the Z-indices from the automated offsets.txt in auto_transform_dir.
2. If a manual transform exists in --manual_transforms_dir for this pair:
   a. Warps the moving slice with the manual transform.
   b. Runs a tight image-based registration on the warped pair.
   c. Composes manual o delta into a single output transform (source = "manual_refined").
   d. Writes transform.tfm, offsets.txt, pairwise_registration_metrics.json to out_dir.
3. If no manual transform exists, copies auto_transform_dir to out_dir unchanged.

Intended to be called once per pair by Nextflow (parallel execution).
"""

import linumpy.config.threads  # noqa: F401

import argparse
import json
import logging
import re
import shutil
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from linumpy.cli.args import add_overwrite_arg
from linumpy.io.zarr import read_omezarr
from linumpy.registration.refinement import register_refinement
from linumpy.registration.transforms import create_transform

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("fixed_zarr", help="Path to fixed slice OME-Zarr (common space)")
    p.add_argument("moving_zarr", help="Path to moving slice OME-Zarr (common space)")
    p.add_argument("auto_transform_dir", help="Automated register_pairwise output dir for this pair")
    p.add_argument("out_dir", help="Output directory for this pair")

    p.add_argument(
        "--manual_transforms_dir",
        default=None,
        help="Directory with manually corrected transforms (slice_z##/transform.tfm)",
    )
    p.add_argument(
        "--max_translation_px",
        type=float,
        default=10.0,
        help="Max residual translation to search during refinement [%(default)s px]",
    )
    p.add_argument(
        "--max_rotation_deg",
        type=float,
        default=2.0,
        help="Max residual rotation to search during refinement [%(default)s degrees]",
    )
    add_overwrite_arg(p)
    return p


def _normalize(image: np.ndarray) -> np.ndarray:
    """Normalize image to [0, 1] using 5th / 95th percentile of non-zero values."""
    valid = image > 0
    if not np.any(valid):
        return np.zeros_like(image, dtype=np.float32)
    pmin = float(np.percentile(image[valid], 5))
    pmax = float(np.percentile(image[valid], 95))
    if pmax <= pmin:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image.astype(np.float32) - pmin) / (pmax - pmin), 0, 1).astype(np.float32)


def _load_manual_transform(tfm_path: Path) -> tuple[float, float, float, float, float]:
    """Return (tx, ty, rot_deg, cx, cy) from a SimpleITK Euler3DTransform file.

    Warns if the stored transform has non-planar Euler components (rx or ry
    non-zero, or tz non-zero) -- the pairwise refinement is 2D rigid and
    cannot represent non-planar rotations, so those components would be
    silently dropped by the composition. Hand-edited .tfm files containing
    such components should be authored via the manual alignment plugin
    instead, which only emits planar transforms.
    """
    tfm = sitk.ReadTransform(str(tfm_path))
    params = tfm.GetParameters()
    # Euler3DTransform params: [rx, ry, rz, tx, ty, tz]
    non_planar_rot = any(abs(float(params[i])) > 1e-6 for i in (0, 1))
    non_planar_t = len(params) > 5 and abs(float(params[5])) > 1e-6
    if non_planar_rot or non_planar_t:
        logger.warning(
            "  manual transform %s has non-planar Euler components "
            "(rx=%.4g rad, ry=%.4g rad, tz=%.4g px); they will be dropped "
            "during 2D refinement composition.",
            tfm_path,
            float(params[0]),
            float(params[1]),
            float(params[5]) if len(params) > 5 else 0.0,
        )
    rot_deg = float(np.degrees(params[2]))
    tx = float(params[3])
    ty = float(params[4])
    fixed_params = tfm.GetFixedParameters()
    cx = float(fixed_params[0]) if len(fixed_params) > 0 else 0.0
    cy = float(fixed_params[1]) if len(fixed_params) > 1 else 0.0
    return tx, ty, rot_deg, cx, cy


def _compose_rigid_2d(
    man_tx: float,
    man_ty: float,
    man_rot_deg: float,
    man_cx: float,
    man_cy: float,
    delta_tx: float,
    delta_ty: float,
    delta_rot_deg: float,
    final_cx: float,
    final_cy: float,
) -> tuple[float, float, float]:
    """Compose manual o delta as a single 2D rigid transform about (final_cx, final_cy).

    Manual: T_m(p) = R_m (p - c_m) + c_m + t_m      (centre = (man_cx, man_cy))
    Delta:  T_d(p) = R_d (p - c_f) + c_f + t_d      (centre = (final_cx, final_cy))
    Final:  T_f(p) = R_f (p - c_f) + c_f + t_f      with R_f = R_d R_m

    We solve for (t_f, theta_f) so that T_f(p) = T_d(T_m(p)) for all p. For 2D
    planar rotations theta_f = theta_m + theta_d; evaluating at p = c_f gives t_f in
    closed form without sampling or a numerical fit:

        t_f = R_delta (T_m(c_f) - c_f) + t_delta

    Returns (tx, ty, rot_deg).
    """

    def _rot(theta_rad: float) -> np.ndarray:
        c = float(np.cos(theta_rad))
        s = float(np.sin(theta_rad))
        return np.array([[c, -s], [s, c]])

    c_final = np.array([final_cx, final_cy])
    c_manual = np.array([man_cx, man_cy])
    t_manual = np.array([man_tx, man_ty])
    t_delta = np.array([delta_tx, delta_ty])

    r_manual = _rot(np.radians(man_rot_deg))
    r_delta = _rot(np.radians(delta_rot_deg))

    # T_m(c_final):
    p_manual = r_manual @ (c_final - c_manual) + c_manual + t_manual
    t_final = r_delta @ (p_manual - c_final) + t_delta

    return float(t_final[0]), float(t_final[1]), float(man_rot_deg + delta_rot_deg)


def _warp_moving(moving: np.ndarray, tx: float, ty: float, rot_deg: float, cx: float, cy: float) -> np.ndarray:
    """Apply a 2D rigid transform to *moving* using SimpleITK.

    The resampling uses SimpleITK's standard output->input convention -- the
    same convention used by linumpy.mosaic.stacking.apply_2d_transform
    (the downstream consumer of the refined tfm) and by
    linum_register_pairwise.py (the automated producer). Positive tx
    therefore shifts content LEFT in the output (equivalent to
    scipy.ndimage.shift with [-ty, -tx]).

    Parameters
    ----------
    moving : np.ndarray
        Input image with shape (H, W).
    tx : float
        Full-resolution pixel translation X in SimpleITK convention.
    ty : float
        Full-resolution pixel translation Y in SimpleITK convention.
    rot_deg : float
        Rotation in degrees (CCW positive).
    cx : float
        Rotation centre X coordinate (column).
    cy : float
        Rotation centre Y coordinate (row).
    """
    out = moving.astype(np.float32)
    if abs(rot_deg) < 0.01 and abs(tx) < 1e-6 and abs(ty) < 1e-6:
        return out

    img = sitk.GetImageFromArray(out)
    tfm = sitk.Euler2DTransform()
    tfm.SetCenter([float(cx), float(cy)])
    tfm.SetAngle(float(np.radians(rot_deg)))
    tfm.SetTranslation([float(tx), float(ty)])

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(img)
    resampler.SetTransform(tfm)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0.0)
    warped = sitk.GetArrayFromImage(resampler.Execute(img))
    return warped.astype(np.float32)


def _write_metrics(
    out_dir: Path,
    tx: float,
    ty: float,
    rot_deg: float,
    delta_tx: float,
    delta_ty: float,
    delta_rot: float,
    z_correlation: float,
    fixed_z: int,
    fixed_path: Path,
    moving_path: Path,
    max_translation_px: float,
    max_rotation_deg: float,
) -> None:
    """Write pairwise_registration_metrics.json with source='manual_refined'."""
    mag = float(np.sqrt(tx**2 + ty**2))
    metrics = {
        "step_name": "pairwise_registration",
        "output_path": str(out_dir),
        "source": "manual_refined",
        "metrics": {
            "translation_x": {"value": tx, "unit": "pixels"},
            "translation_y": {"value": ty, "unit": "pixels"},
            "translation_magnitude": {"value": mag, "unit": "pixels"},
            "rotation": {"value": rot_deg, "unit": "degrees"},
            "registration_confidence": {"value": 1.0},
            "z_correlation": {"value": z_correlation},
            "registration_error": {"value": 0.0},
        },
        "overall_status": "ok",
        "refinement": {
            "delta_tx": delta_tx,
            "delta_ty": delta_ty,
            "delta_rot_deg": delta_rot,
            "max_translation_px": max_translation_px,
            "max_rotation_deg": max_rotation_deg,
            "fixed_path": str(fixed_path) if fixed_path is not None else None,
            "moving_path": str(moving_path) if moving_path is not None else None,
            "fixed_z": fixed_z,
        },
    }
    (out_dir / "pairwise_registration_metrics.json").write_text(json.dumps(metrics, indent=2))


def main() -> None:
    """Run function."""
    p = _build_arg_parser()
    args = p.parse_args()

    fixed_zarr = Path(args.fixed_zarr)
    moving_zarr = Path(args.moving_zarr)
    auto_transform_dir = Path(args.auto_transform_dir)
    out_dir = Path(args.out_dir)

    if out_dir.exists() and not args.overwrite:
        p.error(f"Output directory exists: {out_dir}. Use -f to overwrite.")

    # Extract slice_id from the moving zarr filename (e.g. slice_z05_normalize.ome.zarr -> 5)
    m = re.search(r"z(\d+)", moving_zarr.name)
    if m is None:
        p.error(f"Cannot extract slice ID from moving zarr filename: {moving_zarr.name}")
    slice_id = int(m.group(1))

    # Locate manual transform for this pair (optional)
    manual_tfm_path: Path | None = None
    if args.manual_transforms_dir:
        candidate = Path(args.manual_transforms_dir) / f"slice_z{slice_id:02d}" / "transform.tfm"
        if candidate.exists():
            manual_tfm_path = candidate

    if manual_tfm_path is None:
        # No manual transform -- copy automated result unchanged
        logger.info("z%d: no manual transform, copying automated", slice_id)
        if out_dir.exists():
            shutil.rmtree(out_dir)
        shutil.copytree(auto_transform_dir, out_dir)
        return

    logger.info("z%d: refining from manual transform", slice_id)

    # Load Z-indices from automated offsets.txt
    auto_offsets_path = auto_transform_dir / "offsets.txt"
    if auto_offsets_path.exists():
        offsets_arr = np.loadtxt(str(auto_offsets_path), dtype=int)
        fixed_z = int(offsets_arr[0]) if offsets_arr.size >= 1 else 0
        moving_z = int(offsets_arr[1]) if offsets_arr.size >= 2 else 0
    else:
        fixed_z, moving_z = 0, 0
        logger.warning("z%d: offsets.txt missing, using z=0 for both slices", slice_id)

    # Load zarr volumes and extract the relevant 2D slices
    fixed_vol, _res = read_omezarr(fixed_zarr)
    moving_vol, _res = read_omezarr(moving_zarr)

    fixed_z = max(0, min(fixed_z, fixed_vol.shape[0] - 1))
    moving_z = max(0, min(moving_z, moving_vol.shape[0] - 1))

    fixed_slice = _normalize(np.array(fixed_vol[fixed_z]))
    moving_slice = _normalize(np.array(moving_vol[moving_z]))

    # Load manual transform parameters (full-resolution pixels)
    man_tx, man_ty, man_rot, man_cx, man_cy = _load_manual_transform(manual_tfm_path)
    logger.info("z%d: manual tx=%.1f ty=%.1f rot=%.3f deg", slice_id, man_tx, man_ty, man_rot)

    # Warp moving slice with manual transform so it is approximately aligned
    warped_moving = _warp_moving(moving_slice, man_tx, man_ty, man_rot, man_cx, man_cy)

    # Run tight refinement on the warped pair
    delta_tx, delta_ty, delta_rot, _metric = register_refinement(
        fixed_slice,
        warped_moving,
        enable_rotation=True,
        max_rotation_deg=args.max_rotation_deg,
        max_translation_px=args.max_translation_px,
    )
    logger.info("z%d: refinement delta tx=%.2f ty=%.2f rot=%.3f deg", slice_id, delta_tx, delta_ty, delta_rot)

    # Compose manual o delta about the fixed-slice centre.
    # The refinement runs in the fixed-slice reference frame with rotation
    # centre at its geometric centre, so the composite must be re-expressed
    # about that same centre for the saved .tfm to round-trip correctly.
    final_center = [fixed_slice.shape[1] / 2.0, fixed_slice.shape[0] / 2.0]
    final_tx, final_ty, final_rot = _compose_rigid_2d(
        man_tx,
        man_ty,
        man_rot,
        man_cx,
        man_cy,
        delta_tx,
        delta_ty,
        delta_rot,
        final_center[0],
        final_center[1],
    )
    logger.info("z%d: final    tx=%.2f ty=%.2f rot=%.3f deg", slice_id, final_tx, final_ty, final_rot)

    # Write output. The manual tfm, the refinement delta, and the composed
    # final tfm are all in SimpleITK output->input (point-map) convention.
    out_dir.mkdir(parents=True, exist_ok=True)
    final_tfm = create_transform(final_tx, final_ty, final_rot, final_center)
    sitk.WriteTransform(final_tfm, str(out_dir / "transform.tfm"))
    np.savetxt(str(out_dir / "offsets.txt"), [fixed_z, moving_z], fmt="%d")

    # Estimate z_correlation from the warped pair for metrics
    z_correlation = float(np.corrcoef(fixed_slice.ravel(), warped_moving.ravel())[0, 1])
    z_correlation = max(0.0, z_correlation)

    _write_metrics(
        out_dir=out_dir,
        tx=final_tx,
        ty=final_ty,
        rot_deg=final_rot,
        delta_tx=delta_tx,
        delta_ty=delta_ty,
        delta_rot=delta_rot,
        z_correlation=z_correlation,
        fixed_z=fixed_z,
        fixed_path=fixed_zarr,
        moving_path=moving_zarr,
        max_translation_px=args.max_translation_px,
        max_rotation_deg=args.max_rotation_deg,
    )
    logger.info("z%d: done", slice_id)


if __name__ == "__main__":
    main()
