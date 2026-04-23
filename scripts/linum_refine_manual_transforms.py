#!/usr/bin/env python3
"""
Refine manually-corrected pairwise slice transforms with image-based registration.

For each slice pair where a manual transform exists, this script:
1. Loads the corresponding common-space zarr slices at the Z-indices from the
   automated offsets.txt.
2. Warps the moving slice with the manual transform so it is approximately aligned
   to the fixed slice.
3. Runs a tight image-based registration (small search window) on the warped pair
   to correct any remaining sub-pixel / sub-degree residuals.
4. Composes the manual transform with the refinement result into a single output
   transform (source = "manual_refined").

For pairs without a manual transform the automated transform is copied unchanged.

Output directory structure mirrors the input transforms_dir:
    out_dir/
        slice_z04/
            transform.tfm
            offsets.txt
            pairwise_registration_metrics.json
        slice_z05/
            ...
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
    p.add_argument("slices_dir", help="Directory containing slice_z##.ome.zarr files (common space)")
    p.add_argument("transforms_dir", help="Directory of automated register_pairwise outputs")
    p.add_argument("out_dir", help="Output directory (same slice_z## structure)")

    p.add_argument(
        "--manual_transforms_dir",
        required=True,
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
        help="Max residual rotation to search during refinement [%(default)s°]",
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
    non-zero, or tz non-zero) — the pairwise refinement is 2D rigid and
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
    """Compose manual ∘ delta as a single 2D rigid transform about ``(final_cx, final_cy)``.

    Manual: T_m(p) = R_m (p - c_m) + c_m + t_m      (centre = (man_cx, man_cy))
    Delta:  T_δ(p) = R_δ (p - c_f) + c_f + t_δ      (centre = (final_cx, final_cy))
    Final:  T_f(p) = R_f (p - c_f) + c_f + t_f      with R_f = R_δ R_m

    We solve for (t_f, θ_f) so that T_f(p) = T_δ(T_m(p)) for all p. For 2D
    planar rotations θ_f = θ_m + θ_δ; evaluating at p = c_f gives t_f in
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
    same convention used by ``linumpy.mosaic.stacking.apply_2d_transform``
    (the downstream consumer of the refined tfm) and by
    ``linum_register_pairwise.py`` (the automated producer). Positive ``tx``
    therefore shifts content LEFT in the output (equivalent to
    ``scipy.ndimage.shift`` with ``[-ty, -tx]``).

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


def _discover_slice_zarrs(slices_dir: Path) -> dict[int, Path]:
    """Return {slice_id: zarr_path} for all slice_z##.ome.zarr in *slices_dir*."""
    pattern = re.compile(r"z(\d+)")
    result: dict[int, Path] = {}
    for p in sorted(slices_dir.iterdir()):
        if p.name.endswith(".ome.zarr"):
            m = pattern.search(p.name)
            if m:
                result[int(m.group(1))] = p
    return dict(sorted(result.items()))


def _discover_transform_dirs(transforms_dir: Path) -> dict[int, Path]:
    """Return {slice_id: dir_path} for automated transform subdirectories."""
    pattern = re.compile(r"z(\d+)")
    result: dict[int, Path] = {}
    for p in sorted(transforms_dir.iterdir()):
        if p.is_dir():
            m = pattern.search(p.name)
            if m:
                tfm_files = list(p.glob("*.tfm"))
                if tfm_files:
                    result[int(m.group(1))] = p
    return dict(sorted(result.items()))


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
    fixed_path: str,
    moving_path: str,
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
            "fixed_path": fixed_path,
            "moving_path": moving_path,
            "fixed_z": fixed_z,
        },
    }
    (out_dir / "pairwise_registration_metrics.json").write_text(json.dumps(metrics, indent=2))


def main() -> None:
    """Run function."""
    p = _build_arg_parser()
    args = p.parse_args()

    slices_dir = Path(args.slices_dir)
    transforms_dir = Path(args.transforms_dir)
    manual_dir = Path(args.manual_transforms_dir)
    out_dir = Path(args.out_dir)

    if out_dir.exists() and not args.overwrite:
        p.error(f"Output directory exists: {out_dir}. Use -f to overwrite.")
    out_dir.mkdir(parents=True, exist_ok=True)

    slice_zarrs = _discover_slice_zarrs(slices_dir)
    auto_transforms = _discover_transform_dirs(transforms_dir)

    if not auto_transforms:
        p.error(f"No transform subdirectories found in {transforms_dir}")

    sorted_slice_ids = sorted(slice_zarrs.keys())
    n_refined = 0
    n_copied = 0

    for slice_id, auto_dir in sorted(auto_transforms.items()):
        manual_tfm_path = manual_dir / f"slice_z{slice_id:02d}" / "transform.tfm"
        pair_out = out_dir / auto_dir.name  # preserve original subdir name

        if not manual_tfm_path.exists():
            # No manual transform — copy automated result unchanged
            logger.info("  z%d: no manual transform, copying automated", slice_id)
            if pair_out.exists():
                shutil.rmtree(pair_out)
            shutil.copytree(auto_dir, pair_out)
            n_copied += 1
            continue

        # Find fixed slice: the one immediately before in the sorted list
        idx = sorted_slice_ids.index(slice_id) if slice_id in sorted_slice_ids else -1
        if idx <= 0:
            logger.warning("  z%d: cannot determine fixed slice (id not in zarrs or first slice), copying", slice_id)
            if pair_out.exists():
                shutil.rmtree(pair_out)
            shutil.copytree(auto_dir, pair_out)
            n_copied += 1
            continue

        fixed_id = sorted_slice_ids[idx - 1]
        if fixed_id not in slice_zarrs or slice_id not in slice_zarrs:
            logger.warning("  z%d: zarr missing for pair (%s→%s), copying", slice_id, fixed_id, slice_id)
            if pair_out.exists():
                shutil.rmtree(pair_out)
            shutil.copytree(auto_dir, pair_out)
            n_copied += 1
            continue

        logger.info("  z%d: refining from manual transform (fixed=z%d)", slice_id, fixed_id)

        # Load Z-indices from automated offsets.txt
        auto_offsets_path = auto_dir / "offsets.txt"
        if auto_offsets_path.exists():
            offsets_arr = np.loadtxt(str(auto_offsets_path), dtype=int)
            fixed_z = int(offsets_arr[0]) if offsets_arr.size >= 1 else 0
            moving_z = int(offsets_arr[1]) if offsets_arr.size >= 2 else 0
        else:
            fixed_z, moving_z = 0, 0
            logger.warning("  z%d: offsets.txt missing, using z=0 for both slices", slice_id)

        # Load zarr volumes and extract the relevant 2D slices
        fixed_vol, _res = read_omezarr(str(slice_zarrs[fixed_id]))
        moving_vol, _res = read_omezarr(str(slice_zarrs[slice_id]))

        fixed_z = max(0, min(fixed_z, fixed_vol.shape[0] - 1))
        moving_z = max(0, min(moving_z, moving_vol.shape[0] - 1))

        fixed_slice = _normalize(np.array(fixed_vol[fixed_z]))
        moving_slice = _normalize(np.array(moving_vol[moving_z]))

        # Load manual transform parameters (full-resolution pixels)
        man_tx, man_ty, man_rot, man_cx, man_cy = _load_manual_transform(manual_tfm_path)
        logger.info("  z%d: manual tx=%.1f ty=%.1f rot=%.3f°", slice_id, man_tx, man_ty, man_rot)

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
        logger.info("  z%d: refinement delta tx=%.2f ty=%.2f rot=%.3f°", slice_id, delta_tx, delta_ty, delta_rot)

        # Compose manual ∘ delta about the fixed-slice centre.
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
        logger.info("  z%d: final    tx=%.2f ty=%.2f rot=%.3f°", slice_id, final_tx, final_ty, final_rot)

        # Write output. The manual tfm, the refinement delta, and the
        # composed final tfm are all in SimpleITK output->input (point-map)
        # convention now that _warp_moving and the manual-align widget both
        # produce/consume sitk-convention translations.
        pair_out.mkdir(parents=True, exist_ok=True)
        final_tfm = create_transform(final_tx, final_ty, final_rot, final_center)
        sitk.WriteTransform(final_tfm, str(pair_out / "transform.tfm"))
        np.savetxt(str(pair_out / "offsets.txt"), [fixed_z, moving_z], fmt="%d")

        # Estimate z_correlation from the warped pair for metrics
        z_correlation = float(np.corrcoef(fixed_slice.ravel(), warped_moving.ravel())[0, 1])
        z_correlation = max(0.0, z_correlation)

        _write_metrics(
            out_dir=pair_out,
            tx=final_tx,
            ty=final_ty,
            rot_deg=final_rot,
            delta_tx=delta_tx,
            delta_ty=delta_ty,
            delta_rot=delta_rot,
            z_correlation=z_correlation,
            fixed_z=fixed_z,
            fixed_path=str(slice_zarrs[fixed_id]),
            moving_path=str(slice_zarrs[slice_id]),
            max_translation_px=args.max_translation_px,
            max_rotation_deg=args.max_rotation_deg,
        )
        n_refined += 1

    logger.info("Done: %s pairs refined from manual transforms, %s copied unchanged", n_refined, n_copied)


if __name__ == "__main__":
    main()
