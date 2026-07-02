"""Tests for linumpy.stack_alignment.motor_stack."""

import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from linumpy.stack_alignment.motor_stack import (
    accumulate_pairwise_translations,
    compute_output_shape,
    load_registration_transforms,
)

# ---------------------------------------------------------------------------
# compute_output_shape
# ---------------------------------------------------------------------------


def test_compute_output_shape_single_slice_no_shift():
    ny, nx, x0, y0 = compute_output_shape({0: "slice0"}, {0: (0.0, 0.0)}, (10, 20, 30))
    assert (ny, nx) == (20, 30)
    assert (x0, y0) == (0, 0)


def test_compute_output_shape_expands_for_negative_offset():
    cumsum_px = {0: (0.0, 0.0), 1: (5.0, -3.0)}
    ny, nx, x0, y0 = compute_output_shape({0: "slice0", 1: "slice1"}, cumsum_px, (10, 20, 30))
    assert (ny, nx) == (23, 35)
    assert (x0, y0) == (0, -3)


# ---------------------------------------------------------------------------
# load_registration_transforms
# ---------------------------------------------------------------------------


def _write_transform_dir(tmp_path: Path, slice_id: int, metrics: dict | None) -> None:
    transform_dir = tmp_path / f"slice_z{slice_id:02d}"
    transform_dir.mkdir()
    sitk.WriteTransform(sitk.Euler2DTransform(), str(transform_dir / "transform.tfm"))
    (transform_dir / "offsets.txt").write_text("4\n8\n")
    if metrics is not None:
        (transform_dir / "pairwise_registration_metrics.json").write_text(json.dumps(metrics))


def test_load_registration_transforms_basic(tmp_path: Path):
    metrics = {
        "overall_status": "ok",
        "metrics": {
            "registration_confidence": {"value": 0.75},
            "translation_x": {"value": 2.0},
            "translation_y": {"value": -1.0},
            "z_correlation": {"value": 0.6},
            "rotation": {"value": 0.2},
        },
    }
    _write_transform_dir(tmp_path, 1, metrics)

    transforms, pairwise = load_registration_transforms(tmp_path, [0, 1])

    assert transforms[1] is not None
    _tfm, fixed_z, moving_z, confidence = transforms[1]
    assert fixed_z == 4
    assert moving_z == 8
    assert abs(confidence - 0.75) < 1e-9
    assert pairwise[1] == (2.0, -1.0, 0.6)


def test_load_registration_transforms_metric_gating_rejects_low_zcorr(tmp_path: Path):
    metrics = {
        "overall_status": "ok",
        "metrics": {
            "registration_confidence": {"value": 0.9},
            "translation_x": {"value": 1.0},
            "translation_y": {"value": 1.0},
            "z_correlation": {"value": 0.1},
            "rotation": {"value": 0.1},
        },
    }
    _write_transform_dir(tmp_path, 1, metrics)

    transforms, pairwise = load_registration_transforms(tmp_path, [0, 1], load_min_zcorr=0.5, load_max_rotation=1.0)

    assert transforms[1] is None
    # Translation is still recovered for accumulation even though the
    # transform itself was gated out.
    assert pairwise[1] == (1.0, 1.0, 0.1)


def test_load_registration_transforms_missing_dir_is_none(tmp_path: Path):
    transforms, pairwise = load_registration_transforms(tmp_path, [0, 1])
    assert transforms[1] is None
    assert pairwise == {}


# ---------------------------------------------------------------------------
# accumulate_pairwise_translations
# ---------------------------------------------------------------------------


def test_accumulate_pairwise_translations_basic_cumsum():
    available_ids = [0, 1, 2]
    all_pairwise_translations = {1: (1.0, 2.0, 0.9), 2: (1.0, 2.0, 0.9)}

    accumulated = accumulate_pairwise_translations(
        available_ids,
        registration_transforms={},
        all_pairwise_translations=all_pairwise_translations,
    )

    assert accumulated[1] == (1.0, 2.0)
    assert accumulated[2] == (2.0, 4.0)


def test_accumulate_pairwise_translations_zcorr_filter_skips_low_confidence():
    available_ids = [0, 1, 2]
    all_pairwise_translations = {1: (1.0, 2.0, 0.05), 2: (1.0, 2.0, 0.9)}

    accumulated = accumulate_pairwise_translations(
        available_ids,
        registration_transforms={},
        all_pairwise_translations=all_pairwise_translations,
        translation_min_zcorr=0.2,
    )

    # Slice 1's translation is below threshold and skipped -> stays at 0.
    assert accumulated[1] == (0.0, 0.0)
    # Slice 2's translation is accepted and accumulated on top of slice 1's zero.
    assert accumulated[2] == (1.0, 2.0)


def test_accumulate_pairwise_translations_boundary_exclusion():
    available_ids = [0, 1]
    all_pairwise_translations = {1: (10.0, 0.0, 0.9)}

    accumulated = accumulate_pairwise_translations(
        available_ids,
        registration_transforms={},
        all_pairwise_translations=all_pairwise_translations,
        max_pairwise_translation=10.0,
    )

    # 10.0 >= 0.95 * 10.0 boundary -> excluded (zeroed).
    assert accumulated[1] == (0.0, 0.0)


def test_accumulate_pairwise_translations_drift_cap_clamps_magnitude():
    available_ids = [0, 1]
    all_pairwise_translations = {1: (10.0, 0.0, 0.9)}

    accumulated = accumulate_pairwise_translations(
        available_ids,
        registration_transforms={},
        all_pairwise_translations=all_pairwise_translations,
        max_cumulative_drift_px=5.0,
    )

    ox, oy = accumulated[1]
    assert abs(np.sqrt(ox**2 + oy**2) - 5.0) < 1e-9
