#!/usr/bin/env python3
"""Tests for ``scripts/linum_align_to_ras.py``.

The script is loaded via :mod:`importlib` so we can test its pure-Python helper
functions (no ``zarr`` I/O) without relying on the console entry point.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "linum_align_to_ras.py"


@pytest.fixture(scope="module")
def align_module():
    """Load ``linum_align_to_ras.py`` as a module."""
    spec = importlib.util.spec_from_file_location("linum_align_to_ras", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# CLI help
# ---------------------------------------------------------------------------


def test_help(script_runner):
    ret = script_runner.run(["linum_align_to_ras.py", "--help"])
    assert ret.success


# ---------------------------------------------------------------------------
# sitk_transform_to_affine_matrix
# ---------------------------------------------------------------------------


class TestSitkTransformToAffine:
    def test_identity_transform_yields_identity_matrix(self, align_module):
        t = sitk.Euler3DTransform()
        mat = align_module.sitk_transform_to_affine_matrix(t)
        assert mat.shape == (4, 4)
        np.testing.assert_allclose(mat, np.eye(4), atol=1e-12)

    def test_pure_translation_is_permuted_to_zyx(self, align_module):
        t = sitk.Euler3DTransform()
        # SITK translation in (X, Y, Z) = (1, 2, 3)
        t.SetTranslation((1.0, 2.0, 3.0))
        mat = align_module.sitk_transform_to_affine_matrix(t)
        # After conversion to NGFF (Z, Y, X) order, translation must be (3, 2, 1).
        np.testing.assert_allclose(mat[:3, 3], [3.0, 2.0, 1.0], atol=1e-12)
        np.testing.assert_allclose(mat[:3, :3], np.eye(3), atol=1e-12)

    def test_rotation_is_permuted_to_zyx(self, align_module):
        """A pure rotation around SITK X (=numpy axis 2) should appear as a
        rotation around the last axis of the NGFF matrix (axis Z→row 2)."""
        t = sitk.Euler3DTransform()
        t.SetRotation(np.pi / 4, 0.0, 0.0)  # rotate around SITK X
        mat = align_module.sitk_transform_to_affine_matrix(t)
        # Rotation around numpy axis 2 (X in NGFF) leaves column/row 2 unchanged.
        assert mat.shape == (4, 4)
        np.testing.assert_allclose(mat[2, 2], 1.0, atol=1e-9)
        np.testing.assert_allclose(mat[2, :3], [0.0, 0.0, 1.0], atol=1e-9)
        np.testing.assert_allclose(mat[:3, 2], [0.0, 0.0, 1.0], atol=1e-9)


# ---------------------------------------------------------------------------
# compute_centered_reference_and_transform
# ---------------------------------------------------------------------------


class TestComputeCenteredReferenceAndTransform:
    @staticmethod
    def _make_moving(shape=(20, 20, 20), spacing=(0.1, 0.1, 0.1)):
        """A small ellipsoid brain so the resampled output has known volume."""
        z, y, x = np.indices(shape, dtype=np.float32)
        cz, cy, cx = shape[0] / 2, shape[1] / 2, shape[2] / 2
        rz, ry, rx = shape[0] * 0.3, shape[1] * 0.3, shape[2] * 0.3
        mask = ((z - cz) / rz) ** 2 + ((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2 < 1
        arr = mask.astype(np.float32)
        img = sitk.GetImageFromArray(arr)
        img.SetSpacing((spacing[2], spacing[1], spacing[0]))  # SITK XYZ
        img.SetOrigin((0.0, 0.0, 0.0))
        img.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
        return img, int(mask.sum())

    def test_reference_origin_is_zero(self, align_module):
        moving, _ = self._make_moving()
        t = sitk.Euler3DTransform()
        ref, _ = align_module.compute_centered_reference_and_transform(moving, t)
        assert ref.GetOrigin() == pytest.approx((0.0, 0.0, 0.0))

    def test_reference_spacing_matches_moving_by_default(self, align_module):
        moving, _ = self._make_moving(spacing=(0.125, 0.1, 0.2))
        t = sitk.Euler3DTransform()
        ref, _ = align_module.compute_centered_reference_and_transform(moving, t)
        assert ref.GetSpacing() == pytest.approx(moving.GetSpacing())

    def test_reference_spacing_override(self, align_module):
        moving, _ = self._make_moving()
        t = sitk.Euler3DTransform()
        ref, _ = align_module.compute_centered_reference_and_transform(moving, t, output_spacing=(0.05, 0.05, 0.05))
        assert ref.GetSpacing() == pytest.approx((0.05, 0.05, 0.05))

    def test_identity_transform_roundtrip(self, align_module):
        """For T = identity, resampling through the composite should recover
        the original brain volume (no information loss)."""
        moving, brain_voxels = self._make_moving()
        t = sitk.Euler3DTransform()  # identity
        ref, composite = align_module.compute_centered_reference_and_transform(moving, t)

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(ref)
        resampler.SetTransform(composite)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0.0)
        out = resampler.Execute(moving)
        arr = sitk.GetArrayFromImage(out)

        nonzero = (arr > 0.5).sum()
        assert abs(int(nonzero) - brain_voxels) / brain_voxels < 0.05

    def test_rotation_preserves_brain_volume(self, align_module):
        """A rigid rotation + translation preserves the brain voxel count."""
        moving, brain_voxels = self._make_moving()

        t = sitk.Euler3DTransform()
        t.SetRotation(np.deg2rad(15), np.deg2rad(-10), np.deg2rad(30))
        t.SetTranslation((0.3, -0.2, 0.1))
        center_mm = moving.TransformContinuousIndexToPhysicalPoint([s / 2.0 for s in moving.GetSize()])
        t.SetCenter(center_mm)

        ref, composite = align_module.compute_centered_reference_and_transform(moving, t)

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(ref)
        resampler.SetTransform(composite)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0.0)
        out = resampler.Execute(moving)
        arr = sitk.GetArrayFromImage(out)

        nonzero = (arr > 0.5).sum()
        # Rigid transform preserves volume; allow 5% tolerance for interpolation.
        assert abs(int(nonzero) - brain_voxels) / brain_voxels < 0.05

    def test_composite_transform_semantics(self, align_module):
        """The composite must compute T(shift(p)), not shift(T(p)).

        Sample points at the origin of output space (0, 0, 0) and along each
        axis; verify the composite maps them to the same physical point as the
        *manually* composed ``T ∘ shift``.
        """
        moving, _ = self._make_moving()
        t = sitk.Euler3DTransform()
        t.SetRotation(np.deg2rad(20), 0.0, np.deg2rad(-5))
        t.SetTranslation((0.25, 0.1, -0.15))
        center_mm = moving.TransformContinuousIndexToPhysicalPoint([s / 2.0 for s in moving.GetSize()])
        t.SetCenter(center_mm)

        ref, composite = align_module.compute_centered_reference_and_transform(moving, t)

        # Rebuild the shift transform the helper used: offset == pts_min
        # recovered from the composite (last-added transform is the shift).
        sample_points = [
            (0.0, 0.0, 0.0),
            tuple(ref.GetSpacing()),
            tuple(np.array(ref.GetSize(), dtype=float) * ref.GetSpacing() / 2),
        ]
        for p in sample_points:
            actual = np.array(composite.TransformPoint(p))
            # Compose T(shift(p)) manually.  Retrieve shift from the 2-member
            # composite: ITK applies transforms in reverse order, so nth = last
            # added = shift.
            shift = composite.GetNthTransform(1)
            expected = np.array(t.TransformPoint(shift.TransformPoint(p)))
            np.testing.assert_allclose(actual, expected, atol=1e-9)

            # Sanity check that the *wrong* ordering ``shift(T(p))`` does NOT
            # match (unless the transform is degenerate).
            wrong = np.array(shift.TransformPoint(t.TransformPoint(p)))
            assert not np.allclose(actual, wrong, atol=1e-4), "Composite accidentally matches the buggy order shift(T(p))"


# ---------------------------------------------------------------------------
# store_transform_in_metadata (skipped unless zarr fixture available)
# ---------------------------------------------------------------------------


class TestStoreTransformInMetadata:
    """Smoke test: ensure the metadata writer builds a valid affine block."""

    def test_affine_block_written_to_zattrs(self, align_module, tmp_path):
        import json

        # Create a minimal OME-Zarr v0.4 directory with a .zattrs file.
        store_path = tmp_path / "test.ome.zarr"
        store_path.mkdir()
        initial_attrs = {
            "multiscales": [
                {
                    "version": "0.4",
                    "axes": [
                        {"name": "z", "type": "space", "unit": "millimeter"},
                        {"name": "y", "type": "space", "unit": "millimeter"},
                        {"name": "x", "type": "space", "unit": "millimeter"},
                    ],
                    "datasets": [{"path": "0", "coordinateTransformations": []}],
                }
            ]
        }
        (store_path / ".zattrs").write_text(json.dumps(initial_attrs))

        t = sitk.Euler3DTransform()
        t.SetTranslation((0.5, 1.5, 2.5))

        align_module.store_transform_in_metadata(str(store_path), t)

        with (store_path / ".zattrs").open() as f:
            metadata = json.load(f)

        ms = metadata["multiscales"][0]
        ds = ms["datasets"][0]
        ctfs = ds["coordinateTransformations"]
        affines = [c for c in ctfs if c.get("type") == "affine"]
        assert len(affines) == 1
        mat = np.array(affines[0]["affine"]).reshape(4, 4)
        assert mat.shape == (4, 4)
        np.testing.assert_allclose(mat[3], [0, 0, 0, 1], atol=1e-12)
        # Translation must be permuted to NGFF (Z, Y, X) ordering.
        np.testing.assert_allclose(mat[:3, 3], [2.5, 1.5, 0.5], atol=1e-12)
