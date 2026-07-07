from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from linumpy.imaging.orientation import (
    compute_centered_reference_and_transform,
    sitk_transform_to_affine_matrix,
    store_transform_in_metadata,
)


class TestSitkTransformToAffineMatrix:
    def test_identity_transform_yields_identity_matrix(self):
        transform = sitk.Euler3DTransform()
        matrix = sitk_transform_to_affine_matrix(transform)
        assert matrix.shape == (4, 4)
        np.testing.assert_allclose(matrix, np.eye(4), atol=1e-12)

    def test_affine_transform_translation_is_permuted_to_zyx(self):
        transform = sitk.AffineTransform(3)
        transform.SetTranslation((1.0, 2.0, 3.0))
        matrix = sitk_transform_to_affine_matrix(transform)
        np.testing.assert_allclose(matrix[:3, 3], [3.0, 2.0, 1.0], atol=1e-12)

    def test_unsupported_transform_type_raises(self):
        transform = sitk.TranslationTransform(3)
        with pytest.raises(ValueError, match="Unsupported transform type"):
            sitk_transform_to_affine_matrix(transform)


class TestComputeCenteredReferenceAndTransform:
    @staticmethod
    def _make_moving(shape=(20, 20, 20), spacing=(0.1, 0.1, 0.1)):
        z, y, x = np.indices(shape, dtype=np.float32)
        cz, cy, cx = shape[0] / 2, shape[1] / 2, shape[2] / 2
        rz, ry, rx = shape[0] * 0.3, shape[1] * 0.3, shape[2] * 0.3
        mask = ((z - cz) / rz) ** 2 + ((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2 < 1
        image = sitk.GetImageFromArray(mask.astype(np.float32))
        image.SetSpacing((spacing[2], spacing[1], spacing[0]))
        image.SetOrigin((0.0, 0.0, 0.0))
        image.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
        return image, int(mask.sum())

    def test_reference_origin_is_zero(self):
        moving, _ = self._make_moving()
        reference, _ = compute_centered_reference_and_transform(moving, sitk.Euler3DTransform())
        assert reference.GetOrigin() == pytest.approx((0.0, 0.0, 0.0))

    def test_rotation_preserves_brain_volume(self):
        moving, brain_voxels = self._make_moving()
        transform = sitk.Euler3DTransform()
        transform.SetRotation(np.deg2rad(15), np.deg2rad(-10), np.deg2rad(30))
        transform.SetTranslation((0.3, -0.2, 0.1))
        center_mm = moving.TransformContinuousIndexToPhysicalPoint([axis / 2.0 for axis in moving.GetSize()])
        transform.SetCenter(center_mm)

        reference, composite = compute_centered_reference_and_transform(moving, transform)

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(reference)
        resampler.SetTransform(composite)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0.0)
        output = resampler.Execute(moving)
        array = sitk.GetArrayFromImage(output)

        nonzero = (array > 0.5).sum()
        assert abs(int(nonzero) - brain_voxels) / brain_voxels < 0.05


def test_store_transform_in_metadata_writes_affine_block(tmp_path: Path):
    import json

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

    transform = sitk.Euler3DTransform()
    transform.SetTranslation((0.5, 1.5, 2.5))

    store_transform_in_metadata(store_path, transform)

    with (store_path / ".zattrs").open() as handle:
        metadata = json.load(handle)

    affine = metadata["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]
    matrix = np.array(affine["affine"]).reshape(4, 4)
    np.testing.assert_allclose(matrix[:3, 3], [2.5, 1.5, 0.5], atol=1e-12)
