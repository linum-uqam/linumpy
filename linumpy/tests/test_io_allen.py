"""Tests for linumpy/io/allen.py -- orientation handling and registration.

The Allen template download is monkey-patched to return a synthetic PIR-oriented
volume with a deliberately asymmetric tissue distribution.  That keeps these
tests offline and lets us verify that ``download_template_ras_aligned`` really
produces a RAS+ volume (``+X = Right``, ``+Y = Anterior``, ``+Z = Superior``).
"""

from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from linumpy.reference import allen

# ---------------------------------------------------------------------------
# Synthetic PIR-oriented Allen template
# ---------------------------------------------------------------------------


def _make_synthetic_pir_template(resolution_um: int = 100) -> sitk.Image:
    """Build a small synthetic volume that mimics the Allen CCF nrrd layout.

    Allen CCF v3 stores the template in PIR:
        nrrd axis 0 = AP  (+=Posterior)
        nrrd axis 1 = DV  (+=Inferior)
        nrrd axis 2 = ML  (+=Right)

    ``sitk.ReadImage`` maps nrrd axis k to SITK axis k, so the returned
    SITK image has ``(X, Y, Z) = (AP, DV, ML)``.  Each axis is given a
    unique, monotonically increasing gradient so we can identify the
    resulting orientation unambiguously after the RAS reorientation.
    """
    # Pick axis sizes that are all distinct so permutations are detectable.
    ap_size, dv_size, ml_size = 12, 8, 10

    # numpy shape (Z, Y, X) for sitk.GetImageFromArray:
    #   numpy Z ↔ SITK Z = ML
    #   numpy Y ↔ SITK Y = DV
    #   numpy X ↔ SITK X = AP
    ap = np.arange(ap_size, dtype=np.float32)[None, None, :] * 1.0  # unit step
    dv = np.arange(dv_size, dtype=np.float32)[None, :, None] * 100.0
    ml = np.arange(ml_size, dtype=np.float32)[:, None, None] * 10000.0

    arr = ap + dv + ml  # each axis contributes a distinct decimal place

    vol = sitk.GetImageFromArray(arr)
    r_mm = resolution_um / 1e3
    vol.SetSpacing((r_mm, r_mm, r_mm))
    vol.SetOrigin((0.0, 0.0, 0.0))
    vol.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
    return vol


# ---------------------------------------------------------------------------
# download_template_ras_aligned -- orientation
# ---------------------------------------------------------------------------


class TestDownloadTemplateRasAligned:
    """Verify the RAS reorientation of the Allen template."""

    @pytest.fixture
    def ras_template(self, monkeypatch):
        def fake_download_template(resolution, cache=True, cache_dir=".data/"):
            return _make_synthetic_pir_template(resolution)

        monkeypatch.setattr(allen, "download_template", fake_download_template)
        return allen.download_template_ras_aligned(100)

    def test_spacing_is_isotropic_and_in_mm(self, ras_template):
        spacing = ras_template.GetSpacing()
        assert spacing == pytest.approx((0.1, 0.1, 0.1))

    def test_origin_is_zero(self, ras_template):
        assert ras_template.GetOrigin() == pytest.approx((0.0, 0.0, 0.0))

    def test_direction_is_identity(self, ras_template):
        assert ras_template.GetDirection() == pytest.approx((1, 0, 0, 0, 1, 0, 0, 0, 1))

    def test_size_reflects_permutation(self, ras_template):
        """After ``PermuteAxes((2, 0, 1))`` the SITK size becomes (ML, AP, DV)."""
        # Input sizes: AP=12, DV=8, ML=10  →  output (ML, AP, DV) = (10, 12, 8)
        assert ras_template.GetSize() == (10, 12, 8)

    def test_positive_x_is_right(self, ras_template):
        """+X must point toward Right (originally +ML in nrrd)."""
        arr = sitk.GetArrayFromImage(ras_template)
        # numpy axis 2 = SITK X; ML gradient was the `10000` coefficient.
        col = arr[0, 0, :]
        diffs = np.diff(col)
        # Gradient along X in RAS-aligned volume should increase monotonically.
        assert np.all(diffs > 0), f"+X is not monotonic along ML (Right): {col}"

    def test_positive_y_is_anterior(self, ras_template):
        """+Y must point toward Anterior (originally -AP in nrrd).

        Raw AP gradient increases with +Posterior, so after reorientation the
        AP gradient should DECREASE along +Y (since +Y = Anterior).
        """
        arr = sitk.GetArrayFromImage(ras_template)
        # numpy axis 1 = SITK Y; AP gradient was the `1.0` coefficient.
        # Extract AP component by taking the modulo-100 decimal of a single X,Z column.
        col = arr[0, :, 0] % 100.0  # keep only AP contribution (0 .. 11)
        diffs = np.diff(col)
        assert np.all(diffs < 0), f"+Y is not anterior (AP should decrease): {col}"

    def test_positive_z_is_superior(self, ras_template):
        """+Z must point toward Superior (originally -DV in nrrd).

        Raw DV gradient increases with +Inferior, so after reorientation the
        DV gradient should DECREASE along +Z (since +Z = Superior).
        """
        arr = sitk.GetArrayFromImage(ras_template)
        # numpy axis 0 = SITK Z; DV gradient was the `100` coefficient.
        # Extract DV component using (value % 10000) // 100.
        col = (arr[:, 0, 0] % 10000.0) // 100.0  # 0 .. 7
        diffs = np.diff(col)
        assert np.all(diffs < 0), f"+Z is not superior (DV should decrease): {col}"


# ---------------------------------------------------------------------------
# numpy_to_sitk_image
# ---------------------------------------------------------------------------


class TestNumpyToSitkImage:
    def test_roundtrip_preserves_values(self):
        arr = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
        img = allen.numpy_to_sitk_image(arr, spacing=(0.1, 0.2, 0.3))
        back = sitk.GetArrayFromImage(img)
        np.testing.assert_array_equal(back, arr)

    def test_spacing_is_permuted_to_xyz(self):
        arr = np.zeros((2, 3, 4), dtype=np.float32)
        img = allen.numpy_to_sitk_image(arr, spacing=(0.1, 0.2, 0.3))
        # spacing=(res_z, res_y, res_x) → SITK GetSpacing=(res_x, res_y, res_z)
        assert img.GetSpacing() == pytest.approx((0.3, 0.2, 0.1))

    def test_size_is_reversed_from_numpy_shape(self):
        arr = np.zeros((2, 3, 4), dtype=np.float32)
        img = allen.numpy_to_sitk_image(arr, spacing=(1.0, 1.0, 1.0))
        assert img.GetSize() == (4, 3, 2)

    def test_origin_and_direction_are_identity(self):
        arr = np.zeros((2, 3, 4), dtype=np.float32)
        img = allen.numpy_to_sitk_image(arr, spacing=(1.0, 1.0, 1.0))
        assert img.GetOrigin() == (0.0, 0.0, 0.0)
        assert img.GetDirection() == (1, 0, 0, 0, 1, 0, 0, 0, 1)

    def test_cast_dtype_produces_float32(self):
        arr = np.ones((2, 3, 4), dtype=np.uint16)
        img = allen.numpy_to_sitk_image(arr, spacing=(1.0, 1.0, 1.0), cast_dtype=np.float32)
        assert img.GetPixelID() == sitk.sitkFloat32

    def test_no_cast_preserves_dtype(self):
        arr = np.ones((2, 3, 4), dtype=np.uint16)
        img = allen.numpy_to_sitk_image(arr, spacing=(1.0, 1.0, 1.0))
        assert img.GetPixelID() == sitk.sitkUInt16

    def test_input_array_not_modified(self):
        arr = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        original = arr.copy()
        allen.numpy_to_sitk_image(arr, spacing=(1.0, 1.0, 1.0), cast_dtype=np.float32)
        np.testing.assert_array_equal(arr, original)


# ---------------------------------------------------------------------------
# Real Allen template -- regression test against the cached nrrd
# ---------------------------------------------------------------------------


# Allen ships ``average_template_<res>.nrrd`` in what its docs call **ASL**
# (axis 0 starts Anterior, axis 1 Superior, axis 2 Left -- each axis named by
# where index 0 is anatomically). Under the NIfTI/radiology convention used
# elsewhere in linumpy the same layout is **PIR** (each axis named by the
# direction it points toward). The legacy informatics-archive nrrd that
# ``download_template`` fetches and the newer ABC-Atlas ``average_template_10``
# share this layout. Anatomical landmarks for a sanity check (verified on the
# 100 µm cached file, voxel-count threshold > 50):
#   * SITK X axis (numpy dim 2) length 132 -- olfactory bulb tip at low index
#     (~28 k voxels in a 20-slab) versus cerebellum at high index (~69 k).
#   * SITK Y axis (numpy dim 1) length 80 -- DV; Inferior at low, Superior high.
#   * SITK Z axis (numpy dim 0) length 114 -- ML; near-symmetric L/R.
ALLEN_NRRD_100UM = Path(".data") / "allen_template_100um.nrrd"


@pytest.mark.skipif(
    not ALLEN_NRRD_100UM.is_file(),
    reason=f"Allen template not cached at {ALLEN_NRRD_100UM}; skip real-NRRD regression test",
)
class TestRealAllenTemplateOrientation:
    """Lock down RAS conversion of the actual Allen 100 µm template.

    These checks run only when the nrrd is already cached locally so the
    test stays offline-friendly. They guard against silent regressions in
    ``download_template_ras_aligned``: if a future change to the permute /
    flip recipe (or to upstream Allen file orientation) leaves the volume
    in some other configuration, the anatomical-landmark assertions below
    will fail loudly.
    """

    def test_raw_template_is_pir_aka_asl(self):
        """The raw nrrd really has Anterior at low SITK X (PIR == Allen 'ASL')."""
        vol = sitk.ReadImage(str(ALLEN_NRRD_100UM))
        assert vol.GetSize() == (132, 80, 114)
        arr = sitk.GetArrayFromImage(vol)  # numpy (Z, Y, X) = (ML, DV, AP)
        # numpy axis 2 == SITK X == AP. Olfactory tip = LOW index, cerebellum = HIGH.
        front = (arr[:, :, 10:30] > 50).sum()  # anterior end
        back = (arr[:, :, -30:-10] > 50).sum()  # posterior end
        assert front < back, f"Raw nrrd is not PIR/ASL (anterior tip {front} should be smaller than posterior {back})"
        # numpy axis 0 == SITK Z == ML: should be roughly symmetric.
        left = (arr[10:30, :, :] > 50).sum()
        right = (arr[-30:-10, :, :] > 50).sum()
        assert left == pytest.approx(right, rel=0.10), f"ML asymmetry too large: L={left} R={right}"

    def test_ras_aligned_has_anterior_at_high_dim1(self):
        """After RAS reorientation, +dim1 must point toward Anterior."""
        ras = allen.download_template_ras_aligned(100)
        # RAS+ numpy ordering is (S, A, R): dim0=S, dim1=A, dim2=R.
        assert ras.GetSize() == (114, 132, 80)
        arr = sitk.GetArrayFromImage(ras)
        assert arr.shape == (80, 132, 114)
        front = (arr[:, -30:-10, :] > 50).sum()  # +A end (anterior tip / olfactory)
        back = (arr[:, 10:30, :] > 50).sum()  # -A end (posterior / cerebellum)
        assert front < back, f"+dim1 is not Anterior (anterior tip {front} >= posterior {back})"

    def test_ras_aligned_is_left_right_symmetric(self):
        """Numpy dim2 (+R) should be roughly mirror-symmetric in mass."""
        ras = allen.download_template_ras_aligned(100)
        arr = sitk.GetArrayFromImage(ras)
        left = (arr[:, :, 10:30] > 50).sum()
        right = (arr[:, :, -30:-10] > 50).sum()
        assert left == pytest.approx(right, rel=0.10), f"L/R asymmetry too large: L={left} R={right}"


# ---------------------------------------------------------------------------
# register_3d_rigid_to_allen -- end-to-end self-registration
# ---------------------------------------------------------------------------


def _make_synthetic_brain(shape=(24, 24, 24), spacing=(0.2, 0.2, 0.2)):
    """Small asymmetric synthetic brain with a unique intensity pattern per axis."""
    z, y, x = np.indices(shape, dtype=np.float32)
    # Ellipsoid mask offset from centre, asymmetric along each axis.
    cz, cy, cx = shape[0] * 0.55, shape[1] * 0.5, shape[2] * 0.45
    rz, ry, rx = shape[0] * 0.35, shape[1] * 0.3, shape[2] * 0.4
    mask = ((z - cz) / rz) ** 2 + ((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2 < 1
    brain = np.zeros(shape, dtype=np.float32)
    # Distinct gradient along each axis so registration has more than a single
    # rotationally symmetric blob to work with.
    brain[mask] = 1.0 + 0.3 * (z[mask] / shape[0]) + 0.5 * (y[mask] / shape[1]) + 0.7 * (x[mask] / shape[2])
    return brain


class TestRegisterRigidToAllen:
    """End-to-end registration tests using a synthetic Allen template."""

    @pytest.fixture(autouse=True)
    def patch_allen(self, monkeypatch):
        def fake_download_template(resolution, cache=True, cache_dir=".data/"):
            return _make_synthetic_pir_template(resolution)

        monkeypatch.setattr(allen, "download_template", fake_download_template)

    def test_self_registration_recovers_identity(self):
        """Registering the RAS Allen template against itself yields ~identity."""
        target = allen.download_template_ras_aligned(100)
        moving = sitk.GetArrayFromImage(target)  # numpy (Z, Y, X)
        # SITK spacing is (X, Y, Z); moving_spacing is (res_z, res_y, res_x)
        sx, sy, sz = target.GetSpacing()
        transform, stop, _err = allen.register_3d_rigid_to_allen(
            moving_image=moving,
            moving_spacing=(sz, sy, sx),
            allen_resolution=100,
            metric="MSE",
            max_iterations=50,
            verbose=False,
        )
        params = transform.GetParameters()
        rotation = np.array(params[:3])
        translation = np.array(params[3:6])
        # The MSE minimum is at identity; allow generous tolerances because the
        # synthetic volume is tiny.
        assert np.max(np.abs(rotation)) < 0.1, f"Rotation too large: {rotation}"
        assert np.max(np.abs(translation)) < 1.0, f"Translation too large: {translation}"
        assert stop  # non-empty stop-condition string

    def test_downsamples_allen_when_moving_is_coarser(self, capsys):
        """If moving resolution > allen resolution, allen must be downsampled."""
        # Moving at 200 µm, allen synthetic at 100 µm → expect downsampling.
        shape = (10, 10, 10)
        moving = _make_synthetic_brain(shape, spacing=(0.2, 0.2, 0.2))
        _, _, _ = allen.register_3d_rigid_to_allen(
            moving_image=moving,
            moving_spacing=(0.2, 0.2, 0.2),
            allen_resolution=100,
            metric="MSE",
            max_iterations=3,
            verbose=True,
        )
        captured = capsys.readouterr().out
        assert "Downsampled Allen atlas" in captured

    def test_does_not_downsample_when_already_coarse(self, capsys):
        """If moving resolution ≤ allen resolution, allen must NOT be downsampled."""
        shape = (10, 10, 10)
        moving = _make_synthetic_brain(shape, spacing=(0.05, 0.05, 0.05))
        _, _, _ = allen.register_3d_rigid_to_allen(
            moving_image=moving,
            moving_spacing=(0.05, 0.05, 0.05),
            allen_resolution=100,
            metric="MSE",
            max_iterations=3,
            verbose=True,
        )
        captured = capsys.readouterr().out
        assert "Downsampled Allen atlas" not in captured

    def test_crop_offset_reported_in_verbose_output(self, capsys):
        """The ``crop_origin_mm`` restoration must add an offset proportional to
        the leading zero-padding of the moving volume.  We use a plain cube so
        the non-zero bounding box equals the cube's shape exactly, making the
        expected crop origin easy to compute.
        """
        # A fully filled cube -- nonzero bbox equals the full cube shape.
        cube_size = 12
        cube = np.ones((cube_size, cube_size, cube_size), dtype=np.float32)
        leading_pad = (20, 15, 25)  # (pad_z, pad_y, pad_x); each > 10 (margin)
        canvas = np.pad(cube, [(p, 5) for p in leading_pad], mode="constant", constant_values=0)

        _, _, _ = allen.register_3d_rigid_to_allen(
            moving_image=canvas,
            moving_spacing=(0.1, 0.1, 0.1),
            allen_resolution=100,
            metric="MSE",
            max_iterations=0,
            verbose=True,
        )
        captured = capsys.readouterr().out
        # Expected crop start per numpy axis (voxels): pad_axis - margin = pad - 10.
        margin = 10
        spacing = 0.1
        expected_numpy = tuple((p - margin) * spacing for p in leading_pad)
        # SITK XYZ = numpy axes (X=2, Y=1, Z=0)
        expected_sitk_xyz = (expected_numpy[2], expected_numpy[1], expected_numpy[0])
        expected_log = (
            "Adjusted translation for crop: +["
            f"{expected_sitk_xyz[0]:.3f}, {expected_sitk_xyz[1]:.3f}, {expected_sitk_xyz[2]:.3f}"
            "] mm (SITK XYZ)"
        )
        assert expected_log in captured, f"Expected log not found. Got:\n{captured}"
