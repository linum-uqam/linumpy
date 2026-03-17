# -*- coding: utf-8 -*-
"""Tests for linumpy/utils/orientation.py"""

import numpy as np
import pytest

from linumpy.utils.orientation import (
    apply_orientation_transform,
    parse_orientation_code,
    reorder_resolution,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gradient_vol(shape=(4, 6, 8)):
    """Create a volume where each voxel value encodes its (z, x, y) index."""
    z, x, y = np.indices(shape)
    # Unique encoding that allows axis identification
    return (z * 1000 + x * 100 + y).astype(np.float32)


# ---------------------------------------------------------------------------
# parse_orientation_code — valid codes
# ---------------------------------------------------------------------------

class TestParseOrientationCodeValid:
    def test_identity_SRA(self):
        """SRA is the native target order → identity permutation."""
        perm, flips = parse_orientation_code('SRA')
        assert perm == (0, 1, 2)
        assert flips == (1, 1, 1)

    def test_identity_lowercase(self):
        """Input is case-insensitive."""
        perm, flips = parse_orientation_code('sra')
        assert perm == (0, 1, 2)
        assert flips == (1, 1, 1)

    def test_PIR(self):
        """PIR is a common OCT orientation."""
        perm, flips = parse_orientation_code('PIR')
        assert perm == (1, 2, 0)
        assert flips == (-1, 1, -1)

    def test_RAS(self):
        """RAS orientation (Allen/NIfTI default, but dim0=R not S)."""
        perm, flips = parse_orientation_code('RAS')
        # R→target-dim1, A→target-dim2, S→target-dim0
        # source: dim0=R, dim1=A, dim2=S
        # target order (S, R, A): dim0←source_dim2, dim1←source_dim0, dim2←source_dim1
        assert perm == (2, 0, 1)
        assert flips == (1, 1, 1)

    def test_LPS(self):
        """LPS (opposite of RAS)."""
        perm, flips = parse_orientation_code('LPS')
        # L→target-dim1(flip), P→target-dim2(flip), S→target-dim0
        # source: dim0=L, dim1=P, dim2=S
        # S in dim2 → target dim0, so source_dim2 for target dim0
        # L in dim0 → target dim1, flip; P in dim1 → target dim2, flip
        assert perm == (2, 0, 1)
        assert flips == (1, -1, -1)

    def test_all_flipped_ILP(self):
        """ILP: all three axes need to be flipped (I→S, L→R, P→A)."""
        perm, flips = parse_orientation_code('ILP')
        # I at dim0 → target dim0 (Superior), flip; L at dim1 → target dim1 (Right), flip;
        # P at dim2 → target dim2 (Anterior), flip
        assert all(f == -1 for f in flips)
        assert sorted(perm) == [0, 1, 2]

    def test_AIR(self):
        """AIR: A in dim0, I in dim1, R in dim2."""
        perm, flips = parse_orientation_code('AIR')
        # A at dim0 → target dim2, sign=+1
        # I at dim1 → target dim0, sign=-1
        # R at dim2 → target dim1, sign=+1
        # target_to_source: {0: (1, -1), 1: (2, 1), 2: (0, 1)}
        assert perm == (1, 2, 0)
        assert flips == (-1, 1, 1)

    def test_output_type_is_tuple(self):
        perm, flips = parse_orientation_code('SRA')
        assert isinstance(perm, tuple)
        assert isinstance(flips, tuple)

    def test_output_perm_length_3(self):
        perm, flips = parse_orientation_code('PIR')
        assert len(perm) == 3
        assert len(flips) == 3

    def test_perm_is_valid_permutation(self):
        """axis_permutation must be a valid permutation of (0,1,2)."""
        for code in ('SRA', 'PIR', 'RAS', 'LPS', 'AIR', 'ILP', 'SAR'):
            perm, _ = parse_orientation_code(code)
            assert sorted(perm) == [0, 1, 2], f"Bad permutation for {code}: {perm}"

    def test_flips_only_1_or_minus1(self):
        for code in ('SRA', 'PIR', 'RAS', 'LPS', 'AIR', 'ILP', 'SAR'):
            _, flips = parse_orientation_code(code)
            for f in flips:
                assert f in (1, -1), f"Unexpected flip value {f} for {code}"


# ---------------------------------------------------------------------------
# parse_orientation_code — error cases
# ---------------------------------------------------------------------------

class TestParseOrientationCodeErrors:
    def test_too_short(self):
        with pytest.raises(ValueError, match="3 letters"):
            parse_orientation_code('SR')

    def test_too_long(self):
        with pytest.raises(ValueError, match="3 letters"):
            parse_orientation_code('SRAX')

    def test_invalid_letter(self):
        with pytest.raises(ValueError, match="Invalid orientation letter"):
            parse_orientation_code('XRA')

    def test_duplicate_axis_same_direction(self):
        """RRS has R mapping to target-dim1 twice."""
        with pytest.raises(ValueError):
            parse_orientation_code('RRS')

    def test_duplicate_axis_opposite_direction(self):
        """RLS has R=dim1 and L=dim1 — same target axis."""
        with pytest.raises(ValueError):
            parse_orientation_code('RLS')

    def test_missing_axis(self):
        """SAI uses neither R nor L so target-dim1 is missing."""
        # S→dim0, A→dim2, I→dim0 — actually duplicate! Let's use a truly missing case.
        # SAP: S→0, A→2, P→2 — duplicate (A and P both target dim2).
        with pytest.raises(ValueError):
            parse_orientation_code('SAP')


# ---------------------------------------------------------------------------
# apply_orientation_transform
# ---------------------------------------------------------------------------

class TestApplyOrientationTransform:
    def test_identity_permutation_no_flip(self):
        vol = _make_gradient_vol((4, 6, 8))
        result = apply_orientation_transform(vol, (0, 1, 2), (1, 1, 1))
        np.testing.assert_array_equal(result, vol)

    def test_permutation_changes_shape(self):
        vol = np.zeros((4, 6, 8))
        result = apply_orientation_transform(vol, (1, 2, 0), (1, 1, 1))
        assert result.shape == (6, 8, 4)

    def test_flip_axis0(self):
        vol = np.arange(24).reshape(4, 3, 2).astype(np.float32)
        result = apply_orientation_transform(vol, (0, 1, 2), (-1, 1, 1))
        np.testing.assert_array_equal(result, vol[::-1, :, :])

    def test_flip_axis1(self):
        vol = np.arange(24).reshape(4, 3, 2).astype(np.float32)
        result = apply_orientation_transform(vol, (0, 1, 2), (1, -1, 1))
        np.testing.assert_array_equal(result, vol[:, ::-1, :])

    def test_flip_axis2(self):
        vol = np.arange(24).reshape(4, 3, 2).astype(np.float32)
        result = apply_orientation_transform(vol, (0, 1, 2), (1, 1, -1))
        np.testing.assert_array_equal(result, vol[:, :, ::-1])

    def test_permutation_and_flip(self):
        """Permute (1,0,2) then flip axis0."""
        vol = np.arange(24).reshape(4, 3, 2).astype(np.float32)
        result = apply_orientation_transform(vol, (1, 0, 2), (-1, 1, 1))
        expected = np.transpose(vol, (1, 0, 2))[::-1, :, :]
        np.testing.assert_array_equal(result, expected)

    def test_does_not_modify_input(self):
        vol = np.arange(24).reshape(4, 3, 2).astype(np.float32)
        original = vol.copy()
        apply_orientation_transform(vol, (1, 2, 0), (-1, 1, -1))
        np.testing.assert_array_equal(vol, original)


# ---------------------------------------------------------------------------
# Roundtrip: applying orientation + inverse gives back the original
# ---------------------------------------------------------------------------

class TestOrientationRoundtrip:
    def _inverse_permutation(self, perm):
        """Compute the inverse of a permutation tuple."""
        inv = [0] * len(perm)
        for i, p in enumerate(perm):
            inv[p] = i
        return tuple(inv)

    def test_roundtrip_PIR(self):
        vol = _make_gradient_vol((5, 7, 9))
        perm, flips = parse_orientation_code('PIR')

        # Forward: source → target (SRA)
        forward = apply_orientation_transform(vol, perm, flips)

        # Inverse permutation and de-flip
        inv_perm = self._inverse_permutation(perm)
        # After inverse permutation flips need to be in the final axis order
        # The flip axes in the forward result correspond to the target axes.
        # To undo: first undo flips (same flips since flip is its own inverse),
        # then apply inverse permutation.
        unflipped = apply_orientation_transform(forward, (0, 1, 2), flips)  # flip back
        recovered = apply_orientation_transform(unflipped, inv_perm, (1, 1, 1))

        np.testing.assert_array_equal(recovered, vol)

    def test_roundtrip_RAS(self):
        vol = _make_gradient_vol((3, 5, 7))
        perm, flips = parse_orientation_code('RAS')

        forward = apply_orientation_transform(vol, perm, flips)

        inv_perm = self._inverse_permutation(perm)
        unflipped = apply_orientation_transform(forward, (0, 1, 2), flips)
        recovered = apply_orientation_transform(unflipped, inv_perm, (1, 1, 1))

        np.testing.assert_array_equal(recovered, vol)

    def test_roundtrip_all_flipped_ILP(self):
        """A code with all axes needing a flip."""
        vol = _make_gradient_vol((4, 6, 8))
        perm, flips = parse_orientation_code('ILP')

        forward = apply_orientation_transform(vol, perm, flips)

        inv_perm = self._inverse_permutation(perm)
        unflipped = apply_orientation_transform(forward, (0, 1, 2), flips)
        recovered = apply_orientation_transform(unflipped, inv_perm, (1, 1, 1))

        np.testing.assert_array_equal(recovered, vol)


# ---------------------------------------------------------------------------
# Semantic correctness: after reorientation the expected anatomical axis
# lands in the expected output dimension.
# ---------------------------------------------------------------------------

class TestOrientationSemantics:
    """
    For a volume whose signal varies along a known anatomical axis,
    confirm that after reorientation the variation is in the expected
    output dimension.
    """

    def test_SRA_dim0_is_superior(self):
        """With 'SRA', dim0 is already Superior.  Reorientation is identity."""
        # Volume increases only along dim0 (Superior direction)
        vol = np.zeros((10, 5, 5), dtype=np.float32)
        vol[:, 2, 2] = np.arange(10)

        perm, flips = parse_orientation_code('SRA')
        result = apply_orientation_transform(vol, perm, flips)

        # After identity reorientation, variation should still be along dim0
        assert result.shape[0] == 10
        col = result[:, 2, 2]
        assert col[-1] > col[0], "Superior direction should still increase along dim0"

    def test_IRA_superior_flipped_to_dim0(self):
        """With 'IRA', dim0 is Inferior → after reorientation it becomes Superior (flipped)."""
        vol = np.zeros((10, 5, 5), dtype=np.float32)
        vol[:, 2, 2] = np.arange(10)  # value increases in Inferior direction

        perm, flips = parse_orientation_code('IRA')
        result = apply_orientation_transform(vol, perm, flips)

        # 'IRA': I at dim0 → target dim0 with flip=-1 (Inferior→Superior).
        # Values increasing along Inferior (dim0 source) should decrease along dim0 output.
        slice_col = result[:, 2, 2]
        assert slice_col[0] > slice_col[-1], (
            "After I→S flip, values should decrease along output dim0 (Superior direction)"
        )

    def test_PIR_output_shape(self):
        """PIR → output shape should be a permutation of input shape."""
        shape = (10, 15, 20)
        vol = np.zeros(shape)
        perm, flips = parse_orientation_code('PIR')
        result = apply_orientation_transform(vol, perm, flips)
        # perm=(1,2,0): output shape = (input[1], input[2], input[0]) = (15, 20, 10)
        assert result.shape == (shape[perm[0]], shape[perm[1]], shape[perm[2]])


# ---------------------------------------------------------------------------
# reorder_resolution
# ---------------------------------------------------------------------------

class TestReorderResolution:
    def test_identity_permutation(self):
        res = (0.01, 0.02, 0.03)
        assert reorder_resolution(res, (0, 1, 2)) == res

    def test_cyclic_permutation(self):
        res = (0.01, 0.02, 0.03)
        reordered = reorder_resolution(res, (1, 2, 0))
        # index 0 of output ← res[1], index 1 ← res[2], index 2 ← res[0]
        assert reordered == (0.02, 0.03, 0.01)

    def test_reverse_permutation(self):
        res = (1.0, 2.0, 3.0)
        reordered = reorder_resolution(res, (2, 1, 0))
        assert reordered == (3.0, 2.0, 1.0)

    def test_result_is_tuple(self):
        res = (0.025, 0.025, 0.025)
        result = reorder_resolution(res, (0, 1, 2))
        assert isinstance(result, tuple)

    def test_matches_orientation_permutation(self):
        """reorder_resolution must be consistent with parse_orientation_code."""
        # For 'PIR': perm=(1,2,0)
        # Source resolution: (res_z=0.01, res_x=0.02, res_y=0.03) in (P, I, R) order
        # After reorientation to (S, R, A):
        #   target_dim0 = source_dim1 (I), so resolution[target0] = 0.02
        #   target_dim1 = source_dim2 (R), so resolution[target1] = 0.03
        #   target_dim2 = source_dim0 (P), so resolution[target2] = 0.01
        perm, _ = parse_orientation_code('PIR')
        source_res = (0.01, 0.02, 0.03)
        result = reorder_resolution(source_res, perm)
        assert result == (0.02, 0.03, 0.01)

    def test_reorder_preserves_len(self):
        perm, _ = parse_orientation_code('AIR')
        res = (0.025, 0.025, 0.025)
        result = reorder_resolution(res, perm)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Integration: parse → apply → reorder gives anatomically consistent result
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_isotropic_resolution_unchanged_by_reorder(self):
        """For isotropic data, resolution is the same regardless of permutation."""
        perm, _ = parse_orientation_code('PIR')
        res = (0.025, 0.025, 0.025)
        reordered = reorder_resolution(res, perm)
        assert all(r == 0.025 for r in reordered)

    def test_volume_shape_after_permutation_matches_reordered_resolution(self):
        """
        After applying orientation transform, each output dimension's physical
        size (shape * resolution) should equal the source physical size for that
        anatomical axis.
        """
        shape = (10, 20, 30)  # (P direction, I direction, R direction) in PIR
        res = (0.01, 0.02, 0.03)   # resolutions in (P, I, R) order

        vol = np.ones(shape)
        perm, flips = parse_orientation_code('PIR')
        result = apply_orientation_transform(vol, perm, flips)
        reordered_res = reorder_resolution(res, perm)

        # Physical extent in each target dimension
        for i in range(3):
            src_dim = perm[i]
            assert result.shape[i] == shape[src_dim]
            assert reordered_res[i] == res[src_dim]
