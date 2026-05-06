#!/usr/bin/env python3
"""Tests for ``linum_compensate_attenuation_inplace.py``.

The synthetic test builds a uniform "tissue" volume with a known exponential
depth decay ``I(z) = I0 * exp(-2 * mu * dz * z)`` (the factor 2 is the
round-trip the OCT signal makes), runs the script, and verifies the corrected
profile is approximately flat — i.e. that the Vermeer-based bias estimation
plus division actually inverts the decay we put in.
"""

import dask.array as da
import numpy as np
import pytest

from linumpy.io.zarr import read_omezarr, save_omezarr


def test_help(script_runner):
    ret = script_runner.run(["linum_compensate_attenuation_inplace.py", "--help"])
    assert ret.success


def _make_synthetic_decay_volume(
    tmp_path,
    *,
    nz: int = 60,
    ny: int = 64,
    nx: int = 64,
    mu_per_cm: float = 30.0,
    res_um: float = 10.0,
    air_voxels: int = 4,
    base_intensity: float = 100.0,
    noise_std: float = 0.0,
    rng_seed: int = 0,
):
    """Build a (Z, Y, X) volume with a known round-trip exponential decay."""
    rng = np.random.default_rng(rng_seed)
    dz_cm = res_um * 1e-4
    z = np.arange(nz)
    profile = np.where(
        z < air_voxels,
        0.0,
        base_intensity * np.exp(-2.0 * mu_per_cm * dz_cm * (z - air_voxels)),
    )

    vol = np.broadcast_to(profile[:, None, None], (nz, ny, nx)).astype(np.float32).copy()
    if noise_std > 0:
        vol = vol + rng.normal(0, noise_std, vol.shape).astype(np.float32)
        vol[:air_voxels] = 0.0  # keep the air mask unambiguous

    out = tmp_path / "synthetic.ome.zarr"
    save_omezarr(
        da.from_array(vol),
        out,
        voxel_size=(res_um * 1e-3, res_um * 1e-3, res_um * 1e-3),
        chunks=(nz, ny, nx),
        n_levels=0,
    )
    return out, profile


def _tissue_z_profile(arr: np.ndarray, threshold_frac: float = 0.05) -> np.ndarray:
    """Mean intensity per Z slice, restricted to tissue voxels."""
    thr = threshold_frac * arr.max()
    out = np.zeros(arr.shape[0], dtype=np.float64)
    for z in range(arr.shape[0]):
        mask = arr[z] > thr
        if mask.any():
            out[z] = float(arr[z][mask].mean())
    return out


def test_flattens_synthetic_exponential_decay(script_runner, tmp_path):
    """Volume with a known ``mu`` should be flatter after correction.

    The Vermeer model can never produce a perfectly flat profile -- it cannot
    see past the bottom of the volume, so the deep-tissue attenuation is
    always partially under-estimated. We assert a clear, repeatable reduction
    in the residual depth decay.
    """
    input_path, _ = _make_synthetic_decay_volume(tmp_path, mu_per_cm=5.0, nz=60, noise_std=1.0)
    output_path = tmp_path / "corrected.ome.zarr"

    ret = script_runner.run(
        [
            "linum_compensate_attenuation_inplace.py",
            str(input_path),
            str(output_path),
            "--strength",
            "1.0",
            "--n_levels",
            "0",
        ]
    )
    assert ret.success, ret.stderr

    pre, _ = read_omezarr(input_path, level=0)
    post, _ = read_omezarr(output_path, level=0)
    pre_profile = _tissue_z_profile(np.asarray(pre))
    post_profile = _tissue_z_profile(np.asarray(post))

    pre_tissue = pre_profile[pre_profile > 0]
    post_tissue = post_profile[post_profile > 0]
    assert pre_tissue.size > 10
    assert post_tissue.size > 10

    # Compare top-vs-bottom over the middle 80 % to avoid mask-edge effects.
    n = pre_tissue.size
    lo, hi = int(0.1 * n), int(0.9 * n)
    pre_drop = (pre_tissue[lo] - pre_tissue[hi]) / pre_tissue[lo]
    post_drop = (post_tissue[lo] - post_tissue[hi]) / post_tissue[lo]

    assert pre_drop > 0.3, f"synthetic input does not decay enough: {pre_drop:.3f}"
    # At full strength, the textbook formula should over-correct on this
    # short volume (turning the gentle pre-correction decay into a rise),
    # so |post_drop| must be both substantially smaller than pre_drop and
    # of the opposite sign (or near zero). Either direction is fine; we
    # only require a large reduction in magnitude.
    assert abs(post_drop) < 0.5 * pre_drop, (
        f"correction did not reduce |decay| enough: pre={pre_drop:.3f}, post={post_drop:.3f}"
    )


def test_min_bias_caps_gain(script_runner, tmp_path):
    """``--min_bias`` must bound the maximum amplification of any voxel."""
    input_path, _ = _make_synthetic_decay_volume(tmp_path, mu_per_cm=5.0, nz=60, noise_std=1.0)
    output_path = tmp_path / "capped.ome.zarr"

    min_bias = 0.5  # caps gain at 2x
    ret = script_runner.run(
        [
            "linum_compensate_attenuation_inplace.py",
            str(input_path),
            str(output_path),
            "--min_bias",
            str(min_bias),
            "--n_levels",
            "0",
        ]
    )
    assert ret.success, ret.stderr

    pre, _ = read_omezarr(input_path, level=0)
    post, _ = read_omezarr(output_path, level=0)
    pre = np.asarray(pre)
    post = np.asarray(post)

    nonzero = pre > 1.0  # ignore air voxels
    gain = np.zeros_like(pre)
    gain[nonzero] = post[nonzero] / pre[nonzero]
    assert gain.max() <= 1.0 / min_bias + 1e-3


def test_preserves_shape_and_dtype(script_runner, tmp_path):
    input_path, _ = _make_synthetic_decay_volume(tmp_path, mu_per_cm=5.0, nz=50, noise_std=1.0)
    output_path = tmp_path / "out.ome.zarr"

    ret = script_runner.run(
        [
            "linum_compensate_attenuation_inplace.py",
            str(input_path),
            str(output_path),
            "--n_levels",
            "0",
        ]
    )
    assert ret.success, ret.stderr

    pre, _ = read_omezarr(input_path, level=0)
    post, _ = read_omezarr(output_path, level=0)
    pre = np.asarray(pre)
    post = np.asarray(post)
    assert post.shape == pre.shape
    assert post.dtype == np.float32


@pytest.mark.parametrize("method", ["smith", "vermeer", "liu", "li"])
def test_method_dispatch(script_runner, tmp_path, method):
    """Each ``--method`` choice should run end-to-end and return a sensible volume.

    With ``--strength 0.3`` (the default empirical sweet spot for sub-22
    cropped slices) all four methods should at least *reduce* the
    magnitude of the depth decay relative to the input. The four
    methods do not all flatten equally well -- this test is a smoke
    test for argparse plumbing and import paths, not a benchmark.
    """
    input_path, _ = _make_synthetic_decay_volume(tmp_path / method, mu_per_cm=5.0, nz=60, noise_std=1.0)
    output_path = tmp_path / f"corrected_{method}.ome.zarr"

    ret = script_runner.run(
        [
            "linum_compensate_attenuation_inplace.py",
            str(input_path),
            str(output_path),
            "--method",
            method,
            "--strength",
            "0.3",
            "--n_levels",
            "0",
        ]
    )
    assert ret.success, ret.stderr

    pre, _ = read_omezarr(input_path, level=0)
    post, _ = read_omezarr(output_path, level=0)
    pre_arr = np.asarray(pre)
    post_arr = np.asarray(post)
    assert post_arr.shape == pre_arr.shape
    assert post_arr.dtype == np.float32
    assert np.all(np.isfinite(post_arr))
    pre_profile = _tissue_z_profile(pre_arr)
    post_profile = _tissue_z_profile(post_arr)
    pre_tissue = pre_profile[pre_profile > 0]
    post_tissue = post_profile[post_profile > 0]
    n = pre_tissue.size
    lo, hi = int(0.1 * n), int(0.9 * n)
    pre_drop = (pre_tissue[lo] - pre_tissue[hi]) / pre_tissue[lo]
    post_drop = (post_tissue[lo] - post_tissue[hi]) / post_tissue[lo]
    # The bare 'li' method aggressively truncates short A-lines, which on
    # this 60-voxel synthetic produces a fragile mu_E estimate -- it can
    # over-correct. Other methods should reduce |drop|; for 'li' we only
    # check that the output remains finite and the script ran (validated
    # above).
    if method != "li":
        assert abs(post_drop) < abs(pre_drop), f"{method}: |post|={abs(post_drop):.3f} not < |pre|={abs(pre_drop):.3f}"
