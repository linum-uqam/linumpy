"""Tests for illumination-sweep core logic in ``linumpy/intensity/sweep.py``."""

import numpy as np
import pytest

from linumpy.intensity.sweep import (
    assemble_from_tiles,
    build_sweep_grid,
    parse_bool_list,
    parse_float_none_list,
    run_one_config,
    split_into_tiles,
)

# ---------------------------------------------------------------------------
# parse_float_none_list / parse_bool_list
# ---------------------------------------------------------------------------


def test_parse_float_none_list_parses_mixed_none_and_floats():
    assert parse_float_none_list("none,99.0,99.5") == [None, 99.0, 99.5]


def test_parse_bool_list_parses_true_false():
    assert parse_bool_list("true,false") == [True, False]


def test_parse_bool_list_invalid_token_raises():
    with pytest.raises(ValueError, match="Cannot parse bool value"):
        parse_bool_list("maybe")


# ---------------------------------------------------------------------------
# split_into_tiles / assemble_from_tiles
# ---------------------------------------------------------------------------


def test_split_assemble_tiles_roundtrip():
    rng = np.random.default_rng(0)
    plane = rng.random((8, 12)).astype(np.float32)
    tile_shape = (4, 4)
    tiles = split_into_tiles(plane, tile_shape)
    assert tiles.shape == (6, 4, 4)
    rebuilt = assemble_from_tiles(tiles, plane.shape, tile_shape)
    np.testing.assert_array_equal(rebuilt, plane)


# ---------------------------------------------------------------------------
# build_sweep_grid
# ---------------------------------------------------------------------------


def test_build_sweep_grid_dedupes_irrelevant_darkfield_axes():
    # use_darkfield=False makes df_percentile/darkfield_smooth_sigma irrelevant,
    # so the two df_percs values should collapse into one config.
    configs = build_sweep_grid(
        p_maxes=[None],
        use_darks=[False],
        df_percs=[2.0, 5.0],
        fit_samps=[100],
        max_iters=[10],
        smooth_ffs=[None],
        working_sizes=[None],
        per_z_fits=[False],
        df_smooth_sigmas=[0.0],
        df_z_windows=[0],
        ff_smooth_sigmas=[0.0],
    )
    assert len(configs) == 1


def test_build_sweep_grid_keeps_relevant_axes_distinct():
    configs = build_sweep_grid(
        p_maxes=[None],
        use_darks=[True],
        df_percs=[2.0, 5.0],
        fit_samps=[100],
        max_iters=[10],
        smooth_ffs=[None],
        working_sizes=[None],
        per_z_fits=[False],
        df_smooth_sigmas=[0.0],
        df_z_windows=[0],
        ff_smooth_sigmas=[0.0],
    )
    assert len(configs) == 2


# ---------------------------------------------------------------------------
# run_one_config (requires optional linum_basic dependency)
# ---------------------------------------------------------------------------


def test_run_one_config_fits_and_corrects_small_volume():
    pytest.importorskip("linum_basic")
    rng = np.random.default_rng(0)
    vol = (rng.random((4, 16, 16)).astype(np.float32) * 100.0) + 10.0
    tile_shape = (8, 8)

    corrected, flatfield, darkfield, stats = run_one_config(
        vol,
        tile_shape,
        percentile_max=None,
        use_darkfield=False,
        darkfield_percentile=5.0,
        fit_max_samples=10,
        max_iterations=5,
        smoothness_flatfield=None,
        working_size=None,
        apply_z=[0],
        preview_z=0,
    )

    assert set(corrected.keys()) == {0}
    assert corrected[0].shape == (16, 16)
    assert flatfield.shape == (8, 8)
    assert darkfield is None or np.allclose(darkfield, 0.0)
    assert stats["n_fit_tiles"] > 0
