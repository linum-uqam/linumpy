#!/usr/bin/env python3
"""Tests for ``scripts/conversion/linum_fix_galvo_shift_zarr.py``.

The script is loaded via :mod:`importlib` so we can test its pure shift-
detection/correction helpers on small synthetic OME-Zarr fixtures (no full
acquisition mosaic) without relying on the console entry point. Locks
behavior before extraction to ``linumpy/geometry/galvo.py`` (D-85).
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from linumpy.io.zarr import OmeZarrWriter, read_omezarr

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "conversion" / "linum_fix_galvo_shift_zarr.py"


@pytest.fixture(scope="module")
def fix_galvo_module():
    """Load ``linum-fix-galvo-shift-zarr`` as a module."""
    spec = importlib.util.spec_from_file_location("linum_fix_galvo_shift_zarr", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Synthetic fixture
# ---------------------------------------------------------------------------

NZ, CHUNK_X, CHUNK_Y, N_CX, N_CY = 3, 30, 8, 2, 1
BAND_START, BAND_WIDTH = 20, 6
BRIGHT, DARK = 100.0, 10.0


def _write_synthetic_mosaic(tmp_path: Path) -> Path:
    """Write a small synthetic mosaic OME-Zarr with a dark galvo band in every tile."""
    profile = np.full(CHUNK_X, BRIGHT, dtype=np.float32)
    profile[BAND_START : BAND_START + BAND_WIDTH] = DARK
    tile = np.tile(profile[:, None], (1, CHUNK_Y))

    vol = np.zeros((NZ, CHUNK_X * N_CX, CHUNK_Y * N_CY), dtype=np.float32)
    for kx in range(N_CX):
        vol[:, kx * CHUNK_X : (kx + 1) * CHUNK_X, :] = tile

    zarr_path = tmp_path / "mosaic.ome.zarr"
    writer = OmeZarrWriter(zarr_path, shape=vol.shape, chunk_shape=(NZ, CHUNK_X, CHUNK_Y), dtype=vol.dtype, overwrite=True)
    writer[:] = vol
    writer.finalize([1.0, 1.0, 1.0], n_levels=0)
    return zarr_path


# ---------------------------------------------------------------------------
# CLI help
# ---------------------------------------------------------------------------


def test_help(script_runner):
    ret = script_runner.run(["linum-fix-galvo-shift-zarr", "--help"])
    assert ret.success


# ---------------------------------------------------------------------------
# _auto_detect
# ---------------------------------------------------------------------------


def test_auto_detect_finds_band(tmp_path, fix_galvo_module):
    zarr_path = _write_synthetic_mosaic(tmp_path)

    band_start, band_width, confidence = fix_galvo_module._auto_detect(zarr_path, n_extra=BAND_WIDTH)

    assert band_start == 19
    assert band_width == BAND_WIDTH
    assert confidence == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _apply_fix
# ---------------------------------------------------------------------------


def test_apply_fix_rolls_band_to_edge(tmp_path, fix_galvo_module):
    zarr_path = _write_synthetic_mosaic(tmp_path)
    band_start, band_width, _confidence = fix_galvo_module._auto_detect(zarr_path, n_extra=BAND_WIDTH)

    out_path = tmp_path / "fixed.ome.zarr"
    fix_galvo_module._apply_fix(
        zarr_root=zarr_path,
        output_path=out_path,
        band_start=band_start,
        band_width=band_width,
        mode="fix",
        undo_shift=None,
        overwrite=True,
        workers=1,
        n_extra=BAND_WIDTH,
    )

    arr_out, _res = read_omezarr(out_path)
    out = np.asarray(arr_out[:])

    expected_profile = np.full(CHUNK_X, BRIGHT, dtype=np.float32)
    expected_profile[[0, 25, 26, 27, 28, 29]] = DARK

    for kx in range(N_CX):
        tile_profile = out[0, kx * CHUNK_X : (kx + 1) * CHUNK_X, 0]
        np.testing.assert_array_equal(tile_profile, expected_profile)


def test_apply_fix_undo_reverses_fix(tmp_path, fix_galvo_module):
    zarr_path = _write_synthetic_mosaic(tmp_path)
    band_start, band_width, _confidence = fix_galvo_module._auto_detect(zarr_path, n_extra=BAND_WIDTH)

    fixed_path = tmp_path / "fixed.ome.zarr"
    fix_galvo_module._apply_fix(
        zarr_root=zarr_path,
        output_path=fixed_path,
        band_start=band_start,
        band_width=band_width,
        mode="fix",
        undo_shift=None,
        overwrite=True,
        workers=1,
        n_extra=BAND_WIDTH,
    )

    applied_shift = CHUNK_X - band_start - band_width
    undone_path = tmp_path / "undone.ome.zarr"
    fix_galvo_module._apply_fix(
        zarr_root=fixed_path,
        output_path=undone_path,
        band_start=0,
        band_width=0,
        mode="undo",
        undo_shift=applied_shift,
        overwrite=True,
        workers=1,
    )

    original_arr, _res = read_omezarr(zarr_path)
    undone_arr, _res = read_omezarr(undone_path)
    np.testing.assert_array_equal(np.asarray(undone_arr[:]), np.asarray(original_arr[:]))


# ---------------------------------------------------------------------------
# _parse_skip_tiles
# ---------------------------------------------------------------------------


def test_parse_skip_tiles_empty(fix_galvo_module):
    assert fix_galvo_module._parse_skip_tiles("") == frozenset()


def test_parse_skip_tiles_parses_pairs(fix_galvo_module):
    result = fix_galvo_module._parse_skip_tiles("13,4;13,8;3,3")
    assert result == frozenset({(13, 4), (13, 8), (3, 3)})
