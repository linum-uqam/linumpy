"""Unit tests for the zarr-path galvo-shift helpers extracted to linumpy.geometry.galvo (RECON-04)."""

import numpy as np

from linumpy.geometry.galvo import aggregate_band_detections, decide_tile_shift

CHUNK_X = 30
BAND_START, BAND_WIDTH = 20, 6
BRIGHT, DARK = 100.0, 10.0


def _tile_aip_with_band() -> np.ndarray:
    profile = np.full(CHUNK_X, BRIGHT, dtype=np.float32)
    profile[BAND_START : BAND_START + BAND_WIDTH] = DARK
    return np.tile(profile[:, None], (1, 8))


# ---------------------------------------------------------------------------
# aggregate_band_detections
# ---------------------------------------------------------------------------


def test_aggregate_band_detections_empty_returns_zero():
    assert aggregate_band_detections([], chunk_size=CHUNK_X) == (0, 0, 0.0)


def test_aggregate_band_detections_single_detection_is_unchanged():
    assert aggregate_band_detections([(20.0, 6.0, 0.9)], chunk_size=CHUNK_X) == (20, 6, 0.9)


def test_aggregate_band_detections_penalises_outlier():
    detections = [(20.0, 6.0, 0.9), (19.0, 6.0, 0.8), (60.0, 6.0, 0.2)]
    band_start, band_width, confidence = aggregate_band_detections(detections, chunk_size=CHUNK_X)
    assert band_start == 20
    assert band_width == 6
    # Consistency penalty must strictly reduce confidence below the raw max (0.9).
    assert confidence < 0.9


# ---------------------------------------------------------------------------
# decide_tile_shift
# ---------------------------------------------------------------------------


def test_decide_tile_shift_gradient_pair_detector():
    tile_aip = _tile_aip_with_band()
    shift, confidence, used_per_tile = decide_tile_shift(tile_aip, default_shift=4, min_confidence=0.5, n_extra=BAND_WIDTH)
    assert shift == 5
    assert confidence == 1.0
    assert used_per_tile is True


def test_decide_tile_shift_threshold_fallback_detector():
    tile_aip = _tile_aip_with_band()
    shift, confidence, used_per_tile = decide_tile_shift(tile_aip, default_shift=4, min_confidence=0.5, n_extra=None)
    assert shift == CHUNK_X - BAND_START - BAND_WIDTH
    assert confidence == 1.0
    assert used_per_tile is True


def test_decide_tile_shift_falls_back_when_no_band_detected():
    flat_tile = np.full((CHUNK_X, 8), 50.0, dtype=np.float32)
    shift, confidence, used_per_tile = decide_tile_shift(flat_tile, default_shift=4, min_confidence=0.5, n_extra=BAND_WIDTH)
    assert shift == 4
    assert confidence == 0.0
    assert used_per_tile is False
