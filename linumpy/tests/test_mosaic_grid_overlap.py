"""Tests for MosaicGrid overlap similarity and crop behavior."""

import numpy as np
import pytest

from linumpy.mosaic.grid import MosaicGrid


def _make_grid(tiles: list[list[np.ndarray]], overlap_fraction: float) -> MosaicGrid:
    tile_size_x, tile_size_y = tiles[0][0].shape
    image = np.zeros((len(tiles) * tile_size_x, len(tiles[0]) * tile_size_y), dtype=np.float32)

    for x, column in enumerate(tiles):
        for y, tile in enumerate(column):
            x0 = x * tile_size_x
            y0 = y * tile_size_y
            image[x0 : x0 + tile_size_x, y0 : y0 + tile_size_y] = tile

    return MosaicGrid(image=image, tile_shape=(tile_size_x, tile_size_y), overlap_fraction=overlap_fraction)


def test_ncc_illumination_invariant() -> None:
    scene = np.arange(14 * 8, dtype=np.float32).reshape(14, 8)
    tile_1 = scene[:8, :]
    tile_2 = scene[6:14, :] * 2.5 + 17.0
    grid = _make_grid([[tile_1], [tile_2]], overlap_fraction=0.25)

    cost = grid.global_overlap_similarity(random_fraction=1.0)

    assert cost < 1e-3


def test_overlap_outlier_pair_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    grid = _make_grid([[np.ones((8, 8), dtype=np.float32)]], overlap_fraction=0.25)

    base = np.arange(36, dtype=np.float32).reshape(6, 6)
    good_pairs = [
        (base, base * 1.5 + 3.0),
        (base + 4.0, (base + 4.0) * 0.8 + 9.0),
        (base * 2.0 + 1.0, (base * 2.0 + 1.0) * 1.2 + 5.0),
    ]
    bad_pair = (base, np.flipud(base) * -1.0)
    overlaps = [*good_pairs, bad_pair]

    def fake_get_neighbors_list(neighborhood_type: str = "N4") -> list[tuple[tuple[int, int], tuple[int, int]]]:
        del neighborhood_type
        neighbors = [((0, 0), (0, 0))] * len(overlaps)
        grid.neighbors_list = neighbors
        return neighbors

    def fake_get_neighbor_overlap(index: int) -> tuple[np.ndarray, np.ndarray, None, None]:
        overlap_1, overlap_2 = overlaps[index]
        return overlap_1, overlap_2, None, None

    monkeypatch.setattr(grid, "get_neighbors_list", fake_get_neighbors_list)
    monkeypatch.setattr(grid, "get_neighbor_overlap", fake_get_neighbor_overlap)

    cost = grid.global_overlap_similarity(random_fraction=1.0)

    assert cost < 0.05


def test_post_crop_overlap_adjusted() -> None:
    scene = np.arange(16 * 8, dtype=np.float32).reshape(16, 8)
    tile_1 = scene[:8, :]
    tile_2 = scene[8:16, :]
    grid = _make_grid([[tile_1], [tile_2]], overlap_fraction=0.25)

    grid.crop_tiles(xlim=(1, -2), ylim=(1, -2))

    assert grid.tile_shape == (6, 6)
    assert grid.overlap_fraction == pytest.approx(8.0 * 0.25 / 6.0)
