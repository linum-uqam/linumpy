# -*- coding: utf-8 -*-
"""Tests for linumpy/utils/visualization.py"""
import re

import numpy as np
import pytest

from linumpy.utils.visualization import (
    add_z_slice_labels,
    estimate_n_slices_from_zarr,
    save_annotated_views,
    save_orthogonal_views,
)


def _make_volume(shape=(16, 32, 32)):
    rng = np.random.default_rng(42)
    return rng.random(shape).astype(np.float32)


# ---------------------------------------------------------------------------
# save_orthogonal_views
# ---------------------------------------------------------------------------

def test_save_orthogonal_views_creates_file(tmp_path):
    vol = _make_volume((16, 24, 24))
    out = tmp_path / "views.png"
    save_orthogonal_views(vol, str(out))
    assert out.exists()
    assert out.stat().st_size > 0


def test_save_orthogonal_views_custom_slices(tmp_path):
    vol = _make_volume((20, 30, 30))
    out = tmp_path / "views_custom.png"
    save_orthogonal_views(vol, str(out), z_slice=5, x_slice=10, y_slice=15)
    assert out.exists()


# ---------------------------------------------------------------------------
# estimate_n_slices_from_zarr
# ---------------------------------------------------------------------------

def test_estimate_n_slices_from_zarr_no_file(tmp_path):
    result = estimate_n_slices_from_zarr(str(tmp_path / "nonexistent.ome.zarr"))
    assert result is None


def test_estimate_n_slices_from_zarr_sibling_files(tmp_path):
    """Estimate from sibling slice_z*.ome.zarr files."""
    for i in [0, 1, 2, 3, 4]:
        (tmp_path / f"slice_z{i:02d}.ome.zarr").mkdir()
    result = estimate_n_slices_from_zarr(str(tmp_path / "slice_z00.ome.zarr"))
    assert result == 5


def test_estimate_n_slices_from_zarr_non_contiguous(tmp_path):
    """Non-contiguous slice numbering: max - min + 1."""
    for i in [0, 3, 7]:
        (tmp_path / f"slice_z{i:02d}.ome.zarr").mkdir()
    result = estimate_n_slices_from_zarr(str(tmp_path / "slice_z00.ome.zarr"))
    assert result == 8   # 7 - 0 + 1


# ---------------------------------------------------------------------------
# add_z_slice_labels
# ---------------------------------------------------------------------------

def test_add_z_slice_labels_runs_without_error():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.imshow(np.zeros((100, 50)), cmap='gray')
    add_z_slice_labels(ax, n_input_slices=5, img_height=100, font_size=6)
    plt.close(fig)


def test_add_z_slice_labels_with_slice_ids():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.imshow(np.zeros((100, 50)), cmap='gray')
    add_z_slice_labels(ax, n_input_slices=3, img_height=100,
                       slice_ids=['01', '05', '09'])
    plt.close(fig)


def test_add_z_slice_labels_label_every():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.imshow(np.zeros((100, 50)), cmap='gray')
    # label_every=2: only even indices should be labelled
    add_z_slice_labels(ax, n_input_slices=6, img_height=100, label_every=2)
    plt.close(fig)


# ---------------------------------------------------------------------------
# save_annotated_views
# ---------------------------------------------------------------------------

def test_save_annotated_views_creates_file(tmp_path):
    vol = _make_volume((16, 24, 24))
    out = tmp_path / "annotated.png"
    save_annotated_views(vol, str(out), n_input_slices=4)
    assert out.exists()
    assert out.stat().st_size > 0


def test_save_annotated_views_with_slice_ids(tmp_path):
    vol = _make_volume((16, 24, 24))
    out = tmp_path / "annotated_ids.png"
    save_annotated_views(vol, str(out), n_input_slices=4,
                         slice_ids=['00', '01', '02', '03'])
    assert out.exists()


def test_save_annotated_views_auto_detect_slices(tmp_path):
    vol = _make_volume((16, 24, 24))
    out = tmp_path / "annotated_auto.png"
    # Create sibling files so estimate_n_slices_from_zarr can find them
    zarr_path = tmp_path / "slice_z00.ome.zarr"
    zarr_path.mkdir()
    for i in [1, 2, 3]:
        (tmp_path / f"slice_z{i:02d}.ome.zarr").mkdir()
    save_annotated_views(vol, str(out), zarr_path=str(zarr_path))
    assert out.exists()
