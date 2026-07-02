#!/usr/bin/env python3
import pytest

from linumpy.io.test_data import get_data


def test_help(script_runner):
    ret = script_runner.run(["linum-detect-focal-curvature", "--help"])
    assert ret.success


# BaSiCPy initializes a native thread pool at import time that conflicts
# with threads left by other in-process tests (e.g. dask from mosaic grid).
# Run in a subprocess to ensure a clean process-level state.
@pytest.mark.script_launch_mode("subprocess")
def test_execute(script_runner, tmp_path):
    input = get_data("mosaic_3d_omezarr")
    output = tmp_path / "fix_focal.ome.zarr"
    ret = script_runner.run(["linum-detect-focal-curvature", input, output])
    assert ret.success


@pytest.mark.script_launch_mode("subprocess")
def test_preserves_voxel_size_n_levels_0(script_runner, tmp_path):
    """Round-trip physical resolution through the focal CLI at --n_levels 0."""
    from linumpy.io.zarr import read_omezarr

    # mosaic_3d_omezarr is cached under LINUMPY_HOME; a pre-fix save_omezarr store
    # would read back [1.0, 1.0, 1.0]. Delete and rebuild if stale before asserting.
    input_path = get_data("mosaic_3d_omezarr")
    _, input_res = read_omezarr(input_path, level=0)
    expected_res = [0.001, 0.001, 0.001]
    if input_res != expected_res:
        import shutil

        shutil.rmtree(input_path)
        input_path = get_data("mosaic_3d_omezarr")
        _, input_res = read_omezarr(input_path, level=0)
    assert input_res == expected_res

    output = tmp_path / "fix_focal_n0.ome.zarr"
    ret = script_runner.run(
        [
            "linum-detect-focal-curvature",
            input_path,
            output,
            "--n_levels",
            "0",
            "--no-use_gpu",
        ]
    )
    assert ret.success, ret.stderr

    _, output_res = read_omezarr(output, level=0)
    assert output_res == expected_res
