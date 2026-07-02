#!/usr/bin/env python3
"""Characterization tests for ``scripts/diagnostics/linum_diagnose_pipeline.py``.

The script is loaded via :mod:`importlib` so we can test its pure-Python
helper functions and gate-check/aggregation logic without depending on real
hardware (GPU, Nextflow, BaSiCPy) or subprocess calls (see
``scripts/tests/stitching/test_align_to_ras.py`` for the established pattern).

These tests lock CURRENT behavior (terminal-width fallback, Nextflow
parameter-suggestion formula, BaSiC-error classification messages, and
summary aggregation) before the core logic is extracted into
``linumpy.diagnostics.pipeline`` (D-85).
"""

import importlib.util
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "diagnostics" / "linum_diagnose_pipeline.py"


@pytest.fixture(scope="module")
def mod():
    """Load ``linum-diagnose-pipeline`` as a module."""
    spec = importlib.util.spec_from_file_location("linum_diagnose_pipeline", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_diag(mod, total_cores, total_gb, gpu_available=False):
    diag = mod.SystemDiagnostics()
    diag.results["cpu"]["total_cores"] = total_cores
    diag.results["memory"]["total_gb"] = total_gb
    diag.results["gpu"]["available"] = gpu_available
    return diag


# ---------------------------------------------------------------------------
# CLI help
# ---------------------------------------------------------------------------


def test_help(script_runner):
    ret = script_runner.run(["linum-diagnose-pipeline", "--help"])
    assert ret.success


# ---------------------------------------------------------------------------
# get_terminal_width
# ---------------------------------------------------------------------------


def test_get_terminal_width_returns_positive_int(mod):
    width = mod.get_terminal_width()
    assert isinstance(width, int)
    assert width > 0


# ---------------------------------------------------------------------------
# SystemDiagnostics.check_nextflow_config -- Nextflow suggestion formula
# ---------------------------------------------------------------------------


def test_nextflow_suggestions_reserved_cpus_floor(mod):
    """reserved_cpus is max(2, total_cpus // 12)."""
    diag = _make_diag(mod, total_cores=8, total_gb=32.0)
    diag.check_nextflow_config()
    suggestions = diag.results["nextflow"]["suggestions"]
    assert suggestions["params.reserved_cpus"] == 2  # floor at 2, not 8 // 12 == 0


def test_nextflow_suggestions_reserved_cpus_scales(mod):
    diag = _make_diag(mod, total_cores=48, total_gb=64.0)
    diag.check_nextflow_config()
    suggestions = diag.results["nextflow"]["suggestions"]
    assert suggestions["params.reserved_cpus"] == 4  # 48 // 12
    assert suggestions["params.processes"] == 14  # min(16, max(1, (48 - 4) // 3))


def test_nextflow_suggestions_capped_at_16_processes(mod):
    diag = _make_diag(mod, total_cores=256, total_gb=512.0)
    diag.check_nextflow_config()
    suggestions = diag.results["nextflow"]["suggestions"]
    assert suggestions["params.processes"] == 16


def test_nextflow_suggestions_use_gpu_reflects_availability(mod):
    diag = _make_diag(mod, total_cores=8, total_gb=32.0, gpu_available=True)
    diag.check_nextflow_config()
    assert diag.results["nextflow"]["suggestions"]["params.use_gpu"] is True


# ---------------------------------------------------------------------------
# SystemDiagnostics._handle_basic_error -- BaSiC error classification
# ---------------------------------------------------------------------------


def test_handle_basic_error_missing_basicpy(mod, capsys):
    diag = mod.SystemDiagnostics()
    diag._handle_basic_error("ModuleNotFoundError: No module named 'basicpy'")
    out = capsys.readouterr().out
    assert "basicpy not installed" in out


def test_handle_basic_error_missing_torch(mod, capsys):
    diag = mod.SystemDiagnostics()
    diag._handle_basic_error("ModuleNotFoundError: No module named 'torch'")
    out = capsys.readouterr().out
    assert "PyTorch not installed" in out


def test_handle_basic_error_cuda_oom(mod, capsys):
    diag = mod.SystemDiagnostics()
    diag._handle_basic_error("RuntimeError: CUDA error: out of memory")
    out = capsys.readouterr().out
    assert "GPU out of memory" in out


def test_handle_basic_error_generic(mod, capsys):
    diag = mod.SystemDiagnostics()
    diag._handle_basic_error("some traceback\nERROR:something weird happened\nmore text")
    out = capsys.readouterr().out
    assert "something weird happened" in out


# ---------------------------------------------------------------------------
# SystemDiagnostics.generate_report -- summary aggregation
# ---------------------------------------------------------------------------


def test_generate_report_lists_issues(mod, capsys):
    diag = mod.SystemDiagnostics()
    diag.results["cpu"]["total_cores"] = 8
    diag.results["memory"]["total_gb"] = 16.0
    diag.results["gpu"]["available"] = False
    diag.results["gpu"]["cupy_working"] = False
    diag.results["issues"].append("Low available memory: 2.0 GB")

    results = diag.generate_report()
    out = capsys.readouterr().out

    assert results is diag.results
    assert "Low available memory: 2.0 GB" in out
    assert "CPU cores: 8" in out
    assert "NVIDIA GPU: Not available" in out


def test_generate_report_no_issues_skips_section(mod, capsys):
    diag = mod.SystemDiagnostics()
    diag.results["cpu"]["total_cores"] = 4
    diag.results["memory"]["total_gb"] = 8.0
    diag.results["gpu"]["available"] = True
    diag.results["gpu"]["cupy_working"] = True

    diag.generate_report()
    out = capsys.readouterr().out

    assert "Issues Found" not in out
    assert "NVIDIA GPU: Available" in out
    assert "CuPy GPU (linumpy): Working" in out
