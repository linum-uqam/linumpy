"""Direct-import tests for ``linumpy.diagnostics.pipeline``.

Exercises the extracted diagnostics library directly (no ``importlib`` script
loading), covering the same gate-check/aggregation behavior locked by
``scripts/tests/diagnostics/test_diagnose_pipeline.py``.
"""

from linumpy.diagnostics.pipeline import SystemDiagnostics, get_terminal_width


def _make_diag(total_cores, total_gb, gpu_available=False):
    diag = SystemDiagnostics()
    diag.results["cpu"]["total_cores"] = total_cores
    diag.results["memory"]["total_gb"] = total_gb
    diag.results["gpu"]["available"] = gpu_available
    return diag


def test_get_terminal_width_returns_positive_int():
    width = get_terminal_width()
    assert isinstance(width, int)
    assert width > 0


def test_nextflow_suggestions_reserved_cpus_floor():
    diag = _make_diag(total_cores=8, total_gb=32.0)
    diag.check_nextflow_config()
    suggestions = diag.results["nextflow"]["suggestions"]
    assert suggestions["params.reserved_cpus"] == 2  # floor at 2, not 8 // 12 == 0


def test_nextflow_suggestions_reserved_cpus_scales():
    diag = _make_diag(total_cores=48, total_gb=64.0)
    diag.check_nextflow_config()
    suggestions = diag.results["nextflow"]["suggestions"]
    assert suggestions["params.reserved_cpus"] == 4  # 48 // 12
    assert suggestions["params.processes"] == 14  # min(16, max(1, (48 - 4) // 3))


def test_nextflow_suggestions_capped_at_16_processes():
    diag = _make_diag(total_cores=256, total_gb=512.0)
    diag.check_nextflow_config()
    suggestions = diag.results["nextflow"]["suggestions"]
    assert suggestions["params.processes"] == 16


def test_nextflow_suggestions_use_gpu_reflects_availability():
    diag = _make_diag(total_cores=8, total_gb=32.0, gpu_available=True)
    diag.check_nextflow_config()
    assert diag.results["nextflow"]["suggestions"]["params.use_gpu"] is True


def test_handle_basic_error_missing_basicpy(capsys):
    diag = SystemDiagnostics()
    diag._handle_basic_error("ModuleNotFoundError: No module named 'basicpy'")
    out = capsys.readouterr().out
    assert "basicpy not installed" in out


def test_handle_basic_error_missing_torch(capsys):
    diag = SystemDiagnostics()
    diag._handle_basic_error("ModuleNotFoundError: No module named 'torch'")
    out = capsys.readouterr().out
    assert "PyTorch not installed" in out


def test_handle_basic_error_cuda_oom(capsys):
    diag = SystemDiagnostics()
    diag._handle_basic_error("RuntimeError: CUDA error: out of memory")
    out = capsys.readouterr().out
    assert "GPU out of memory" in out


def test_handle_basic_error_generic(capsys):
    diag = SystemDiagnostics()
    diag._handle_basic_error("some traceback\nERROR:something weird happened\nmore text")
    out = capsys.readouterr().out
    assert "something weird happened" in out


def test_generate_report_lists_issues(capsys):
    diag = SystemDiagnostics()
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


def test_generate_report_no_issues_skips_section(capsys):
    diag = SystemDiagnostics()
    diag.results["cpu"]["total_cores"] = 4
    diag.results["memory"]["total_gb"] = 8.0
    diag.results["gpu"]["available"] = True
    diag.results["gpu"]["cupy_working"] = True

    diag.generate_report()
    out = capsys.readouterr().out

    assert "Issues Found" not in out
    assert "NVIDIA GPU: Available" in out
    assert "CuPy GPU (linumpy): Working" in out
