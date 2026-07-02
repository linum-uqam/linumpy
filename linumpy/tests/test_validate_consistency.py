"""Tests for Phase 3 consistency catalog validator (.planning/scripts/validate_consistency.py)."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
VALIDATOR = REPO_ROOT / ".planning" / "scripts" / "validate_consistency.py"


def _load_validator_module():
    spec = importlib.util.spec_from_file_location("validate_consistency", VALIDATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("mode", ["structure", "catalog", "gate", "all"])
def test_validate_consistency_mode_exits_zero(mode: str) -> None:
    """Each validator mode passes after INFRA-01 thread-config fixes."""
    assert VALIDATOR.is_file(), f"missing validator script: {VALIDATOR}"
    result = subprocess.run(
        [sys.executable, str(VALIDATOR), mode],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (result.stderr or result.stdout).strip()
    assert f"OK: {mode}" in result.stdout


def test_check_gate_passes_when_high_target_phase_6_fixed() -> None:
    """Gate check passes once all HIGH target_phase=6 thread_config rows are fixed."""
    mod = _load_validator_module()
    reasons = mod.check_gate(mod.repo_root())
    high_phase6 = [r for r in reasons if "HIGH Phase-6 open row" in r]
    assert not high_phase6, high_phase6


def test_validate_consistency_csv_header_matches_schema() -> None:
    """Canonical CSV header matches validator module constant (no schema drift)."""
    mod = _load_validator_module()
    index_path = REPO_ROOT / ".planning" / "consistency-index.csv"
    first_line = index_path.read_text(encoding="utf-8").splitlines()[0]
    assert first_line == mod.CANONICAL_HEADER
