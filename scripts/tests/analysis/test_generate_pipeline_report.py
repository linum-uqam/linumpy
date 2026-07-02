#!/usr/bin/env python3
"""Characterization tests for ``scripts/analysis/linum_generate_pipeline_report.py``.

The script is loaded via :mod:`importlib` so we can test its pure aggregation
and report-formatting helper functions without relying on the console entry
point (see ``scripts/tests/stitching/test_align_to_ras.py`` for the
established pattern).

These tests lock the CURRENT behavior of the pure helpers before extraction
into ``linumpy/diagnostics/pipeline_report.py`` (D-85).
"""

import importlib.util
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "analysis" / "linum_generate_pipeline_report.py"


@pytest.fixture(scope="module")
def mod():
    """Load ``linum-generate-pipeline-report`` as a module."""
    spec = importlib.util.spec_from_file_location("linum_generate_pipeline_report", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# CLI help
# ---------------------------------------------------------------------------


def test_help(script_runner):
    ret = script_runner.run(["linum-generate-pipeline-report", "--help"])
    assert ret.success


# ---------------------------------------------------------------------------
# get_status_color / get_status_emoji
# ---------------------------------------------------------------------------


def test_get_status_color_known_statuses(mod):
    assert mod.get_status_color("ok") == "#28a745"
    assert mod.get_status_color("warning") == "#ffc107"
    assert mod.get_status_color("error") == "#dc3545"


def test_get_status_color_unknown_falls_back(mod):
    assert mod.get_status_color("bogus") == mod.get_status_color("unknown")


def test_get_status_emoji_known_statuses(mod):
    assert mod.get_status_emoji("ok") == "✓"
    assert mod.get_status_emoji("error") == "✗"


# ---------------------------------------------------------------------------
# format_value
# ---------------------------------------------------------------------------


def test_format_value_normal_float(mod):
    assert mod.format_value(1.23456) == "1.2346"


def test_format_value_scientific_for_small(mod):
    assert "e" in mod.format_value(0.00001)


def test_format_value_scientific_for_large(mod):
    assert "e" in mod.format_value(123456.0)


def test_format_value_long_list_summarized(mod):
    assert mod.format_value([1, 2, 3, 4, 5, 6]) == "[6 items]"


def test_format_value_non_float_str(mod):
    assert mod.format_value("abc") == "abc"


# ---------------------------------------------------------------------------
# slug / extract_slice_id
# ---------------------------------------------------------------------------


def test_slug_normalizes(mod):
    assert mod.slug("Slice Quality Assessment!") == "slice-quality-assessment"


def test_extract_slice_id_from_path(mod):
    assert mod.extract_slice_id("/data/slice_z03/tile.json") == "slice_z03"


def test_extract_slice_id_fallback_to_stem(mod):
    assert mod.extract_slice_id("/data/foo/bar.json") == "bar"


# ---------------------------------------------------------------------------
# sort_steps
# ---------------------------------------------------------------------------


def test_sort_steps_orders_known_steps_first(mod):
    aggregated = {"stack_slices": [1], "slice_quality_assessment": [1], "unknown_step": [1]}
    ordered = list(mod.sort_steps(aggregated).keys())
    assert ordered == ["slice_quality_assessment", "stack_slices", "unknown_step"]


# ---------------------------------------------------------------------------
# parse_issue / group_issues
# ---------------------------------------------------------------------------


def test_parse_issue_extracts_value_and_threshold(mod):
    parsed = mod.parse_issue("slice_z00: ssim: 0.5 < 0.8 (warning)")
    assert parsed["source"] == "slice_z00"
    assert parsed["metric"] == "ssim"
    assert parsed["value"] == 0.5
    assert parsed["op"] == "<"
    assert parsed["threshold"] == 0.8


def test_parse_issue_handles_malformed_string(mod):
    parsed = mod.parse_issue("not a well formed issue")
    assert parsed["value"] is None


def test_group_issues_groups_by_metric(mod):
    issues = [
        "slice_z00: ssim: 0.5 < 0.8 (warning)",
        "slice_z01: ssim: 0.6 < 0.8 (warning)",
    ]
    grouped = mod.group_issues(issues)
    assert len(grouped) == 1
    assert grouped[0]["metric"] == "ssim"
    assert grouped[0]["count"] == 2


# ---------------------------------------------------------------------------
# separate_metrics_by_type
# ---------------------------------------------------------------------------


def test_separate_metrics_by_type(mod):
    metrics_list = [
        {
            "metrics": {
                "ssim": {"value": 0.9, "status": "ok", "unit": ""},
                "resolution": {"value": 3.5, "status": "info", "unit": "um"},
            }
        }
    ]
    quality, info = mod.separate_metrics_by_type(metrics_list)
    assert "ssim" in quality
    assert "resolution" in info
    assert info["resolution"]["is_constant"] is True


# ---------------------------------------------------------------------------
# compute_overall_status / get_step_status / collect_issues
# ---------------------------------------------------------------------------


def test_compute_overall_status_counts(mod):
    aggregated = {
        "step_a": [{"overall_status": "ok"}, {"overall_status": "warning"}],
        "step_b": [{"overall_status": "error"}],
    }
    _, errors, warnings, ok = mod.compute_overall_status(aggregated)
    assert errors == 1
    assert warnings == 1
    assert ok == 1


def test_get_step_status_prioritizes_error(mod):
    assert mod.get_step_status([{"overall_status": "ok"}, {"overall_status": "error"}]) == "error"


def test_collect_issues_prefixes_source(mod):
    metrics_list = [{"source_file": "/x/slice_z00.json", "warnings": ["low ssim"], "errors": []}]
    warnings, errors = mod.collect_issues(metrics_list)
    assert warnings == ["slice_z00: low ssim"]
    assert errors == []


# ---------------------------------------------------------------------------
# compute_cross_slice_trends
# ---------------------------------------------------------------------------


def test_compute_cross_slice_trends_pairwise_registration(mod):
    aggregated = {
        "pairwise_registration": [
            {"source_file": "a", "metrics": {"translation_x": {"value": 1.0}, "translation_y": {"value": 0.5}}},
            {"source_file": "b", "metrics": {"translation_x": {"value": 2.0}, "translation_y": {"value": 1.0}}},
        ]
    }
    trends = mod.compute_cross_slice_trends(aggregated)
    assert "registration_drift" in trends
    tx_series = next(s for s in trends["registration_drift"]["series"] if s["name"] == "Cumulative tx (px)")
    assert tx_series["values"] == [1.0, 3.0]


def test_compute_cross_slice_trends_empty_when_no_matching_steps(mod):
    assert mod.compute_cross_slice_trends({}) == {}


# ---------------------------------------------------------------------------
# generate_text_report smoke (small synthetic aggregated input)
# ---------------------------------------------------------------------------


def test_generate_text_report_smoke(mod):
    aggregated = {
        "slice_quality_assessment": [
            {
                "source_file": "slice_z00.json",
                "overall_status": "ok",
                "metrics": {"ssim": {"value": 0.9, "status": "ok", "unit": ""}},
                "warnings": [],
                "errors": [],
            }
        ]
    }
    report = mod.generate_text_report(aggregated, title="Test Report")
    assert "Test Report" in report
    assert "SLICE QUALITY ASSESSMENT" in report
