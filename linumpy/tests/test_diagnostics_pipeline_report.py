"""Direct-import tests for ``linumpy.diagnostics.pipeline_report``.

Exercises the extracted diagnostics library directly (no ``importlib`` script
loading), covering the same pure-helper behavior locked by
``scripts/tests/analysis/test_generate_pipeline_report.py``.
"""

from linumpy.diagnostics.pipeline_report import (
    collect_issues,
    compute_cross_slice_trends,
    compute_overall_status,
    extract_slice_id,
    format_value,
    generate_text_report,
    get_status_color,
    get_status_emoji,
    get_step_status,
    group_issues,
    parse_issue,
    separate_metrics_by_type,
    slug,
    sort_steps,
)


def test_get_status_color_known_statuses():
    assert get_status_color("ok") == "#28a745"
    assert get_status_color("error") == "#dc3545"


def test_get_status_emoji_known_statuses():
    assert get_status_emoji("ok") == "✓"
    assert get_status_emoji("error") == "✗"


def test_format_value_scientific_for_small():
    assert "e" in format_value(0.00001)


def test_slug_normalizes():
    assert slug("Slice Quality Assessment!") == "slice-quality-assessment"


def test_extract_slice_id_from_path():
    assert extract_slice_id("/data/slice_z03/tile.json") == "slice_z03"


def test_sort_steps_orders_known_steps_first():
    aggregated = {"stack_slices": [1], "slice_quality_assessment": [1], "unknown_step": [1]}
    ordered = list(sort_steps(aggregated).keys())
    assert ordered == ["slice_quality_assessment", "stack_slices", "unknown_step"]


def test_parse_issue_extracts_value_and_threshold():
    parsed = parse_issue("slice_z00: ssim: 0.5 < 0.8 (warning)")
    assert parsed["value"] == 0.5
    assert parsed["threshold"] == 0.8


def test_group_issues_groups_by_metric():
    issues = [
        "slice_z00: ssim: 0.5 < 0.8 (warning)",
        "slice_z01: ssim: 0.6 < 0.8 (warning)",
    ]
    grouped = group_issues(issues)
    assert len(grouped) == 1
    assert grouped[0]["count"] == 2


def test_separate_metrics_by_type():
    metrics_list = [
        {
            "metrics": {
                "ssim": {"value": 0.9, "status": "ok", "unit": ""},
                "resolution": {"value": 3.5, "status": "info", "unit": "um"},
            }
        }
    ]
    quality, info = separate_metrics_by_type(metrics_list)
    assert "ssim" in quality
    assert info["resolution"]["is_constant"] is True


def test_compute_overall_status_counts():
    aggregated = {
        "step_a": [{"overall_status": "ok"}, {"overall_status": "warning"}],
        "step_b": [{"overall_status": "error"}],
    }
    _, errors, warnings, ok = compute_overall_status(aggregated)
    assert errors == 1
    assert warnings == 1
    assert ok == 1


def test_get_step_status_prioritizes_error():
    assert get_step_status([{"overall_status": "ok"}, {"overall_status": "error"}]) == "error"


def test_collect_issues_prefixes_source():
    metrics_list = [{"source_file": "/x/slice_z00.json", "warnings": ["low ssim"], "errors": []}]
    warnings, errors = collect_issues(metrics_list)
    assert warnings == ["slice_z00: low ssim"]
    assert errors == []


def test_compute_cross_slice_trends_pairwise_registration():
    aggregated = {
        "pairwise_registration": [
            {"source_file": "a", "metrics": {"translation_x": {"value": 1.0}, "translation_y": {"value": 0.5}}},
            {"source_file": "b", "metrics": {"translation_x": {"value": 2.0}, "translation_y": {"value": 1.0}}},
        ]
    }
    trends = compute_cross_slice_trends(aggregated)
    assert "registration_drift" in trends


def test_generate_text_report_smoke():
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
    report = generate_text_report(aggregated, title="Test Report")
    assert "Test Report" in report
    assert "SLICE QUALITY ASSESSMENT" in report
