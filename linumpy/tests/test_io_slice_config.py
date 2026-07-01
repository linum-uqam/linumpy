"""Tests for linumpy/io/slice_config.py."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from linumpy.io import slice_config


def _write(path: Path, header: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in header})


def _read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open() as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def test_normalize_slice_id_variants():
    assert slice_config.normalize_slice_id(1) == "01"
    assert slice_config.normalize_slice_id("1") == "01"
    assert slice_config.normalize_slice_id("01") == "01"
    assert slice_config.normalize_slice_id("1.0") == "01"
    assert slice_config.normalize_slice_id(" 7 ") == "07"
    assert slice_config.normalize_slice_id("") == ""
    assert slice_config.normalize_slice_id("a_custom_id") == "a_custom_id"


def test_read_round_trip(tmp_path: Path):
    path = tmp_path / "slice_config.csv"
    _write(
        path,
        ["slice_id", "use", "notes"],
        [
            {"slice_id": "00", "use": "true", "notes": ""},
            {"slice_id": "01", "use": "false", "notes": "bad"},
        ],
    )
    rows = slice_config.read(path)
    assert list(rows.keys()) == ["00", "01"]
    assert rows["01"]["use"] == "false"
    assert rows["01"]["notes"] == "bad"


def test_read_normalises_ids(tmp_path: Path):
    path = tmp_path / "slice_config.csv"
    _write(
        path,
        ["slice_id", "use"],
        [
            {"slice_id": "1", "use": "true"},
            {"slice_id": "2.0", "use": "false"},
        ],
    )
    rows = slice_config.read(path)
    assert set(rows) == {"01", "02"}


def test_write_orders_canonical_first(tmp_path: Path):
    path = tmp_path / "slice_config.csv"
    slice_config.write(
        path,
        [
            {"slice_id": "02", "use": True, "custom": "extra", "interpolated": "true"},
            {"slice_id": "01", "use": False, "custom": "foo"},
        ],
    )
    header, rows = _read_rows(path)
    assert header[0] == "slice_id"
    assert "use" in header
    assert "interpolated" in header
    assert "custom" in header
    assert header.index("use") < header.index("custom")
    assert header.index("interpolated") < header.index("custom")
    assert [r["slice_id"] for r in rows] == ["01", "02"]
    assert rows[0]["use"] == "false"
    assert rows[1]["use"] == "true"


def test_stamp_updates_existing_row(tmp_path: Path):
    path_in = tmp_path / "in.csv"
    path_out = tmp_path / "out.csv"
    _write(
        path_in,
        ["slice_id", "use"],
        [{"slice_id": "00", "use": "true"}, {"slice_id": "01", "use": "true"}],
    )
    slice_config.stamp(path_in, path_out, "01", rehomed=True, rehoming_reliable=0)
    rows = slice_config.read(path_out)
    assert rows["01"]["rehomed"] == "true"
    assert rows["01"]["rehoming_reliable"] == "0"
    assert rows["00"].get("rehomed", "") == ""


def test_stamp_adds_unknown_slice(tmp_path: Path):
    path_in = tmp_path / "in.csv"
    path_out = tmp_path / "out.csv"
    _write(path_in, ["slice_id", "use"], [{"slice_id": "00", "use": "true"}])
    slice_config.stamp(path_in, path_out, "03", interpolated=True)
    rows = slice_config.read(path_out)
    assert "03" in rows
    assert rows["03"]["use"] == "false"
    assert rows["03"]["interpolated"] == "true"


def test_merge_fragments(tmp_path: Path):
    base = tmp_path / "base.csv"
    out = tmp_path / "out.csv"
    _write(
        base,
        ["slice_id", "use", "notes"],
        [
            {"slice_id": "00", "use": "true", "notes": ""},
            {"slice_id": "01", "use": "false", "notes": "bad"},
            {"slice_id": "02", "use": "true", "notes": ""},
        ],
    )
    frag1 = tmp_path / "frag1.csv"
    _write(
        frag1,
        ["slice_id", "method_used", "fallback_reason"],
        [{"slice_id": "01", "method_used": "zmorph", "fallback_reason": ""}],
    )
    frag2 = tmp_path / "frag2.csv"
    _write(
        frag2,
        ["slice_id", "method_used"],
        [{"slice_id": "05", "method_used": "weighted"}],
    )
    slice_config.merge_fragments(
        base,
        [frag1, frag2],
        out,
        column_map={
            "method_used": "interpolation_method_used",
            "fallback_reason": "interpolation_fallback_reason",
        },
    )
    rows = slice_config.read(out)
    assert rows["01"]["interpolation_method_used"] == "zmorph"
    assert rows["01"]["notes"] == "bad"
    assert rows["05"]["use"] == "false"
    assert rows["05"]["interpolation_method_used"] == "weighted"


def test_filter_slices_to_use(tmp_path: Path):
    path = tmp_path / "sc.csv"
    _write(
        path,
        ["slice_id", "use"],
        [
            {"slice_id": "00", "use": "true"},
            {"slice_id": "01", "use": "false"},
            {"slice_id": "02", "use": "YES"},
            {"slice_id": "03", "use": ""},
        ],
    )
    assert slice_config.filter_slices_to_use(path) == {"00", "02"}


def test_force_skip_slices(tmp_path: Path):
    path = tmp_path / "sc.csv"
    _write(
        path,
        ["slice_id", "use", "auto_excluded"],
        [
            {"slice_id": "00", "use": "true", "auto_excluded": "false"},
            {"slice_id": "01", "use": "false", "auto_excluded": "false"},
            {"slice_id": "02", "use": "true", "auto_excluded": "true"},
        ],
    )
    assert slice_config.force_skip_slices(path) == {"01", "02"}


def test_is_interpolated(tmp_path: Path):
    path = tmp_path / "sc.csv"
    _write(
        path,
        ["slice_id", "use", "interpolated"],
        [
            {"slice_id": "00", "use": "true", "interpolated": "false"},
            {"slice_id": "01", "use": "false", "interpolated": "true"},
        ],
    )
    assert slice_config.is_interpolated(path, "01") is True
    assert slice_config.is_interpolated(path, 0) is False
    assert slice_config.is_interpolated(path, 99) is False


def test_read_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        slice_config.read(tmp_path / "does_not_exist.csv")


def test_stamp_preserves_unknown_extra_columns(tmp_path: Path):
    path_in = tmp_path / "in.csv"
    path_out = tmp_path / "out.csv"
    _write(
        path_in,
        ["slice_id", "use", "legacy_metric"],
        [{"slice_id": "00", "use": "true", "legacy_metric": "42.0"}],
    )
    slice_config.stamp(path_in, path_out, "00", interpolated=True)
    rows = slice_config.read(path_out)
    assert rows["00"]["legacy_metric"] == "42.0"
    assert rows["00"]["interpolated"] == "true"
