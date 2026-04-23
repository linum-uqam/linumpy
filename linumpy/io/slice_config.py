"""
Shared helpers for reading, writing and stamping ``slice_config.csv``.

``slice_config.csv`` is the single per-slice trace file threaded through
the reconstruction pipeline. Each stage that makes a per-slice decision
(quality assessment, rehoming correction, auto-exclusion, missing-slice
interpolation, ...) stamps its flag columns via this module and hands
the enriched file to the next stage.

Only pipeline-*decision* columns live here; raw metrics belong in the
pipeline report and per-stage diagnostics JSON.

Concurrency model
-----------------

This module does **not** implement any file locking. Safe concurrent use
depends on the upstream Nextflow pipeline's channel discipline:

* Every process receives ``slice_config.csv`` as an immutable input
  staged into its own work directory. Nothing reads and writes the same
  file at the same time.
* Per-slice stages (interpolation, pairwise registration, ...) emit
  per-slice fragment files (``slice_z{NN}_manifest.csv``). Those fragments
  are collected and merged sequentially in a single downstream process
  (``finalise_interpolation``), so the CSV writer always runs on a single
  worker.
* Stamping helpers (:func:`stamp` / :func:`merge_fragments`) always produce
  a *new* CSV at ``slice_config_out`` rather than updating in place, so a
  reader on the old version is never in a torn state.

If you ever need to call these helpers outside of Nextflow (e.g. ad-hoc
scripts running in parallel), make sure each writer targets a distinct
output path; otherwise the last writer wins.
"""

from __future__ import annotations

import csv
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from pathlib import Path

CANONICAL_COLUMNS: list[str] = [
    "slice_id",
    "use",
    "exclude_reason",
    "quality_score",
    "galvo_confidence",
    "galvo_fix",
    "notes",
    "rehomed",
    "rehoming_reliable",
    "auto_excluded",
    "auto_exclude_reason",
    "interpolated",
    "interpolation_failed",
    "interpolation_method_used",
    "interpolation_fallback_reason",
]

TRUE_STRINGS = frozenset({"true", "1", "yes", "y", "t"})
FALSE_STRINGS = frozenset({"false", "0", "no", "n", "f", ""})


def normalize_slice_id(slice_id: object) -> str:
    """Return ``slice_id`` as a two-digit zero-padded string (``"01"``, ``"17"``).

    Accepts int / str / float ("1.0") inputs. Falls back to ``str(slice_id).strip()``
    for non-numeric ids.
    """
    if slice_id is None:
        return ""
    if isinstance(slice_id, (int,)):
        return f"{int(slice_id):02d}"
    text = str(slice_id).strip()
    if not text:
        return ""
    try:
        return f"{int(float(text)):02d}"
    except ValueError:
        return text


def _coerce_bool(value: object) -> bool:
    """Coerce a CSV cell to bool; empty / unknown => False."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    if text in TRUE_STRINGS:
        return True
    if text in FALSE_STRINGS:
        return False
    return False


def read(path: str | Path) -> OrderedDict[str, dict[str, str]]:
    """Read ``slice_config.csv``; return ``slice_id -> row`` with normalized ids.

    Raises :class:`FileNotFoundError` if the file does not exist.
    Row values are kept as strings (CSV native); use :func:`get_flag` for bool
    coercion.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"slice_config not found: {path}")
    rows: OrderedDict[str, dict[str, str]] = OrderedDict()
    with path.open() as f:
        reader = csv.DictReader(f)
        for raw in reader:
            sid = normalize_slice_id(raw.get("slice_id", ""))
            if not sid:
                continue
            cleaned = {k: ("" if v is None else str(v)) for k, v in raw.items()}
            cleaned["slice_id"] = sid
            rows[sid] = cleaned
    return rows


def read_header(path: str | Path) -> list[str]:
    """Return the header row of ``path`` (empty list if file has no header)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"slice_config not found: {path}")
    with path.open() as f:
        reader = csv.reader(f)
        try:
            return next(reader)
        except StopIteration:
            return []


def _as_cell(value: object) -> str:
    """Stringify a value for CSV storage (bool -> 'true'/'false')."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return str(value)


def _build_header(rows: Iterable[Mapping[str, object]], extra_columns: Iterable[str]) -> list[str]:
    """Build header: canonical columns (in order) + any other columns seen in
    rows or in ``extra_columns``, preserving insertion order.
    """
    seen: list[str] = []
    seen_set: set[str] = set()
    for col in CANONICAL_COLUMNS:
        if col not in seen_set:
            seen.append(col)
            seen_set.add(col)
    for col in extra_columns:
        if col not in seen_set:
            seen.append(col)
            seen_set.add(col)
    for row in rows:
        for col in row:
            if col not in seen_set:
                seen.append(col)
                seen_set.add(col)
    return seen


def write(
    path: str | Path,
    rows: Iterable[Mapping[str, object]],
    extra_columns: Iterable[str] = (),
) -> None:
    """Atomically write ``rows`` to ``path``.

    - The header always starts with :data:`CANONICAL_COLUMNS` (in that order);
      any extra columns come after. Missing canonical columns are emitted
      empty.
    - Rows are sorted by ``slice_id``.
    - ``slice_id`` is normalised to a 2-digit string.
    """
    rows_list = [dict(r) for r in rows]
    for r in rows_list:
        r["slice_id"] = normalize_slice_id(r.get("slice_id", ""))
    rows_list.sort(key=lambda r: r.get("slice_id", ""))

    header = _build_header(rows_list, extra_columns)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows_list:
            writer.writerow({col: _as_cell(row.get(col, "")) for col in header})
    tmp.replace(path)


def stamp(
    path_in: str | Path,
    path_out: str | Path,
    slice_id: object,
    **flags: object,
) -> None:
    """Stamp a single slice: read ``path_in``, update ``slice_id`` with
    ``flags``, write to ``path_out``.

    New slice rows are appended with ``use=false`` when the row is absent.
    """
    stamp_many(path_in, path_out, {normalize_slice_id(slice_id): dict(flags)})


def stamp_many(
    path_in: str | Path,
    path_out: str | Path,
    updates: Mapping[str, Mapping[str, object]],
) -> None:
    """Stamp multiple slices at once.

    ``updates`` maps ``slice_id -> {column: value}``. Unknown slices are
    appended with ``use=false`` unless the caller supplies a ``use`` key.
    """
    rows = read(path_in)
    for raw_sid, flags in updates.items():
        sid = normalize_slice_id(raw_sid)
        if not sid:
            continue
        existing = rows.get(sid)
        if existing is None:
            new_row: dict[str, str] = {"slice_id": sid, "use": "false"}
            for k, v in flags.items():
                new_row[k] = _as_cell(v)
            rows[sid] = new_row
        else:
            for k, v in flags.items():
                existing[k] = _as_cell(v)
    write(path_out, rows.values())


def merge_fragments(
    path_in: str | Path,
    fragment_paths: Iterable[str | Path],
    path_out: str | Path,
    column_map: Mapping[str, str] | None = None,
) -> None:
    """Merge per-slice CSV fragments into ``path_in`` and write to ``path_out``.

    Each fragment is a small CSV with at least a ``slice_id`` column. Columns
    from the fragment are stamped onto the matching slice row, renamed via
    ``column_map`` if provided (``{fragment_col: target_col}``).

    Fragments that reference slices absent from the base config add new rows
    (``use=false``).
    """
    updates: dict[str, dict[str, object]] = {}
    for frag in fragment_paths:
        frag_path = Path(frag)
        if not frag_path.exists():
            continue
        with frag_path.open() as f:
            reader = csv.DictReader(f)
            for raw in reader:
                sid = normalize_slice_id(raw.get("slice_id", ""))
                if not sid:
                    continue
                entry = updates.setdefault(sid, {})
                for col, val in raw.items():
                    if col == "slice_id" or val is None:
                        continue
                    target = column_map.get(col, col) if column_map else col
                    if target:
                        entry[target] = val
    stamp_many(path_in, path_out, updates)


def filter_slices_to_use(path: str | Path) -> set[str]:
    """Return the set of slice IDs whose ``use`` column is truthy.

    When ``slice_config.csv`` is missing this raises :class:`FileNotFoundError`
    — callers should guard on ``path.exists()`` or pass an optional path
    themselves.
    """
    rows = read(path)
    return {sid for sid, row in rows.items() if _coerce_bool(row.get("use", ""))}


def get_flag(row: Mapping[str, object], column: str, default: bool = False) -> bool:
    """Return a boolean flag from a config row (default when absent/empty)."""
    if column not in row:
        return default
    value = row.get(column, "")
    if value is None or value == "":
        return default
    return _coerce_bool(value)


def is_interpolated(path: str | Path, slice_id: object) -> bool:
    """Convenience: is ``slice_id`` flagged as interpolated in ``path``?"""
    sid = normalize_slice_id(slice_id)
    rows = read(path)
    row = rows.get(sid)
    if row is None:
        return False
    return get_flag(row, "interpolated")


def force_skip_slices(path: str | Path) -> set[str]:
    """Return slice IDs that stacking should treat as motor-only (force-skip
    their pairwise transforms).

    A slice is force-skipped when it is explicitly excluded (``use=false``)
    or was flagged by auto-exclude (``auto_excluded=true``).
    """
    rows = read(path)
    skip: set[str] = set()
    for sid, row in rows.items():
        used = _coerce_bool(row.get("use", "true")) if row.get("use", "") != "" else True
        if not used or get_flag(row, "auto_excluded"):
            skip.add(sid)
    return skip
