"""CSV loading and cumulative shift computation."""

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from linumpy.stack_alignment.units import detect_shift_units


def load_shifts_csv(shifts_path: Path) -> tuple[dict, list]:
    """Load shifts CSV and build cumulative shift lookup.

    The shifts file contains pairwise shifts: fixed_id -> moving_id in mm.
    Accumulates these to get absolute positions from the first slice.

    Parameters
    ----------
    shifts_path : str or Path
        Path to CSV file with columns: fixed_id, moving_id, x_shift_mm, y_shift_mm

    Returns
    -------
    cumsum : dict
        Mapping from slice_id to (cumulative_dx_mm, cumulative_dy_mm)
    all_ids : list
        Sorted list of all slice IDs
    """
    df = pd.read_csv(shifts_path)

    all_ids = sorted(set(df["fixed_id"].tolist() + df["moving_id"].tolist()))

    shift_lookup = {}
    for _, row in df.iterrows():
        fixed_id = int(row["fixed_id"])
        moving_id = int(row["moving_id"])
        shift_lookup[(fixed_id, moving_id)] = (row["x_shift_mm"], row["y_shift_mm"])

    cumsum = {all_ids[0]: (0.0, 0.0)}
    for i in range(len(all_ids) - 1):
        fixed_id = all_ids[i]
        moving_id = all_ids[i + 1]

        dx_mm, dy_mm = shift_lookup.get((fixed_id, moving_id), (0.0, 0.0))

        prev_dx, prev_dy = cumsum[fixed_id]
        cumsum[moving_id] = (prev_dx + dx_mm, prev_dy + dy_mm)

    return cumsum, all_ids


def build_cumulative_shifts(
    shifts_df: pd.DataFrame,
    selected_slice_ids: list,
    resolution: Sequence[float],
    center_drift: bool = True,
) -> dict:
    """Build cumulative pixel shifts for selected slices.

    Handles skipped slices by accumulating intermediate steps.
    Converts mm shifts to pixels using the provided resolution.

    Parameters
    ----------
    shifts_df : pd.DataFrame
        DataFrame with columns: fixed_id, moving_id, x_shift_mm, y_shift_mm
    selected_slice_ids : list
        Sorted list of slice IDs to process.
    resolution : tuple
        Resolution (res_z, res_y, res_x) from read_omezarr; auto-detects mm vs µm.
    center_drift : bool
        If True, center cumulative drift around the middle slice.

    Returns
    -------
    dict
        Mapping from slice_id to (cumulative_dx_px, cumulative_dy_px).
    """
    shift_lookup = {}
    for _, row in shifts_df.iterrows():
        fixed_id = int(row["fixed_id"])
        moving_id = int(row["moving_id"])
        shift_lookup[(fixed_id, moving_id)] = (row["x_shift_mm"], row["y_shift_mm"])

    all_slice_ids = set()
    for _, row in shifts_df.iterrows():
        all_slice_ids.add(int(row["fixed_id"]))
        all_slice_ids.add(int(row["moving_id"]))
    all_slice_ids = sorted(all_slice_ids)

    cumsum_all = {all_slice_ids[0]: (0.0, 0.0)}
    for i in range(len(all_slice_ids) - 1):
        fixed_id = all_slice_ids[i]
        moving_id = all_slice_ids[i + 1]
        dx_mm, dy_mm = shift_lookup.get((fixed_id, moving_id), (0.0, 0.0))
        prev_dx, prev_dy = cumsum_all[fixed_id]
        cumsum_all[moving_id] = (prev_dx + dx_mm, prev_dy + dy_mm)

    res_x_um, res_y_um = detect_shift_units(resolution)
    mm_to_px_x = 1000.0 / res_x_um
    mm_to_px_y = 1000.0 / res_y_um

    cumsum_selected = {}
    for slice_id in selected_slice_ids:
        if slice_id in cumsum_all:
            dx_mm, dy_mm = cumsum_all[slice_id]
            cumsum_selected[slice_id] = (dx_mm * mm_to_px_x, dy_mm * mm_to_px_y)
        else:
            cumsum_selected[slice_id] = (0.0, 0.0)

    if center_drift and len(cumsum_selected) > 0:
        middle_idx = len(selected_slice_ids) // 2
        middle_id = selected_slice_ids[middle_idx]
        center_dx, center_dy = cumsum_selected[middle_id]
        for slice_id in cumsum_selected:
            dx, dy = cumsum_selected[slice_id]
            cumsum_selected[slice_id] = (dx - center_dx, dy - center_dy)

    return cumsum_selected
