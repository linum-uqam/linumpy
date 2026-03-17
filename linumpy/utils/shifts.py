# -*- coding: utf-8 -*-
"""
XY shift utilities for serial-section alignment.

Consolidated from linum_stack_motor_only.py, linum_stack_slices_motor.py,
and linum_align_mosaics_3d_from_shifts.py.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def load_shifts_csv(shifts_path) -> Tuple[Dict, List]:
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

    all_ids = sorted(set(df['fixed_id'].tolist() + df['moving_id'].tolist()))

    shift_lookup = {}
    for _, row in df.iterrows():
        fixed_id = int(row['fixed_id'])
        moving_id = int(row['moving_id'])
        shift_lookup[(fixed_id, moving_id)] = (row['x_shift_mm'], row['y_shift_mm'])

    cumsum = {all_ids[0]: (0.0, 0.0)}
    for i in range(len(all_ids) - 1):
        fixed_id = all_ids[i]
        moving_id = all_ids[i + 1]

        if (fixed_id, moving_id) in shift_lookup:
            dx_mm, dy_mm = shift_lookup[(fixed_id, moving_id)]
        else:
            dx_mm, dy_mm = 0.0, 0.0

        prev_dx, prev_dy = cumsum[fixed_id]
        cumsum[moving_id] = (prev_dx + dx_mm, prev_dy + dy_mm)

    return cumsum, all_ids


def detect_shift_units(resolution) -> Tuple[float, float]:
    """Detect whether resolution is in mm or µm and return (res_x_um, res_y_um).

    OME-Zarr resolution can be reported in either mm (OME-NGFF standard)
    or µm depending on the writer. Detects by magnitude:
    - Values < 1.0 assumed to be mm (e.g. 0.01 mm = 10 µm)
    - Values >= 1.0 assumed to be µm (e.g. 10 µm)

    Parameters
    ----------
    resolution : sequence
        Resolution tuple/list (res_z, res_y, res_x) from read_omezarr.

    Returns
    -------
    res_x_um, res_y_um : float
        X and Y resolution in microns.
    """
    res_x_raw = resolution[-1]
    res_y_raw = resolution[-2] if len(resolution) >= 2 else res_x_raw

    if res_x_raw < 1.0:
        res_x_um = res_x_raw * 1000.0
        res_y_um = res_y_raw * 1000.0
    else:
        res_x_um = float(res_x_raw)
        res_y_um = float(res_y_raw)

    return res_x_um, res_y_um


def convert_shifts_to_pixels(cumsum_mm: Dict, resolution_um: float) -> Dict:
    """Convert mm cumulative shifts to pixel shifts.

    Parameters
    ----------
    cumsum_mm : dict
        Mapping from slice_id to (dx_mm, dy_mm).
    resolution_um : float
        Resolution in microns per pixel (isotropic XY assumed).

    Returns
    -------
    dict
        Mapping from slice_id to (dx_px, dy_px).
    """
    mm_to_px = 1000.0 / resolution_um
    return {
        slice_id: (dx_mm * mm_to_px, dy_mm * mm_to_px)
        for slice_id, (dx_mm, dy_mm) in cumsum_mm.items()
    }


def center_shifts(cumsum_px: Dict, slice_ids: List) -> Dict:
    """Center shifts around the middle slice.

    Subtracts the middle slice's cumulative shift from all slices,
    preventing drift from pushing slices out of the output canvas.

    Parameters
    ----------
    cumsum_px : dict
        Mapping from slice_id to (dx_px, dy_px).
    slice_ids : list
        Sorted list of slice IDs.

    Returns
    -------
    dict
        Centered cumulative shifts.
    """
    if not slice_ids:
        return cumsum_px

    middle_idx = len(slice_ids) // 2
    middle_id = slice_ids[middle_idx]
    center_dx, center_dy = cumsum_px.get(middle_id, (0, 0))

    return {
        slice_id: (dx - center_dx, dy - center_dy)
        for slice_id, (dx, dy) in cumsum_px.items()
    }


def filter_outlier_shifts(shifts_df: pd.DataFrame,
                          max_shift_mm: float = 0.5,
                          method: str = 'iqr',
                          iqr_multiplier: float = 1.5) -> pd.DataFrame:
    """Detect and filter outlier shifts that cause excessive drift.

    Parameters
    ----------
    shifts_df : pd.DataFrame
        DataFrame with columns: fixed_id, moving_id, x_shift_mm, y_shift_mm
    max_shift_mm : float
        Maximum allowed pairwise shift in mm (floor for IQR method)
    method : str
        'clamp', 'median', 'zero', 'local', or 'iqr'
    iqr_multiplier : float
        Multiplier for IQR-based detection (default 1.5)

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with outlier shifts corrected.
    """
    df = shifts_df.copy()
    shift_mag = np.sqrt(df['x_shift_mm']**2 + df['y_shift_mm']**2)

    if method == 'iqr':
        q1 = shift_mag.quantile(0.25)
        q3 = shift_mag.quantile(0.75)
        iqr = q3 - q1
        iqr_bound = q3 + iqr_multiplier * iqr
        upper_bound = max(iqr_bound, max_shift_mm)
        outlier_mask = shift_mag > upper_bound
    else:
        outlier_mask = shift_mag > max_shift_mm

    n_outliers = outlier_mask.sum()
    if n_outliers == 0:
        return df

    if method == 'clamp':
        for idx in df[outlier_mask].index:
            scale = max_shift_mm / shift_mag[idx]
            df.loc[idx, 'x_shift_mm'] *= scale
            df.loc[idx, 'y_shift_mm'] *= scale
            if 'x_shift' in df.columns:
                df.loc[idx, 'x_shift'] *= scale
                df.loc[idx, 'y_shift'] *= scale

    elif method == 'median':
        non_outlier = df[~outlier_mask]
        median_x = non_outlier['x_shift_mm'].median()
        median_y = non_outlier['y_shift_mm'].median()
        for idx in df[outlier_mask].index:
            df.loc[idx, 'x_shift_mm'] = median_x
            df.loc[idx, 'y_shift_mm'] = median_y

    elif method == 'zero':
        for idx in df[outlier_mask].index:
            df.loc[idx, 'x_shift_mm'] = 0.0
            df.loc[idx, 'y_shift_mm'] = 0.0
            if 'x_shift' in df.columns:
                df.loc[idx, 'x_shift'] = 0.0
                df.loc[idx, 'y_shift'] = 0.0

    elif method in ['local', 'iqr']:
        for idx in df[outlier_mask].index:
            pos = df.index.get_loc(idx)
            neighbor_vals_x, neighbor_vals_y = [], []
            for offset in [-2, -1, 1, 2]:
                neighbor_pos = pos + offset
                if 0 <= neighbor_pos < len(df):
                    neighbor_idx = df.index[neighbor_pos]
                    if not outlier_mask[neighbor_idx]:
                        neighbor_vals_x.append(df.loc[neighbor_idx, 'x_shift_mm'])
                        neighbor_vals_y.append(df.loc[neighbor_idx, 'y_shift_mm'])

            if neighbor_vals_x:
                df.loc[idx, 'x_shift_mm'] = np.median(neighbor_vals_x)
                df.loc[idx, 'y_shift_mm'] = np.median(neighbor_vals_y)
            else:
                non_outlier = df[~outlier_mask]
                df.loc[idx, 'x_shift_mm'] = non_outlier['x_shift_mm'].median()
                df.loc[idx, 'y_shift_mm'] = non_outlier['y_shift_mm'].median()

    return df


def filter_step_outliers(shifts_df: pd.DataFrame,
                         max_step_mm: float = 0.0,
                         window: int = 2,
                         method: str = 'local_median',
                         mad_threshold: float = 3.0) -> pd.DataFrame:
    """Fix per-step spikes in shifts, independent of global outlier detection.

    Parameters
    ----------
    shifts_df : pd.DataFrame
        DataFrame with shift columns.
    max_step_mm : float
        Maximum allowed per-step shift in mm. 0 disables (for clamp/local_median).
    window : int
        Neighbor window size.
    method : str
        'clamp', 'local_median', or 'local_mad'.
    mad_threshold : float
        MADs above local median to flag as outlier (for local_mad method).

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.
    """
    df = shifts_df.copy()
    shift_mag = np.sqrt(df['x_shift_mm']**2 + df['y_shift_mm']**2)

    if method == 'local_mad':
        outlier_mask = pd.Series(False, index=df.index)
        for i in range(len(df)):
            lo = max(0, i - window)
            hi = min(len(df), i + window + 1)
            neighbour_mags = np.concatenate([shift_mag.iloc[lo:i].values,
                                             shift_mag.iloc[i + 1:hi].values])
            if len(neighbour_mags) == 0:
                continue
            local_med = float(np.median(neighbour_mags))
            local_mad = float(np.median(np.abs(neighbour_mags - local_med)))
            effective_mad = local_mad if local_mad > 0 else 1e-6
            if shift_mag.iloc[i] > local_med + mad_threshold * effective_mad:
                outlier_mask.iloc[i] = True
        n_outliers = int(outlier_mask.sum())
        if n_outliers == 0:
            return df
    else:
        if max_step_mm is None or max_step_mm <= 0:
            return shifts_df
        outlier_mask = shift_mag > max_step_mm
        n_outliers = int(outlier_mask.sum())
        if n_outliers == 0:
            return df

    for idx in df[outlier_mask].index:
        row = df.loc[idx]
        if method == 'clamp':
            scale = max_step_mm / shift_mag[idx]
            df.loc[idx, 'x_shift_mm'] *= scale
            df.loc[idx, 'y_shift_mm'] *= scale
            if 'x_shift' in df.columns:
                df.loc[idx, 'x_shift'] *= scale
                df.loc[idx, 'y_shift'] *= scale
        else:
            pos = df.index.get_loc(idx)
            neighbor_vals_x, neighbor_vals_y = [], []
            for offset in range(-window, window + 1):
                if offset == 0:
                    continue
                neighbor_pos = pos + offset
                if 0 <= neighbor_pos < len(df):
                    neighbor_idx = df.index[neighbor_pos]
                    neighbor_vals_x.append(df.loc[neighbor_idx, 'x_shift_mm'])
                    neighbor_vals_y.append(df.loc[neighbor_idx, 'y_shift_mm'])
            if neighbor_vals_x:
                df.loc[idx, 'x_shift_mm'] = float(np.median(neighbor_vals_x))
                df.loc[idx, 'y_shift_mm'] = float(np.median(neighbor_vals_y))
                if 'x_shift' in df.columns:
                    neighbor_px_x = [df.loc[df.index[df.index.get_loc(idx) + o], 'x_shift']
                                     for o in range(-window, window + 1)
                                     if o != 0 and 0 <= df.index.get_loc(idx) + o < len(df)
                                     and 'x_shift' in df.columns]
                    neighbor_px_y = [df.loc[df.index[df.index.get_loc(idx) + o], 'y_shift']
                                     for o in range(-window, window + 1)
                                     if o != 0 and 0 <= df.index.get_loc(idx) + o < len(df)
                                     and 'x_shift' in df.columns]
                    if neighbor_px_x:
                        df.loc[idx, 'x_shift'] = float(np.median(neighbor_px_x))
                        df.loc[idx, 'y_shift'] = float(np.median(neighbor_px_y))

    return df


def build_cumulative_shifts(shifts_df: pd.DataFrame,
                            selected_slice_ids: List,
                            resolution,
                            center_drift: bool = True) -> Dict:
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
        fixed_id = int(row['fixed_id'])
        moving_id = int(row['moving_id'])
        shift_lookup[(fixed_id, moving_id)] = (row['x_shift_mm'], row['y_shift_mm'])

    all_slice_ids = set()
    for _, row in shifts_df.iterrows():
        all_slice_ids.add(int(row['fixed_id']))
        all_slice_ids.add(int(row['moving_id']))
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
