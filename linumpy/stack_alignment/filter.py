"""Outlier filtering and tile-offset correction for inter-slice shift fields."""

from typing import cast

import numpy as np
import pandas as pd


def filter_outlier_shifts(
    shifts_df: pd.DataFrame,
    max_shift_mm: float = 0.5,
    method: str = "rehome",
    iqr_multiplier: float = 1.5,
    return_fraction: float = 0.4,
) -> pd.DataFrame:
    """Detect and filter outlier shifts that cause excessive drift.

    Parameters
    ----------
    shifts_df : pd.DataFrame
        DataFrame with columns: fixed_id, moving_id, x_shift_mm, y_shift_mm
    max_shift_mm : float
        Maximum allowed pairwise shift in mm (floor for IQR method)
    method : str
        'clamp', 'median', 'zero', 'local', 'iqr', or 'rehome'.

        'rehome' (recommended default): distinguishes genuine re-homing events
        from encoder glitch spikes.  A step is only corrected if it is large
        AND approximately self-cancelling with an adjacent step.  Specifically,
        a step at position i is treated as a spike when

            |step[i] + step[i±1]| < return_fraction * |step[i]|

        i.e. the adjacent step reverses most of the displacement.  Re-homing
        events (large step followed by small steps that stay at the new
        position) are left untouched.  This makes the filter safe to enable
        by default without manual threshold tuning per subject.
    iqr_multiplier : float
        Multiplier for IQR-based detection (only used by 'iqr' method).
    return_fraction : float
        For 'rehome': fraction threshold below which a round-trip is
        considered self-cancelling (default 0.4 — if the adjacent step
        reverses more than 60 % of a large step, treat as glitch spike).

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with outlier shifts corrected.
    """
    df = shifts_df.copy()
    shift_mag = (df["x_shift_mm"] ** 2 + df["y_shift_mm"] ** 2) ** 0.5

    if method == "iqr":
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

    if method == "clamp":
        for idx in df[outlier_mask].index:
            scale = max_shift_mm / shift_mag[idx]
            df.loc[idx, "x_shift_mm"] *= scale
            df.loc[idx, "y_shift_mm"] *= scale
            if "x_shift" in df.columns:
                df.loc[idx, "x_shift"] *= scale
                df.loc[idx, "y_shift"] *= scale

    elif method == "median":
        non_outlier = df[~outlier_mask]
        median_x = non_outlier["x_shift_mm"].median()
        median_y = non_outlier["y_shift_mm"].median()
        for idx in df[outlier_mask].index:
            df.loc[idx, "x_shift_mm"] = median_x
            df.loc[idx, "y_shift_mm"] = median_y

    elif method == "zero":
        for idx in df[outlier_mask].index:
            df.loc[idx, "x_shift_mm"] = 0.0
            df.loc[idx, "y_shift_mm"] = 0.0
            if "x_shift" in df.columns:
                df.loc[idx, "x_shift"] = 0.0
                df.loc[idx, "y_shift"] = 0.0

    elif method in ["local", "iqr"]:
        for idx in df[outlier_mask].index:
            pos: int = cast("int", df.index.get_loc(idx))
            neighbor_vals_x, neighbor_vals_y = [], []
            for offset in [-2, -1, 1, 2]:
                neighbor_pos = pos + offset
                if 0 <= neighbor_pos < len(df):
                    neighbor_idx = df.index[neighbor_pos]
                    if not outlier_mask[neighbor_idx]:
                        neighbor_vals_x.append(df.loc[neighbor_idx, "x_shift_mm"])
                        neighbor_vals_y.append(df.loc[neighbor_idx, "y_shift_mm"])

            if neighbor_vals_x:
                df.loc[idx, "x_shift_mm"] = np.median(neighbor_vals_x)
                df.loc[idx, "y_shift_mm"] = np.median(neighbor_vals_y)
            else:
                non_outlier = df[~outlier_mask]
                df.loc[idx, "x_shift_mm"] = non_outlier["x_shift_mm"].median()
                df.loc[idx, "y_shift_mm"] = non_outlier["y_shift_mm"].median()

    elif method == "rehome":
        # Only correct steps that are large AND self-cancelling with a neighbour.
        # A step that stays (re-homing event) has a large neighbour sum; a step
        # that returns (encoder glitch) has a near-zero neighbour sum.
        def _is_spike(pos: int, step_x: float, step_y: float, step_mag: float) -> bool:
            for offset in [-1, 1]:
                nb_pos = pos + offset
                if 0 <= nb_pos < len(df):
                    nb_idx = df.index[nb_pos]
                    nb_x = df.loc[nb_idx, "x_shift_mm"]
                    nb_y = df.loc[nb_idx, "y_shift_mm"]
                    roundtrip = np.sqrt((step_x + nb_x) ** 2 + (step_y + nb_y) ** 2)
                    if roundtrip < return_fraction * step_mag:
                        return True
            return False

        for idx in df[outlier_mask].index:
            pos: int = cast("int", df.index.get_loc(idx))
            step_x = df.loc[idx, "x_shift_mm"]
            step_y = df.loc[idx, "y_shift_mm"]
            step_mag = shift_mag[idx]

            if not _is_spike(pos, step_x, step_y, step_mag):
                # Re-homing event — leave it unchanged
                continue

            # Glitch spike — replace with local median of non-outlier neighbours
            neighbor_vals_x, neighbor_vals_y = [], []
            for offset in [-2, -1, 1, 2]:
                neighbor_pos = pos + offset
                if 0 <= neighbor_pos < len(df):
                    neighbor_idx = df.index[neighbor_pos]
                    if not outlier_mask[neighbor_idx]:
                        neighbor_vals_x.append(df.loc[neighbor_idx, "x_shift_mm"])
                        neighbor_vals_y.append(df.loc[neighbor_idx, "y_shift_mm"])

            if not neighbor_vals_x:
                non_outlier = df[~outlier_mask]
                neighbor_vals_x = [non_outlier["x_shift_mm"].median()]
                neighbor_vals_y = [non_outlier["y_shift_mm"].median()]

            df.loc[idx, "x_shift_mm"] = float(np.median(neighbor_vals_x))
            df.loc[idx, "y_shift_mm"] = float(np.median(neighbor_vals_y))
            if "x_shift" in df.columns:
                nb_px_x, nb_px_y = [], []
                for offset in [-2, -1, 1, 2]:
                    nb_pos = pos + offset
                    if 0 <= nb_pos < len(df):
                        nb_idx = df.index[nb_pos]
                        if not outlier_mask[nb_idx]:
                            nb_px_x.append(df.loc[nb_idx, "x_shift"])
                            nb_px_y.append(df.loc[nb_idx, "y_shift"])
                if nb_px_x:
                    df.loc[idx, "x_shift"] = float(np.median(nb_px_x))
                    df.loc[idx, "y_shift"] = float(np.median(nb_px_y))

    return df


def correct_tile_offset_shifts(
    shifts_df: pd.DataFrame,
    tile_fov_x_mm: float,
    tile_fov_y_mm: float | None = None,
    tolerance: float = 0.05,
    min_step_mm: float = 0.0,
) -> tuple[pd.DataFrame, list[int]]:
    """Correct pairwise shifts that are spurious integer multiples of an artifact step.

    The XY shifts file records ``xmin_mm[fixed] - xmin_mm[moving]``, where
    ``xmin_mm[i]`` is the **left-edge position of the mosaic grid** for slice
    ``i``.  After each slice the acquisition software calls ``detect_mosaic`` to
    find the tissue boundary; if the boundary has moved, ``mosaic_xmin_mm`` is
    reset to the new position (minus a margin).  This repositioning is recorded
    in the shifts file as an apparent lateral tissue drift even though the tissue
    itself did not move.  The magnitude equals however far the detected tissue
    boundary shifted, which is determined by tissue geometry and the ROI
    detection algorithm — **not** by the overlap-corrected tile step or any
    stage hardware quantum.

    .. note::
        The artifact step ``tile_fov_x_mm`` must be **empirically determined**
        from the shifts_xy.csv data.  It is **not** equal to
        ``tile_size_um × (1 - overlap_fraction) / 1000`` (the stitching tile
        step).  To find the correct value, inspect the x_shift_mm column for a
        cluster of near-equal large steps; that common value is the artifact step.

    These steps are persistent (not self-cancelling) and therefore survive the
    spike detector in ``filter_outlier_shifts`` unmodified.  This function
    strips the integer-artifact-step component from each shift, leaving only
    the true inter-slice tissue drift.

    This function checks each pairwise step independently: if the X component
    is within ``tolerance`` of N × ``tile_fov_x_mm`` (for integer N ≠ 0), the
    offset N × tile_fov_x_mm is subtracted, recovering the true tissue drift.
    The same is done for the Y component independently.

    Parameters
    ----------
    shifts_df : pd.DataFrame
        DataFrame with columns: fixed_id, moving_id, x_shift_mm, y_shift_mm
        (and optionally x_shift, y_shift in pixels).
    tile_fov_x_mm : float
        Empirically determined artifact step size in X (mm).  Must be found
        from the shifts data — see note above.
    tile_fov_y_mm : float, optional
        Tile field-of-view width in Y (mm).  Defaults to ``tile_fov_x_mm``.
    tolerance : float
        Fractional tolerance: a component is treated as a tile-multiple when
        ``|component - N × fov| / fov < tolerance``.  Default 0.05 (5 %).
    min_step_mm : float
        Only inspect steps whose magnitude exceeds this value (mm).
        Default 0 — all steps are checked.

    Returns
    -------
    pd.DataFrame
        Corrected DataFrame.
    List[int]
        Indices of rows that were modified.
    """
    if tile_fov_y_mm is None:
        tile_fov_y_mm = tile_fov_x_mm

    df = shifts_df.copy()
    corrected_indices = []

    for idx in df.index:
        dx = df.loc[idx, "x_shift_mm"]
        dy = df.loc[idx, "y_shift_mm"]
        mag = float(np.sqrt(dx**2 + dy**2))

        if mag < min_step_mm:
            continue

        modified = False

        # Check X component
        if tile_fov_x_mm > 0:
            nx = round(dx / tile_fov_x_mm)
            if nx != 0 and abs(dx - nx * tile_fov_x_mm) / tile_fov_x_mm < tolerance:
                offset_x_mm = nx * tile_fov_x_mm
                if "x_shift" in df.columns and abs(dx) > 1e-9:
                    df.loc[idx, "x_shift"] -= offset_x_mm * (df.loc[idx, "x_shift"] / dx)
                df.loc[idx, "x_shift_mm"] -= offset_x_mm
                modified = True

        # Check Y component
        if tile_fov_y_mm > 0:
            dy_cur = df.loc[idx, "y_shift_mm"]  # may differ from dy if X was corrected
            ny = round(dy_cur / tile_fov_y_mm)
            if ny != 0 and abs(dy_cur - ny * tile_fov_y_mm) / tile_fov_y_mm < tolerance:
                offset_y_mm = ny * tile_fov_y_mm
                if "y_shift" in df.columns and abs(dy) > 1e-9:
                    df.loc[idx, "y_shift"] -= offset_y_mm * (df.loc[idx, "y_shift"] / dy)
                df.loc[idx, "y_shift_mm"] -= offset_y_mm
                modified = True

        if modified:
            corrected_indices.append(idx)

    return df, corrected_indices


def filter_step_outliers(
    shifts_df: pd.DataFrame,
    max_step_mm: float = 0.0,
    window: int = 2,
    method: str = "local_median",
    mad_threshold: float = 3.0,
    return_fraction: float = 0.0,
) -> pd.DataFrame:
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
    return_fraction : float
        For all methods: if a flagged large step is NOT self-cancelling with an
        adjacent step (round-trip > return_fraction * step_mag), it is treated
        as a re-homing event and left unchanged.  Set to 0 to disable this
        guard (legacy behaviour).

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.
    """
    df = shifts_df.copy()
    shift_mag = (df["x_shift_mm"] ** 2 + df["y_shift_mm"] ** 2) ** 0.5

    if method == "local_mad":
        outlier_mask = pd.Series(False, index=df.index)
        for i in range(len(df)):
            lo = max(0, i - window)
            hi = min(len(df), i + window + 1)
            neighbour_mags = np.concatenate([shift_mag.iloc[lo:i].to_numpy(), shift_mag.iloc[i + 1 : hi].to_numpy()])
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
        pos: int = cast("int", df.index.get_loc(idx))
        step_x = df.loc[idx, "x_shift_mm"]
        step_y = df.loc[idx, "y_shift_mm"]
        step_mag = float(shift_mag.iloc[pos])

        # Re-homing guard: skip correction if the step is NOT self-cancelling.
        # A re-homing event has a large neighbour sum (position stays); a glitch
        # spike returns to baseline (neighbour sum ≈ 0).
        if return_fraction > 0:
            is_spike = False
            for offset in [-1, 1]:
                nb_pos = pos + offset
                if 0 <= nb_pos < len(df):
                    nb_idx = df.index[nb_pos]
                    nb_x = df.loc[nb_idx, "x_shift_mm"]
                    nb_y = df.loc[nb_idx, "y_shift_mm"]
                    roundtrip = np.sqrt((step_x + nb_x) ** 2 + (step_y + nb_y) ** 2)
                    if roundtrip < return_fraction * step_mag:
                        is_spike = True
                        break
            if not is_spike:
                continue  # Re-homing event — leave unchanged

        if method == "clamp":
            scale = max_step_mm / shift_mag[idx]
            df.loc[idx, "x_shift_mm"] *= scale
            df.loc[idx, "y_shift_mm"] *= scale
            if "x_shift" in df.columns:
                df.loc[idx, "x_shift"] *= scale
                df.loc[idx, "y_shift"] *= scale
        else:
            pos = cast("int", df.index.get_loc(idx))
            neighbor_vals_x, neighbor_vals_y = [], []
            for offset in range(-window, window + 1):
                if offset == 0:
                    continue
                neighbor_pos = pos + offset
                if 0 <= neighbor_pos < len(df):
                    neighbor_idx = df.index[neighbor_pos]
                    neighbor_vals_x.append(df.loc[neighbor_idx, "x_shift_mm"])
                    neighbor_vals_y.append(df.loc[neighbor_idx, "y_shift_mm"])
            if neighbor_vals_x:
                df.loc[idx, "x_shift_mm"] = float(np.median(neighbor_vals_x))
                df.loc[idx, "y_shift_mm"] = float(np.median(neighbor_vals_y))
                if "x_shift" in df.columns:
                    neighbor_px_x = [
                        df.loc[df.index[pos + o], "x_shift"]
                        for o in range(-window, window + 1)
                        if o != 0 and 0 <= pos + o < len(df) and "x_shift" in df.columns
                    ]
                    neighbor_px_y = [
                        df.loc[df.index[pos + o], "y_shift"]
                        for o in range(-window, window + 1)
                        if o != 0 and 0 <= pos + o < len(df) and "x_shift" in df.columns
                    ]
                    if neighbor_px_x:
                        df.loc[idx, "x_shift"] = float(np.median(neighbor_px_x))
                        df.loc[idx, "y_shift"] = float(np.median(neighbor_px_y))

    return df
