"""Unit conversion and centring for inter-slice shift fields."""

from collections.abc import Sequence


def detect_shift_units(resolution: Sequence[float]) -> tuple[float, float]:
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



def convert_shifts_to_pixels(cumsum_mm: dict, resolution_um: float) -> dict:
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
    return {slice_id: (dx_mm * mm_to_px, dy_mm * mm_to_px) for slice_id, (dx_mm, dy_mm) in cumsum_mm.items()}



def center_shifts(cumsum_px: dict, slice_ids: list) -> dict:
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

    return {slice_id: (dx - center_dx, dy - center_dy) for slice_id, (dx, dy) in cumsum_px.items()}
