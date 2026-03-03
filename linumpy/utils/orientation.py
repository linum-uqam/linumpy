# -*- coding: utf-8 -*-

"""
Utilities for handling 3D volume orientation codes and transformations.

Orientation convention used throughout:
  - numpy dim 0 → SITK Z → Allen S (Superior)
  - numpy dim 1 → SITK X → Allen R (Right)
  - numpy dim 2 → SITK Y → Allen A (Anterior)

The RAS target orientation maps:
  - output dim 0  ←→  Superior (S)
  - output dim 1  ←→  Right (R)
  - output dim 2  ←→  Anterior (A)
"""

import numpy as np


def parse_orientation_code(orientation: str) -> tuple:
    """
    Parse an orientation code and return axis permutation and flips for RAS alignment.

    Parameters
    ----------
    orientation : str
        3-letter code (R/L, A/P, S/I) describing what each *source* axis points to.
        Example: 'AIR' means dim0→Anterior, dim1→Inferior, dim2→Right.

    Returns
    -------
    axis_permutation : tuple of int
        Source indices for each target dimension, such that
        ``np.transpose(volume, axis_permutation)`` produces a volume whose axes are
        ordered (S, R, A) — matching the numpy_to_sitk_image convention where:
          - numpy dim 0 → SITK Z → Allen S (Superior)
          - numpy dim 1 → SITK X → Allen R (Right)
          - numpy dim 2 → SITK Y → Allen A (Anterior)
    axis_flips : tuple of int
        Sign for each axis **after** permutation: -1 means flip that axis, +1 means keep.

    Raises
    ------
    ValueError
        If the orientation code is not exactly 3 letters, contains invalid letters,
        or has duplicate axis directions.

    Examples
    --------
    >>> parse_orientation_code('SRA')  # source already in (S, R, A) order — identity
    ((0, 1, 2), (1, 1, 1))
    >>> parse_orientation_code('PIR')  # common OCT orientation
    ((1, 2, 0), (-1, 1, -1))
    """
    if len(orientation) != 3:
        raise ValueError(f"Orientation code must be 3 letters, got '{orientation}'")

    orientation = orientation.upper()

    # Map each letter to the TARGET numpy dimension and the sign for that direction.
    # Target dimensions (after permutation):
    #   dim 0 → S (Superior)    letter 'S' → same direction, 'I' → flipped
    #   dim 1 → R (Right)       letter 'R' → same direction, 'L' → flipped
    #   dim 2 → A (Anterior)    letter 'A' → same direction, 'P' → flipped
    letter_map = {
        'S': (0,  1), 'I': (0, -1),   # target dim 0 (Superior)
        'R': (1,  1), 'L': (1, -1),   # target dim 1 (Right)
        'A': (2,  1), 'P': (2, -1),   # target dim 2 (Anterior)
    }

    source_to_target = {}
    axes_used = set()

    for source_dim, letter in enumerate(orientation):
        if letter not in letter_map:
            raise ValueError(
                f"Invalid orientation letter '{letter}'. Use R/L, A/P, or S/I."
            )
        target_dim, sign = letter_map[letter]
        if target_dim in axes_used:
            raise ValueError(
                f"Duplicate axis direction in orientation code '{orientation}': "
                f"letter '{letter}' maps to an already-used target axis."
            )
        axes_used.add(target_dim)
        source_to_target[source_dim] = (target_dim, sign)

    if axes_used != {0, 1, 2}:
        raise ValueError(
            f"Orientation code '{orientation}' must specify all three axes (S/I, R/L, A/P)."
        )

    # Build target_dim -> (source_dim, sign)
    target_to_source = {v[0]: (k, v[1]) for k, v in source_to_target.items()}

    axis_permutation = tuple(target_to_source[i][0] for i in range(3))
    axis_flips = tuple(target_to_source[i][1] for i in range(3))

    return axis_permutation, axis_flips


def apply_orientation_transform(
    volume: np.ndarray,
    permutation: tuple,
    flips: tuple
) -> np.ndarray:
    """
    Reorient a 3D volume by applying an axis permutation followed by axis flips.

    Parameters
    ----------
    volume : np.ndarray
        Input 3-D volume (any shape).
    permutation : tuple of int
        Axis permutation as returned by :func:`parse_orientation_code`.
        ``np.transpose(volume, permutation)`` is applied first.
    flips : tuple of int
        Sign for each axis after permutation.  A value of -1 means that axis is
        flipped (``np.flip``); +1 means the axis is kept as-is.

    Returns
    -------
    np.ndarray
        Reoriented volume.  The returned array may share memory with *volume*
        for the non-contiguous transpose, but ``np.flip`` produces a view, so
        callers should copy if in-place modification is needed.
    """
    result = np.transpose(volume, permutation)
    for axis, flip in enumerate(flips):
        if flip < 0:
            result = np.flip(result, axis=axis)
    return result


def reorder_resolution(resolution: tuple, permutation: tuple) -> tuple:
    """
    Reorder a per-axis resolution tuple to match the axis permutation.

    Parameters
    ----------
    resolution : tuple of float
        Per-axis resolution values, one per spatial dimension.
    permutation : tuple of int
        Axis permutation as returned by :func:`parse_orientation_code`.

    Returns
    -------
    tuple of float
        Resolution values reordered so that ``reordered[i] == resolution[permutation[i]]``,
        i.e. the resolution now corresponds to the target axis ordering.
    """
    return tuple(resolution[permutation[i]] for i in range(len(permutation)))
