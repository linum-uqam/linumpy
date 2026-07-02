"""Galvanometric XY shift detection and correction."""

import numpy as np
from scipy.ndimage import median_filter


def detect_galvo_shift(aip: np.ndarray, n_pixel_return: int = 40) -> int:
    """Detect the galvo shift in the AIP.

    Parameters
    ----------
    aip : ndarray
        AIP of the OCT volume containing both the image and the galvo return. This assumes that the first axis is the
        A-line axis, and the second axis is the B-scan axis, and the average was taken over the depth axis.
    n_pixel_return : int
        Number of pixels used for the galvo returns.

    Returns
    -------
    int
        Shift in pixels
    """
    # Compute the average a-line
    profile = aip.mean(axis=1)
    profile = median_filter(profile, 9)

    # Compute the intensity difference between the start and end of the a-line for various shifts.
    # A wrong shift would result in values close to zero as they would be close by in the actual scan
    differences = []
    for s in range(len(profile)):
        d = np.abs(profile[s] - profile[-1 + s])
        differences.append(d)

    # If we find the right shift, both the beginning and the end of galvo return will result in high differences
    similarities = []
    for s in range(len(profile) - n_pixel_return):
        foo = differences[s] * differences[s + n_pixel_return]
        similarities.append(foo)

    shift = np.argmax(similarities)
    shift = len(profile) - shift - n_pixel_return

    return int(shift)


def fix_galvo_shift(vol: np.ndarray, shift: int = 0, axis: int = 1) -> np.ndarray:
    """Fix the galvo shift in an OCT volume."""
    if shift == 0:
        return vol
    else:
        return np.roll(vol, shift, axis=axis)
