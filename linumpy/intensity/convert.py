"""Datatype conversion utilities for OCT volumes."""

import numpy as np


def convert_to_8bit(vol: np.ndarray) -> np.ndarray:
    """Convert a volume to 8-bit unsigned integer representation."""
    return (255 * (vol - vol.min()) / float(vol.max() - vol.min())).astype(np.uint8)
