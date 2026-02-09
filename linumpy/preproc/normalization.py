#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intensity normalization functions for OCT volumes.

This module provides functions for normalizing OCT volume intensities
based on agarose background detection.
"""

from typing import Tuple

import numpy as np


def normalize_volume(vol: np.ndarray,
                     agarose_mask: np.ndarray,
                     percentile_max: float = 99.9,
                     min_contrast_fraction: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize volume intensities based on agarose background.

    Intensities for each z-slice are rescaled between the minimum value
    inside agarose and the value defined by the percentile_max argument.

    Parameters
    ----------
    vol : np.ndarray
        Input volume with shape (Z, X, Y).
    agarose_mask : np.ndarray
        2D binary mask indicating agarose regions (shape X, Y).
    percentile_max : float
        Values above this percentile will be clipped. Default 99.9.
    min_contrast_fraction : float
        Minimum contrast (max-min) as a fraction of the global max.
        Slices with lower contrast will use this threshold to avoid
        over-amplification of noise in weak/bad slices. Default 0.1.

    Returns
    -------
    tuple
        (normalized_volume, background_thresholds)
        - normalized_volume: The normalized volume
        - background_thresholds: Array of background threshold per slice
    """
    # Clip to percentile max per slice
    pmax = np.percentile(vol, percentile_max, axis=(1, 2))
    vol = np.clip(vol, None, pmax[:, None, None])

    # Compute background threshold per slice from agarose regions
    background_thresholds = []
    for curr_slice in vol:
        agarose = curr_slice[agarose_mask]
        bg_median = np.median(agarose)
        background_thresholds.append(bg_median)

    background_thresholds = np.array(background_thresholds)
    vol = np.clip(vol, background_thresholds[:, None, None], None)

    # Rescale to [0, 1]
    vol = vol - np.min(vol, axis=(1, 2), keepdims=True)
    vmax = np.max(vol, axis=(1, 2))

    # Compute minimum acceptable contrast based on global statistics
    # This prevents over-amplification of slices with very weak signal
    global_max = np.max(vmax)
    min_contrast = global_max * min_contrast_fraction

    # For slices with sufficient contrast, normalize normally
    # For weak slices, use the minimum contrast threshold to avoid over-amplification
    effective_max = np.maximum(vmax, min_contrast)

    # Apply normalization
    for i in range(vol.shape[0]):
        if effective_max[i] > 0:
            vol[i] = vol[i] / effective_max[i]

    return vol, background_thresholds
