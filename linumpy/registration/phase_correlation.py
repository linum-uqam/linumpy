"""Phase-correlation registration for tile pairs."""

from typing import Literal, overload

import numpy as np
from skimage.feature import peak_local_max

from linumpy.mosaic.overlap import get_overlap


@overload
def pair_wise_phase_correlation(
    vol1: np.ndarray, vol2: np.ndarray, n_peaks: int = ..., return_cc: Literal[False] = ...
) -> list[int]: ...
@overload
def pair_wise_phase_correlation(
    vol1: np.ndarray, vol2: np.ndarray, n_peaks: int = ..., return_cc: Literal[True] = ...
) -> tuple[list[int], float]: ...
def pair_wise_phase_correlation(
    vol1: np.ndarray, vol2: np.ndarray, n_peaks: int = 8, return_cc: bool = False
) -> list[int] | tuple[list[int], float]:  # TODO: Test for 3D images
    """Find the translation between image pairs using phase correlation and cross-correlation.

    Parameters
    ----------
    vol1 : ndimage
        Fixed image / volume
    vol2 : ndimage
        Moving image / volume
    n_peaks : int
        Number of phase correlation peaks to sample
    return_cc : bool
        Return cross-correlation score

    Returns
    -------
    list
        Translation of vol2 -/- vol1 in each direction

    Notes
    -----
    - Works in 2D for now. Needs to be tested in 3D.

    References
    ----------
    Preibisch S. et al. (2008) Fast Stitching of Large 3D Biological Datasets (ImageJ Proceesings)
    """
    # Extend images by 1/4 of their size in each direction
    vol_shape = vol1.shape
    new_shape = np.array(vol_shape) * 1.25
    pad_size = np.ceil(0.5 * (new_shape - vol_shape)).astype(int)
    pad_width = [(pad, pad) for pad in pad_size]
    vol1_p = np.pad(vol1, pad_width, mode="reflect")
    vol2_p = np.pad(vol2, pad_width, mode="reflect")

    # Apply a window on the image extension
    vol1_p = apply_hanning_window(vol1_p, pad_size)
    vol2_p = apply_hanning_window(vol2_p, pad_size)

    # TODO: Add zero-padding up to the next power of two or up to a given size ...

    # Phase correlation matrix Q computation
    Q_num = np.fft.fft2(vol2_p) * np.conjugate(np.fft.fft2(vol1_p))
    Q_denum = np.abs(Q_num)
    with np.errstate(divide="ignore"):
        Q_freq = np.divide(Q_num, Q_denum)
        Q_freq[Q_denum == 0] = 0
    Q = np.fft.ifft2(Q_freq)

    # Find the main peak
    pmax = np.amax(Q)
    indices = np.where(pmax == Q)

    # Find the first N peaks
    coordinates = peak_local_max(
        np.abs(Q), min_distance=1, num_peaks=n_peaks, exclude_border=False
    )  # max value in the whole image

    deltas_list = []
    for indices in coordinates:
        deltas = []
        for idx, s in zip(indices, vol1_p.shape, strict=False):
            deltas.append(int(-idx + s / 2))

        # Check if it is outside the original image
        for ii in range(len(deltas)):
            if deltas[ii] > vol_shape[ii]:
                print(("deltas larger than imshape", deltas[ii], vol_shape[ii]))
                deltas[ii] -= vol_shape[ii]
        deltas_list.append(deltas)

    # Try all translation permutations and find which one has the highest correlation.
    translations = []
    for deltas in deltas_list:
        if vol1.ndim == 2:
            dx, dy = deltas[:]
            translations.extend(
                [
                    [dx, dy],
                    [dx - np.sign(dx) * int(vol1_p.shape[0] / 2), dy],
                    [dx, dy - np.sign(dy) * int(vol1_p.shape[1] / 2)],
                    [
                        dx - np.sign(dx) * int(vol1_p.shape[0] / 2),
                        dy - np.sign(dy) * int(vol1_p.shape[1] / 2),
                    ],
                ]
            )
        else:
            dx, dy, dz = deltas[:]
            nxp = np.sign(dx) * int(vol1_p.shape[0] / 2)
            nyp = np.sign(dy) * int(vol1_p.shape[1] / 2)
            nzp = np.sign(dz) * int(vol1_p.shape[2] / 2)
            translations.extend(
                [
                    [dx, dy, dz],
                    [dx - nxp, dy, dz],
                    [dx, dy - nyp, dz],
                    [dx - nxp, dy - nyp, dz],
                    [dx, dy, dz - nzp],
                    [dx, dy - nyp, dz - nzp],
                    [dx - nxp, dy, dz - nzp],
                    [dx - nxp, dy - nyp, dz - nzp],
                ]
            )
    corr_score = []
    for this_delta in translations:
        pos1 = tuple([0] * vol1.ndim)
        ov1, ov2, _, _ = get_overlap(vol1, vol2, pos1, this_delta)
        try:
            corr = cross_correlation(ov1, ov2)
        except Exception:
            corr = 0
        corr_score.append(corr)

    corr_score = np.array(corr_score)
    corr_score[np.isnan(corr_score)] = 0

    idx = np.where(corr_score == corr_score.max())[0][0]

    if return_cc:
        return translations[idx], corr_score[idx]
    else:
        return translations[idx]


def cross_correlation(vol1: np.ndarray, vol2: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Compute the normalized cross-correlation between two ndarrays.

    Parameters
    ----------
    vol1 : ndarray
        Fixed volume
    vol2 : ndarray
        Moving volume
    mask : ndarray
        Mask where the cross-correlation is computed. Assumed to be everywhere.

    Returns
    -------
    float
        Cross correlation between the volumes

    Notes
    -----
    - If a mask is given, the weighted NCC is computed instead of the NCC.
    - vol1, vol2 and mask should have the same shape.
    - mask is normalized before using it in the NCC computation.
    """
    if mask is None:
        mask = np.ones_like(vol1, dtype=float)

    # Normalizing the mask
    if mask.sum() > 0:
        mask = mask / float(mask.sum())
    else:
        return 0.0  # The mask is empty

    try:  # Using the WNCC, i.e. using a weighted sum instead of an average.
        cov_ab = np.sum((vol1 - np.sum(vol1 * mask)) * (vol2 - np.sum(vol2 * mask)) * mask)
        sA = np.sqrt(np.sum((vol1 - np.sum(vol1 * mask)) ** 2.0 * mask))
        sB = np.sqrt(np.sum((vol2 - np.sum(vol2 * mask)) ** 2.0 * mask))

        return cov_ab / float(sA * sB)
    except Exception:
        return 0.0


def apply_hanning_window(im: np.ndarray, padshape: np.ndarray | tuple[int, ...]) -> np.ndarray:
    """Apply an hanning window to image.

    Parameters
    ----------
    im : ndarray
         ndarray to modify
    padshape : ndarray or tuple of int
        Padding size for each dimension.

    Returns
    -------
    ndarray
        Modified ndarray.

    """
    ndim = im.ndim
    if ndim == 2:
        im = np.reshape(im, (im.shape[0], im.shape[1], 1))

    nx, ny, nz = im.shape
    for ii in range(ndim):
        pad = padshape[ii]
        s = im.shape[ii]

        h = np.hanning(pad * 2)
        h_full = np.ones((s,))
        h_full[0:pad] = h[0:pad]
        h_full[-pad::] = h[pad::]

        # Reshape and tile
        reshape_size = [1, 1, 1]
        reshape_size[ii] = s
        tile_size = [nx, ny, nz]
        tile_size[ii] = 1

        h_full = np.tile(np.reshape(h_full, reshape_size), tile_size)
        im = im * h_full
    if ndim == 2:
        im = np.squeeze(im)

    return im
