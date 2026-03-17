"""
GPU-accelerated FFT operations for linumpy.

Provides GPU versions of FFT-based operations including phase correlation
for image registration and stitching.
"""

import numpy as np

from . import GPU_AVAILABLE, to_cpu


def phase_correlation(vol1, vol2, n_peaks=8, use_gpu=True):
    """
    GPU-accelerated phase correlation for finding translation between images.
    
    Parameters
    ----------
    vol1 : np.ndarray
        Fixed image (2D or 3D)
    vol2 : np.ndarray
        Moving image (2D or 3D)
    n_peaks : int
        Number of peaks to sample for refinement
    use_gpu : bool
        Whether to use GPU acceleration
        
    Returns
    -------
    list
        Translation [dx, dy] or [dx, dy, dz] of vol2 relative to vol1
    float
        Cross-correlation score
    """
    if use_gpu and GPU_AVAILABLE:
        return _phase_correlation_gpu(vol1, vol2, n_peaks)
    else:
        return _phase_correlation_cpu(vol1, vol2, n_peaks)


def _phase_correlation_gpu(vol1, vol2, n_peaks=8):
    """GPU implementation of phase correlation."""
    import cupy as cp
    from cupyx.scipy.ndimage import uniform_filter

    vol_shape = vol1.shape
    ndim = vol1.ndim

    # Transfer to GPU
    vol1_gpu = cp.asarray(vol1, dtype=cp.float32)
    vol2_gpu = cp.asarray(vol2, dtype=cp.float32)

    # Extend images by 1/4 of their size (padding)
    new_shape = tuple(int(s * 1.25) for s in vol_shape)
    pad_size = tuple((int(np.ceil(0.5 * (n - s))),) * 2
                     for s, n in zip(vol_shape, new_shape))

    vol1_p = cp.pad(vol1_gpu, pad_size, mode='reflect')
    vol2_p = cp.pad(vol2_gpu, pad_size, mode='reflect')

    # Apply Hanning window
    vol1_p = _apply_hanning_window_gpu(vol1_p, [p[0] for p in pad_size])
    vol2_p = _apply_hanning_window_gpu(vol2_p, [p[0] for p in pad_size])

    # Phase correlation using cuFFT
    if ndim == 2:
        fft_func = cp.fft.fft2
        ifft_func = cp.fft.ifft2
    else:
        fft_func = cp.fft.fftn
        ifft_func = cp.fft.ifftn

    Q_num = fft_func(vol2_p) * cp.conj(fft_func(vol1_p))
    Q_denum = cp.abs(Q_num)

    # Avoid division by zero
    Q_freq = cp.where(Q_denum > 1e-10, Q_num / Q_denum, 0)
    Q = ifft_func(Q_freq)
    Q_abs = cp.abs(Q)

    # Find peaks
    from cupyx.scipy.ndimage import maximum_filter

    # Local maxima detection
    local_max = maximum_filter(Q_abs, size=3)
    peaks_mask = (Q_abs == local_max)

    # Get top n_peaks
    flat_indices = cp.argsort(Q_abs.ravel())[-n_peaks:]
    coordinates = cp.unravel_index(flat_indices, Q_abs.shape)
    coordinates = cp.stack(coordinates, axis=1)

    # Try all translation permutations
    best_translation = None
    best_score = -1

    coordinates_cpu = to_cpu(coordinates)
    vol1_cpu = to_cpu(vol1_gpu)
    vol2_cpu = to_cpu(vol2_gpu)

    for indices in coordinates_cpu:
        deltas = []
        for idx, s in zip(indices, vol1_p.shape):
            deltas.append(int(-idx + s / 2))

        # Check bounds
        for ii in range(len(deltas)):
            if abs(deltas[ii]) > vol_shape[ii]:
                deltas[ii] -= int(np.sign(deltas[ii]) * vol_shape[ii])

        # Generate candidate translations
        if ndim == 2:
            dx, dy = deltas
            candidates = [
                [dx, dy],
                [dx - int(np.sign(dx) * vol1_p.shape[0] / 2), dy],
                [dx, dy - int(np.sign(dy) * vol1_p.shape[1] / 2)],
                [dx - int(np.sign(dx) * vol1_p.shape[0] / 2),
                 dy - int(np.sign(dy) * vol1_p.shape[1] / 2)],
            ]
        else:
            dx, dy, dz = deltas
            nxp = int(np.sign(dx) * vol1_p.shape[0] / 2)
            nyp = int(np.sign(dy) * vol1_p.shape[1] / 2)
            nzp = int(np.sign(dz) * vol1_p.shape[2] / 2)
            candidates = [
                [dx, dy, dz],
                [dx - nxp, dy, dz],
                [dx, dy - nyp, dz],
                [dx - nxp, dy - nyp, dz],
                [dx, dy, dz - nzp],
                [dx, dy - nyp, dz - nzp],
                [dx - nxp, dy, dz - nzp],
                [dx - nxp, dy - nyp, dz - nzp],
            ]

        for trans in candidates:
            score = _compute_correlation_score(vol1_cpu, vol2_cpu, trans)
            if score > best_score:
                best_score = score
                best_translation = trans

    return best_translation, best_score


def _apply_hanning_window_gpu(vol, pad_sizes):
    """Apply Hanning window on GPU."""
    import cupy as cp

    ndim = vol.ndim
    result = vol.copy()

    for axis, pad in enumerate(pad_sizes):
        if pad <= 0:
            continue

        s = vol.shape[axis]
        h = cp.hanning(pad * 2)
        h_full = cp.ones(s)
        h_full[:pad] = h[:pad]
        h_full[-pad:] = h[pad:]

        # Reshape for broadcasting
        shape = [1] * ndim
        shape[axis] = s
        h_full = h_full.reshape(shape)

        result = result * h_full

    return result


def _compute_correlation_score(vol1, vol2, translation):
    """Compute normalized cross-correlation score for a translation."""
    ndim = vol1.ndim

    # Compute overlap region
    slices1 = []
    slices2 = []

    for i, t in enumerate(translation):
        t = int(t)
        if t >= 0:
            slices1.append(slice(t, None))
            slices2.append(slice(None, vol2.shape[i] - t if t > 0 else None))
        else:
            slices1.append(slice(None, vol1.shape[i] + t))
            slices2.append(slice(-t, None))

    try:
        ov1 = vol1[tuple(slices1)]
        ov2 = vol2[tuple(slices2)]

        if ov1.size == 0 or ov2.size == 0:
            return 0

        # Normalized cross-correlation
        ov1_norm = ov1 - np.mean(ov1)
        ov2_norm = ov2 - np.mean(ov2)

        std1 = np.std(ov1_norm)
        std2 = np.std(ov2_norm)

        if std1 < 1e-10 or std2 < 1e-10:
            return 0

        return float(np.mean(ov1_norm * ov2_norm) / (std1 * std2))
    except:
        return 0


def _phase_correlation_cpu(vol1, vol2, n_peaks=8):
    """CPU fallback for phase correlation - calls existing implementation."""
    from linumpy.stitching.registration import pairWisePhaseCorrelation
    return pairWisePhaseCorrelation(vol1, vol2, nPeaks=n_peaks, returnCC=True)


def fft2(image, use_gpu=True):
    """
    GPU-accelerated 2D FFT.
    
    Parameters
    ----------
    image : np.ndarray
        Input 2D image
    use_gpu : bool
        Whether to use GPU
        
    Returns
    -------
    np.ndarray
        FFT result (complex)
    """
    if use_gpu and GPU_AVAILABLE:
        import cupy as cp
        img_gpu = cp.asarray(image)
        result = cp.fft.fft2(img_gpu)
        return to_cpu(result)
    else:
        return np.fft.fft2(image)


def ifft2(spectrum, use_gpu=True):
    """
    GPU-accelerated 2D inverse FFT.
    
    Parameters
    ----------
    spectrum : np.ndarray
        Input spectrum (complex)
    use_gpu : bool
        Whether to use GPU
        
    Returns
    -------
    np.ndarray
        Inverse FFT result
    """
    if use_gpu and GPU_AVAILABLE:
        import cupy as cp
        spec_gpu = cp.asarray(spectrum)
        result = cp.fft.ifft2(spec_gpu)
        return to_cpu(result)
    else:
        return np.fft.ifft2(spectrum)
