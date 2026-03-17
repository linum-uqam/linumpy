"""GPU-accelerated correction operations for linumpy."""

import numpy as np

from . import GPU_AVAILABLE, to_cpu


def fix_galvo_shift(volume, shift, axis=1, use_gpu=True):
    """
    GPU-accelerated galvo shift correction.
    
    Parameters
    ----------
    volume : np.ndarray
        Input volume
    shift : int
        Shift amount in pixels
    axis : int
        Axis along which to shift
    use_gpu : bool
        Whether to use GPU
        
    Returns
    -------
    np.ndarray
        Corrected volume
    """
    if shift == 0:
        return volume

    if use_gpu and GPU_AVAILABLE:
        import cupy as cp
        vol_gpu = cp.asarray(volume)
        result = cp.roll(vol_gpu, shift, axis=axis)
        return to_cpu(result)
    else:
        return np.roll(volume, shift, axis=axis)


def detect_and_fix_galvo_shift(volume, n_pixel_return=40, threshold=0.5,
                               axis=1, use_gpu=True):
    """
    Detect and conditionally fix galvo shift.
    
    Note: Detection uses CPU (GPU offers no benefit). Only the fix uses GPU.
    
    Parameters
    ----------
    volume : np.ndarray
        Input volume (3D)
    n_pixel_return : int
        Number of pixels in galvo return region
    threshold : float
        Confidence threshold for applying fix (default 0.5, higher = more conservative)
    axis : int
        A-line axis
    use_gpu : bool
        Whether to use GPU for the fix operation
        
    Returns
    -------
    np.ndarray
        Corrected volume (or original if no fix needed)
    dict
        Detection results with 'shift', 'confidence', 'fixed' keys
    """
    from linumpy.preproc.xyzcorr import detect_galvo_shift

    # Compute AIP
    aip = np.mean(volume, axis=0)

    # Detect shift using CPU (GPU offers no benefit for detection)
    shift, confidence = detect_galvo_shift(aip, n_pixel_return)

    result = {
        'shift': shift,
        'confidence': confidence,
        'fixed': False
    }

    if confidence >= threshold:
        volume = fix_galvo_shift(volume, shift, axis=axis, use_gpu=use_gpu)
        result['fixed'] = True

    return volume, result
