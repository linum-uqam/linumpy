"""GPU implementation of tissue interface detection.

Mirrors the CPU path in :func:`linumpy.geometry.interface.find_tissue_interface`
but routes the heavy filters through ``cupyx.scipy.ndimage`` to avoid
host-device round trips when the caller already holds GPU data or when
the volume is large enough that the transfer cost is amortised.
"""

import numpy as np

from . import to_cpu


def find_tissue_interface_gpu(
    vol: np.ndarray,
    s_xy: int = 15,
    s_z: int = 2,
    use_log: bool = True,
    order: int = 1,
    detect_cutting_errors: bool = False,
) -> np.ndarray:
    """GPU equivalent of ``find_tissue_interface`` (no mask path).

    Always returns a NumPy array of integer depths so callers do not need
    to be aware of the device.
    """
    import cupy as cp
    from cupyx.scipy.ndimage import gaussian_filter1d, uniform_filter

    vol_gpu = cp.asarray(vol)
    if use_log:
        vol_p = vol_gpu.astype(cp.float32, copy=True)
        positive = vol_gpu > 0
        vol_p[positive] = cp.log(vol_gpu[positive])
    else:
        vol_p = vol_gpu

    vol_p = uniform_filter(vol_p, (s_xy, s_xy, 0))
    vol_g = gaussian_filter1d(vol_p, s_z, axis=2, order=order)
    z0 = cp.ceil(vol_g.argmax(axis=2) + s_z * 0.5).astype(cp.int64)

    if detect_cutting_errors:
        vol_p = gaussian_filter1d(vol_p, s_z, axis=2, order=0)
        z0_p = cp.abs(vol_p).argmax(axis=2)
        mask_max = z0_p < z0
        z0 = cp.where(mask_max, z0_p, z0)

    return to_cpu(z0)
