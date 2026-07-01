"""GPU-accelerated helpers for N4 bias field correction pre/post-processing.

Provides block-mean downsampling, bias field upsampling, and chunked
element-wise division on GPU (CuPy + PyTorch).  All functions fall back to
CPU (NumPy + SciPy) when ``GPU_AVAILABLE`` is False.
"""

from __future__ import annotations

import numpy as np

from . import GPU_AVAILABLE


def downsample_gpu(vol: np.ndarray, shrink_factor: int, use_gpu: bool = True) -> np.ndarray:
    """Block-mean spatial downsampling by an integer factor.

    Parameters
    ----------
    vol : np.ndarray
        Float32 input (Z, Y, X).
    shrink_factor : int
        Isotropic downsampling factor.  The output shape is
        ``ceil(s / shrink_factor)`` on each axis.
    use_gpu : bool
        Use CuPy when GPU is available.

    Returns
    -------
    np.ndarray
        Downsampled float32 array.
    """
    if use_gpu and GPU_AVAILABLE:
        try:
            import cupy as cp

            arr = cp.asarray(vol, dtype=cp.float32)
            z, y, x = arr.shape
            f = shrink_factor
            # Trim to multiple of shrink_factor on each axis
            arr = arr[: z - z % f or z, : y - y % f or y, : x - x % f or x]
            z2, y2, x2 = arr.shape
            out = arr.reshape(z2 // f, f, y2 // f, f, x2 // f, f).mean(axis=(1, 3, 5))
            return cp.asnumpy(out).astype(np.float32)
        except Exception:
            pass  # fall through to CPU

    # CPU fallback — scipy zoom with anti-aliasing via block-mean
    from scipy.ndimage import zoom

    factor = 1.0 / shrink_factor
    return zoom(vol.astype(np.float32), (factor, factor, factor), order=1, prefilter=False)


def upsample_bias_gpu(
    bias_low: np.ndarray,
    target_shape: tuple[int, int, int],
    use_gpu: bool = True,
) -> np.ndarray:
    """Trilinear upsampling of a low-resolution bias field to *target_shape*.

    Parameters
    ----------
    bias_low : np.ndarray
        Low-resolution bias field (Z', Y', X'), float32.
    target_shape : tuple of int
        Desired output shape (Z, Y, X).
    use_gpu : bool
        Use PyTorch trilinear interpolation when GPU is available.

    Returns
    -------
    np.ndarray
        Upsampled float32 bias field of shape *target_shape*.
    """
    if use_gpu and GPU_AVAILABLE:
        try:
            import torch

            device = torch.device("cuda")
            t = torch.from_numpy(bias_low[np.newaxis, np.newaxis]).to(device, dtype=torch.float32)
            out = torch.nn.functional.interpolate(t, size=target_shape, mode="trilinear", align_corners=False)
            return out[0, 0].cpu().numpy()
        except Exception:
            pass  # fall through to CPU

    # CPU fallback
    from scipy.ndimage import zoom

    factors = tuple(t / s for t, s in zip(target_shape, bias_low.shape, strict=True))
    return zoom(bias_low.astype(np.float32), factors, order=1, prefilter=False)


def apply_bias_field_gpu(
    vol: np.ndarray,
    bias_field: np.ndarray,
    chunk_z: int = 50,
    floor: float = 1e-6,
    use_gpu: bool = True,
) -> np.ndarray:
    """Element-wise division ``vol / bias_field`` processed in Z-chunks on GPU.

    Parameters
    ----------
    vol : np.ndarray
        Float32 input volume (Z, Y, X).
    bias_field : np.ndarray
        Multiplicative bias field, same shape as *vol*.
    chunk_z : int
        Number of Z-planes per GPU chunk.
    floor : float
        Minimum divisor to avoid division by zero.
    use_gpu : bool
        Use CuPy when GPU is available.

    Returns
    -------
    np.ndarray
        Corrected float32 volume, same shape as *vol*.
    """
    if use_gpu and GPU_AVAILABLE:
        try:
            import cupy as cp

            out = np.empty_like(vol, dtype=np.float32)
            for z_start in range(0, vol.shape[0], chunk_z):
                z_end = min(z_start + chunk_z, vol.shape[0])
                v = cp.asarray(vol[z_start:z_end], dtype=cp.float32)
                b = cp.asarray(bias_field[z_start:z_end], dtype=cp.float32)
                out[z_start:z_end] = cp.asnumpy(v / cp.maximum(b, floor))
            return out
        except Exception:
            pass  # fall through to CPU

    # CPU fallback
    from linumpy.intensity.bias_field import apply_bias_field

    return apply_bias_field(vol, bias_field, floor=floor)
