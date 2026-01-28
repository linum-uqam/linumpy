"""
GPU-accelerated registration operations for linumpy.

Provides a hybrid approach where metric computation is done on GPU
while the optimizer runs on CPU (SimpleITK).
"""

import numpy as np

from . import GPU_AVAILABLE, to_cpu
from .interpolation import affine_transform


class GPUAcceleratedRegistration:
    """
    Hybrid GPU/CPU registration class.
    
    Uses GPU for:
    - Image resampling/transformation
    - Metric computation (MSE, NCC)
    
    Uses CPU (SimpleITK) for:
    - Optimization loop
    - Transform management
    
    Parameters
    ----------
    use_gpu : bool
        Whether to use GPU for metric computation
    metric : str
        Registration metric: 'mse', 'ncc', 'mi'
    """

    def __init__(self, use_gpu=True, metric='mse'):
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.metric = metric.lower()

        if self.use_gpu:
            import cupy as cp
            self._cp = cp

    def compute_metric(self, fixed, moving):
        """
        Compute registration metric between two images.
        
        Parameters
        ----------
        fixed : np.ndarray
            Fixed image
        moving : np.ndarray
            Moving image (already transformed)
            
        Returns
        -------
        float
            Metric value (lower is better for MSE, higher for NCC)
        """
        if self.use_gpu:
            return self._compute_metric_gpu(fixed, moving)
        else:
            return self._compute_metric_cpu(fixed, moving)

    def _compute_metric_gpu(self, fixed, moving):
        """GPU implementation of metric computation."""
        cp = self._cp

        fixed_gpu = cp.asarray(fixed.astype(np.float32))
        moving_gpu = cp.asarray(moving.astype(np.float32))

        # Create mask for valid pixels
        mask = (fixed_gpu > 0) & (moving_gpu > 0)

        if self.metric == 'mse':
            diff = fixed_gpu - moving_gpu
            mse = cp.mean(diff[mask] ** 2)
            return float(mse.get())

        elif self.metric == 'ncc':
            # Normalized cross-correlation
            fixed_masked = fixed_gpu[mask]
            moving_masked = moving_gpu[mask]

            fixed_norm = fixed_masked - cp.mean(fixed_masked)
            moving_norm = moving_masked - cp.mean(moving_masked)

            std_fixed = cp.std(fixed_norm)
            std_moving = cp.std(moving_norm)

            if std_fixed < 1e-10 or std_moving < 1e-10:
                return 0.0

            ncc = cp.mean(fixed_norm * moving_norm) / (std_fixed * std_moving)
            return float(ncc.get())

        elif self.metric == 'mi':
            # Mutual information (simplified histogram-based)
            return self._compute_mi_gpu(fixed_gpu, moving_gpu, mask)

        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def _compute_mi_gpu(self, fixed, moving, mask, bins=32):
        """Compute mutual information on GPU."""
        cp = self._cp

        # Normalize to [0, bins-1]
        fixed_masked = fixed[mask]
        moving_masked = moving[mask]

        f_min, f_max = cp.min(fixed_masked), cp.max(fixed_masked)
        m_min, m_max = cp.min(moving_masked), cp.max(moving_masked)

        if f_max - f_min < 1e-10 or m_max - m_min < 1e-10:
            return 0.0

        fixed_binned = ((fixed_masked - f_min) / (f_max - f_min) * (bins - 1)).astype(cp.int32)
        moving_binned = ((moving_masked - m_min) / (m_max - m_min) * (bins - 1)).astype(cp.int32)

        fixed_binned = cp.clip(fixed_binned, 0, bins - 1)
        moving_binned = cp.clip(moving_binned, 0, bins - 1)

        # Joint histogram
        joint_hist = cp.zeros((bins, bins), dtype=cp.float32)
        for i in range(len(fixed_binned)):
            joint_hist[fixed_binned[i], moving_binned[i]] += 1

        # Normalize
        joint_hist /= joint_hist.sum()

        # Marginal histograms
        p_fixed = joint_hist.sum(axis=1)
        p_moving = joint_hist.sum(axis=0)

        # Mutual information
        mi = 0.0
        for i in range(bins):
            for j in range(bins):
                if joint_hist[i, j] > 1e-10:
                    mi += joint_hist[i, j] * cp.log(
                        joint_hist[i, j] / (p_fixed[i] * p_moving[j] + 1e-10) + 1e-10
                    )

        return float(mi.get())

    def _compute_metric_cpu(self, fixed, moving):
        """CPU fallback for metric computation."""
        mask = (fixed > 0) & (moving > 0)

        if self.metric == 'mse':
            diff = fixed - moving
            return float(np.mean(diff[mask] ** 2))

        elif self.metric == 'ncc':
            fixed_masked = fixed[mask]
            moving_masked = moving[mask]

            fixed_norm = fixed_masked - np.mean(fixed_masked)
            moving_norm = moving_masked - np.mean(moving_masked)

            std_fixed = np.std(fixed_norm)
            std_moving = np.std(moving_norm)

            if std_fixed < 1e-10 or std_moving < 1e-10:
                return 0.0

            return float(np.mean(fixed_norm * moving_norm) / (std_fixed * std_moving))

        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def transform_image(self, image, transform_matrix, output_shape=None):
        """
        Apply transformation to image using GPU.
        
        Parameters
        ----------
        image : np.ndarray
            Input image
        transform_matrix : np.ndarray
            Transformation matrix
        output_shape : tuple, optional
            Output shape
            
        Returns
        -------
        np.ndarray
            Transformed image
        """
        return affine_transform(image, transform_matrix, output_shape,
                                order=1, use_gpu=self.use_gpu)


def register_2d_gpu(fixed, moving, method='affine', metric='mse',
                    max_iterations=1000, use_gpu=True):
    """
    GPU-accelerated 2D image registration.
    
    Uses SimpleITK optimizer with GPU metric computation.
    
    Parameters
    ----------
    fixed : np.ndarray
        Fixed image
    moving : np.ndarray
        Moving image
    method : str
        Transform type: 'translation', 'euler', 'affine'
    metric : str
        Metric: 'mse', 'ncc', 'mi'
    max_iterations : int
        Maximum optimizer iterations
    use_gpu : bool
        Whether to use GPU acceleration
        
    Returns
    -------
    transform : sitk.Transform
        Computed transform
    str
        Optimizer stop condition
    float
        Final metric value
    """

    # For now, use SimpleITK's built-in registration
    # GPU acceleration is applied via pre/post processing

    # Normalize images on GPU if available
    if use_gpu and GPU_AVAILABLE:
        import cupy as cp

        fixed_gpu = cp.asarray(fixed.astype(np.float32))
        moving_gpu = cp.asarray(moving.astype(np.float32))

        # Normalize
        fixed_norm = (fixed_gpu - cp.min(fixed_gpu)) / (cp.max(fixed_gpu) - cp.min(fixed_gpu) + 1e-10)
        moving_norm = (moving_gpu - cp.min(moving_gpu)) / (cp.max(moving_gpu) - cp.min(moving_gpu) + 1e-10)

        fixed = to_cpu(fixed_norm)
        moving = to_cpu(moving_norm)

    # Use existing CPU registration
    from linumpy.stitching.registration import register_2d_images_sitk

    return register_2d_images_sitk(
        fixed, moving,
        method=method,
        metric='MSE' if metric.lower() == 'mse' else metric.upper(),
        max_iterations=max_iterations
    )


def apply_transform_gpu(image, transform, use_gpu=True):
    """
    Apply SimpleITK transform to image using GPU resampling.
    
    Parameters
    ----------
    image : np.ndarray
        Input image
    transform : sitk.Transform
        SimpleITK transform
    use_gpu : bool
        Whether to use GPU
        
    Returns
    -------
    np.ndarray
        Transformed image
    """

    # For complex transforms, use SimpleITK
    # Could potentially extract matrix and use GPU affine_transform

    if use_gpu and GPU_AVAILABLE and _is_affine_transform(transform):
        # Extract affine matrix and use GPU
        matrix, offset = _sitk_transform_to_matrix(transform, image.shape)
        return affine_transform(image, matrix, use_gpu=True)
    else:
        # Fall back to SimpleITK
        from linumpy.stitching.registration import apply_transform
        return apply_transform(image, transform)


def _is_affine_transform(transform):
    """Check if transform can be represented as affine matrix."""
    import SimpleITK as sitk
    return isinstance(transform, (sitk.AffineTransform,
                                  sitk.Euler2DTransform,
                                  sitk.Euler3DTransform,
                                  sitk.TranslationTransform))


def _sitk_transform_to_matrix(transform, image_shape):
    """Convert SimpleITK transform to affine matrix."""
    import SimpleITK as sitk

    ndim = len(image_shape)

    if isinstance(transform, sitk.TranslationTransform):
        matrix = np.eye(ndim)
        offset = np.array(transform.GetOffset())
        return matrix, offset

    elif isinstance(transform, sitk.Euler2DTransform):
        angle = transform.GetAngle()
        center = np.array(transform.GetCenter())
        translation = np.array(transform.GetTranslation())

        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

        # Affine: y = R(x - c) + c + t = Rx + (c - Rc + t)
        offset = center - rotation @ center + translation

        return rotation, offset

    elif isinstance(transform, sitk.AffineTransform):
        matrix = np.array(transform.GetMatrix()).reshape(ndim, ndim)
        offset = np.array(transform.GetTranslation())
        return matrix, offset

    else:
        raise ValueError(f"Cannot convert {type(transform)} to matrix")
