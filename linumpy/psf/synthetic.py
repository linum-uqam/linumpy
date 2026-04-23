"""Synthetic PSF generation from Gaussian-beam parameters."""

from collections.abc import Sequence

import numpy as np

from linumpy.intensity.psf_model import confocal_psf


def synthesize_3d_psf(zf: float, zr: float, res: float, volshape: Sequence[int]) -> np.ndarray:
    """Generate a 3D PSF based on Gaussian beam parameters.

    Parameters
    ----------
    zf : float
        Focal depth in microns
    zr : float
        Rayleigh length in microns
    res : float
        Axial resolution in micron / pixel
    volshape : (3,) list of int
        Output volume shape in pixel

    Returns
    -------
    ndarray
        3D PSF of shape 'volshape'
    """
    # TODO: Invert axes to agree with OME-zarr convention?
    nx, ny, nz = volshape[0:3]
    z = np.linspace(0, res * nz, nz)
    psf = confocal_psf(z, zf, zr)
    psf = np.tile(np.reshape(psf, (1, 1, nz)), (nx, ny, 1))

    return psf
