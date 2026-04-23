"""Removal of high-frequency artifacts and specular reflections."""

import numpy as np
from scipy.interpolate import interpn
from scipy.ndimage import gaussian_filter


def remove_hf_intensity_artifact(vol: np.ndarray, sigma: int = 5, mask: np.ndarray | None = None) -> np.ndarray:
    """Remove high-frequency axial intensity artifacts from a volume."""
    nx, ny, nz = vol.shape
    maxI = vol.max()
    minI = vol.min()
    vol_zeros = vol == 0

    # Compute Intensity depth profile
    if mask is None:
        i_profile = vol.mean(axis=(0, 1))
    else:
        i_profile = np.zeros((nz,))
        if mask.ndim == 2:
            for z in range(nz):
                i_profile[z] = np.mean(vol[:, :, z][mask])

        else:
            i_profile = np.zeros((nz,))
            for z in range(nz):
                i_profile[z] = np.mean(vol[:, :, z][mask[:, :, z]])

    # Low pass filter of the intensity profile
    lp_profile = gaussian_filter(i_profile, sigma)
    hf_profile = i_profile - lp_profile

    # Removing the hf component from the original data
    hf_3dprofile = np.tile(np.reshape(hf_profile, (1, 1, nz)), (nx, ny, 1))
    vol_p = vol - hf_3dprofile
    vol_p = (maxI - minI) * (vol_p - vol_p.min()) / float(vol_p.max() - vol_p.min()) + minI
    vol_p[vol_zeros] = 0

    return vol_p.astype(vol.dtype)



def remove_reflection(vol: np.ndarray, z0: int, radius: int = 3) -> np.ndarray:
    """Remove a specular reflection at a given depth using 3D interpolation."""
    vol_p = np.copy(vol)
    nx, ny, nz = vol.shape
    x = np.arange(nx)
    y = np.arange(ny)
    z = np.arange(nz)

    # Compute the zmin and zmax index used for the interpolation
    zmin = z0 - radius
    zmax = z0 + radius
    z_list = list(range(zmin, zmax))
    z = np.delete(z, z_list)
    vol_p = np.delete(vol_p, z_list, axis=2)

    # 3D Interpolation
    xx, yy, zz = np.meshgrid(x, y, z_list, indexing="ij")
    new_pos = np.stack((xx, yy, zz), axis=3)
    vol_roi = interpn((x, y, z), vol_p, new_pos, method="linear")

    # Update the vol_p
    vol_p = np.concatenate((vol_p[:, :, 0:zmin], vol_roi, vol_p[:, :, zmax::]), axis=-1)

    return vol_p
