"""Volume cropping and interface-based masking."""

import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import binary_closing, binary_fill_holes
from skimage.filters import threshold_otsu

from linumpy.geometry.interface import find_tissue_interface


def crop_volume(
    vol: np.ndarray, xlim: list[int] | None = None, ylim: list[int] | None = None, zlim: list[int] | None = None
) -> np.ndarray:
    """Crops the given volume according to the range given as input.

    Parameters
    ----------
    vol : ndarray
        Volume to crop
    xlim : (2,) list
        x range to keep
    ylim : (2,) list
        y range to keep
    zlim : (2,) list
        z range to keep

    Returns
    -------
    ndarray
        Cropped volume

    Notes
    -----
    * xlim=[0,-1] means that the whole volume in the x dimension will be returned.

    """
    if zlim is None:
        zlim = [0, -1]
    if ylim is None:
        ylim = [0, -1]
    if xlim is None:
        xlim = [0, -1]
    nx, ny = vol.shape[:2]
    xlim = list(xlim)
    ylim = list(ylim)
    zlim = list(zlim)
    if xlim[1] == -1:
        xlim[1] = nx
    if ylim[1] == -1:
        ylim[1] = ny

    if vol.ndim == 3:
        nz = vol.shape[2]
        if zlim[1] == -1:
            zlim[1] = nz
        return vol[xlim[0] : xlim[1], ylim[0] : ylim[1], zlim[0] : zlim[1]]

    elif vol.ndim == 2:
        return vol[xlim[0] : xlim[1], ylim[0] : ylim[1]]

    return vol


def crop_z0_whole_slice(
    vol: np.ndarray,
    dz: float = 20.0,
    nz: float = 200.0,
    voxdim: tuple[int, int, int] = (1, 1, 1),
    z0: float | None = None,
    verbose: bool = False,
    mask: np.ndarray | None = None,
    return_z0: bool = False,
) -> np.ndarray | tuple[np.ndarray, float]:
    """Crop whole slice in the z direction.

    Parameters
    ----------
    vol : ndarray
        Input volume to crop.
    dz : float
        Margin in microns under the interface to crop (to remove cutting deformations).
    nz : float
        Size of the slice to crop in microns.
    voxdim : tuple of int
        Dimension of each voxel in micron/pixel.
    z0 : float, optional
        Interface position in microns. If None, it is detected automatically.
    verbose : bool
        If True, print debug information.
    mask : ndarray, optional
        Tissue mask. If provided, used to compute the tissue interface.
    return_z0 : bool
        If True, also return the interface position z0.

    Returns
    -------
    ndarray
        Cropped array
    """
    if z0 is None:
        # Computing tissue mask
        if mask is not None:
            mask = vol.std(axis=2) > threshold_otsu(vol.std(axis=2))  # Using otsu on A-line intensity std
            mask = binary_fill_holes(binary_closing(mask))  # Closing and filling holes
        else:
            mask = np.ones(vol.shape[0:2], dtype=bool)

        # Computing tissue interface
        interface = find_tissue_interface(vol)

        # Use median interface
        z0 = np.median(interface)
    else:
        # Compute z0 in pixel
        z0 = np.floor(z0 / (1.0 * voxdim[2])).astype(int)

    # Finding crop limits
    zmin = np.floor((z0 * voxdim[2] + dz) / (1.0 * voxdim[2])).astype(int)
    zmax = np.floor((zmin * voxdim[2] + nz) / (1.0 * voxdim[2])).astype(int)

    if verbose:
        print(f"Crop limits are : [{zmin * voxdim[2]:.2f}, {zmax * voxdim[2]:.2f}] microns")
        print(f"Crop limits are : [{zmin}, {zmax}] pixels")

    # Cropping
    if return_z0:
        return crop_volume(vol, zlim=[zmin, zmax]), z0
    else:
        return crop_volume(vol, zlim=[zmin, zmax])


def mask_under_interface(vol: np.ndarray, interface: np.ndarray, return_mask: bool = False) -> np.ndarray:
    """Create a boolean mask for all voxels at or below the interface depth."""
    nx, ny, nz = vol.shape
    _, _, zz = np.meshgrid(list(range(nx)), list(range(ny)), list(range(nz)), indexing="ij")
    interface_3d = np.tile(np.reshape(interface, (nx, ny, 1)), (1, 1, nz))
    mask = zz >= interface_3d
    if return_mask:
        return mask
    else:
        return vol * mask


def apply_interface_correction(
    vol: np.ndarray, interface: np.ndarray
) -> np.ndarray:  # TODO: Test this algorithm to make sure it works well.
    """Apply interface depth correction using linear interpolation.

    Parameters
    ----------
    vol : ndarray
        Volume to fix.
    interface : ndarray
        Tissue interface depth.

    Returns
    -------
    ndarray
        Fixed volume.

    """
    nx, ny, nz = vol.shape
    z_range = np.around(interface.max() - interface.min())
    fixed_vol = np.zeros((nx, ny, nz - z_range), dtype=vol.dtype)

    # Loop over XY
    for x in range(nx):
        for y in range(ny):
            z = interface[x, y]
            real_z = np.linspace(-z, -z + nz, nz)
            new_z = list(range(int(nz - z_range)))
            z_interp = interp1d(real_z, vol[x, y, :], fill_value=0, bounds_error=False, kind="quadratic")
            fixed_vol[x, y, :] = z_interp(new_z)

    return fixed_vol


def crop_below_interface(
    vol_zxy: np.ndarray,
    depth_um: float,
    resolution_um: float,
    sigma_xy: float = 3.0,
    sigma_z: float = 2.0,
    crop_before_interface: bool = False,
    percentile_clip: float | None = None,
) -> tuple[np.ndarray, int]:
    """Crop an OME-Zarr volume to a specified depth below the tissue interface.

    Detects the water/tissue interface using gradient analysis, then crops
    the volume to retain only ``depth_um`` microns below the interface.

    Parameters
    ----------
    vol_zxy : np.ndarray
        Volume with shape (Z, X, Y) as returned by read_omezarr.
    depth_um : float
        Target depth below interface in microns.
    resolution_um : float
        Z resolution in microns per voxel.
    sigma_xy : float
        XY smoothing sigma for interface detection.
    sigma_z : float
        Z smoothing sigma for interface detection.
    crop_before_interface : bool
        If True, also crop the volume above the detected interface.
    percentile_clip : float or None
        If provided, clip values above this percentile before interface detection.

    Returns
    -------
    np.ndarray
        Cropped volume (Z', X, Y).
    int
        Detected interface depth in Z voxels.
    """
    from linumpy.geometry.interface import detect_interface_z

    vol_f = np.abs(vol_zxy) if np.iscomplexobj(vol_zxy) else np.asarray(vol_zxy, dtype=np.float32)

    vol_xyz = np.transpose(vol_f, (1, 2, 0))

    if percentile_clip is not None:
        vol_xyz = np.clip(vol_xyz, None, np.percentile(vol_xyz, percentile_clip))

    avg_iface = detect_interface_z(vol_xyz, sigma_xy=sigma_xy, sigma_z=sigma_z)

    depth_px = round(depth_um / resolution_um)
    surface_idx = max(0, min(avg_iface, vol_zxy.shape[0] - 1))
    end_idx = surface_idx + depth_px

    start_idx = surface_idx if crop_before_interface else 0
    vol_crop = vol_zxy[start_idx:end_idx, :, :]

    return vol_crop, avg_iface
