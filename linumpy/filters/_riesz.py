import numpy as np
from scipy.fftpack import fftshift, ifftshift, fftfreq, fftn, ifftn, fft2, ifft2
from scipy.ndimage import gaussian_filter


def get_riesz_kernels(shape=(512, 512)):
    """Generate ND Riesz kernels

    Parameters
    ----------
    shape : tuple
        Kernel shape (example : (NxM)). The number of elements is the number of dimensions K.
    Returns
    -------
    riesz_kernels : ndarray
        Returns the Riesz kernel in K dimensions (example : (NxMxK)

    """

    n_dims = len(shape)  # Number of dimensions

    # Frequency components
    w_list = []
    for dim in range(n_dims):
        w_list.append(fftshift(fftfreq(shape[dim])))
    w = np.stack(np.meshgrid(*w_list, indexing="ij"), axis=n_dims)
    w_norm = np.sqrt((w ** 2).sum(axis=n_dims, keepdims=True))

    # Computing the Riesz kernels
    with np.errstate(divide='ignore', invalid='ignore'):
        riesz_kernels = - 1j * w / w_norm

    # Replacing the singularity by zero
    riesz_kernels[np.isnan(riesz_kernels)] = 0

    # FIXME : Clipping to deal with the singularity (validate this)
    riesz_kernels = np.clip(riesz_kernels, -1j, 1j)
    return riesz_kernels


def get_higher_order_riesz(riesz_kernels, order):
    """Computes the nth order Riesz kernel

    Parameters
    ----------
    riesz_kernels : ndarray
        Zero-th order Riesz kernel of shape (NxMx2) in 2D
    order : tuple
        Riesz kernel order along each dimension (ex : (0,1) for R01)
    Returns
    -------
    riesz_kernels : ndarray
        Returns the Riesz kernel in K dimensions (example : (NxMxK)
    """

    kernel = riesz_kernels ** np.array(order)
    kernel = np.prod(kernel, axis=-1)
    return kernel


def riesz(img, order):
    """Apply the kth order N-dimensional Riesz transform

    Parameters
    ----------
    img : ndarray
        Grayscale input array to process; can be 2D or 3D.
    order : tuple
        Riesz kernel order along each dimension (ex : (0,1) for R01)
    Returns
    -------
    output : ndarray (float32)
        The Riesz filter response.

    Notes
    -----
    * The Riesz transform is applied in the Fourier space.
    * Only the real part of the filter response is returned

    """

    # Get Riesz kernels
    kernels = get_riesz_kernels(img.shape)

    # Generate the order tuple
    r = get_higher_order_riesz(kernels, order)

    # Apply the Riesz transform
    img_fft = fftshift(fftn(img))
    output = ifftn(ifftshift(r * img_fft)).real.astype(np.float32)

    return output


def riesz_orientation_structure_tensor_2d(img, sigma=3):
    """Extract the local orientation with the Riesz-based 2D structure tensor
    Parameters
    ----------
    img : ndarray
        2D image to process
    sigma : int
        Gaussian neighborhood size
    Returns
    -------
    (2,) tuple
        Contains the local amplitude and angle as ndarrays
    """
    # Normalizing the input image
    img = (img.astype(np.float32) - img.min()) / (img.max() - img.min())

    # Compute the image Fourier transform (and normalizing the input)
    img_fft = fftshift(fft2((img)))

    # Generate the Riesz kernels
    rieszKernels = get_riesz_kernels(img.shape)

    # Generate the 1st derivatives
    r10 = get_higher_order_riesz(rieszKernels, (1, 0))
    r01 = get_higher_order_riesz(rieszKernels, (0, 1))

    # Compute the structure tensor elements
    j11 = ifft2(ifftshift(r10 * img_fft)).real ** 2
    j12 = ifft2(ifftshift(r10 * img_fft)).real * ifft2(ifftshift(r01 * img_fft)).real
    j22 = ifft2(ifftshift(r01 * img_fft)).real ** 2

    # Local averaging
    j11 = gaussian_filter(j11, sigma)
    j12 = gaussian_filter(j12, sigma)
    j22 = gaussian_filter(j22, sigma)

    # Compute the amplitude and orientation
    amplitude = j11 + j22
    orientation = np.stack([j22 - j11, 2 * j12], axis=-1)
    angle = (np.arctan2(orientation[..., 1], orientation[..., 0]) + np.pi) / 2 + np.pi / 2
    angle[angle > np.pi] -= np.pi

    return (amplitude, angle)


def riesz_orientation_hessian_2d(img):
    """Extract the local orientation with the Riesz-based 2D Hessian matrix
    Parameters
    ----------
    img : ndarray
        2D image to process
    Returns
    -------
    (2,) tuple
        Contains the local amplitude and angle as ndarrays
    """
    # Normalizing the input image
    img = (img.astype(np.float32) - img.min()) / (img.max() - img.min())

    # Compute the image Fourier transform (and normalizing the input)
    img_fft = fftshift(fft2((img)))

    # Generate the Riesz kernels
    rieszKernels = get_riesz_kernels(img.shape)

    # Generate the 2nd derivatives
    r11 = get_higher_order_riesz(rieszKernels, (1, 1))
    r20 = get_higher_order_riesz(rieszKernels, (2, 0))
    r02 = get_higher_order_riesz(rieszKernels, (0, 2))

    # Compute the structure tensor elements
    h11 = ifft2(ifftshift(r20 * img_fft)).real
    h12 = ifft2(ifftshift(r11 * img_fft)).real
    h22 = ifft2(ifftshift(r02 * img_fft)).real

    # Compute the amplitude and orientation
    amplitude = h11 + h22
    orientation = np.stack([h22 - h11, 2 * h12], axis=-1)
    angle = (np.arctan2(orientation[..., 1], orientation[..., 0]) + np.pi) / 2

    return (amplitude, angle)


def riesz_orientation_tkeo_2d(img):
    """Extract the local orientation with the Riesz-based 2D TKEO
    Parameters
    ----------
    img : ndarray
        2D image to process
    Returns
    -------
    (2,) tuple
        Contains the local energy and angle as ndarrays
    """
    # Normalizing the input image
    img = (img.astype(np.float32) - img.min()) / (img.max() - img.min())

    # Compute the image Fourier transform (and normalizing the input)
    img_fft = fftshift(fft2(img))

    # Generate the Riesz kernels
    rieszKernels = get_riesz_kernels(img.shape)

    # Generate the 2nd derivatives
    r10 = get_higher_order_riesz(rieszKernels, (1, 0))
    r01 = get_higher_order_riesz(rieszKernels, (0, 1))
    r11 = get_higher_order_riesz(rieszKernels, (1, 1))
    r20 = get_higher_order_riesz(rieszKernels, (2, 0))
    r02 = get_higher_order_riesz(rieszKernels, (0, 2))

    # Apply the Riesz transforms
    f10 = ifft2(ifftshift(r10 * img_fft)).real
    f01 = ifft2(ifftshift(r01 * img_fft)).real
    f11 = ifft2(ifftshift(r11 * img_fft)).real
    f20 = ifft2(ifftshift(r20 * img_fft)).real
    f02 = ifft2(ifftshift(r02 * img_fft)).real

    # Compute the energy and orientation
    energy = (f10 ** 2 - f01 ** 2 - img * (f20 - f02)) + 2j * (f10 * f01 - f11 * img)
    amplitude = np.abs(energy)
    angle = -(np.angle(energy) + np.pi) / 2 + np.pi

    return (amplitude, angle)


def riesz_orientation_structure_tensor_3d(vol, sigma=1):
    """Extract the local orientation with the Riesz-based 3D structure tensor
    Parameters
    ----------
    img : ndarray
        3D volume to process
    sigma : int
        Gaussian neighborhood size
    Returns
    -------
    (2,) tuple
        Contains the fractional anisotropy and local orientation as ndarrays
    """
    # Calcul des noyaux de Riesz
    rieszKernels = get_riesz_kernels(shape=vol.shape)
    R100 = get_higher_order_riesz(rieszKernels, (1, 0, 0))
    R010 = get_higher_order_riesz(rieszKernels, (0, 1, 0))
    R001 = get_higher_order_riesz(rieszKernels, (0, 0, 1))

    # Application des filtres de Riesz
    vol_fft = fftshift(fftn(vol))
    f100 = ifftn(ifftshift(R100 * vol_fft)).real
    f010 = ifftn(ifftshift(R010 * vol_fft)).real
    f001 = ifftn(ifftshift(R001 * vol_fft)).real

    # Création du tenseur de structure 3D
    tensor = np.zeros((*vol.shape, 3, 3))
    tensor[..., 0, 0] = gaussian_filter(f100 ** 2, sigma)
    tensor[..., 1, 1] = gaussian_filter(f010 ** 2, sigma)
    tensor[..., 2, 2] = gaussian_filter(f001 ** 2, sigma)
    tensor[..., 0, 1] = gaussian_filter(f100 * f010, sigma)
    tensor[..., 1, 0] = tensor[..., 0, 1]
    tensor[..., 0, 2] = gaussian_filter(f100 * f001, sigma)
    tensor[..., 2, 0] = tensor[..., 1, 2]
    tensor[..., 1, 2] = gaussian_filter(f010 * f001, sigma)
    tensor[..., 2, 1] = tensor[..., 1, 2]

    # Eigendecomposition
    eig_values, eig_vectors = np.linalg.eig(tensor)
    eig_values = eig_values.real

    # Principal orientation is associated with the min eigval
    nx, ny, nz = vol.shape
    idx = np.argmin(np.abs(eig_values), axis=-1)
    orientation = np.zeros((*vol.shape, 3), dtype=np.float32)

    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                this_idx = idx[x, y, z]
                orientation[x, y, z, :] = eig_vectors[x, y, z, :, this_idx].real

    # Compute fractional anisotropy FA
    eig_values.sort(axis=3)  # Sorting the eigen values from min to max

    # Computing FA (see khan2015)
    t1 = eig_values[..., 2].real  # largest
    t2 = eig_values[..., 1].real  # middle
    t3 = eig_values[..., 0].real  # small
    p1 = (t1 - t2) ** 2  # max - mid
    p2 = (t2 - t3) ** 2  # mid - min
    p3 = (t3 - t1) ** 2

    fa = np.sqrt(0.5 * (p1 + p2 + p3) / (t1 ** 2 + t2 ** 2 + t3 ** 2))
    fa[np.isnan(fa)] = 0
    fa = fa.real.astype(np.float32)

    return (fa, orientation)


def riesz_orientation_hessian_3d(vol):
    """Extract the local orientation with the Riesz-based 3D Hessian matrix
    Parameters
    ----------
    vol : ndarray
        3D volume to process
    Returns
    -------
    (2,) tuple
        Contains the fractional anisotropy and local orientation as ndarrays
    """
    # Calcul des noyaux de Riesz
    rieszKernels = get_riesz_kernels(shape=vol.shape)
    r200 = get_higher_order_riesz(rieszKernels, (2, 0, 0))
    r020 = get_higher_order_riesz(rieszKernels, (0, 2, 0))
    r002 = get_higher_order_riesz(rieszKernels, (0, 0, 2))
    r110 = get_higher_order_riesz(rieszKernels, (1, 1, 0))
    r101 = get_higher_order_riesz(rieszKernels, (1, 0, 1))
    r011 = get_higher_order_riesz(rieszKernels, (0, 1, 1))

    # Transformation de Riesz
    vol_fft = fftshift(fftn(vol))
    f200 = ifftn(ifftshift(r200 * vol_fft)).real
    f020 = ifftn(ifftshift(r020 * vol_fft)).real
    f002 = ifftn(ifftshift(r002 * vol_fft)).real
    f110 = ifftn(ifftshift(r110 * vol_fft)).real
    f101 = ifftn(ifftshift(r101 * vol_fft)).real
    f011 = ifftn(ifftshift(r011 * vol_fft)).real

    # Création de la matrice Hessienne 3D
    hessian = np.zeros((*vol.shape, 3, 3))
    hessian[..., 0, 0] = f200
    hessian[..., 1, 1] = f020
    hessian[..., 2, 2] = f002
    hessian[..., 0, 1] = f110
    hessian[..., 1, 0] = f110
    hessian[..., 0, 2] = f101
    hessian[..., 2, 0] = f101
    hessian[..., 1, 2] = f011
    hessian[..., 2, 1] = f011

    # Eigendecomposition
    eig_values, eig_vectors = np.linalg.eig(hessian)
    eig_values = eig_values.real

    # Principal orientation is associated with the min eigval
    nx, ny, nz = vol.shape
    idx = np.argmin(np.abs(eig_values), axis=-1)
    orientation = np.zeros((*vol.shape, 3), dtype=np.float32)

    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                this_idx = idx[x, y, z]
                orientation[x, y, z, :] = eig_vectors[x, y, z, :, this_idx].real

    # Compute fractional anisotropy FA
    eig_values.sort(axis=3)  # Sorting the eigen values from min to max

    # Computing FA (see khan2015)
    t1 = eig_values[..., 2].real
    t2 = eig_values[..., 1].real
    t3 = eig_values[..., 0].real
    p1 = (t1 - t2) ** 2  # max - mid
    p2 = (t2 - t3) ** 2  # mid - min
    p3 = (t3 - t1) ** 2

    fa = np.sqrt(0.5 * (p1 + p2 + p3) / (t1 ** 2 + t2 ** 2 + t3 ** 2))
    fa[np.isnan(fa)] = 0
    fa = fa.real.astype(np.float32)

    return (fa, orientation)


def riesz_orientation_tkeo_3d(vol):
    """Extract the local orientation with the Riesz-based 3D TKEO
    Parameters
    ----------
    img : ndarray
        3D volume to process
    Returns
    -------
    (2,) tuple
        Contains the local energy and orientation as ndarrays
    """
    # Computing the first and second derivatives
    rieszKernels = get_riesz_kernels(shape=vol.shape)
    r100 = get_higher_order_riesz(rieszKernels, (1, 0, 0))
    r010 = get_higher_order_riesz(rieszKernels, (0, 1, 0))
    r001 = get_higher_order_riesz(rieszKernels, (0, 0, 1))
    r200 = get_higher_order_riesz(rieszKernels, (2, 0, 0))
    r020 = get_higher_order_riesz(rieszKernels, (0, 2, 0))
    r002 = get_higher_order_riesz(rieszKernels, (0, 0, 2))
    r110 = get_higher_order_riesz(rieszKernels, (1, 1, 0))
    r101 = get_higher_order_riesz(rieszKernels, (1, 0, 1))
    # r011 = getHigherOrderRiesz(rieszKernels, (0, 1, 1))

    # Riesz transforms
    vol_fft = fftshift(fftn(vol))
    f100 = ifftn(ifftshift(r100 * vol_fft)).real
    f010 = ifftn(ifftshift(r010 * vol_fft)).real
    f001 = ifftn(ifftshift(r001 * vol_fft)).real
    f110 = ifftn(ifftshift(r110 * vol_fft)).real
    f101 = ifftn(ifftshift(r101 * vol_fft)).real
    # f011 = ifftn(ifftshift(r011 * vol_fft)).real
    f200 = ifftn(ifftshift(r200 * vol_fft)).real
    f020 = ifftn(ifftshift(r020 * vol_fft)).real
    f002 = ifftn(ifftshift(r002 * vol_fft)).real

    # Création TKEO multi-dimensionnel (quaternion)
    q0 = f100 ** 2 - vol * f200 - (f010 ** 2 - vol * f020 + f001 ** 2 - vol * f002)
    q1 = 2 * (f100 * f010 - vol * f110)  # Il manque un nombre imaginaire i
    q2 = 2 * (f100 * f001 - vol * f101)  # Il manque un nombre imaginaire j

    # Computing the energy / orientation
    energy = np.sqrt(q0 ** 2 + q1 ** 2 + q2 ** 2)
    orientation = np.stack([q0 / energy, q1 / energy, q2 / energy], axis=3)

    return energy, orientation
