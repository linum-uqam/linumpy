"""Gaussian based 2D steerable filters based on Freeman1991"""

import numpy as np
from functools import partial
from scipy.ndimage import convolve
from tqdm import tqdm

__all__ = ["steerable_gaussian_2d",
           "steerable_hilbert_gaussian_2d",
           "steerable_oriented_energy_2d"]


def _generic_interpolation_function(theta, A=1, n=1, m=1):
    """Generic interpolation function used for the steerable filters

    Parameters
    ----------
    theta : float
        Angle in radian
    A : float
        Multiplicative normalization constant
    n : int
        Cosinus power
    m : int
        Sinus power

    Returns
    -------
    float
        The evaluated function

    Notes
    -----
    - The generic interpolation function implements

    f = A * cos^n(theta) * sin^n(theta)

    References
    ----------
    - Adapted from Freeman, W. T., & Adelson, E. H. (1991). The design and use of steerable filters.
    IEEE Transactions on Pattern Analysis and Machine Intelligence, 13(9), 891‑906. https://doi.org/10.1109/34.93808

    """

    return A * np.cos(theta) ** n * np.sin(theta) ** m


def _get_gaussian_basis(order=2, size=21, range=3):
    assert order in [1, 2, 4], "Order must be in [1,2,4]"

    x, y = np.meshgrid(np.linspace(-range, range, size),
                       np.linspace(-range, range, size),
                       indexing="ij")

    basis_set = []
    interp_functions = []
    G = np.exp(-(x ** 2 + y ** 2))
    if order == 1:
        # 1st derivative of Gaussian
        G1a = -2 * x * G
        G1b = -2 * y * G
        basis_set = [G1a, G1b]

        # Associated interpolation functions
        ka = partial(_generic_interpolation_function, A=1, n=1, m=0)
        kb = partial(_generic_interpolation_function, A=1, n=0, m=1)
        interp_functions = [ka, kb]

    elif order == 2:
        # 2nd derivative of Gaussian
        G2a = 0.9213 * (2 * x ** 2 - 1) * G
        G2b = 1.8430 * x * y * G
        G2c = 0.9213 * (2 * y ** 2 - 1) * G
        basis_set = [G2a, G2b, G2c]

        # Associated interpolation functions
        ka = partial(_generic_interpolation_function, A=1, n=2, m=0)
        kb = partial(_generic_interpolation_function, A=-2, n=1, m=1)
        kc = partial(_generic_interpolation_function, A=1, n=0, m=2)
        interp_functions = [ka, kb, kc]

    elif order == 4:
        # 4th derivative of Gaussian
        G4a = 1.246 * (0.75 - 3 * x ** 2 + x ** 4) * G
        G4b = 1.246 * (-1.5 * x + x ** 3) * y * G
        G4c = 1.246 * (x ** 2 - 0.5) * (y ** 2 - 0.5) * G
        G4d = 1.246 * (-1.5 * y + y ** 3) * x * G
        G4e = 1.246 * (0.75 - 3 * y ** 2 + y ** 4) * G
        basis_set = [G4a, G4b, G4c, G4d, G4e]

        # Associated interpolation functions
        ka = partial(_generic_interpolation_function, A=1, n=4, m=0)
        kb = partial(_generic_interpolation_function, A=-4, n=3, m=1)
        kc = partial(_generic_interpolation_function, A=6, n=2, m=2)
        kd = partial(_generic_interpolation_function, A=-4, n=1, m=3)
        ke = partial(_generic_interpolation_function, A=1, n=0, m=4)
        interp_functions = [ka, kb, kc, kd, ke]

    return (basis_set, interp_functions)


def _get_hilbert_gaussian_basis(order=2, size=21, range=3):
    assert order in [2, 4], "Order must be in [2, 4]"

    x, y = np.meshgrid(np.linspace(-range, range, size),
                       np.linspace(-range, range, size),
                       indexing="ij")

    basis_set = []
    interp_functions = []
    G = np.exp(-(x ** 2 + y ** 2))
    if order == 2:
        # Hilbert transform of the 2nd derivative of Gaussian
        H2a = 0.9780 * (-2.254 * x + x ** 3) * G
        H2b = 0.9780 * (-0.7515 + x ** 2) * y * G
        H2c = 0.9780 * (-0.7515 + y ** 2) * x * G
        H2d = 0.9780 * (-2.254 * y + y ** 3) * G
        basis_set = [H2a, H2b, H2c, H2d]

        # Associated interpolation functions
        ka = partial(_generic_interpolation_function, A=1, n=3, m=0)
        kb = partial(_generic_interpolation_function, A=-3, n=2, m=1)
        kc = partial(_generic_interpolation_function, A=3, n=1, m=2)
        kd = partial(_generic_interpolation_function, A=-1, n=0, m=3)
        interp_functions = [ka, kb, kc, kd]

    elif order == 4:
        # 4th derivative of Gaussian
        H4a = 0.3975 * (7.189 * x - 7.501 * x ** 3 + x ** 5) * G
        H4b = 0.3975 * (1.438 - 4.501 * x ** 2 + x ** 4) * y * G
        H4c = 0.3975 * (x ** 3 - 2.225 * x) * (y ** 2 - 0.6638) * G
        H4d = 0.3975 * (y ** 3 - 2.225 * y) * (x ** 2 - 0.6638) * G
        H4e = 0.3975 * (1.438 - 4.501 * y ** 2 + y ** 4) * x * G
        H4f = 0.3975 * (7.189 * y - 7.501 * y ** 3 + y ** 5) * G
        basis_set = [H4a, H4b, H4c, H4d, H4e, H4f]

        # Associated interpolation functions
        ka = partial(_generic_interpolation_function, A=1, n=5, m=0)
        kb = partial(_generic_interpolation_function, A=-5, n=4, m=1)
        kc = partial(_generic_interpolation_function, A=10, n=3, m=2)
        kd = partial(_generic_interpolation_function, A=-10, n=2, m=3)
        ke = partial(_generic_interpolation_function, A=5, n=1, m=4)
        kf = partial(_generic_interpolation_function, A=-1, n=0, m=5)
        interp_functions = [ka, kb, kc, kd, ke, kf]

    return (basis_set, interp_functions)


def steerable_gaussian_2d(img, thetas, size=9, order=2):
    """Steerable 2D Gaussian derivatives

    Parameters
    ----------
    img : ndarray
        Image to filter. Must be of type np.float32 and normalized between 0.0 and 1.0
    thetas : list of floats
        The steering angles in radians between 0 and pi
    size : int
        Kernel size
    order : int
        The gaussian derivative order (must be 1, 2 or 4)

    Returns
    -------
    output : ndarray
        The filter's response of shape (KxNxM), where NxM is the shape of the original image,
        and K is the number of angles in thetas

    """
    # Get the bases and interpolation functions
    G, K = _get_gaussian_basis(order=order, size=size)

    # Apply the basis convolutions
    R = []
    for g in G:
        R.append(convolve(img, g))

    # Prepare the output
    n_angles = len(thetas)
    output = np.zeros((n_angles, *img.shape), dtype=np.float32)
    for i, t in enumerate(thetas):
        for r, k in zip(R, K):
            output[i, ...] += k(t) * r

    return output


def steerable_hilbert_gaussian_2d(img, thetas, size=9, order=2):
    """Steerable 2D Hilbert transform of Gaussian derivatives

    Parameters
    ----------
    img : ndarray
        Image to filter. Must be of type np.float32 and normalized between 0.0 and 1.0
    thetas : list of floats
        The steering angles in radians between 0 and pi
    size : int
        Kernel size
    order : int
        The gaussian derivative order (must be 2 or 4)

    Returns
    -------
    output : ndarray
        The filter's response of shape (KxNxM), where NxM is the shape of the original image,
        and K is the number of angles in thetas

    """
    # Get the bases and interpolation functions
    h_basis, k_functions = _get_hilbert_gaussian_basis(order=order, size=size)

    # Apply the basis convolutions
    R = []
    for g in h_basis:
        R.append(convolve(img, g))

    # Prepare the output
    n_angles = len(thetas)
    output = np.zeros((n_angles, *img.shape), dtype=np.float32)
    for i, t in enumerate(thetas):
        for r, k in zip(R, k_functions):
            output[i, ...] += k(t) * r

    return output


def steerable_oriented_energy_2d(img, thetas, size=9, order=2):
    """Computer the 2D oriented energy using Gaussian-based steerable filters

    Parameters
    ----------
    img : ndarray of shape NxM
        Image to filter. Must be of type np.float32 and normalized between 0.0 and 1.0
    thetas : list of floats of length K
        The steering angles in radians between 0 and pi
    size : int
        Kernel size
    order : int
        The gaussian derivative order (must be 2 or 4)

    Returns
    -------
    energy : ndarray of shape KxNxM
        The steerable filter's oriented energy along each directions given in `thetas`

    Notes ----- - The 2D oriented energy is similar to a local orientation distribution function (ODF) - The oriented
    energy is defined as E_n(theta) = G_n(theta)^2 + H_n(theta)^2, where G_n and H_n are the filter responses for the
    steerable nth-order derivative of the Gaussian and the steerable Hilbert transform of the nth-order derivative of
    the Gaussian, respectively.

    References
    ----------
    - Adapted from Freeman, W. T., & Adelson, E. H. (1991). The design and use of steerable filters.
    IEEE Transactions on Pattern Analysis and Machine Intelligence, 13(9), 891‑906. https://doi.org/10.1109/34.93808

    """

    G = steerable_gaussian_2d(img, thetas=thetas, order=order, size=size)
    H = steerable_hilbert_gaussian_2d(img, thetas=thetas, order=order, size=size)
    energy = G ** 2 + H ** 2
    return energy


def steerable_gaussian_3d(img, dirs, size):
    """"Steerable 3D Gaussian 2nd derivative
    Parameters
    ----------
    img : ndarray
        Volume to filter. Must be of type np.float32 and normalized between 0.0 and 1.0
    dirs : list of (3,) floats
        The steering directions expressed as cosine directions
    size : int
        Kernel size

    Returns
    -------
    output : ndarray
        The filter's response of shape (NxMxPxK), where NxMxP is the shape of the original image,
        and K is the number of cosine directions in dirs

    References
    ----------
    - Adapted from: Derpanis, K. G., Gryn, J. M. (2005). Three-dimensional nth derivative of Gaussian separable
    steerable filters. IEEE International Conference on Image Processing 2005, Genova, Italy, 2005, p. III–553.
    https://doi.org/10.1109/ICIP.2005.1530451

    """

    # TODO: First implementation, naive 3D convolutions.
    if len(dirs) == 3:
        dirs = [dirs]

    # 3D Basis functions
    x, y, z = np.meshgrid(
        np.linspace(-3, 3, size),  # x
        np.linspace(-3, 3, size),  # y
        np.linspace(-3, 3, size),  # z
        indexing="ij"
    )
    N = 2 / np.sqrt(3) * (2 / np.pi) ** (3 / 4)
    gaussian = np.exp(-(x ** 2 + y ** 2 + z ** 2))
    kernels = []
    kernels.append(N * (2 * x ** 2 - 1) * gaussian)
    kernels.append(N * (2 * x * y) * gaussian)
    kernels.append(N * (2 * y ** 2 - 1) * gaussian)
    kernels.append(N * (2 * x * z) * gaussian)
    kernels.append(N * (2 * y * z) * gaussian)
    kernels.append(N * (2 * z ** 2 - 1) * gaussian)

    # Get the filter's response
    responses = []
    for kernel in kernels:
        responses.append(convolve(img, kernel))

    # Compute the interpolation function a given orientation
    interps = []
    for d in dirs:
        a, b, c = d[:]
        foo = []
        foo.append(a ** 2)
        foo.append(2 * a * b)
        foo.append(b ** 2)
        foo.append(2 * a * c)
        foo.append(2 * b * c)
        foo.append(c ** 2)
        interps.append(foo)

    outputs = np.zeros((*img.shape, len(dirs)))
    for j, d in tqdm(enumerate(dirs), desc="Steering 3D Gaussians"):
        output = None
        for r, i in zip(responses, interps[j]):

            foo = i * r
            if output is None:
                output = foo
            else:
                output += foo

        outputs[..., j] = output

    return outputs


def steerable_hilbert_gaussian_3d(img, dirs, size):
    """"Steerable 3D Hilbert transform of the Gaussian 2nd derivative
    Parameters
    ----------
    img : ndarray
        Volume to filter. Must be of type np.float32 and normalized between 0.0 and 1.0
    dirs : list of (3,) floats
        The steering directions expressed as cosine directions
    size : int
        Kernel size

    Returns
    -------
    output : ndarray
        The filter's response of shape (NxMxPxK), where NxMxP is the shape of the original image,
        and K is the number of cosine directions in dirs

    References
    ----------
    - Adapted from: Derpanis, K. G., Gryn, J. M. (2005). Three-dimensional nth derivative of Gaussian separable
    steerable filters. IEEE International Conference on Image Processing 2005, Genova, Italy, 2005, p. III–553.
    https://doi.org/10.1109/ICIP.2005.1530451

    """

    # TODO: First implementation, naive 3D convolutions.
    if len(dirs) == 3:
        dirs = [dirs]

    # 3D Basis functions
    x, y, z = np.meshgrid(
        np.linspace(-3, 3, size),  # x
        np.linspace(-3, 3, size),  # y
        np.linspace(-3, 3, size),  # z
        indexing="ij"
    )
    N = 0.877776

    gaussian = np.exp(-(x ** 2 + y ** 2 + z ** 2))
    kernels = []
    kernels.append(N * (x ** 3 - 2.254 * x) * gaussian)
    kernels.append(N * y * (x ** 2 - 0.751333) * gaussian)
    kernels.append(N * x * (y ** 2 - 0.751333) * gaussian)
    kernels.append(N * (y ** 3 - 2.254 * y) * gaussian)
    kernels.append(N * z * (x ** 2 - 0.751333) * gaussian)
    kernels.append(N * x * y * z * gaussian)
    kernels.append(N * z * (y ** 2 - 0.751333) * gaussian)
    kernels.append(N * x * (z ** 2 - 0.751333) * gaussian)
    kernels.append(N * y * (z ** 2 - 0.751333) * gaussian)
    kernels.append(N * (z ** 3 - 2.254 * z) * gaussian)

    # Get the filter's response
    responses = []
    for kernel in kernels:
        responses.append(convolve(img, kernel))

    # Compute the interpolation function a given orientation
    interps = []
    for d in dirs:
        a, b, c = d[:]
        foo = []
        foo.append(a ** 3)
        foo.append(3 * a ** 2 * b)
        foo.append(3 * a * b ** 2)
        foo.append(b ** 3)
        foo.append(3 * a ** 2 * c)
        foo.append(6 * a * b * c)
        foo.append(3 * b ** 2 * c)
        foo.append(3 * a * c ** 2)
        foo.append(3 * b * c ** 2)
        foo.append(c ** 3)
        interps.append(foo)

    outputs = np.zeros((*img.shape, len(dirs)))
    for j, d in tqdm(enumerate(dirs), desc="Steering 3D Hilbert of Gaussians"):
        output = None
        for r, i in zip(responses, interps[j]):

            foo = i * r
            if output is None:
                output = foo
            else:
                output += foo

        outputs[..., j] = output

    return outputs


def steerable_oriented_energy_3d(img, dirs, size=9):
    """Computer the 3D oriented energy using Gaussian-based steerable filters of the 2nd order

    Parameters
    ----------
    img : ndarray of shape NxMxP
        Image to filter. Must be of type np.float32 and normalized between 0.0 and 1.0
    dirs : list of (3,) floats
        The steering directions expressed as cosine directions
    size : int
        Kernel size

    Returns
    -------
    energy : ndarray of shape NxMxPxK
        The steerable filter's oriented energy along each directions given in `thetas`

    Notes
    -----
    - The 3D oriented energy is similar to a local orientation distribution function (ODF)
    - The oriented energy is defined as E_n(theta) = G_n(theta)^2 + H_n(theta)^2, where G_n and H_n are the filter
    responses for the steerable nth-order derivative of the Gaussian and the steerable Hilbert transform of the
    nth-order derivative of the Gaussian, respectively.

    References
    ----------
    - Adapted from Freeman, W. T., & Adelson, E. H. (1991). The design and use of steerable filters.
    IEEE Transactions on Pattern Analysis and Machine Intelligence, 13(9), 891‑906. https://doi.org/10.1109/34.93808

    """

    G = steerable_gaussian_3d(img, dirs=dirs, size=size)
    H = steerable_hilbert_gaussian_3d(img, dirs=dirs, size=size)
    energy = G ** 2 + H ** 2
    return energy
