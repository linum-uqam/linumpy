"""Vignette estimation models."""

import numpy as np
from scipy.optimize import minimize


def vignette_gauss(pos: list, x0: float, y0: float, sx: float, sy: float, a: float, b: float) -> np.ndarray:
    """Evaluate a 2D Gaussian vignette model."""
    return np.exp(-((pos[0] - x0) ** 2) / (2.0 * sx**2.0) - (pos[1] - y0) ** 2 / (2.0 * sy**2.0)) * a + b


def vignette_gauss_lin(pos: list, x0: float, y0: float, s: float, a: float, b: float, c: float) -> np.ndarray:
    """Evaluate a Gaussian-linear mixed vignette model."""
    gauss_surf = np.exp(-((pos[0] - x0) ** 2) / (2.0 * s**2.0) - (pos[1] - y0) ** 2 / (2.0 * s**2.0))
    lin_surf = pos[0] * a + pos[1] * b + c
    return gauss_surf * lin_surf


def vignette_quad(pos: list, a: float, b: float, c: float, d: float, e: float, f: float) -> np.ndarray:
    """Evaluate a quadratic vignette model."""
    x = pos[0]
    y = pos[1]
    return a * x + b * y + c * x * y + d * x**2 + e * y**2 + f


def get_vignette(
    vol: np.ndarray, return_params: bool = False, mask_z: np.ndarray | None = None, method: str = "gauss"
) -> dict | np.ndarray:
    """Estimate the vignette correction field from a volume."""
    if method == "gauss":

        def f_opt(x: np.ndarray, y: np.ndarray, pos: list) -> float:
            return np.mean((y - vignette_gauss(pos, x[0], x[2], x[1], x[3], x[4], x[5])) ** 2.0)
    elif method == "gauss_lin":

        def f_opt(x: np.ndarray, y: np.ndarray, pos: list) -> float:
            return np.mean((y - vignette_gauss_lin(pos, x[0], x[1], x[2], x[3], x[4], x[5])) ** 2.0)
    else:

        def f_opt(x: np.ndarray, y: np.ndarray, pos: list) -> float:
            return np.mean((y - vignette_quad(pos, x[0], x[1], x[2], x[3], x[4], x[5])) ** 2.0)

    # Computing position in this mosaic.
    xx, yy = np.meshgrid(
        np.linspace(-1, 1, vol.shape[0]),
        np.linspace(-1, 1, vol.shape[1]),
        indexing="ij",
    )
    pos = [xx, yy]

    # Find center before
    img = vol.mean(axis=2)
    img /= float(img.max())
    if method == "gauss":
        p0 = [0.0, 0.5, 0.0, 0.5, 1.0, 0.0]
    elif method == "gauss_lin":
        p0 = [0.0, 0.0, 0.5, 0.0, 0.0, 1.0]
    else:
        p0 = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    popt_0 = minimize(f_opt, p0, args=(img, pos))

    print(popt_0)

    w_list = []
    params_list = []
    if mask_z is None:
        mask_z = np.ones((vol.shape[2],))
    for z in range(vol.shape[2]):
        if mask_z[z]:
            p0 = popt_0.x
            img = vol[:, :, z]
            img /= float(img.max())

            popt = minimize(f_opt, p0, args=(img, pos))

            params_list.append(popt.x)
            w_list.append(f_opt(popt.x, img, pos))

    optimized_vignette_params = np.median(np.array(params_list), axis=0)
    print(np.array(params_list))
    print(optimized_vignette_params)

    if return_params:
        output = {"method": method, "params": optimized_vignette_params}
        return output
    else:
        if method == "gauss":
            x0, sx, y0, sy, a, b = optimized_vignette_params
            vignette = vignette_gauss(pos, x0, y0, sx, sy, 1, 0)
        elif method == "gauss_lin":
            x0, y0, s, a, b, c = optimized_vignette_params
            vignette = vignette_gauss_lin(pos, x0, y0, s, a, b, c)
        else:
            a, b, c, d, e, f = optimized_vignette_params
            vignette = vignette_quad(pos, a, b, c, d, e, f)
        return vignette
