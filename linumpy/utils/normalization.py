import numpy as np


def normalize(x, n_max=255, n_min=0):
    """Normalize array-like values
    Parameters
    ----------
    x : list, ndarray
        array-like to normalize
    n_max : int, float
        new max value
    n_min : int, float
        new min value
    Returns
    -------
    normalized : ndarray
        ndarray of normalized x input
    """
    x_c = np.array(x).astype(float)

    return ((x_c - x_c.min()) * (n_max - n_min) / (x_c.max() - x_c.min())) + n_min
