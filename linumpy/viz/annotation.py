import numpy as np
from linumpy import util


def merge_channel(gray, red=None, green=None, blue=None, ratio=0.5):
    '''Merge channels for image or volume (normalized between 0 and 1)
    Parameters
    ----------
    gray : ndarray
        gray channel
    red : ndarray
        red channel
    green : ndarray
        green channel
    blue : ndarray
        blue channel
    ratio : float
        ratio for gray channel intensity
    Returns
    -------
    merged_channel : ndarray
        stack of red, green and blue channel
    '''
    if red is None and green is None and blue is None:
        return util.normalize(gray, 1)  # weirdo...

    gray_ch = util.normalize(gray, ratio)
    red_ch = gray_ch.copy()
    green_ch = gray_ch.copy()
    blue_ch = gray_ch.copy()

    if red is not None:
        red_ch = red_ch + util.normalize(red, 1 - ratio)

    if green is not None:
        green_ch = green_ch + util.normalize(green, 1 - ratio)

    if blue is not None:
        blue_ch = red_ch + util.normalize(blue, 1 - ratio)

    return np.stack([red_ch, green_ch, blue_ch], axis=gray.ndim)
