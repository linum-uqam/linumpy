# -*- coding:utf-8 -*-
import numpy as np


def samples_from_sigma(sigma):
    return np.arange(-int(np.ceil(sigma * 3)), int(np.ceil(sigma * 3)) + 1)


def gaussian(sigma):
    r = samples_from_sigma(sigma)
    ret = 1.0 / np.sqrt(2.0 * np.pi * sigma**2) * np.exp(-r**2 / 2.0 / sigma**2)
    ret = 1.0 / np.sqrt(np.sum(ret**2)) * ret
    return ret


def gaussian_derivative(sigma):
    r = samples_from_sigma(sigma)
    ret = -r / sigma**2 * gaussian(sigma)
    ret = 1.0 / np.sqrt(np.sum(ret**2)) * ret
    return ret


def make_xfilter(f):
    out = np.reshape(f, (-1, 1, 1))
    return out


def make_yfilter(f):
    out = np.reshape(f, (1, -1, 1))
    return out


def make_zfilter(f):
    out = np.reshape(f, (1, 1, -1))
    return out
