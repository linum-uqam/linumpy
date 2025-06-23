# -*- coding:utf8 -*-
import logging
from functools import partial
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed

from dipy.data import get_sphere
from dipy.reconst.shm import sh_to_sf_matrix, sph_harm_ind_list
from scipy.special import eval_legendre
from scipy.ndimage import correlate
import numpy as np
from tqdm import tqdm
import zarr


G_RESPONSE_KEY = "G_RESPONSE"
H_RESPONSE_KEY = "H_RESPONSE"
Q_RESPONSE_KEY = "Q_RESPONSE"
G_FILTERED_BASE_KEY = "G"
H_FILTERED_BASE_KEY = "H"
ODF_SH_KEY = "ODF_SH"


def G_key(idx):
    return f"{G_FILTERED_BASE_KEY}_{idx}"


def H_key(idx):
    return f"{H_FILTERED_BASE_KEY}_{idx}"


def _gaussian_1d(r):
    out = np.exp(-r**2, dtype=np.float64)
    return out


def _gaussian_2d(r1, r2):
    out = np.exp(-r1**2, dtype=np.float64)*\
          np.exp(-r2**2, dtype=np.float64)
    return out


def _make_xfilter(f):
    out = np.reshape(f, (-1, 1, 1))
    return out


def _make_yfilter(f):
    out = np.reshape(f, (1, -1, 1))
    return out


def _make_zfilter(f):
    out = np.reshape(f, (1, 1, -1))
    return out


def _make_xyfilter(f):
    _sx, _sy = f.shape
    out = np.reshape(f, (_sx, _sy, 1))
    return out


def _make_xzfilter(f):
    _sx, _sz = f.shape
    out = np.reshape(f, (_sx, 1, _sz))
    return out


def _make_yzfilter(f):
    _sy, _sz = f.shape
    out = np.reshape(f, (1, _sy, _sz))
    return out


def _kappa_fct(alpha, beta, gamma, coeff=1.0, pow_alpha=0.0,
               pow_beta=0.0, pow_gamma=0.0):
    out = coeff*alpha**pow_alpha*beta**pow_beta*gamma**pow_gamma
    out = np.reshape(out, (1, 1, 1, -1))
    return out.astype(np.float32)


def equalize_filter(wfilter):
    sum_pos = np.sum(wfilter[wfilter > 0])
    sum_neg = np.sum(np.abs(wfilter[wfilter < 0]))
    wfilter_eq = wfilter
    wfilter_eq[wfilter < 0] = wfilter[wfilter < 0] / sum_neg * sum_pos
    return wfilter_eq


def convolve_with_bank(image, filter_bank, norm=1.0, mode='reflect'):
    # the sum of negatives must equal the sum of positives
    out = image * norm
    for wfilter in filter_bank:
        wfilter_eq = equalize_filter(wfilter)
        out = correlate(out, wfilter_eq, mode=mode)
    return out


class Steerable4thOrderGaussianQuadratureFilter():
    """
    Steerable 4th-order derivative of Gaussian Quadrature filters.

    :type image: numpy ndarray
    :param image: Input image to process.
    :type halfwidth: int
    :param halfwidth: Half-width of filtering kernel.
    :type sphere_name: str
    :param sphere_name: Name of DIPY sphere defining sampled directions.
    :type sh_order_max: int > 0, even
    :param sh_order_max:
        Maximum order for projecting filter amplitudes to SH coefficients.
    :type mode: str
    :param mode:
        Padding mode for convolution. One of: `reflect`, `constant`,
        `nearest`, `mirror` or `wrap`.
    :type chunk_shape: tuple of 4 int
    :param chunk_shape: Chunk shape for processing and saving data.
    :type num_processes: int
    :param num_processes: Number of processes.
    """
    def __init__(self, image, halfwidth, sphere_name, sh_order_max,
                 mode, chunk_shape, num_processes):
        argtol = 3.0
        samples = np.linspace(0, argtol, halfwidth+1)
        samples = np.append(-samples[:0:-1], samples)
        self.image_shape = image.shape
        self.samples = samples
        self.n_coeffs = int((sh_order_max + 2) * (sh_order_max + 1)) / 2
        _, l = sph_harm_ind_list(sh_order_max)
        self.FRT = np.diag(2.0*np.pi*eval_legendre(l, 0))

        sphere = get_sphere(sphere_name)
        _, self.b_inv = sh_to_sf_matrix(sphere, sh_order_max)
        self.directions = sphere.vertices
        self.num_processes = num_processes

        # additional normalization constant to generate 0-area-under-curve G filters
        self._C_forGFilters = 4.0*np.sqrt(210)/(105*np.sqrt(np.pi))

        self._N_2Dto3D = np.float_power(2.0/np.pi, 0.25).astype(np.float64)
        self._sampling_delta = np.float_power(argtol / halfwidth, 3)

        self._g4_funcs_list = [
            self._g4a, self._g4b, self._g4c, self._g4d, self._g4e, self._g4f,
            self._g4g, self._g4h, self._g4i, self._g4j, self._g4k, self._g4l,
            self._g4m, self._g4n, self._g4o
        ]
        self._h4_funcs_list = [
            self._h4a, self._h4b, self._h4c, self._h4d, self._h4e, self._h4f,
            self._h4g, self._h4h, self._h4i, self._h4j, self._h4k, self._h4l,
            self._h4m, self._h4n, self._h4o, self._h4p, self._h4q, self._h4r,
            self._h4s, self._h4t, self._h4u
        ]

        self._g4_kappas = []
        self._h4_kappas = []

        # store intermediate files in temp zarr file
        self.zarr_store = zarr.TempStore()
        self.zarr_root = zarr.open(self.zarr_store, mode='w')

        self.chunkshape = chunk_shape
        self.n_chunks = [
            np.ceil(float(image.shape[i]) / self.chunkshape[i]).astype(int)
            for i in range(3)
        ]

        # TODO: Process in chunks (handle edge effects)
        # TODO: Move processing to compute_odf_sh method
        for it, _g_func in enumerate(tqdm(self._g4_funcs_list)):
            _kappa, _filters = _g_func(samples)
            self._g4_kappas.append(_kappa)
            data = convolve_with_bank(
                image, _filters, self._N_2Dto3D * self._sampling_delta, mode)
            self.zarr_root.array(G_key(it), data[..., None],
                                 chunks=self.chunkshape[:3] + (1,),
                                 dtype=np.float32,
                                 write_empty_chunks=False)

        for it, _h_func in enumerate(tqdm(self._h4_funcs_list)):
            _kappa, _filters = _h_func(samples)
            self._h4_kappas.append(_kappa)
            data = convolve_with_bank(
                image, _filters, self._N_2Dto3D * self._sampling_delta, mode)
            self.zarr_root.array(H_key(it), data[..., None],
                                 chunks=self.chunkshape[:3] + (1,),
                                 dtype=np.float32,
                                 write_empty_chunks=False)

    def compute_odf_sh(self):
        """
        Compute ODF from precomputed bank of images.

        :type out_zarr: zarr array
        :return out_zarr: Zarr array containing SH coefficients.
        """
        logging.info('Compute quadrature response')
        self.zarr_root.zeros(Q_RESPONSE_KEY,
                             shape=self.image_shape + (len(self.directions),),
                             chunks=self.chunkshape, dtype=np.float32)
        out_zarr = self.zarr_root.zeros(ODF_SH_KEY,
                                        shape=self.image_shape + (self.n_coeffs,),
                                        chunks=self.chunkshape, dtype=np.float32)

        self._compute_G_response()
        self._compute_H_response()

        futures = []
        with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
            for (i, j, k) in product(*[range(_i) for _i in self.n_chunks]):
                futures.append(executor.submit(self._sh_from_quadrature_response, i, j, k))
            for f in as_completed(futures):
                f.result()  # this step for throwing exceptions in main thread
        return out_zarr

    def _compute_G_response(self):
        out_zarr = self.zarr_root.zeros(G_RESPONSE_KEY,
                                        shape=self.image_shape + (len(self.directions),),
                                        chunks=self.chunkshape, dtype=np.float32)

        logging.info('Interpolate feature maps for G response')
        # Interpolate feature maps in blocks
        for it, _kappa in enumerate(tqdm(self._g4_kappas)):
            alpha = self.directions[:, 0]
            beta = self.directions[:, 1]
            gamma = self.directions[:, 2]
            with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
                futures = {}
                for (i, j, k) in product(*[range(_i) for _i in self.n_chunks]):
                    futures[executor.submit(
                        self._add_response, G_RESPONSE_KEY, G_key(it),
                        _kappa(alpha, beta, gamma), i, j, k)] = (i, j, k)
                for f in as_completed(futures):
                    f.result()

        # Square the resulting image
        with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
                futures = {}
                for (i, j, k) in product(*[range(_i) for _i in self.n_chunks]):
                    futures[executor.submit(
                        self._square_response, G_RESPONSE_KEY, i, j, k)] = (i, j, k)
                for f in as_completed(futures):
                    f.result()

        # output zarr image
        return out_zarr

    def _compute_H_response(self):
        out_zarr = self.zarr_root.zeros(H_RESPONSE_KEY,
                                        shape=self.image_shape + (len(self.directions),),
                                        chunks=self.chunkshape, dtype=np.float32)

        logging.info('Interpolate feature maps for H response')
        # Interpolate feature maps in blocks
        for it, _kappa in enumerate(tqdm(self._h4_kappas)):
            alpha = self.directions[:, 0]
            beta = self.directions[:, 1]
            gamma = self.directions[:, 2]
            with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
                futures = {}
                for (i, j, k) in product(*[range(_i) for _i in self.n_chunks]):
                    futures[executor.submit(
                        self._add_response, H_RESPONSE_KEY, H_key(it),
                        _kappa(alpha, beta, gamma), i, j, k)] = (i, j, k)
                for f in as_completed(futures):
                    f.result()

        # Square the resulting image
        with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
                futures = {}
                for (i, j, k) in product(*[range(_i) for _i in self.n_chunks]):
                    futures[executor.submit(
                        self._square_response, H_RESPONSE_KEY, i, j, k)] = (i, j, k)
                for _ in as_completed(futures):
                    f.result()

        # output zarr image
        return out_zarr

    def _add_response(self, response_key, base_filtered_key, kappa_arr, i, j, k):
        response = self.zarr_root[response_key]
        base_image = self.zarr_root[base_filtered_key]
        response.blocks[i, j, k] += kappa_arr * base_image.blocks[i, j, k]

    def _square_response(self, response_key, i, j, k):
        response = self.zarr_root[response_key]
        response.blocks[i, j, k] = response.blocks[i, j, k]**2

    def _sh_from_quadrature_response(self, i, j, k):
        g_response = self.zarr_root[G_RESPONSE_KEY]
        h_response = self.zarr_root[H_RESPONSE_KEY]
        q_response = self.zarr_root[Q_RESPONSE_KEY]
        odf_sh = self.zarr_root[ODF_SH_KEY]
        q_response.blocks[i, j, k] =\
            g_response.blocks[i, j, k] + h_response.blocks[i, j, k]

        odf_sh.blocks[i, j, k] = np.asarray(
            [e.dot(self.b_inv) for e in q_response.blocks[i, j, k]]
        )
        odf_sh.blocks[i, j, k] = odf_sh.blocks[i, j, k].dot(self.FRT)

    def _g4a(self, r):
        kappa = partial(_kappa_fct, pow_gamma=4)
        xfilter = _make_xfilter(_gaussian_1d(r))
        yfilter = _make_yfilter(_gaussian_1d(r))
        zfilter = _make_zfilter(self._C_forGFilters*(4.0*r**4 - 12.0*r**2 + 3.0)*_gaussian_1d(r))
        return kappa, (xfilter, yfilter, zfilter)

    def _g4b(self, r):
        kappa = partial(_kappa_fct, coeff=4, pow_alpha=1.0, pow_gamma=3.0)
        xfilter = _make_xfilter(r*_gaussian_1d(r))
        yfilter = _make_yfilter(self._C_forGFilters*_gaussian_1d(r))
        zfilter = _make_zfilter(r*(4.0*r**2 - 6.0)*_gaussian_1d(r))
        return kappa, (xfilter, yfilter, zfilter)

    def _g4c(self, r):
        kappa = partial(_kappa_fct, coeff=6.0, pow_alpha=2, pow_gamma=2)
        xfilter = _make_xfilter((2*r**2 - 1)*_gaussian_1d(r))
        yfilter = _make_yfilter(self._C_forGFilters*_gaussian_1d(r))
        zfilter = _make_zfilter((2*r**2 - 1)*_gaussian_1d(r))
        return kappa, (xfilter, yfilter, zfilter)

    def _g4d(self, r):
        kappa = partial(_kappa_fct, coeff=4.0, pow_alpha=3, pow_gamma=1)
        xfilter = _make_xfilter(r*(4.0*r**2 - 6.0)*_gaussian_1d(r))
        yfilter = _make_yfilter(self._C_forGFilters*_gaussian_1d(r))
        zfilter = _make_zfilter(r*_gaussian_1d(r))
        return kappa, (xfilter, yfilter, zfilter)

    def _g4e(self, r):
        kappa = partial(_kappa_fct, pow_alpha=4)
        xfilter = _make_xfilter((4.0*r**4 - 12.0*r**2 + 3.0)*_gaussian_1d(r))
        yfilter = _make_yfilter(self._C_forGFilters*_gaussian_1d(r))
        zfilter = _make_zfilter(_gaussian_1d(r))
        return kappa, (xfilter, yfilter, zfilter)

    def _g4f(self, r):
        kappa = partial(_kappa_fct, coeff=4, pow_beta=1, pow_gamma=3)
        xfilter = _make_xfilter(self._C_forGFilters*_gaussian_1d(r))
        yfilter = _make_yfilter(r*_gaussian_1d(r))
        zfilter = _make_zfilter(r*(4.0*r**2 - 6.0)*_gaussian_1d(r))
        return kappa, (xfilter, yfilter, zfilter)

    def _g4g(self, r):
        kappa = partial(_kappa_fct, coeff=12, pow_alpha=1, pow_beta=1, pow_gamma=2)
        xfilter = _make_xfilter(self._C_forGFilters*r*_gaussian_1d(r))
        yfilter = _make_yfilter(r*_gaussian_1d(r))
        zfilter = _make_zfilter((4.0*r**2 - 2.0)*_gaussian_1d(r))
        return kappa, (xfilter, yfilter, zfilter)

    def _g4h(self, r):
        kappa = partial(_kappa_fct, coeff=12, pow_alpha=2, pow_beta=1, pow_gamma=1)
        xfilter = _make_xfilter((4.0*r**2 - 2.0)*_gaussian_1d(r))
        yfilter = _make_yfilter(r*_gaussian_1d(r))
        zfilter = _make_zfilter(self._C_forGFilters*r*_gaussian_1d(r))
        return kappa, (xfilter, yfilter, zfilter)

    def _g4i(self, r):
        kappa = partial(_kappa_fct, coeff=4, pow_alpha=3, pow_beta=1)
        xfilter = _make_xfilter(r*(4.0*r**2 - 6.0)*_gaussian_1d(r))
        yfilter = _make_yfilter(self._C_forGFilters*r*_gaussian_1d(r))
        zfilter = _make_zfilter(_gaussian_1d(r))
        return kappa, (xfilter, yfilter, zfilter)

    def _g4j(self, r):
        kappa = partial(_kappa_fct, coeff=6, pow_beta=2, pow_gamma=2)
        xfilter = _make_xfilter(self._C_forGFilters*_gaussian_1d(r))
        yfilter = _make_yfilter((2*r**2 - 1)*_gaussian_1d(r))
        zfilter = _make_zfilter((2*r**2 - 1)*_gaussian_1d(r))
        return kappa, (xfilter, yfilter, zfilter)

    def _g4k(self, r):
        kappa = partial(_kappa_fct, coeff=12, pow_alpha=1, pow_beta=2, pow_gamma=1)
        xfilter = _make_xfilter(self._C_forGFilters*r*_gaussian_1d(r))
        yfilter = _make_yfilter((4.0*r**2 - 2.0)*_gaussian_1d(r))
        zfilter = _make_zfilter(r*_gaussian_1d(r))
        return kappa, (xfilter, yfilter, zfilter)

    def _g4l(self, r):
        kappa = partial(_kappa_fct, coeff=6, pow_alpha=2, pow_beta=2)
        xfilter = _make_xfilter((2*r**2 - 1)*_gaussian_1d(r))
        yfilter = _make_yfilter((2*r**2 - 1)*_gaussian_1d(r))
        zfilter = _make_zfilter(self._C_forGFilters*_gaussian_1d(r))
        return kappa, (xfilter, yfilter, zfilter)

    def _g4m(self, r):
        kappa = partial(_kappa_fct, coeff=4, pow_beta=3, pow_gamma=1)
        xfilter = _make_xfilter(self._C_forGFilters*_gaussian_1d(r))
        yfilter = _make_yfilter(r*(4.0*r**2 - 6.0)*_gaussian_1d(r))
        zfilter = _make_zfilter(r*_gaussian_1d(r))
        return kappa, (xfilter, yfilter, zfilter)

    def _g4n(self, r):
        kappa = partial(_kappa_fct, coeff=4, pow_alpha=1, pow_beta=3)
        xfilter = _make_xfilter(self._C_forGFilters*r*_gaussian_1d(r))
        yfilter = _make_yfilter(r*(4.0*r**2 - 6.0)*_gaussian_1d(r))
        zfilter = _make_zfilter(_gaussian_1d(r))
        return kappa, (xfilter, yfilter, zfilter)

    def _g4o(self, r):
        kappa = partial(_kappa_fct, pow_beta=4)
        xfilter = _make_xfilter(self._C_forGFilters*_gaussian_1d(r))
        yfilter = _make_yfilter((4.0*r**4 - 12.0*r**2 + 3.0)*_gaussian_1d(r))
        zfilter = _make_zfilter(_gaussian_1d(r))
        return kappa, (xfilter, yfilter, zfilter)

    def _h4a(self, r):
        kappa = partial(_kappa_fct, pow_gamma=5)
        xfilter = _make_xfilter(_gaussian_1d(r))
        yfilter = _make_yfilter(_gaussian_1d(r))
        zfilter = _make_zfilter(r*(0.3975*r**4 - 2.982*r**2 + 2.858)*_gaussian_1d(r))
        return kappa, (xfilter, yfilter, zfilter)

    def _h4b(self, r):
        kappa = partial(_kappa_fct, coeff=5, pow_alpha=1.0, pow_gamma=4.0)
        xfilter = _make_xfilter(r*_gaussian_1d(r))
        yfilter = _make_yfilter(_gaussian_1d(r))
        zfilter = _make_zfilter((0.3975*r**4 - 1.7892*r**2 + 0.5716)*_gaussian_1d(r))
        return kappa, (xfilter, yfilter, zfilter)

    def _h4c(self, r):
        _x, _z = np.meshgrid(r, r, indexing='ij')
        kappa = partial(_kappa_fct, coeff=10, pow_alpha=2, pow_gamma=3)
        # this filter is not x-z separable, so the output is a 2D filter
        xzfilter = _make_xzfilter(_z*(0.3975*_x**2*_z**2 - 0.8946*_x**2 -0.2982*_z**2 + 0.5716)*_gaussian_2d(_x, _z))
        yfilter = _make_yfilter(_gaussian_1d(r))
        return kappa, (xzfilter, yfilter)

    def _h4d(self, r):
        _x, _z = np.meshgrid(r, r, indexing='ij')
        kappa = partial(_kappa_fct, coeff=10, pow_alpha=3, pow_gamma=2)
        xzfilter = _make_xzfilter(_x*(0.3975*_x**2*_z**2 - 0.8946*_z**2 -0.2982*_x**2 + 0.5716)*_gaussian_2d(_x, _z))
        yfilter = _make_yfilter(_gaussian_1d(r))
        return kappa, (xzfilter, yfilter)

    def _h4e(self, r):
        kappa = partial(_kappa_fct, coeff=5, pow_alpha=4, pow_gamma=1)
        xfilter = _make_xfilter((0.3975*r**4 - 1.7892*r**2 + 0.5716)*_gaussian_1d(r))
        yfilter = _make_yfilter(_gaussian_1d(r))
        zfilter = _make_zfilter(r*_gaussian_1d(r))
        return kappa, (xfilter, yfilter, zfilter)

    def _h4f(self, r):
        kappa = partial(_kappa_fct, pow_alpha=5)
        xfilter = _make_xfilter(r*(0.3975*r**4 - 2.982*r**2 + 2.858)*_gaussian_1d(r))
        yfilter = _make_yfilter(_gaussian_1d(r))
        zfilter = _make_zfilter(_gaussian_1d(r))
        return kappa, (xfilter, yfilter, zfilter)

    def _h4g(self, r):
        kappa = partial(_kappa_fct, coeff=5, pow_beta=1, pow_gamma=4)
        xfilter = _make_xfilter(_gaussian_1d(r))
        yfilter = _make_yfilter(r*_gaussian_1d(r))
        zfilter = _make_zfilter((0.3975*r**4 - 1.7892*r**2 + 0.5716)*_gaussian_1d(r))
        return kappa, (xfilter, yfilter, zfilter)

    def _h4h(self, r):
        kappa = partial(_kappa_fct, coeff=20, pow_alpha=1, pow_beta=1, pow_gamma=3)
        xfilter = _make_xfilter(r*_gaussian_1d(r))
        yfilter = _make_yfilter(r*_gaussian_1d(r))
        zfilter = _make_zfilter(r*(0.3975*r**2 - 0.8946)*_gaussian_1d(r))
        return kappa, (xfilter, yfilter, zfilter)

    def _h4i(self, r):
        _x, _z = np.meshgrid(r, r, indexing='ij')
        kappa = partial(_kappa_fct, coeff=30, pow_alpha=2, pow_beta=1, pow_gamma=2)
        xzfilter = _make_xzfilter((0.3975*_x**2*_z**2 - 0.2982*_x**2 - 0.2982*_z**2 + 0.19053)*_gaussian_2d(_x, _z))
        yfilter = _make_yfilter(r*_gaussian_1d(r))
        return kappa, (xzfilter, yfilter)

    def _h4j(self, r):
        kappa = partial(_kappa_fct, coeff=20, pow_alpha=3, pow_beta=1, pow_gamma=1)
        xfilter = _make_xfilter(r*(0.3975*r**2 - 0.8946)*_gaussian_1d(r))
        yfilter = _make_yfilter(r*_gaussian_1d(r))
        zfilter = _make_zfilter(r*_gaussian_1d(r))
        return kappa, (xfilter, yfilter, zfilter)

    def _h4k(self, r):
        kappa = partial(_kappa_fct, coeff=5, pow_alpha=4, pow_beta=1)
        xfilter = _make_xfilter((0.3975*r**4 - 1.7892*r**2 + 0.5716)*_gaussian_1d(r))
        yfilter = _make_yfilter(r*_gaussian_1d(r))
        zfilter = _make_zfilter(_gaussian_1d(r))
        return kappa, (xfilter, yfilter, zfilter)

    def _h4l(self, r):
        _y, _z = np.meshgrid(r, r, indexing='ij')
        kappa = partial(_kappa_fct, coeff=10, pow_beta=2, pow_gamma=3)
        xfilter = _make_xfilter(_gaussian_1d(r))
        yzfilter = _make_yzfilter(_z*(0.3975*_y**2*_z**2 - 0.8946*_y**2 - 0.2982*_z**2 + 0.5716)*_gaussian_2d(_y, _z))
        return kappa, (xfilter, yzfilter)

    def _h4m(self, r):
        _y, _z = np.meshgrid(r, r, indexing='ij')
        kappa = partial(_kappa_fct, coeff=30, pow_alpha=1, pow_beta=2, pow_gamma=2)
        xfilter = _make_xfilter(r*_gaussian_1d(r))
        yzfilter = _make_yzfilter((0.3975*_z**2*_y**2 - 0.2982*_z**2 - 0.2982*_y**2 + 0.19053)*_gaussian_2d(_y, _z))
        return kappa, (xfilter, yzfilter)

    def _h4n(self, r):
        _x, _y = np.meshgrid(r, r, indexing='ij')
        kappa = partial(_kappa_fct, coeff=30, pow_alpha=2, pow_beta=2, pow_gamma=1)
        xyfilter = _make_xyfilter((0.3975*_x**2*_y**2 - 0.2982*_x**2 - 0.2982*_y**2 + 0.19053)*_gaussian_2d(_x, _y))
        zfilter = _make_zfilter(r*_gaussian_1d(r))
        return kappa, (xyfilter, zfilter)

    def _h4o(self, r):
        _x, _y = np.meshgrid(r, r, indexing='ij')
        kappa = partial(_kappa_fct, coeff=10, pow_alpha=3, pow_gamma=2)
        xyfilter = _make_xyfilter(_x*(0.3975*_x**2*_y**2 - 0.2982*_x**2 - 0.8946*_y**2 + 0.5716)*_gaussian_2d(_x, _y))
        zfilter = _make_zfilter(_gaussian_1d(r))
        return kappa, (xyfilter, zfilter)

    def _h4p(self, r):
        _y, _z = np.meshgrid(r, r, indexing='ij')
        kappa = partial(_kappa_fct, coeff=10, pow_beta=3, pow_gamma=2)
        xfilter = _make_xfilter(_gaussian_1d(r))
        yzfilter = _make_yzfilter(_y*(0.3975*_y**2*_z**2 - 0.2982*_y**2 - 0.8946*_z**2 + 0.5716)*_gaussian_2d(_y, _z))
        return kappa, (xfilter, yzfilter)

    def _h4q(self, r):
        kappa = partial(_kappa_fct, coeff=20, pow_alpha=1, pow_beta=3, pow_gamma=1)
        xfilter = _make_xfilter(r*_gaussian_1d(r))
        yfilter = _make_yfilter(r*(0.3975*r**2 - 0.8946)*_gaussian_1d(r))
        zfilter = _make_zfilter(r*_gaussian_1d(r))
        return kappa, (xfilter, yfilter, zfilter)

    def _h4r(self, r):
        _x, _y = np.meshgrid(r, r, indexing='ij')
        kappa = partial(_kappa_fct, coeff=10, pow_alpha=2, pow_beta=3)
        xyfilter = _make_xyfilter(_y*(0.3975*_x**2*_y**2 - 0.8946*_x**2 - 0.2982*_y**2 + 0.5716)*_gaussian_2d(_x, _y))
        zfilter = _make_zfilter(_gaussian_1d(r))
        return kappa, (xyfilter, zfilter)

    def _h4s(self, r):
        kappa = partial(_kappa_fct, coeff=5, pow_beta=4, pow_gamma=1)
        xfilter = _make_xfilter(_gaussian_1d(r))
        yfilter = _make_yfilter((0.3975*r**4 - 1.7892*r**2 + 0.5716)*_gaussian_1d(r))
        zfilter = _make_zfilter(r*_gaussian_1d(r))
        return kappa, (xfilter, yfilter, zfilter)

    def _h4t(self, r):
        kappa = partial(_kappa_fct, coeff=5, pow_alpha=1, pow_beta=4)
        xfilter = _make_xfilter(r*_gaussian_1d(r))
        yfilter = _make_yfilter((0.3975*r**4 - 1.7892*r**2 + 0.5716)*_gaussian_1d(r))
        zfilter = _make_zfilter(_gaussian_1d(r))
        return kappa, (xfilter, yfilter, zfilter)

    def _h4u(self, r):
        kappa = partial(_kappa_fct, pow_beta=5)
        xfilter = _make_xfilter(_gaussian_1d(r))
        yfilter = _make_yfilter(r*(0.3975*r**4 - 2.982*r**2 + 2.858)*_gaussian_1d(r))
        zfilter = _make_zfilter(_gaussian_1d(r))
        return kappa, (xfilter, yfilter, zfilter)
