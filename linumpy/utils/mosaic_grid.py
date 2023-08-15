#!/usr/bin/python3
# -*- coding:utf-8 -*-

import SimpleITK as sitk
import numpy as np
import scipy.ndimage
import scipy.ndimage.morphology as morpho
from scipy.ndimage import gaussian_filter
from skimage.morphology import disk, ball
from skimage import metrics
from tqdm import tqdm
from scipy import optimize


# TODO: Add an algorithm to estimate the affine transform parameters


class MosaicGrid():
    """This class is used to manage and process mosaic grid images. A mosaic grid is a 2D image containing all the tiles
    for a given mosaic, without any overlap. This class can be used for instance to apply processing to all tiles, to
    optimize the affine transform matrix describing the tile position, and to stitch the tiles together to obtain the
    reconstructed mosaic.

    .. note::
        This class can only deal with 2D mosaic grids for now. To generate a 2D mosaic grid from a collection of volumetric
        tiles for a given slice, you can use the :ref:`script-linum-create-mosaic-grid` script.

    """

    def __init__(self, image: np.ndarray, tile_shape: tuple = (512, 512), overlap_fraction: float = 0.2):
        """Constructor method
        """
        self.tile_shape = tile_shape
        self.tile_size_x = self.tile_shape[0]
        self.tile_size_y = self.tile_shape[1]
        self.overlap_fraction = overlap_fraction
        self.blending_method = None
        self.dtype = image.dtype
        self.imin = image.min()
        self.imax = image.max()
        #self.image = (image - self.imin) / (self.imax - self.imin)
        self.image = image

        self.compute_mosaic_shape()
        self.set_affine(self.overlap_fraction)

    def set_affine(self, overlap_fraction: float = 0.2) -> None:
        """Sets the affine matrix given an overlap fraction.

        :param overlap_fraction: An overlap fraction between 0 and 1, defaults to 0.2
        :type overlap_fraction: float, optional
        """
        self.affine = np.eye(2) * (1 - overlap_fraction) * np.array(self.tile_shape[0:2]).T
        self.overlap_fraction = overlap_fraction

    def set_blending_method(self, method="none"):
        """To set the blending method. Available methodes are 'none' and 'average', 'diffusion'"""
        available_methods = ['none', 'average', 'diffusion']
        method = str(method).lower()
        assert method in available_methods, f"Available blending methods are : {available_methods}"
        if method == "none":
            self.blending_method = None
        else:
            self.blending_method = method

    def get_image(self):
        """To get the original image"""
        return self.image

    def compute_mosaic_shape(self):
        """Compute the mosaic grid shape"""
        self.n_tiles_x = self.image.shape[0] // self.tile_size_x
        self.n_tiles_y = self.image.shape[1] // self.tile_size_y

    def get_tiles(self):
        """Returns the tiles from the mosaic grid.

        :return: tuple containing the tiles and the tile positions in the grid.
        """
        tiles = np.zeros((self.n_tiles_x * self.n_tiles_y, self.tile_size_x, self.tile_size_y), dtype=self.image.dtype)
        tiles_pos = []
        i = 0
        for x in range(self.n_tiles_x):
            for y in range(self.n_tiles_y):
                tiles[i, ...] = self.get_tile(x, y)
                tiles_pos.append((x, y))
                i += 1

        return (tiles, tiles_pos)

    def get_neighbors_around_tile(self, x, y, neighborhood_type="N4"):
        positions = []
        neighbors = []

        for i in range(-1, 2):
            for j in range(-1, 2):
                if x + i < 0 or x + i >= self.n_tiles_x:
                    continue
                if y + j < 0 or y + j >= self.n_tiles_y:
                    continue
                # Ignoring center tile
                if i == 0 and j == 0:
                    continue
                if (i, j) in [(-1, 0), (1, 0), (0, -1), (0, 1)] and neighborhood_type in ["N4", "N8"]:
                    positions.append((i + x, j + y))
                    neighbors.append(self.get_tile(i + x, j + y))
                if (i, j) in [(-1, -1), (1, 1), (1, -1), (-1, 1)] and neighborhood_type in ["Nd", "N8"]:
                    positions.append((i + x, j + y))
                    neighbors.append(self.get_tile(i + x, j + y))

        return neighbors, positions

    def get_neighbors_list(self, neighborhood_type: str = "N4"):
        """Returns a list of neighboring tiles.

        :param neighborhood_type: Type of neighborhood to consider. 'N4' for horizontal and vertical neighbors, 'N8' to also consider diagonal neighbors.
        :return: A list of neighbor pairs given by their grid position.

        .. note::
            This also updates the `neighbors_list` object property.

        """
        neighbors_list = []
        # First add horizontal neighbors
        for y in range(self.n_tiles_y):
            for x in range(self.n_tiles_x - 1):
                neighbors_list.append(((x, y), (x + 1, y)))

        # Second add vertical neighbors
        for x in range(self.n_tiles_x):
            for y in range(self.n_tiles_y - 1):
                neighbors_list.append(((x, y), (x, y + 1)))

        if neighborhood_type == "N8":
            # First diagonal neighbors
            for y in range(self.n_tiles_y - 1):
                for x in range(self.n_tiles_x - 1):
                    neighbors_list.append(((x, y), (x + 1, y + 1)))

            # Second diagonal neighbors
            for y in range(self.n_tiles_y - 1):
                for x in range(1, self.n_tiles_x):
                    neighbors_list.append(((x, y), (x - 1, y + 1)))

        self.neighbors_list = neighbors_list

        return self.neighbors_list

    def get_tile(self, x: int, y: int) -> np.ndarray:
        """Extract a tile from the mosaic grid.

        :param x: x position within the mosaic grid
        :param y: y position within the mosaic grid
        :return: 2D tile
        """
        assert x < self.n_tiles_x and y < self.n_tiles_y, f"Mosaic shape is {self.tile_size_x}x{self.tile_size_y}"
        x0 = self.tile_size_x * x
        y0 = self.tile_size_y * y
        xf = x0 + self.tile_size_x
        yf = y0 + self.tile_size_y
        tile = self.image[x0:xf, y0:yf]
        return tile

    def set_tile(self, x: int, y: int, tile: np.ndarray):
        """Set a tile from the mosaic grid.

        :param x: x position within the mosaic grid
        :param y: y position within the mosaic grid
        :param tile: 2D tile
        """
        assert x < self.n_tiles_x and y < self.n_tiles_y, f"Mosaic shape is {self.tile_size_x}x{self.tile_size_y}"
        x0 = self.tile_size_x * x
        y0 = self.tile_size_y * y
        xf = x0 + self.tile_size_x
        yf = y0 + self.tile_size_y
        self.image[x0:xf, y0:yf] = tile

    def get_position(self, x: int, y: int) -> np.ndarray:
        """Compute the cartesian position of a given tile using the internal affine transform.

        :param x: x position within the mosaic grid
        :param y: y position within the mosaic grid
        :return: (2,) array containing the cartesian position of this tile (in pixel)
        """
        pos = np.dot(self.affine, [x, y]).astype(int)
        return pos

    def get_neighbor_tiles(self, n_id: int) -> tuple:
        """ Extract the tiles for a given neighbor pair.

        :param n_id: The neighbor pair id.
        :return: (2,) tuple containing each tile as a np.ndarray.
        """
        if not hasattr(self, "neighbors_list"):
            self.get_neighbors_list()

        assert n_id < len(self.neighbors_list), "The neighbor id exceeds the number of neighbors"
        neighbor = self.neighbors_list[n_id]
        tile_1 = self.get_tile(*neighbor[0])
        tile_2 = self.get_tile(*neighbor[1])
        return (tile_1, tile_2)

    def get_neighbor_overlap_from_pos(self, p1, p2):
        t1 = self.get_tile(*p1)
        t2 = self.get_tile(*p2)

        # Consider both 2D and 3D tiles
        ndim = t1.ndim
        if ndim == 2:
            t1 = np.reshape(t1, (*t1.shape, 1))
            t2 = np.reshape(t2, (*t2.shape, 1))

        # Convert mosaic tile position to cartesian position
        p1 = self.get_position(*p1)
        p2 = self.get_position(*p2)

        # Get the tile shape
        nx, ny = t1.shape[0:2]
        if ndim == 2:
            nz = 1
        else:
            nz = t1.shape[2]

        # Get the min and max coordinates for this mosaic
        x0 = int(min([p1[0], p2[0]]))
        xf = int(max([p1[0], p2[0]]) + nx)
        y0 = int(min([p1[1], p2[1]]))
        yf = int(max([p1[1], p2[1]]) + ny)

        # Create empty mosaics with each tile
        mosaic1 = np.zeros((xf - x0, yf - y0, nz))
        mosaic2 = np.zeros((xf - x0, yf - y0, nz))

        mosaic1[p1[0] - x0:p1[0] - x0 + nx, p1[1] - y0:p1[1] - y0 + ny, :] = t1 + 1
        mosaic2[p2[0] - x0:p2[0] - x0 + nx, p2[1] - y0:p2[1] - y0 + ny, :] = t2 + 1

        # Find intersection
        mask = mosaic1 * mosaic2 >= 1

        # Convert this into t1 and t2 coordinates
        x, y, z = np.where(mask)
        o_xmin = x.min()
        o_ymin = y.min()
        o_xmax = x.max()
        o_ymax = y.max()

        o_pos1 = (o_xmin - (p1[0] - x0), o_ymin - (p1[1] - y0),
                  o_xmax - (p1[0] - x0), o_ymax - (p1[1] - y0))
        o_pos2 = (o_xmin - (p2[0] - x0), o_ymin - (p2[1] - y0),
                  o_xmax - (p2[0] - x0), o_ymax - (p2[1] - y0))

        # Getting overlap
        overlap1 = t1[o_pos1[0]:o_pos1[2], o_pos1[1]:o_pos1[3], :]
        overlap2 = t2[o_pos2[0]:o_pos2[2], o_pos2[1]:o_pos2[3], :]

        if ndim == 2:
            overlap1 = np.squeeze(overlap1)
            overlap2 = np.squeeze(overlap2)

        return (overlap1, overlap2, o_pos1, o_pos2)

    def get_neighbor_overlap(self, n_id):
        """ Extract the tile overlaps for a given neighbor pair.

        :param n_id: The neighbor pair id.
        :return: (4,) tuple containing (overlap1, overlap2, overlap1_position, overlap2_position)
        """
        p1, p2 = self.neighbors_list[n_id]
        return self.get_neighbor_overlap_from_pos(p1, p2)





    def crop_tiles(self, xlim: tuple = (0, -1), ylim: tuple = (0, -1)):
        """Crop all tiles in the mosaic grid.

        :param xlim: (2,) tuple containing the x-axis (row) cropping limits.
        :param ylim: (2,) tuple containing the y-axis (col) cropping limits.

        .. note::
            - This also resets the affine transform using the overlap_fraction.
        """
        xlim = list(xlim)
        ylim = list(ylim)
        if xlim[1] < 0:
            xlim[1] = self.tile_shape[0] + xlim[1] + 1
        if ylim[1] < 0:
            ylim[1] = self.tile_shape[1] + ylim[1] + 1

        nx = xlim[1] - xlim[0]
        ny = ylim[1] - ylim[0]
        new_shape = (self.n_tiles_x * (xlim[1] - xlim[0]), self.n_tiles_y * (ylim[1] - ylim[0]))
        image = np.zeros(new_shape, dtype=np.float32)
        for x in range(self.n_tiles_x):
            for y in range(self.n_tiles_y):
                tile = self.get_tile(x, y)

                # New positions
                x0 = x * nx
                xf = x0 + nx
                y0 = y * ny
                yf = y0 + ny
                image[x0:xf, y0:yf] = tile[xlim[0]:xlim[1], ylim[0]:ylim[1]]

        self.image = image
        self.tile_shape = (nx, ny)
        self.tile_size_x = nx
        self.tile_size_y = ny

        self.set_affine(
            overlap_fraction=self.overlap_fraction)  # FIXME : Overlap fraction need to be adjusted after cropping

    def get_stitched_image(self, blending_method: str = "none") -> np.ndarray:
        """Performs a 2D reconstruction of the mosaic grid.

        :param blending_method: Blending method. Available: 'none', 'average', 'diffusion'.
        :return: Stitched mosaic.

        .. note::
            The affine transform obtained from the overlap fraction or by the affine transform optimization is used
            for the reconstruction."""
        # TODO: Add subpixel reconstruction precision.
        if blending_method is not None:
            self.set_blending_method(blending_method)
        xmin = np.inf
        xmax = -np.inf
        ymin = np.inf
        ymax = -np.inf
        for x in range(self.n_tiles_x):
            for y in range(self.n_tiles_y):
                px, py = self.get_position(x, y)
                if px < xmin:
                    xmin = px
                if px > xmax:
                    xmax = px
                if py < ymin:
                    ymin = py
                if py > ymax:
                    ymax = py
        x_shape = np.ceil(xmax - xmin + self.tile_shape[0]).astype(int)
        y_shape = np.ceil(ymax - ymin + self.tile_shape[1]).astype(int)

        image = np.zeros((x_shape, y_shape, 1), dtype=np.float32)
        # mask = np.zeros((x_shape, y_shape), dtype=np.float32)
        for x in range(self.n_tiles_x):
            for y in range(self.n_tiles_y):
                tile = self.get_tile(x, y)
                if np.all(tile == 0):  # Empty tile, ignore
                    continue
                    print(x, y)
                px, py = self.get_position(x, y)
                x0 = np.floor(px - xmin).astype(int)
                y0 = np.floor(py - ymin).astype(int)
                # xf = x0 + self.tile_shape[0]
                # yf = y0 + self.tile_shape[1]
                # if self.blending_method is None:
                #     image[x0:xf, y0:yf] = tile
                # elif self.blending_method == "average":
                #     image[x0:xf, y0:yf] = image[x0:xf, y0:yf] + tile
                #     mask[x0:xf, y0:yf] += 1.0
                image = addVolumeToMosaic(tile, (x0, y0), image, blendingMethod=self.blending_method)

        # if self.blending_method == "average":
        #    image[mask>0] = image[mask>0] / mask[mask>0]

        return image.squeeze()

    def global_overlap_similarity(self, random_fraction: float = 1.0, threshold: float = None):
        neighbors = self.get_neighbors_list(neighborhood_type="N4")
        n_neighbors = len(neighbors)
        neighbors_ids = list(range(n_neighbors))
        np.random.shuffle(neighbors_ids)

        error = 0.0
        n_samples = 0

        i = 0
        while (i < n_neighbors) and (n_samples / float(n_neighbors) < random_fraction):
            o1, o2, p1, p2 = self.get_neighbor_overlap(neighbors_ids[i])

            if threshold is None:
                if np.all(o1 == 0) or np.all(o2 == 0):  # Ignore empty overlaps
                    continue
            else:
                m1 = o1 < threshold
                m2 = o2 < threshold
                if np.all(m1 == False) or np.all(m2 == False):
                    continue

            # TODO: Test other error metrics, this one doesn't work well when there is illumination inhomogeneity
            error += np.sqrt(((o1 - o2) ** 2).mean())
            n_samples += 1
            i += 1
        if n_samples > 0:
            error = error / float(n_samples)
        return error

    def optimize_overlap(self, step: float = 0.01, omin: float = 0.1, omax: float = 0.5, display: bool = False, random_fraction=1.0, threshold=None):
        """Uses the similarity between every neighboring tiles to estimate the overlap fraction.

        :param step: Overlap fraction steps used for the search.
        :param omin: Minimum overlap fraction to consider.
        :param omax: Maximum overlap fraction to consider.
        :param display: If set to true, the similarity curve will be displayed at the end of the optimization.
        """
        old_overlap = self.overlap_fraction
        overlaps = np.arange(omin, omax, step)
        cost = []
        for o in tqdm(overlaps):
            self.set_affine(overlap_fraction=o)
            c = self.global_overlap_similarity(random_fraction=random_fraction, threshold=threshold)
            cost.append(c)

        # Extract the optimal overlap
        optimal_overlap = overlaps[np.argmin(cost)]
        print("Old overlap : ", old_overlap)
        print("New overlap : ", optimal_overlap)
        self.set_affine(overlap_fraction=optimal_overlap)

        # DEBUG : Repeat for ox vs oy
        old_overlap = self.overlap_fraction
        overlaps = np.arange(omin, omax, step)
        cost = []
        for o in tqdm(overlaps):
            affine = np.array([[self.tile_size_x * (1 - o), 0], [0, self.tile_size_y * (1 - old_overlap)]])
            self.affine = affine
            c = self.global_overlap_similarity(random_fraction=random_fraction, threshold=threshold)
            cost.append(c)

        # Extract the optimal overlap
        optimal_overlap = overlaps[np.argmin(cost)]
        print("Old ox overlap : ", old_overlap)
        print("New ox overlap : ", optimal_overlap)

        old_overlap = self.overlap_fraction
        overlaps = np.arange(omin, omax, step)
        cost = []
        for o in tqdm(overlaps):
            affine = np.array([[self.tile_size_x * (1 - old_overlap), 0], [0, self.tile_size_y * (1 - o)]])
            self.affine = affine
            c = self.global_overlap_similarity(random_fraction=random_fraction, threshold=threshold)
            cost.append(c)

        # Extract the optimal overlap
        optimal_overlap = overlaps[np.argmin(cost)]
        print("Old oy overlap : ", old_overlap)
        print("New oy overlap : ", optimal_overlap)

        if display:
            import matplotlib.pyplot as plt
            plt.plot(overlaps, cost)
            plt.axvline(optimal_overlap, color="r", linestyle="dashed", label=f"Optimal overlap: {optimal_overlap:.4f}")
            plt.xlabel("Overlap fraction")
            plt.ylabel(f"Error")
            plt.legend()
            plt.show()

    def optimize_affine(self, initial_overlap: float = 0.2, random_fraction=1.0, threshold=None):
        """Optimize the mosaic affine transform.

        :param initial_overlap: Initial overlap fraction (between 0 and 1), defaults to 0.2
        :type initial_overlap: float, optional
        """
        def loss(x):
            """Computing the normalized root mse over all the overlaps for a given transform"""
            self.affine = np.array(x).reshape((2, 2))
            c = self.global_overlap_similarity(random_fraction=random_fraction, threshold=threshold)
            return c

        def loss_grad(x):
            """Computing the loss gradient using the 1-pixel wide steps"""
            g = np.zeros_like(x)

            a, b, c, d = x[:]
            # x overlap gradient
            g[0] = (loss([a + 1, b, c, d]) - loss([a - 1, b, c, d])) / 2.0
            g[1] = (loss([a, b + 1, c, d]) - loss([a, b - 1, c, d])) / 2.0
            g[2] = (loss([a, b, c + 1, d]) - loss([a, b, c - 1, d])) / 2.0
            g[3] = (loss([a, b, c, d + 1]) - loss([a, b, c, d - 1])) / 2.0

            return g

        # Initialize the optimizer
        self.set_affine(initial_overlap)
        x0 = self.affine.ravel()
        min_overlap = self.tile_size_x * 0.5
        max_overlap = self.tile_size_x
        result = scipy.optimize.minimize(loss, x0, jac=loss_grad,
                                         bounds=((min_overlap, max_overlap),
                                                 (-64, 64),
                                                 (-64, 64),
                                                 (min_overlap, max_overlap)),
                                         options={'maxiter': 30, 'disp': True})
        if result.success:
            print("The optimization was a success!")
            print("The new affine matrix is:", result.x.reshape((2, 2)))
            self.affine = result.x.reshape((2, 2))
        else:
            print(f"The optimization failed. Using the affine for the {initial_overlap} overlap fraction.")
            self.set_affine(initial_overlap)


def addVolumeToMosaic(volume, pos, mosaic, blendingMethod='diffusion', factor=3, width=1.0):
    """Add a single volume into a mosaic, using the specified blendingMethod.

    :param vol: Volume to add to the mosaic
    :type vol: ndarray
    :param pos: Position of this volume in mosaic coordinates
    :type pos: (2,) tuple
    :param mosaic: Mosaic in which the volume is stitched
    :type mosaic: ndarray
    :param blendingMethod: Blending method to use (available : 'diffusion', 'average', 'none'), defaults to 'diffusion'
    :type blendingMethod: str, optional
    :param factor: Subsampling factor used by the diffusion blending method.
    :type factor: int, optional
    :param width: Blending transition width (between 0 and 1) used by the diffusion blending method, defaults to 1.0.
    :type width: float, optional
    :return: Updated mosaic
    :rtype: ndarray
    """
    # Mask representing the overlap of the mosaic and the new volume
    if volume.ndim == 3:
        nx, ny, nz = volume.shape
    elif volume.ndim == 2:
        nx, ny = volume.shape
        nz = 1
        volume = np.reshape(volume, [nx, ny, nz])

    # Position of tile in mosaic reference frame
    wx = int(pos[0])
    wy = int(pos[1])

    if len(pos) == 3:
        wz = pos[2]
    else:
        wz = 0

    if mosaic.ndim == 3 and mosaic.shape[2] != 1:
        mask = mosaic[wx:wx + nx, wy:wy + ny, wz:wz + nz].mean(axis=2) > 0  # Todo : Use a 3D mask instead.
    else:
        mask = np.squeeze(mosaic[wx:wx + nx, wy:wy + ny, 0]) > 0

    # Computing the blending weights
    if np.any(mask):
        if blendingMethod == 'diffusion':
            alpha = getDiffusionBlendingWeights(mask, factor=factor)

        elif blendingMethod == 'average':
            alpha = getAverageBlendingWeights(mask)

        else:  # Either none of unknown blending method
            alpha = np.ones([nx, ny])

    else:  # No overlap between mosaic and volume.
        alpha = np.ones([nx, ny])

    if width > 0 and width < 1 and blendingMethod == 'diffusion':
        lowThresh = 0.5 * (1.0 - width)
        highThresh = 1.0 - lowThresh
        alpha = (alpha - lowThresh) / float(highThresh - lowThresh)
        alpha[alpha > 1] = 1
        alpha[alpha < 0] = 0

    # Repeating the matrix for each z slice
    alpha = np.tile(np.reshape(alpha, [nx, ny, 1]), [1, 1, nz])

    # Adding the volume to the mosaic using the blending weights computed above
    try:
        mosaic[wx:wx + nx, wy:wy + ny, wz:wz + nz] = volume * alpha + (1 - alpha) * mosaic[wx:wx + nx, wy:wy + ny,
                                                                                    wz:wz + nz]
    except:
        print("Unable to add volume")
    return mosaic


def getAverageBlendingWeights(mask):
    """Computes the average blending weights over the mask in ND.

    :param mask: Bool ndarray describing the overlap.
    :type mask: ndarray
    :return: Blending weights
    :rtype: ndarray
    """
    alpha = np.ones_like(mask, dtype=float)
    alpha[mask] = 0.5
    return alpha


def getDiffusionBlendingWeights(fixedMask: np.ndarray, movingMask: np.ndarray = None, factor: int = 8,
                                nSteps: int = 5e2,
                                convergence_threshold: float = 1e-4, k: int = 1) -> np.ndarray:
    """Computes the diffusion blending (based on laplace equation) in 2D or 3D.

    :param fixedMask: Fixed volume mask to use as basis for the blending weights
    :param movingMask: Moving volume data mask. (If none is given, it assumes that the whole volume contains data.)
    :param factor: Subsampling factor
    :param nSteps: Number of diffusion steps.
    :param convergence_threshold: Convergence threshold used to end the diffusion.
    :param k: Structural element radius used to find the boundary of the mask.
    :return: ND blending weights.
    """

    def laplaceSolverStep(I, mask):
        dI = np.zeros_like(I)
        if I.ndim == 2:
            dI[1:-1, 1:-1] = I[0:-2, 1:-1] + I[2::, 1:-1] + I[1:-1, 0:-2] + I[1:-1, 2::] - 4 * I[1:-1, 1:-1]
            dI *= mask
            return dI / 4.0
        elif I.ndim == 3:
            dI[1:-1, 1:-1, 1:-1] = I[0:-2, 1:-1, 1:-1] + I[2::, 1:-1, 1:-1] + \
                                   I[1:-1, 0:-2, 1:-1] + I[1:-1, 2::, 1:-1] + \
                                   I[1:-1, 1:-1, 0:-2] + I[1:-1, 1:-1, 2::] - \
                                   6 * I[1:-1, 1:-1, 1:-1]
            dI *= mask
            return dI / 6.0

    if movingMask is None:
        movingMask = np.ones_like(fixedMask, dtype=bool)

    # Resampling
    old_shape = fixedMask.shape
    if factor > 1:
        new_shape = list(np.round(np.array(old_shape) / float(factor)).astype(int))
        small_fixedMask = resampleITK(fixedMask, new_shape, interpolator='NN')
        small_movingMask = resampleITK(movingMask, new_shape, interpolator='NN')
    else:
        new_shape = old_shape
        small_fixedMask = fixedMask
        small_movingMask = movingMask

    # Getting the boundary of the mask
    if fixedMask.ndim == 2:
        strel = disk(k)
    elif fixedMask.ndim == 3:
        strel = ball(k)

    small_mask = np.logical_and(small_fixedMask, small_movingMask)
    erodedMask = morpho.binary_erosion(small_mask, structure=strel)
    boundary_moving = np.logical_xor(small_movingMask, morpho.binary_erosion(small_movingMask, structure=strel))
    boundary_fixed = np.logical_xor(small_fixedMask, morpho.binary_erosion(small_fixedMask, structure=strel))
    boundary = np.logical_xor(small_mask, morpho.binary_erosion(small_mask, structure=strel))

    # Getting the boundary conditions
    bc = boundary.copy()
    bc = bc * morpho.binary_erosion(small_fixedMask, strel)

    # bc = morpho.binary_erosion(small_fixedMask, strel)*boundary
    dilatedMask = morpho.binary_dilation(~np.logical_or(small_fixedMask, small_mask), structure=strel)
    bc = np.zeros(new_shape)
    bc[boundary] = (~ dilatedMask[boundary]) * 1.0
    # del dilatedMask

    # Initialize alpha using gaussian smoothing
    alpha = gaussian_filter((bc == 1.0).astype(float), np.array(bc.shape) * 0.1)
    alpha = alpha * small_mask
    alpha = (alpha - alpha.min()) / float(alpha.max() - alpha.min())
    alpha[boundary] = bc[boundary]

    # Solve the Laplace Equation for this geometry
    rms = np.inf
    iStep = 0
    while rms > convergence_threshold and iStep < nSteps:
        dAlpha = laplaceSolverStep(alpha, erodedMask)
        try:
            if np.any(alpha[erodedMask] == 0):
                rms = np.inf
            else:
                rms = np.sqrt(np.mean((dAlpha[erodedMask] / alpha[erodedMask]) ** 2.0))
        except:
            rms = np.inf
        alpha += dAlpha
        iStep += 1

    # Resampling the blending weigths to the original resolution
    alpha[~morpho.binary_dilation(small_mask, strel)] = 1
    alpha[np.logical_xor(small_movingMask, small_mask)] = 0.0
    alpha[np.logical_xor(small_fixedMask, small_mask)] = 1.0
    alpha = 1.0 - alpha

    if factor > 1:
        alpha = resampleITK(alpha, old_shape, interpolator='linear')

    return alpha


def resampleITK(vol: np.ndarray, newshape: tuple, interpolator: str = 'linear') -> np.ndarray:
    """Resamples a volume / image using ITK.

    :param vol: 2D/3D array to resample.
    :param newshape: New shape of the array, or resampling factor (if a single integer is given)
    :param interpolation: Interpolation method to use. Available are: 'NN' (NearestNeighbor) and 'linear'
    :return: Resampled array
    """
    resample = sitk.ResampleImageFilter()

    # Computing newshape if a factor is given
    if isinstance(newshape, int):
        newshape = np.round(np.array(vol.shape) / float(newshape)).astype(int)
    else:
        newshape = [int(x) for x in newshape]

    if vol.dtype == bool:
        isBool = True
        vol = 255 * vol.astype(np.uint8)
    else:
        isBool = False

    if vol.ndim == 3:
        if vol.shape[2] == 1:
            vol = np.squeeze(vol, axis=(2,))
            newshape = newshape[0:2]

    if vol.ndim == 2:
        nx, ny = vol.shape
        ox, oy = newshape
        resample.SetSize([oy, ox])
        resample.SetOutputSpacing([(ny - 1) / float(oy), (nx - 1) / float(ox)])
        if nx / float(ox) > 1 or ny / float(oy) > 1:  # Smoothing if downsampling
            vol = gaussian_filter(vol, sigma=[nx / float(2 * ox), ny / float(2 * oy)])

    elif vol.ndim == 3:
        nx, ny, nz = vol.shape
        ox, oy, oz = newshape
        resample.SetSize([oz, oy, ox])
        resample.SetOutputSpacing([(nz - 1) / float(oz), (ny - 1) / float(oy), (nx - 1) / float(ox)])
        if nx / float(ox) > 1 or ny / float(oy) > 1 or nz / float(oz) > 1:  # Smoothing if downsampling
            vol = gaussian_filter(vol, sigma=[nx / float(2 * ox), ny / float(2 * oy), nz / float(2 * oz)])

    if interpolator == 'NN':
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    elif interpolator == 'linear':
        resample.SetInterpolator(sitk.sitkLinear)
    else:
        resample.SetInterpolator(sitk.sitkLinear)

    vol_itk = sitk.GetImageFromArray(vol)
    output_itk = resample.Execute(vol_itk)

    if isBool:
        vol_p = sitk.GetArrayFromImage(output_itk)
        # vol_p = vol_p == vol_p.max()
        vol_p = vol_p > vol_p.max() * 0.5
    else:
        vol_p = sitk.GetArrayFromImage(output_itk)

    return vol_p
