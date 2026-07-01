#!/usr/bin/python3
"""Mosaic grid management and stitching utilities."""

from collections.abc import Sequence
from typing import Any

import numpy as np
import scipy.ndimage.morphology as morpho
import scipy.optimize
from scipy.ndimage import gaussian_filter
from skimage.morphology import ball, disk
from tqdm import tqdm

from linumpy.geometry.resample import resample_itk

# TODO: Add an algorithm to estimate the affine transform parameters


class MosaicGrid:
    """Manage and process mosaic grid images.

    A mosaic grid is a 2D image containing all the tiles
    for a given mosaic, without any overlap. This class can be used for instance to apply processing to all tiles, to
    optimize the affine transform matrix describing the tile position, and to stitch the tiles together to obtain the
    reconstructed mosaic.


    .. note::
        This class can only deal with 2D mosaic grids for now. To generate a 2D mosaic grid from a collection of volumetric
        tiles for a given slice, you can use the :ref:`script-linum-create-mosaic-grid` script.

    """

    def __init__(self, image: np.ndarray, tile_shape: tuple | Sequence = (512, 512), overlap_fraction: float = 0.2) -> None:
        """Initialize the MosaicGrid instance."""
        self.tile_shape = tile_shape
        self.tile_size_x = self.tile_shape[0]
        self.tile_size_y = self.tile_shape[1]
        self.overlap_fraction = overlap_fraction
        self.blending_method = None
        self.dtype = image.dtype
        self.imin = image.min()
        self.imax = image.max()
        self.image = image

        self.compute_mosaic_shape()
        self.set_affine(self.overlap_fraction)

    def set_affine(self, overlap_fraction: float = 0.2) -> None:
        """Set the affine matrix given an overlap fraction.

        Parameters
        ----------
        overlap_fraction : float, optional
            An overlap fraction between 0 and 1, by default 0.2.
        """
        self.affine = np.eye(2) * (1 - overlap_fraction) * np.array(self.tile_shape[0:2]).T
        self.overlap_fraction = overlap_fraction

    def set_blending_method(self, method: str = "none") -> None:
        """To set the blending method. Available methodes are 'none' and 'average', 'diffusion'."""
        available_methods = ["none", "average", "diffusion"]
        method = str(method).lower()
        assert method in available_methods, f"Available blending methods are : {available_methods}"
        if method == "none":
            self.blending_method = None
        else:
            self.blending_method = method

    def get_image(self) -> np.ndarray:
        """To get the original image."""
        return self.image

    def compute_mosaic_shape(self) -> None:
        """Compute the mosaic grid shape."""
        self.n_tiles_x = self.image.shape[0] // self.tile_size_x
        self.n_tiles_y = self.image.shape[1] // self.tile_size_y

    def get_tiles(self) -> tuple:
        """Return the tiles from the mosaic grid.

        Returns
        -------
        tuple
            Tiles and tile positions in the grid.
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

    def get_neighbors_around_tile(self, x: int, y: int, neighborhood_type: str = "N4") -> tuple:
        """Return all neighboring tiles around a given tile position."""
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

    def get_neighbors_list(self, neighborhood_type: str = "N4") -> list:
        """Return a list of neighboring tiles.

        Parameters
        ----------
        neighborhood_type : str, optional
            Type of neighborhood to consider. 'N4' for horizontal and vertical
            neighbors, 'N8' to also consider diagonal neighbors.

        Returns
        -------
        list
            A list of neighbor pairs given by their grid position.

        Notes
        -----
        This also updates the `neighbors_list` object property.
        """
        neighbors_list = []
        # First add horizontal neighbors
        neighbors_list.extend(((x, y), (x + 1, y)) for y in range(self.n_tiles_y) for x in range(self.n_tiles_x - 1))

        # Second add vertical neighbors
        neighbors_list.extend(((x, y), (x, y + 1)) for x in range(self.n_tiles_x) for y in range(self.n_tiles_y - 1))

        if neighborhood_type == "N8":
            # First diagonal neighbors
            neighbors_list.extend(
                ((x, y), (x + 1, y + 1)) for y in range(self.n_tiles_y - 1) for x in range(self.n_tiles_x - 1)
            )

            # Second diagonal neighbors
            neighbors_list.extend(
                ((x, y), (x - 1, y + 1)) for y in range(self.n_tiles_y - 1) for x in range(1, self.n_tiles_x)
            )

        self.neighbors_list = neighbors_list

        return self.neighbors_list

    def get_tile(self, x: int, y: int) -> np.ndarray:
        """Extract a tile from the mosaic grid.

        Parameters
        ----------
        x : int
            x position within the mosaic grid.
        y : int
            y position within the mosaic grid.

        Returns
        -------
        ndarray
            2D tile.
        """
        assert x < self.n_tiles_x and y < self.n_tiles_y, f"Mosaic shape is {self.tile_size_x}x{self.tile_size_y}"
        x0 = self.tile_size_x * x
        y0 = self.tile_size_y * y
        xf = x0 + self.tile_size_x
        yf = y0 + self.tile_size_y
        tile = self.image[x0:xf, y0:yf]
        return tile

    def set_tile(self, x: int, y: int, tile: np.ndarray) -> None:
        """Set a tile from the mosaic grid.

        Parameters
        ----------
        x : int
            x position within the mosaic grid.
        y : int
            y position within the mosaic grid.
        tile : ndarray
            2D tile.
        """
        assert x < self.n_tiles_x and y < self.n_tiles_y, f"Mosaic shape is {self.tile_size_x}x{self.tile_size_y}"
        x0 = self.tile_size_x * x
        y0 = self.tile_size_y * y
        xf = x0 + self.tile_size_x
        yf = y0 + self.tile_size_y
        self.image[x0:xf, y0:yf] = tile

    def get_position(self, x: int, y: int) -> np.ndarray:
        """Compute the cartesian position of a given tile using the internal affine transform.

        Parameters
        ----------
        x : int
            x position within the mosaic grid.
        y : int
            y position within the mosaic grid.

        Returns
        -------
        ndarray
            (2,) array containing the cartesian position of this tile (in pixel).
        """
        pos = np.dot(self.affine, [x, y]).astype(int)
        return pos

    def get_neighbor_tiles(self, n_id: int) -> tuple:
        """Extract the tiles for a given neighbor pair.

        Parameters
        ----------
        n_id : int
            The neighbor pair id.

        Returns
        -------
        tuple
            (2,) tuple containing each tile as a np.ndarray.
        """
        if not hasattr(self, "neighbors_list"):
            self.get_neighbors_list()

        assert n_id < len(self.neighbors_list), "The neighbor id exceeds the number of neighbors"
        neighbor = self.neighbors_list[n_id]
        tile_1 = self.get_tile(*neighbor[0])
        tile_2 = self.get_tile(*neighbor[1])
        return (tile_1, tile_2)

    def get_neighbor_overlap_from_pos(self, p1: Sequence[int] | np.ndarray, p2: Sequence[int] | np.ndarray) -> tuple:
        """Compute the overlapping regions between two tiles given their positions."""
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
        nz = 1 if ndim == 2 else t1.shape[2]

        # Get the min and max coordinates for this mosaic
        x0 = int(min([p1[0], p2[0]]))
        xf = int(max([p1[0], p2[0]]) + nx)
        y0 = int(min([p1[1], p2[1]]))
        yf = int(max([p1[1], p2[1]]) + ny)

        # Create empty mosaics with each tile
        mosaic1 = np.zeros((xf - x0, yf - y0, nz))
        mosaic2 = np.zeros((xf - x0, yf - y0, nz))

        mosaic1[p1[0] - x0 : p1[0] - x0 + nx, p1[1] - y0 : p1[1] - y0 + ny, :] = t1 + 1
        mosaic2[p2[0] - x0 : p2[0] - x0 + nx, p2[1] - y0 : p2[1] - y0 + ny, :] = t2 + 1

        # Find intersection
        mask = mosaic1 * mosaic2 >= 1

        # Convert this into t1 and t2 coordinates
        x, y, _z = np.where(mask)
        o_xmin = x.min()
        o_ymin = y.min()
        o_xmax = x.max()
        o_ymax = y.max()

        o_pos1 = (o_xmin - (p1[0] - x0), o_ymin - (p1[1] - y0), o_xmax - (p1[0] - x0), o_ymax - (p1[1] - y0))
        o_pos2 = (o_xmin - (p2[0] - x0), o_ymin - (p2[1] - y0), o_xmax - (p2[0] - x0), o_ymax - (p2[1] - y0))

        # Getting overlap
        overlap1 = t1[o_pos1[0] : o_pos1[2], o_pos1[1] : o_pos1[3], :]
        overlap2 = t2[o_pos2[0] : o_pos2[2], o_pos2[1] : o_pos2[3], :]

        if ndim == 2:
            overlap1 = np.squeeze(overlap1)
            overlap2 = np.squeeze(overlap2)

        return (overlap1, overlap2, o_pos1, o_pos2)

    def get_neighbor_overlap(self, n_id: int) -> tuple:
        """Extract the tile overlaps for a given neighbor pair.

        Parameters
        ----------
        n_id : int
            The neighbor pair id.

        Returns
        -------
        tuple
            (4,) tuple containing (overlap1, overlap2, overlap1_position, overlap2_position).
        """
        p1, p2 = self.neighbors_list[n_id]
        return self.get_neighbor_overlap_from_pos(p1, p2)

    def crop_tiles(self, xlim: tuple = (0, -1), ylim: tuple = (0, -1)) -> None:
        """Crop all tiles in the mosaic grid.

        Parameters
        ----------
        xlim : tuple, optional
            (2,) tuple containing the x-axis (row) cropping limits.
        ylim : tuple, optional
            (2,) tuple containing the y-axis (col) cropping limits.

        Notes
        -----
        This also resets the affine transform using the overlap_fraction.
        """
        xlim_mut = list(xlim)
        ylim_mut = list(ylim)
        if xlim_mut[1] < 0:
            xlim_mut[1] = self.tile_shape[0] + xlim_mut[1] + 1
        if ylim_mut[1] < 0:
            ylim_mut[1] = self.tile_shape[1] + ylim_mut[1] + 1

        nx = xlim_mut[1] - xlim_mut[0]
        ny = ylim_mut[1] - ylim_mut[0]
        new_shape = (self.n_tiles_x * (xlim_mut[1] - xlim_mut[0]), self.n_tiles_y * (ylim_mut[1] - ylim_mut[0]))
        image = np.zeros(new_shape, dtype=np.float32)
        for x in range(self.n_tiles_x):
            for y in range(self.n_tiles_y):
                tile = self.get_tile(x, y)

                # New positions
                x0 = x * nx
                xf = x0 + nx
                y0 = y * ny
                yf = y0 + ny
                image[x0:xf, y0:yf] = tile[xlim_mut[0] : xlim_mut[1], ylim_mut[0] : ylim_mut[1]]

        self.image = image
        self.tile_shape = (nx, ny)
        self.tile_size_x = nx
        self.tile_size_y = ny

        self.set_affine(overlap_fraction=self.overlap_fraction)  # FIXME : Overlap fraction need to be adjusted after cropping

    def get_stitched_image(self, blending_method: str = "none") -> np.ndarray:
        """Perform a 2D reconstruction of the mosaic grid.

        Parameters
        ----------
        blending_method : str, optional
            Blending method. Available: 'none', 'average', 'diffusion'.

        Returns
        -------
        ndarray
            Stitched mosaic.

        Notes
        -----
        The affine transform obtained from the overlap fraction or by the affine transform optimization is used
        for the reconstruction.
        """
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

        image = np.zeros((1, x_shape, y_shape), dtype=np.float32)
        for x in range(self.n_tiles_x):
            for y in range(self.n_tiles_y):
                tile = self.get_tile(x, y)
                if np.all(tile == 0):  # Empty tile, ignore
                    continue
                    print(x, y)
                px, py = self.get_position(x, y)
                x0 = np.floor(px - xmin).astype(int)
                y0 = np.floor(py - ymin).astype(int)
                image = add_volume_to_mosaic(tile, (x0, y0), image, blending_method=self.blending_method)

        return image.squeeze()

    def global_overlap_similarity(self, random_fraction: float = 1.0, threshold: float | None = None) -> float:
        """Compute a global overlap similarity error across all tile pairs."""
        neighbors = self.get_neighbors_list(neighborhood_type="N4")
        n_neighbors = len(neighbors)
        neighbors_ids = list(range(n_neighbors))
        np.random.shuffle(neighbors_ids)

        error = 0.0
        n_samples = 0

        i = 0
        while (i < n_neighbors) and (n_samples / float(n_neighbors) < random_fraction):
            o1, o2, _p1, _p2 = self.get_neighbor_overlap(neighbors_ids[i])

            if threshold is None:
                if np.all(o1 == 0) or np.all(o2 == 0):  # Ignore empty overlaps
                    continue
            else:
                m1 = o1 < threshold
                m2 = o2 < threshold
                if np.all(not m1) or np.all(not m2):
                    continue

            # TODO: Test other error metrics, this one doesn't work well when there is illumination inhomogeneity
            error += np.sqrt(((o1 - o2) ** 2).mean())
            n_samples += 1
            i += 1
        if n_samples > 0:
            error = error / float(n_samples)
        return error

    def optimize_overlap(
        self,
        step: float = 0.01,
        omin: float = 0.1,
        omax: float = 0.5,
        display: bool = False,
        random_fraction: float = 1.0,
        threshold: float | None = None,
    ) -> None:
        """Use the similarity between every neighboring tiles to estimate the overlap fraction.

        Parameters
        ----------
        step : float, optional
            Overlap fraction steps used for the search.
        omin : float, optional
            Minimum overlap fraction to consider.
        omax : float, optional
            Maximum overlap fraction to consider.
        display : bool, optional
            If set to true, the similarity curve will be displayed at the end of the optimization.
        random_fraction : float, optional
            Fraction of tiles to use for the similarity computation, by default 1.0.
        threshold : float, optional
            Threshold for the similarity computation, by default None.
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
            plt.ylabel("Error")
            plt.legend()
            plt.show()

    def optimize_affine(
        self,
        initial_overlap: float = 0.2,
        random_fraction: float = 1.0,
        threshold: float | None = None,
    ) -> None:
        """Optimize the mosaic affine transform.

        Parameters
        ----------
        initial_overlap : float, optional
            Initial overlap fraction (between 0 and 1), by default 0.2.
        random_fraction : float, optional
            Fraction of tiles to use for the similarity computation, by default 1.0.
        threshold : float, optional
            Threshold for the similarity computation, by default None.
        """

        def loss(x: np.ndarray | list) -> float:
            """Compute the normalized root mse over all the overlaps for a given transform."""
            self.affine = np.array(x).reshape((2, 2))
            c = self.global_overlap_similarity(random_fraction=random_fraction, threshold=threshold)
            return c

        def loss_grad(x: np.ndarray) -> np.ndarray:
            """Compute the loss gradient using the 1-pixel wide steps."""
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
        result = scipy.optimize.minimize(
            loss,
            x0,
            jac=loss_grad,
            bounds=((min_overlap, max_overlap), (-64, 64), (-64, 64), (min_overlap, max_overlap)),
            options={"maxiter": 30, "disp": True},
        )
        if result.success:
            print("The optimization was a success!")
            print("The new affine matrix is:", result.x.reshape((2, 2)))
            self.affine = result.x.reshape((2, 2))
        else:
            print(f"The optimization failed. Using the affine for the {initial_overlap} overlap fraction.")
            self.set_affine(initial_overlap)


def add_volume_to_mosaic(
    volume: np.ndarray,
    pos: tuple | list,
    mosaic: Any,
    blending_method: str | None = "diffusion",
    factor: int = 3,
    width: float = 1.0,
) -> np.ndarray:
    """Add a single volume into a mosaic.

    Parameters
    ----------
    volume : ndarray
        Volume to add to the mosaic
    pos : (2,) tuple
        Position of this volume in mosaic coordinates (XY in pixel)
    mosaic : ndarray
        Mosaic in which the volume is stitched
    blending_method : str, optional
        Blending method to use (available : 'diffusion', 'average', 'none')
    factor : int, optional
        Subsampling factor used by the diffusion blending method.
    width : float, optional
        Blending transition width (between 0 and 1) used by the diffusion blending method, defaults to 1.0.

    Returns
    -------
    ndarray
        The updated mosaic
    """
    # Mask representing the overlap of the mosaic and the new volume
    nz = 1
    nx = 0
    ny = 0
    if volume.ndim == 3:
        nz, nx, ny = volume.shape
    elif volume.ndim == 2:
        nx, ny = volume.shape
        nz = 1
        volume = np.reshape(volume, [nz, nx, ny])

    # Position of tile in mosaic reference frame
    wx = int(pos[0])
    wy = int(pos[1])

    wz = pos[2] if len(pos) == 3 else 0

    if mosaic.ndim == 3 and mosaic.shape[0] != 0:
        mask = mosaic[wz : wz + nz, wx : wx + nx, wy : wy + ny].mean(axis=0) > 0
    else:
        mask = np.squeeze(mosaic[wx : wx + nx, wy : wy + ny]) > 0

    # Computing the blending weights
    if np.any(mask):
        if blending_method == "diffusion":
            alpha = get_diffusion_blending_weights(mask, factor=factor)

        elif blending_method == "average":
            alpha = get_average_blending_weights(mask)

        else:  # Either none of unknown blending method
            alpha = np.ones([nx, ny])

    else:  # No overlap between mosaic and volume.
        alpha = np.ones([nx, ny])

    # Adjusting the blending weights for the diffusion method
    if 0 < width < 1 and blending_method == "diffusion":
        low_thresh = 0.5 * (1.0 - width)
        high_thresh = 1.0 - low_thresh
        alpha = (alpha - low_thresh) / float(high_thresh - low_thresh)
        alpha[alpha > 1] = 1
        alpha[alpha < 0] = 0

    # Repeating the matrix for each z slice
    alpha = np.tile(np.reshape(alpha, [1, nx, ny]), [nz, 1, 1])

    # Adding the volume to the mosaic using the blending weights computed above
    if mosaic.ndim == 3:
        mosaic[wz : wz + nz, wx : wx + nx, wy : wy + ny] = (
            volume * alpha + (1 - alpha) * mosaic[wz : wz + nz, wx : wx + nx, wy : wy + ny]
        )
    else:
        mosaic[wx : wx + nx, wy : wy + ny] = volume * alpha + (1 - alpha) * mosaic[wx : wx + nx, wy : wy + ny]

    return mosaic


def get_average_blending_weights(mask: np.ndarray) -> np.ndarray:
    """Compute the average blending weights over the mask in ND.

    Parameters
    ----------
    mask : ndarray
        Bool ndarray describing the overlap.

    Returns
    -------
    ndarray
        Blending weights.
    """
    alpha = np.ones_like(mask, dtype=float)
    alpha[mask] = 0.5
    return alpha


def get_diffusion_blending_weights(
    fixed_mask: np.ndarray,
    moving_mask: np.ndarray | None = None,
    factor: int = 8,
    n_steps: int = 500,
    convergence_threshold: float = 1e-4,
    k: int = 1,
) -> np.ndarray:
    """Compute the diffusion blending (based on laplace equation) in 2D or 3D.

    Parameters
    ----------
    fixed_mask : ndarray
        Fixed volume mask to use as basis for the blending weights.
    moving_mask : ndarray, optional
        Moving volume data mask. If none is given, it assumes that the whole volume contains data.
    factor : int, optional
        Subsampling factor.
    n_steps : int, optional
        Number of diffusion steps.
    convergence_threshold : float, optional
        Convergence threshold used to end the diffusion.
    k : int, optional
        Structural element radius used to find the boundary of the mask.

    Returns
    -------
    ndarray
        ND blending weights.
    """

    def laplace_solver_step(field: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
        dI = np.zeros_like(field)
        if field.ndim == 2:
            dI[1:-1, 1:-1] = (
                field[0:-2, 1:-1] + field[2::, 1:-1] + field[1:-1, 0:-2] + field[1:-1, 2::] - 4 * field[1:-1, 1:-1]
            )
            dI *= mask
            return dI / 4.0
        elif field.ndim == 3:
            dI[1:-1, 1:-1, 1:-1] = (
                field[0:-2, 1:-1, 1:-1]
                + field[2::, 1:-1, 1:-1]
                + field[1:-1, 0:-2, 1:-1]
                + field[1:-1, 2::, 1:-1]
                + field[1:-1, 1:-1, 0:-2]
                + field[1:-1, 1:-1, 2::]
                - 6 * field[1:-1, 1:-1, 1:-1]
            )
            dI *= mask
            return dI / 6.0

    if moving_mask is None:
        moving_mask = np.ones_like(fixed_mask, dtype=bool)

    # Resampling
    old_shape = fixed_mask.shape
    if factor > 1:
        new_shape = list(np.round(np.array(old_shape) / float(factor)).astype(int))
        small_fixed_mask = resample_itk(fixed_mask, new_shape, interpolator="NN")
        small_moving_mask = resample_itk(moving_mask, new_shape, interpolator="NN")
    else:
        new_shape = old_shape
        small_fixed_mask = fixed_mask
        small_moving_mask = moving_mask

    # Getting the boundary of the mask
    strel: np.ndarray | None = None
    if fixed_mask.ndim == 2:
        strel = disk(k)
    elif fixed_mask.ndim == 3:
        strel = ball(k)

    small_mask = np.logical_and(small_fixed_mask, small_moving_mask)
    eroded_mask = morpho.binary_erosion(small_mask, structure=strel)
    boundary = np.logical_xor(small_mask, morpho.binary_erosion(small_mask, structure=strel))

    # Getting the boundary conditions
    bc = boundary.copy()
    bc = bc * morpho.binary_erosion(small_fixed_mask, strel)

    dilated_mask = morpho.binary_dilation(~np.logical_or(small_fixed_mask, small_mask), structure=strel)
    bc = np.zeros(new_shape)
    bc[boundary] = (~dilated_mask[boundary]) * 1.0

    # Initialize alpha using gaussian smoothing
    alpha = gaussian_filter((bc == 1.0).astype(float), np.array(bc.shape) * 0.1)
    alpha = alpha * small_mask
    alpha = (alpha - alpha.min()) / float(alpha.max() - alpha.min())
    alpha[boundary] = bc[boundary]

    # Solve the Laplace Equation for this geometry
    rms = np.inf
    i_step = 0
    while rms > convergence_threshold and i_step < n_steps:
        d_alpha = laplace_solver_step(alpha, eroded_mask)
        if d_alpha is not None:
            try:
                if np.any(alpha[eroded_mask] == 0):
                    rms = np.inf
                else:
                    rms = np.sqrt(np.mean((d_alpha[eroded_mask] / alpha[eroded_mask]) ** 2.0))
            except Exception:
                rms = np.inf
            alpha += d_alpha
        i_step += 1

    # Resampling the blending weigths to the original resolution
    alpha[~morpho.binary_dilation(small_mask, strel)] = 1
    alpha[np.logical_xor(small_moving_mask, small_mask)] = 0.0
    alpha[np.logical_xor(small_fixed_mask, small_mask)] = 1.0
    alpha = 1.0 - alpha

    if factor > 1:
        alpha = resample_itk(alpha, old_shape, interpolator="linear")

    return alpha
