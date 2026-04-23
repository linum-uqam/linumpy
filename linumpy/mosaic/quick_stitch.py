"""Quick-stitch tiles into a single 2D mosaic image and detect tissue ROI."""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from imageio import imwrite
from matplotlib.patches import Rectangle
from scipy.ndimage import binary_fill_holes, median_filter
from skimage.color import label2rgb
from skimage.filters import threshold_otsu
from skimage.transform import resize
from tqdm.auto import tqdm

from linumpy.microscope.oct import OCT
from linumpy.mosaic.discovery import get_largest_cc, get_mosaic_info


def quick_stitch(
    directory: Path,
    z: int,
    overlap_fraction: float = 0.2,
    n_rot: int = 3,
    zmin: int = 0,
    zmax: int = -1,
    use_log: bool = False,
    use_stage_positions: bool = False,
    flip_ud: bool = True,
    flip_lr: bool = False,
    galvo_shift: int | None = None,
    galvo_shift_first_tile: tuple = (0, 0),
) -> np.ndarray:
    """Quickly stitch tiles at a given z slice into a mosaic image."""
    # TODO: accelerate the stitching by preprocessing the tiles in parallel
    input_directory = Path(directory)

    # Get a list of the input tiles
    tiles_to_process = f"*z{z:02d}"
    tiles = list(input_directory.glob(tiles_to_process))

    # Get the tile positions (in pixel and mm)
    file_pattern = r"tile_x(?P<x>\d+)_y(?P<y>\d+)_z(?P<z>\d+)"
    tiles_positions_px = []
    tiles_positions_mm = []
    oct_tile: OCT | None = None
    for t in tiles:
        oct_tile = OCT(t)
        if oct_tile.position_available and use_stage_positions:
            x_mm, y_mm, _ = oct_tile.position
        else:
            # Extract the tile's mosaic position.
            match = re.match(file_pattern, t.name)
            assert match is not None
            mx = int(match.group("x"))
            my = int(match.group("y"))

            # Compute the tile position in mm
            x_mm = oct_tile.dimension[0] * (1 - overlap_fraction) * mx
            y_mm = oct_tile.dimension[1] * (1 - overlap_fraction) * my

        x_px = int(np.floor(x_mm / oct_tile.resolution[0]))
        y_px = int(np.floor(y_mm / oct_tile.resolution[1]))

        tiles_positions_mm.append((x_mm, y_mm))
        tiles_positions_px.append((x_px, y_px))

    assert oct_tile is not None
    # Compute the mosaic shape
    x_min = min([x for x, _ in tiles_positions_px])
    y_min = min([y for _, y in tiles_positions_px])
    x_max = max([x for x, _ in tiles_positions_px]) + oct_tile.shape[0]
    y_max = max([y for _, y in tiles_positions_px]) + oct_tile.shape[1]
    mosaic_nrows = x_max - x_min
    mosaic_ncols = y_max - y_min
    mosaic = np.zeros((mosaic_nrows, mosaic_ncols), dtype=np.float32)

    # Perform stitching
    for i in tqdm(range(len(tiles)), desc="Quick Stitch"):
        oct_tile = OCT(tiles[i])

        # Compute the pixel position within the mosaic
        rmin = tiles_positions_px[i][0] - x_min
        rmax = rmin + oct_tile.shape[0]
        cmin = tiles_positions_px[i][1] - y_min
        cmax = cmin + oct_tile.shape[1]

        # Get the tile id
        match = re.match(file_pattern, tiles[i].name)
        assert match is not None
        mx = int(match.group("x"))
        my = int(match.group("y"))

        apply_shift = True
        if mx < galvo_shift_first_tile[0] or (mx == galvo_shift_first_tile[0] and my < galvo_shift_first_tile[1]):
            apply_shift = False

        # Load the fringes
        img = (
            oct_tile.load_image(fix_galvo_shift=galvo_shift if galvo_shift is not None else True)
            if apply_shift
            else oct_tile.load_image()
        )

        # Log transform
        if use_log:
            img = np.log(img)

        # Compute an AIP
        img = img[zmin:zmax, :, :].mean(axis=0)

        # BUG: there are sometimes missing bscans
        oct_shape_2d = (int(oct_tile.shape[0]), int(oct_tile.shape[1]))
        if img.shape != oct_shape_2d:
            img = np.zeros(oct_shape_2d) if np.any(np.array(img.shape) == 0) else resize(img, oct_shape_2d)

        # Apply rotations
        img = np.rot90(img, k=n_rot)

        # Flips
        if flip_lr:
            img = np.fliplr(img)

        if flip_ud:
            img = np.flipud(img)

        # Add the tile to the mosaic
        mosaic[rmin:rmax, cmin:cmax] = img

    return mosaic


def detect_mosaic(
    directory: Path,
    z: int,
    img: np.ndarray | None = None,
    margin: float = 0.5,
    display: bool = False,
    image_file: Path | None = None,
    roi_file: str | None = None,
    keep_largest_island: bool = False,
    stitching_settings: dict | None = None,
) -> tuple[float, float, float, float]:
    """Detect the tissue in the mosaic and compute the limits of the tissue.

    Parameters
    ----------
    directory : str
        The directory containing the tiles.
    z : int
        The z slices to process
    img : ndarray, optional
        Pre-computed quickstitch image. If None, it will be computed.
    margin : float
        The margin to add to the tissue limits (in mm).
    display : bool
        Display the result in a matplotlib window.
    image_file : str, optional
        The filename to save the quickstitch image.
    roi_file : str, optional
        The filename to save the ROI image.
    keep_largest_island : bool
        Keep the largest connected component in the mask.
    stitching_settings : dict, optional
        Settings dict to pass to the stitching function.
    """
    # Additional parameters
    threshold_size = 1024  # maximum image size to use for the thresholding
    normalization_percentile = 99.7
    median_size = 15  # pixel

    # Extract the parameters
    directory = Path(directory)

    # Get the mosaic information
    info = get_mosaic_info(directory, z=z, use_stage_positions=True)

    # Extract the tile positions from the metadata
    xmin = np.min([p[0] for p in info["tiles_pos_mm"]]) - info["tile_shape_mm"][0] / 2
    ymin = np.min([p[1] for p in info["tiles_pos_mm"]]) - info["tile_shape_mm"][1] / 2
    xmax = np.max([p[0] for p in info["tiles_pos_mm"]]) + info["tile_shape_mm"][0] / 2
    ymax = np.max([p[1] for p in info["tiles_pos_mm"]]) + info["tile_shape_mm"][1] / 2

    # Stitch the image using the tile position
    if img is None:
        extra = stitching_settings if stitching_settings is not None else {}
        img = quick_stitch(directory, z=z, use_stage_positions=True, **extra)

    # Save the quick stitch image
    if image_file is not None:
        save_quickstitch(img, image_file)

    # Rescale the image to a small size
    new_shape = tuple((np.array(img.shape) * threshold_size / np.min(img.shape)).astype(int).tolist())
    img = resize(img, new_shape)

    # Normalize the intensity
    img = (img.astype(np.float32) - img.min()) / (np.percentile(img, normalization_percentile) - img.min())
    img[img > 1] = 1

    # Process the image, to find a mask
    thresh = threshold_otsu(img)
    mask = img > thresh
    mask = median_filter(mask, median_size)

    # Fill holes
    mask = binary_fill_holes(mask)

    # Keep the largest connected component
    if keep_largest_island:
        mask = get_largest_cc(mask)

    # Compute the mosaic limits
    n_rows, n_cols = img.shape
    rows, cols = np.where(mask)
    roi_r_min = rows.min()
    roi_r_max = rows.max()
    roi_c_min = cols.min()
    roi_c_max = cols.max()

    # Convert to mm
    roi_x_min = (xmax - xmin) * roi_r_min / n_rows + xmin
    roi_x_max = (xmax - xmin) * roi_r_max / n_rows + xmin
    roi_y_min = (ymax - ymin) * roi_c_min / n_cols + ymin
    roi_y_max = (ymax - ymin) * roi_c_max / n_cols + ymin

    # Add margin
    roi_x_min_margin = roi_x_min - margin
    roi_x_max_margin = roi_x_max + margin
    roi_y_min_margin = roi_y_min - margin
    roi_y_max_margin = roi_y_max + margin

    # TODO: Make sure the mosaic limits are within the allowed imaging limits

    # Display the result
    if display or roi_file is not None:
        _fig, ax = plt.subplots()
        ax.imshow(label2rgb(mask, img, bg_label=0, colors=["blue"]), extent=(ymin, ymax, xmax, xmin))  # Y axes are inverted

        rect = Rectangle(
            (roi_y_min, roi_x_min),
            width=(roi_y_max - roi_y_min),
            height=(roi_x_max - roi_x_min),
            fill=None,
            edgecolor="red",
            linestyle="dashed",
            label="ROI",
        )
        ax.add_patch(rect)

        rect_margin = Rectangle(
            (roi_y_min_margin, roi_x_min_margin),
            width=(roi_y_max_margin - roi_y_min_margin),
            height=(roi_x_max_margin - roi_x_min_margin),
            fill=None,
            edgecolor="red",
            label="ROI + margin",
        )
        ax.add_patch(rect_margin)

        ax.set_ylabel("x axis (mm)")
        ax.set_xlabel("y axis (mm)")
        title = (
            f"xmin={roi_x_min_margin:.4f}mm, xmax={roi_x_max_margin:.4f}mm\n"
            f"ymin={roi_y_min_margin:.4f}mm, ymax={roi_y_max_margin:.4f}mm"
        )
        ax.set_title(title)
        ax.legend()

        if roi_file is not None:
            filename = Path(roi_file)
            filename.parent.mkdir(exist_ok=True, parents=True)
            plt.savefig(filename, dpi=150, bbox_inches="tight", pad_inches=0)

        if display:
            plt.show()

    return roi_x_min_margin, roi_x_max_margin, roi_y_min_margin, roi_y_max_margin


def save_quickstitch(img: np.ndarray, quickstitch_file: Path) -> None:
    """Normalize and save a quick-stitch mosaic image to disk."""
    filename = Path(quickstitch_file)
    # Normalize the intensity
    mask = img > 0
    imin = img[mask].min()
    imax = np.percentile(img[mask], 99.7)
    mosaic = (img - imin) / (imax - imin)
    mosaic[~mask] = 0.0
    # Save the mosaic
    if filename.name.endswith(".jpg") or filename.name.endswith(".png"):
        mosaic[mosaic > 1] = 1
        mosaic = (255 * mosaic).astype(np.uint8)
    elif filename.name.endswith(".tiff"):
        mosaic = mosaic.astype(np.float32)
    filename.parent.mkdir(exist_ok=True, parents=True)
    imwrite(filename, mosaic)
