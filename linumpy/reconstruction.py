#!/usr/bin/env python3

""""Quick reconstruction and processing methods for the S-OCT data."""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from imageio import imwrite
from matplotlib.patches import Rectangle
from scipy.ndimage import median_filter, binary_fill_holes
from skimage.color import label2rgb
from skimage.filters import threshold_otsu
from skimage.measure import label
from skimage.transform import resize
from tqdm.auto import tqdm

from linumpy.microscope.oct import OCT


def getLargestCC(segmentation: np.ndarray) -> np.ndarray:
    """Get the largest connected component in a binary image.
    Parameters
    ----------
    segmentation : np.ndarray
        The binary image to process.
    Returns
    -------
    np.ndarray
        The largest connected component.
    """
    labels = label(segmentation)
    assert (labels.max() != 0)  # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largestCC



DEFAULT_TILE_FILE_PATTERN = r"tile_x(?P<x>\d+)_y(?P<y>\d+)_z(?P<z>\d+)"


def get_tiles_ids(directory, z: int = None):
    """Analyzes a directory and detects all the tiles in contains"""
    input_directory = Path(directory)

    # Get a list of the input tiles
    if z is not None:
        tiles_to_process = f"*z{z:02d}"
    else:
        tiles_to_process = f"tile_*"
    tiles = list(input_directory.rglob(tiles_to_process))
    tiles = [t for t in tiles if t.name.startswith('tile_')]
    tile_ids = get_tiles_ids_from_list(tiles)

    return tiles, tile_ids


def get_tiles_ids_from_list(tiles_list,
                            file_pattern=DEFAULT_TILE_FILE_PATTERN):
    tiles_list.sort()

    # Get the tile positions
    tile_ids = []
    n_tiles = len(tiles_list)
    for t in tqdm(tiles_list, desc="Extracting tile ids", total=n_tiles):
        # Extract the tile's mosaic position.
        match = re.match(file_pattern, t.name)
        mx = int(match.group("x"))
        my = int(match.group("y"))
        mz = int(match.group("z"))
        tile_ids.append((mx, my, mz))

    return tile_ids



def get_mosaic_info(directory, z: int, overlap_fraction: float = 0.2, use_stage_positions: bool = False):
    # Get a list of the input tiles
    tiles, tile_ids = get_tiles_ids(directory, z)

    # Get the tile positions (in pixel and mm)
    file_pattern = r"tile_x(?P<x>\d+)_y(?P<y>\d+)_z(?P<z>\d+)"
    tiles_positions_px = []
    tiles_positions_mm = []
    mosaic_tile_pos = []
    # Progress bars overlap as the position is the same in all threads. Position is 1 to avoid overlap with outer loop.
    # No better solution has been found.
    for t in tqdm(tiles, desc="Reading mosaic info", leave=False, position=1):
        oct = OCT(t)

        # Extract the tile's mosaic position.
        match = re.match(file_pattern, t.name)
        mx = int(match.group("x"))
        my = int(match.group("y"))

        if oct.position_available and use_stage_positions:
            x_mm, y_mm, _ = oct.position
        else:
            # Compute the tile position in mm
            x_mm = oct.dimension[0] * (1 - overlap_fraction) * mx
            y_mm = oct.dimension[1] * (1 - overlap_fraction) * my

        x_px = int(np.floor(x_mm / oct.resolution[0]))
        y_px = int(np.floor(y_mm / oct.resolution[1]))

        mosaic_tile_pos.append((mx, my))
        tiles_positions_mm.append((x_mm, y_mm))
        tiles_positions_px.append((x_px, y_px))

    # Compute the mosaic shape
    x_min = min([x for x, _ in tiles_positions_px])
    y_min = min([y for _, y in tiles_positions_px])
    x_max = max([x for x, _ in tiles_positions_px]) + oct.shape[0]
    y_max = max([y for _, y in tiles_positions_px]) + oct.shape[1]
    mosaic_nrows = x_max - x_min
    mosaic_ncols = y_max - y_min

    # Get the mosaic grid shape
    n_mx = len(np.unique([x[0] for x in mosaic_tile_pos]))
    n_my = len(np.unique([x[1] for x in mosaic_tile_pos]))

    # Get the mosaic limits in mm
    xmin_mm = np.min([p[0] for p in tiles_positions_mm]) - oct.dimension[0] / 2
    ymin_mm = np.min([p[1] for p in tiles_positions_mm]) - oct.dimension[1] / 2
    xmax_mm = np.max([p[0] for p in tiles_positions_mm]) + oct.dimension[0] / 2
    ymax_mm = np.max([p[1] for p in tiles_positions_mm]) + oct.dimension[1] / 2
    mosaic_center_mm = ((xmin_mm + xmax_mm) / 2, (ymin_mm + ymax_mm) / 2)
    mosaic_width_mm = xmax_mm - xmin_mm
    mosaic_height_mm = ymax_mm - ymin_mm

    info = {
        "tiles": tiles,
        "tiles_pos_px": tiles_positions_px,
        "tiles_pos_mm": tiles_positions_mm,
        "mosaic_tile_pos": mosaic_tile_pos,
        "mosaic_nrows": mosaic_nrows,
        "mosaic_ncols": mosaic_ncols,
        "mosaic_xmin_px": x_min,
        "mosaic_ymin_px": y_min,
        "mosaic_xmax_px": x_max,
        "mosaic_ymax_px": y_max,
        "mosaic_xmin_mm": xmin_mm,
        "mosaic_ymin_mm": ymin_mm,
        "mosaic_xmax_mm": xmax_mm,
        "mosaic_ymax_mm": ymax_mm,
        "mosaic_center_mm": mosaic_center_mm,
        "mosaic_width_mm": mosaic_width_mm,
        "mosaic_height_mm": mosaic_height_mm,
        "mosaic_grid_shape": (n_mx, n_my),
        "tile_shape_px": oct.shape,
        "tile_shape_mm": oct.dimension,
        "tile_resolution": oct.resolution,
    }
    return info


def quick_stitch(directory, z: int, overlap_fraction: float = 0.2, n_rot: int = 3, zmin: int = 0, zmax: int = -1,
                 use_log: bool = False, use_stage_positions: bool = False, flip_ud: bool = True, flip_lr: bool = False,
                 galvo_shift: int = None, galvo_shift_first_tile=(0, 0)):
    # TODO: accelerate the stitching by preprocessing the tiles in parallel
    input_directory = Path(directory)

    # Get a list of the input tiles
    tiles_to_process = f"*z{z:02d}"
    tiles = list(input_directory.glob(tiles_to_process))

    # Get the tile positions (in pixel and mm)
    file_pattern = r"tile_x(?P<x>\d+)_y(?P<y>\d+)_z(?P<z>\d+)"
    tiles_positions_px = []
    tiles_positions_mm = []
    tiles_mx = []
    tiles_my = []
    for t in tiles:
        oct = OCT(t)
        if oct.position_available and use_stage_positions:
            x_mm, y_mm, _ = oct.position
        else:
            # Extract the tile's mosaic position.
            match = re.match(file_pattern, t.name)
            mx = int(match.group("x"))
            my = int(match.group("y"))

            # Compute the tile position in mm
            x_mm = oct.dimension[0] * (1 - overlap_fraction) * mx
            y_mm = oct.dimension[1] * (1 - overlap_fraction) * my

        x_px = int(np.floor(x_mm / oct.resolution[0]))
        y_px = int(np.floor(y_mm / oct.resolution[1]))

        tiles_positions_mm.append((x_mm, y_mm))
        tiles_positions_px.append((x_px, y_px))

    # Compute the mosaic shape
    x_min = min([x for x, _ in tiles_positions_px])
    y_min = min([y for _, y in tiles_positions_px])
    x_max = max([x for x, _ in tiles_positions_px]) + oct.shape[0]
    y_max = max([y for _, y in tiles_positions_px]) + oct.shape[1]
    mosaic_nrows = x_max - x_min
    mosaic_ncols = y_max - y_min
    mosaic = np.zeros((mosaic_nrows, mosaic_ncols), dtype=np.float32)

    # Perform stitching
    for i in tqdm(range(len(tiles)), desc="Quick Stitch"):
        oct = OCT(tiles[i])

        # Compute the pixel position within the mosaic
        rmin = tiles_positions_px[i][0] - x_min
        rmax = rmin + oct.shape[0]
        cmin = tiles_positions_px[i][1] - y_min
        cmax = cmin + oct.shape[1]

        # Get the tile id
        match = re.match(file_pattern, tiles[i].name)
        mx = int(match.group("x"))
        my = int(match.group("y"))

        apply_shift = True
        if mx < galvo_shift_first_tile[0]:
            apply_shift = False
        elif mx == galvo_shift_first_tile[0] and my < galvo_shift_first_tile[1]:
            apply_shift = False

        # Load the fringes
        if apply_shift:
            img = oct.load_image(fix_shift=galvo_shift)
        else:
            img = oct.load_image()

        # Log transform
        if use_log:
            img = np.log(img)

        # Compute an AIP
        img = img[zmin:zmax, :, :].mean(axis=0)

        # BUG: there are sometimes missing bscans
        if img.shape != oct.shape[0:2]:
            if np.any(np.array(img.shape) == 0):
                img = np.zeros(oct.shape[0:2])
            else:
                img = resize(img, oct.shape[0:2])

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


def detect_mosaic(directory: str, z: int, margin: float = 0.5, display: bool = False, image_file: str = None,
                  roi_file: str = None, keep_largest_island: bool = False, stitching_settings:dict = None):
    """Detect the tissue in the mosaic and compute the limits of the tissue.
    Parameters
    ----------
    directory : str
        The directory containing the tiles.
    z : int
        The z slices to process
    margin : float
        The margin to add to the tissue limits (in mm).
    display : bool
        Display the result in a matplotlib window.
    image_file : str
        The filename to save the quickstitch image.
    roi_file : str
        The filename to save the ROI image.
    keep_largest_island : bool
        Keep the largest connected component in the mask.
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
    img = quick_stitch(directory, z=z, use_stage_positions=True, **stitching_settings)

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
        mask = getLargestCC(mask)

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
        fig, ax = plt.subplots()
        ax.imshow(label2rgb(mask, img, bg_label=0, colors=['blue']),
                  extent=(ymin, ymax, xmax, xmin))  # Y axes are inverted

        rect = Rectangle((roi_y_min, roi_x_min),
                         width=(roi_y_max - roi_y_min),
                         height=(roi_x_max - roi_x_min),
                         fill=None, edgecolor="red", linestyle="dashed", label="ROI")
        ax.add_patch(rect)

        rect_margin = Rectangle((roi_y_min_margin, roi_x_min_margin),
                                width=(roi_y_max_margin - roi_y_min_margin),
                                height=(roi_x_max_margin - roi_x_min_margin),
                                fill=None, edgecolor="red", label="ROI + margin")
        ax.add_patch(rect_margin)

        ax.set_ylabel("x axis (mm)")
        ax.set_xlabel("y axis (mm)")
        title = f"xmin={roi_x_min_margin:.4f}mm, xmax={roi_x_max_margin:.4f}mm\nymin={roi_y_min_margin:.4f}mm, ymax={roi_y_max_margin:.4f}mm"
        ax.set_title(title)
        ax.legend()

        if roi_file is not None:
            filename = Path(roi_file)
            filename.parent.mkdir(exist_ok=True, parents=True)
            plt.savefig(filename, dpi=150, bbox_inches="tight", pad_inches=0)

        if display:
            plt.show()

    return roi_x_min_margin, roi_x_max_margin, roi_y_min_margin, roi_y_max_margin


def save_quickstitch(img, quickstitch_file):
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
