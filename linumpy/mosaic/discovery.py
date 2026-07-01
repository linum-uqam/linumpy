"""Discover OCT tile files in a directory and extract mosaic metadata."""

import re
from pathlib import Path

import numpy as np
from skimage.measure import label
from tqdm.auto import tqdm

from linumpy.microscope.oct import OCT


def get_largest_cc(segmentation: np.ndarray) -> np.ndarray:
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
    assert labels.max() != 0  # assume at least 1 CC
    largest_cc = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largest_cc


DEFAULT_TILE_FILE_PATTERN = r"tile_x(?P<x>\d+)_y(?P<y>\d+)_z(?P<z>\d+)"


def get_tiles_ids(directory: Path, z: int | None = None) -> tuple[list, list]:
    """Analyzes a directory and detects all the tiles in contains."""
    input_directory = Path(directory)

    # Get a list of the input tiles
    tiles_to_process = f"*z{z:02d}" if z is not None else "tile_*"
    tiles = list(input_directory.rglob(tiles_to_process))
    tiles = [t for t in tiles if t.name.startswith("tile_") and not t.is_file()]
    tile_ids = get_tiles_ids_from_list(tiles)
    return tiles, tile_ids


def get_tiles_ids_from_list(tiles_list: list, file_pattern: str = DEFAULT_TILE_FILE_PATTERN) -> list:
    """Extract tile (x, y, z) grid positions from a sorted list of tile paths."""
    tiles_list.sort()

    # Get the tile positions
    tile_ids = []
    n_tiles = len(tiles_list)
    for t in tqdm(tiles_list, desc="Extracting tile ids", total=n_tiles):
        # Extract the tile's mosaic position.
        match = re.match(file_pattern, t.name)
        assert match is not None
        mx = int(match.group("x"))
        my = int(match.group("y"))
        mz = int(match.group("z"))
        tile_ids.append((mx, my, mz))

    return tile_ids


def get_mosaic_info(directory: Path, z: int, overlap_fraction: float = 0.2, use_stage_positions: bool = False) -> dict:
    """Return mosaic metadata for all tiles at a given z slice."""
    # Get a list of the input tiles
    tiles, _tile_ids = get_tiles_ids(directory, z)

    # Get the tile positions (in pixel and mm)
    file_pattern = r"tile_x(?P<x>\d+)_y(?P<y>\d+)_z(?P<z>\d+)"
    tiles_positions_px = []
    tiles_positions_mm = []
    mosaic_tile_pos = []
    # Progress bars overlap as the position is the same in all threads. Position is 1 to avoid overlap with outer loop.
    # No better solution has been found.
    oct_tile: OCT | None = None
    for t in tqdm(tiles, desc="Reading mosaic info", leave=False, position=1):
        oct_tile = OCT(t)

        # Extract the tile's mosaic position.
        match = re.match(file_pattern, t.name)
        assert match is not None
        mx = int(match.group("x"))
        my = int(match.group("y"))

        if oct_tile.position_available and use_stage_positions:
            x_mm, y_mm, _ = oct_tile.position
        else:
            # Compute the tile position in mm
            x_mm = oct_tile.dimension[0] * (1 - overlap_fraction) * mx
            y_mm = oct_tile.dimension[1] * (1 - overlap_fraction) * my

        x_px = int(np.floor(x_mm / oct_tile.resolution[0]))
        y_px = int(np.floor(y_mm / oct_tile.resolution[1]))

        mosaic_tile_pos.append((mx, my))
        tiles_positions_mm.append((x_mm, y_mm))
        tiles_positions_px.append((x_px, y_px))

    assert oct_tile is not None
    # Compute the mosaic shape
    x_min = min(x for x, _ in tiles_positions_px)
    y_min = min(y for _, y in tiles_positions_px)
    x_max = max(x for x, _ in tiles_positions_px) + oct_tile.shape[0]
    y_max = max(y for _, y in tiles_positions_px) + oct_tile.shape[1]
    mosaic_nrows = x_max - x_min
    mosaic_ncols = y_max - y_min

    # Get the mosaic grid shape
    n_mx = len(np.unique([x[0] for x in mosaic_tile_pos]))
    n_my = len(np.unique([x[1] for x in mosaic_tile_pos]))

    # Get the mosaic limits in mm
    xmin_mm = np.min([p[0] for p in tiles_positions_mm]) - oct_tile.dimension[0] / 2
    ymin_mm = np.min([p[1] for p in tiles_positions_mm]) - oct_tile.dimension[1] / 2
    xmax_mm = np.max([p[0] for p in tiles_positions_mm]) + oct_tile.dimension[0] / 2
    ymax_mm = np.max([p[1] for p in tiles_positions_mm]) + oct_tile.dimension[1] / 2
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
        "tile_shape_px": oct_tile.shape,
        "tile_shape_mm": oct_tile.dimension,
        "tile_resolution": oct_tile.resolution,
    }
    return info
