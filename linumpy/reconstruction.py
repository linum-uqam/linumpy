from pathlib import Path
from tqdm.auto import tqdm
import re
from linumpy.microscope.oct import OCT
import numpy as np

def get_tiles_ids(directory, z: int = None):
    """Analyzes a directory and detects all the tiles in contains"""
    input_directory = Path(directory)

    # Get a list of the input tiles
    if z is not None:
        tiles_to_process = f"*z{z:02d}"
    else:
        tiles_to_process = f"tile_*"
    tiles = list(input_directory.glob(tiles_to_process))
    tiles.sort()

    # Get the tile positions (in pixel and mm)
    file_pattern = r"tile_x(?P<x>\d+)_y(?P<y>\d+)_z(?P<z>\d+)"
    tile_ids = []
    n_tiles = len(tiles)
    for t in tqdm(tiles, desc="Extracting tile ids", total=n_tiles):
        # Extract the tile's mosaic position.
        match = re.match(file_pattern, t.name)
        mx = int(match.group("x"))
        my = int(match.group("y"))
        mz = int(match.group("z"))
        tile_ids.append((mx, my, mz))

    return tiles, tile_ids

def get_mosaic_info(directory, z: int, overlap_fraction: float = 0.2, use_stage_positions: bool = False):
    input_directory = Path(directory)

    # Get a list of the input tiles
    tiles_to_process = f"*z{z:02d}"
    tiles = list(input_directory.glob(tiles_to_process))

    # Get the tile positions (in pixel and mm)
    file_pattern = r"tile_x(?P<x>\d+)_y(?P<y>\d+)_z(?P<z>\d+)"
    tiles_positions_px = []
    tiles_positions_mm = []
    mosaic_tile_pos = []
    for t in tqdm(tiles, desc="Reading mosaic info"):
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
    mosaic_center_mm = ((xmin_mm+xmax_mm)/2, (ymin_mm+ymax_mm)/2)
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