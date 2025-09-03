import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_erosion, binary_fill_holes, median_filter
from skimage.color import label2rgb
from skimage.filters import threshold_li
from skimage.measure import label
from skimage.morphology import disk
from skimage.transform import rescale, resize
from tqdm import tqdm

from linumpy import reconstruction

# Parameters
z = 31
directory = Path(f"E:\Frans\sub-17-reembedded\slice_z{z:02d}")
tiles_directory = directory / "tiles"
figure_file = directory / "tile_cleaning.png"
tiles_margin = 1
saturation = 99.7
median_size = 5
scaling_factor = 4
keep_largest_island = True


def getLargestCC(segmentation):
    labels = label(segmentation)
    assert (labels.max() != 0)  # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largestCC


# Get a quick reconstruction (without overlap)
mosaic = reconstruction.quick_stitch(tiles_directory, z=z, overlap_fraction=0.0, n_rot=0, flip_lr=False, flip_ud=False)

# Normalize the image
mosaic = (mosaic.astype(np.float32) - mosaic.min()) / (np.percentile(mosaic, saturation) - mosaic.min())
mosaic[mosaic > 1] = 1

# Rescale
mosaic = rescale(mosaic, 1 / scaling_factor, anti_aliasing=True)

# Get the mosaic shape
mosaic_info = reconstruction.get_mosaic_info(tiles_directory, z=z, overlap_fraction=0.0, use_stage_positions=False)

# Mask the tissue
threshold = threshold_li(mosaic)
mask = mosaic > threshold
mask = median_filter(mask, median_size)

# Detect the empty tiles
tiles_grid_mask = np.zeros(mosaic_info['mosaic_grid_shape'], dtype=bool)
for p, p_px in zip(mosaic_info['mosaic_tile_pos'], mosaic_info['tiles_pos_px']):
    rmin = p_px[0] // scaling_factor
    cmin = p_px[1] // scaling_factor
    rmax = rmin + mosaic_info['tile_shape_px'][0] // scaling_factor
    cmax = cmin + mosaic_info['tile_shape_px'][1] // scaling_factor
    foo = mask[rmin:rmax, cmin:cmax]
    if not np.any(foo):
        tiles_grid_mask[p[0], p[1]] = True

# Fill holes
tiles_grid_mask = ~binary_fill_holes(~tiles_grid_mask)

# Keep the largest CC
if keep_largest_island:
    tiles_grid_mask = ~getLargestCC(~tiles_grid_mask)

# Add a margin
tiles_grid_mask = binary_erosion(tiles_grid_mask, disk(tiles_margin), border_value=1)

# Convert to a list of tiles to remove
tiles_to_delete = []
for i in range(len(mosaic_info["tiles"])):
    p = mosaic_info["mosaic_tile_pos"][i]
    if tiles_grid_mask[p[0], p[1]]:
        tiles_to_delete.append(mosaic_info["tiles"][i])

# Display the tiles that will be removed
overlay = label2rgb(resize(tiles_grid_mask, mosaic.shape), mosaic, alpha=0.1)
plt.imshow(overlay)
plt.title(f"Slice z={z}")
plt.savefig(fname=figure_file, dpi=300, bbox_inches='tight', pad_inches=0)
plt.show()

# Wait for user ok
question = f"Do you wish to delete the red tiles in slice z={z} (y/n)? : "
delete_tiles = None
while delete_tiles is None:
    answer = input(question)
    if answer.lower() == "y":
        delete_tiles = True
    elif answer.lower() == "n":
        delete_tiles = False
    else:
        print("Wrong input! Accepted values are: y or n")

# Remove the selected background tiles for this slice
if delete_tiles:
    # TODO: save the tile deletion map for reference

    # Loop over the tiles to delete and remove their data directory
    for t in tqdm(tiles_to_delete, desc="Deleting tiles"):
        shutil.rmtree(t)