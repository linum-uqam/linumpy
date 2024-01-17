from tqdm import tqdm
from pathlib import Path
import nibabel as nib
import numpy as np
from linumpy.utils_images import apply_xy_shift
import re
import pandas
from skimage.filters import threshold_li
from skimage import exposure
import matplotlib.pyplot as plt


directory = Path("/home/joel/data/2023-12-08-F3-Multiorientation-coronal-sagittal-45/reconstruction_2d")
output_volume = directory / "volume.nii"
resolution_xy = 750.0 / 400 # Micron / pixel
resolution_z = 200.0
match_histogram = False

do_crop = True
margin = 500 # micron

# Number of slices
files = list(directory.glob("*_stitched.nii"))
files.sort()

# Extract the slice ids
slice_ids = []
for f in files:
    g = re.match(r".*z(\d+)_.*", f.name)
    slice_ids.append(int(g.groups()[0]))
n_slices = np.max(slice_ids) - np.min(slice_ids) + 1
#n_slices = len(slice_ids)

# Load cvs containing the shift values for each slice
#shift_file = directory / "mosaicgrids_shifts_xy.csv"
shift_file = "/home/joel/data/2023-12-08-F3-Multiorientation-coronal-sagittal-45/mosaicgrids_shift_xy.csv"
df = pandas.read_csv(shift_file)
fixed_id = df["fixed_id"].tolist()
moving_id = df["moving_id"].tolist()
dx_list = df["x_shift"].tolist()
dy_list = df["y_shift"].tolist()
dx_list = np.array(dx_list)
dy_list = np.array(dy_list)

# Compute the volume shape
xmin = []
xmax = []
ymin = []
ymax = []

for i, f in enumerate(files):
    # Get this volume shape
    img = nib.load(f)
    shape = img.shape

    # Get the cumulative shift
    if i == 0:
        xmin.append(0)
        xmax.append(shape[1])
        ymin.append(0)
        ymax.append(shape[0])
    else:
        dx = np.cumsum(dx_list)[i-1]
        dy = np.cumsum(dy_list)[i-1]
        xmin.append(-dx)
        xmax.append(-dx + shape[1])
        ymin.append(-dy)
        ymax.append(-dy + shape[0])

# Get the volume shape
x0 = min(xmin)
y0 = min(ymin)
x1 = max(xmax)
y1 = max(ymax)
#nx = int((x1 - x0)/resolution_xy)
nx = int((x1 - x0))
#ny = int((y1 - y0)/resolution_xy)
ny = int((y1 - y0))
volume_shape = (ny, nx, n_slices)

# Create the volume
volume = np.zeros(volume_shape, dtype=np.float32)

# Reference image
ref = volume[:, :, 0]

# Loop over the slices
dx0 = 0
dy0 = 0
for i in tqdm(range(len(files)), unit="slice", desc="Stacking slices"):
    # Load the slice
    f = files[i]
    z = slice_ids[i]
    img = nib.load(f).get_fdata()

    # Get the shift values for the slice
    #dx = xmin[i] - x0 - ref.shape[1]/2
    #dy = ymin[i] - y0 - ref.shape[0]/2
    if i == 0:
        dx = x0 #+ ref.shape[1]/2
        dy = y0 #+ ref.shape[0]/2
    else:
        dx = np.cumsum(dx_list)[i-1] + x0 #- ref.shape[1]/2
        dy = np.cumsum(dy_list)[i-1] + y0 #- ref.shape[0]/2

    # Apply the shift
    img = apply_xy_shift(img, ref, dx, dy)

    # Math the histogram
    if match_histogram and z > 0:
        img = exposure.match_histograms(img, volume[:, :, z-1])

    # Add the slice to the volume
    volume[:, :, z] = img

    # Add to the reference
    ref = ref + img / n_slices

# Maks the data and crop the volume
if do_crop:
    mask_tissue = ref > 0
    mask = ref > threshold_li(ref[mask_tissue])
    rr, cc = np.where(mask)
    rmin = int(max(rr.min() - margin/resolution_xy, 0))
    rmax = int(min(rr.max() + margin/resolution_xy, volume.shape[0]))
    cmin = int(max(cc.min() - margin/resolution_xy, 0))
    cmax = int(min(cc.max() + margin/resolution_xy, volume.shape[1]))

    # Display the cropped reference
    plt.imshow(ref)
    plt.axhline(rmin, color="r")
    plt.axhline(rmax, color="r")
    plt.axvline(cmin, color="r")
    plt.axvline(cmax, color="r")
    plt.show()

    volume = volume[rmin:rmax, cmin:cmax, :]

# Save the volume as a nifti file
affine = np.eye(4)
affine[0, 0] = resolution_xy
affine[1, 1] = resolution_xy
affine[2, 2] = resolution_z
nib.save(nib.Nifti1Image(volume, affine), output_volume)