from pathlib import Path
import nibabel as nib
import numpy as np
from linumpy import segmentation
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu, threshold_li
from scipy.ndimage import median_filter


directory = Path("/home/joel/data/2023-12-08-F3-Multiorientation-coronal-sagittal-45/reconstruction_2d")
filename = directory / "volume_25um_ras.nii"
mask_filename = directory / "volume_25um_ras_mask.nii.gz"

# Load the volume
img = nib.load(str(filename))
vol = img.get_fdata()

# Display the volume
plt.figure()
plt.imshow(vol.mean(axis=0))
plt.show()

# Create a data mask
mask = np.zeros_like(vol, dtype=bool)
mask[vol > 0] = True

# Compute the threahold only in the data
threshold = threshold_otsu(vol[mask])
#threshold = threshold_li(vol[mask])
print(threshold)

# Update the mask
mask[vol < threshold] = False

# Fill the holes
mask = segmentation.fillHoles_2Dand3D(mask)

# Filter to remove some noise
mask = median_filter(mask, size=5)

# Display the result
plt.figure()
plt.imshow(mask.mean(axis=0))
plt.show()

# Save the mask
img_mask = nib.Nifti1Image(mask.astype(int), img.affine, img.header)
nib.save(img_mask, str(mask_filename))

