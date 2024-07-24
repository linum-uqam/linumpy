from pathlib import Path
import numpy as np
import nibabel as nib
import nrrd
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu, threshold_li
from skimage.transform import resize
from scipy.ndimage import median_filter, binary_fill_holes


# Parameters
directory = Path("/home/joel/data/2023-12-08-F3-Multiorientation-coronal-sagittal-45/reconstruction_2d")
input_volume = directory / "volume.nii"
#input_agarose_mask = directory / "volume_25um_v2_agarose_samples.seg.nrrd"
#input_agarose_mask = directory / "greymatter_segmentation_sample.seg.nrrd"
output_volume = directory / "volume_normalized.nii"

def get_tissue_avg(img):
    # Rescale the image to a small size
    size = 1024
    new_shape = tuple((np.array(img.shape) * size / np.min(img.shape)).astype(int).tolist())
    img = resize(img, new_shape)

    # Normalize the intensity
    #img = (img.astype(np.float32) - img.min()) / (np.percentile(img, 99.7) - img.min())
    #img[img > 1] = 1

    # Process the image, to find a mask
    data_mask = img > 0.0
    thresh = threshold_li(img[data_mask])
    mask = img > thresh
    mask = median_filter(mask, 15)
    mask = binary_fill_holes(mask)

    # Display
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.subplot(1, 2, 2)
    plt.imshow(mask)
    plt.show()

    # Compute the average intensity within the tissue
    avg_intensity = img[mask].mean()
    return avg_intensity

version = 2

# Load the volume and the mask
img = nib.load(str(input_volume))
vol = img.get_fdata()
#mask, _ = nrrd.read(str(input_agarose_mask))
print(vol.shape)

# Plot the average intensity of the volume for each slice
avg_intensity = []
for i in range(vol.shape[2]):
    avg_intensity.append(get_tissue_avg(vol[:, :, i]))

# Compute the average intensity of the agarose mask for each slice
# mask_intensity = []
# for i in range(vol.shape[2]):
#     this_mask = mask[:,:, i] > 0
#     if this_mask.sum() > 0:
#         mask_intensity.append(vol[:, :, i][this_mask].mean())
#     else:
#         mask_intensity.append(np.nan)
# print(mask_intensity)
#
#
# # Plot the average intensity of the mask for each slice
# plt.figure()
# plt.plot(avg_intensity, label="Average intensity")
# plt.plot(mask_intensity, label="Average intensity in agarose")
# plt.legend()
# plt.show()
#
# mask_intensity = np.array(mask_intensity)
#
# if version==1:
#     # Compute the correction
#     avg_agarose = np.nanmean(mask_intensity)
#
#     # Remove NAN
#     mask_intensity[np.isnan(mask_intensity)] = avg_agarose
#
#     # Compute the correction
#     compensation = avg_agarose / mask_intensity
#     plt.plot(compensation)
#     plt.show()
# elif version==2:
compensation = [1.0] * vol.shape[2]
for i in range(1,vol.shape[2]):
    compensation[i] = compensation[-1] * avg_intensity[i-1] / avg_intensity[i]
plt.plot(compensation)
plt.show()


# Vol correction
vol_corrected = vol.copy()
for i in range(vol.shape[2]):
    this_mask = vol[:, :, i] > 0
    vol_corrected[:, :, i] *= compensation[i]
    vol_corrected[:, :, i][this_mask == False] = 0

# Save the volume
img_corrected = nib.Nifti1Image(vol_corrected, img.affine, img.header)
nib.save(img_corrected, str(output_volume))