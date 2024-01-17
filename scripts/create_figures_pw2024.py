import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from pathlib import Path

template_file = "/home/joel/Dropbox/Travail/papers/2023-pw2024-multiorientation-soct/allen_template_25um_ras.nii"

directory = Path("/home/joel/Dropbox/Travail/papers/2023-pw2024-multiorientation-soct/data_ras_25um_slicer_aligned/00_templating_iteration_similarity")
oct_files = [
    directory / "m1_coronal_25um_ras_transformed_Similarity_warped.nii",
    directory / "m3_sagittal_25um_ras_transformed_Similarity_warped.nii",
    directory / "m4_axial_25um_ras_transformed_Similarity_warped.nii",
    directory / "f3_cor-sag-45_25im_ras_transformed_Similarity_warped.nii"
]

slicing_orientation = [
    "Coronal",
    "Sagittal",
    "Axial",
    "CS-45"
]

# Load all the volumes
template = nib.load(template_file).get_fdata()
volumes = []
for f in oct_files:
    volumes.append(nib.load(f).get_fdata())

# Slice to display
coronal_idx = 200
sagittal_idx = 100
axial_idx = 80

# Display the coronal slices
plt.figure()
plt.subplot(1, 5, 1)
plt.imshow(template[:, :, axial_idx].T, cmap="gray")
plt.title("Template")
plt.xticks([])
plt.yticks([])
for i in range(len(volumes)):
    plt.subplot(1, 5, i+2)
    plt.imshow(volumes[i][:, :, axial_idx].T, cmap="gray")
    plt.title(slicing_orientation[i])
    plt.axis("off")
    plt.xticks([])
    plt.yticks([])
plt.tight_layout()
plt.show()

# Display coronal slices
plt.figure()
plt.subplot(1, 5, 1)
plt.imshow(template[:, coronal_idx, :].T, cmap="gray")
plt.title("Template")
plt.xticks([])
plt.yticks([])
for i in range(len(volumes)):
    plt.subplot(1, 5, i+2)
    plt.imshow(volumes[i][:, coronal_idx, :].T, cmap="gray")
    plt.title(slicing_orientation[i])
    plt.axis("off")
    plt.xticks([])
    plt.yticks([])
plt.tight_layout()
plt.show()
