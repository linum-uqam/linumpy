import ants
from pathlib import Path
import nibabel as nib
import numpy as np
from skimage.transform import resize

directory = Path("/home/joel/data/2023-12-01-PW2024-MultiOrientation-SOCT/template_100um")
brain_file = directory / "f3_45_ras_masked.nii"
output_file = directory / "f3_45_ras_masked_warped.nii"
template_file = "/home/joel/data/2023-12-01-PW2024-MultiOrientation-SOCT/allen_template_25micron_ras.nii"

brain_resolution = 100.0  # Micron / Pixel
allen_resolution = 25.0  # Micron / Pixel

# Load the fixed brain
img = nib.load(template_file)
allen = img.get_fdata()
allen_mask = allen > 3.0

# Normalize the intensity
allen = (allen - allen.min()) / (allen.max() - allen.min())

# Resize the template from 25um to 100um
factor = brain_resolution / allen_resolution
new_shape = tuple((np.array(allen.shape) / factor).astype(int).tolist())
allen = resize(allen, new_shape)

# Load the moving brain
img = nib.load(brain_file)
brain = img.get_fdata()
mask = brain > 0.0

# Normalize the intensity
brain = (brain - brain.min()) / (np.percentile(brain, 99.7) - brain.min())
brain[brain > 1] = 1

# Create and display the ants images
fixed = ants.from_numpy(allen)
moving = ants.from_numpy(brain)
fixed_mask = ants.from_numpy((allen_mask).astype(float))
moving_mask = ants.from_numpy((mask).astype(float))
ants.plot(fixed)
ants.plot(fixed_mask)
ants.plot(moving)
ants.plot(moving_mask)

# Register the images
#output_prefix = output_directory / name
transform = ants.registration(fixed=fixed, moving=moving, type_of_transform="Similarity")#,
                              #fixed_mask=fixed_mask, moving_mask=moving_mask, mask_all_stages=True)#,
                              #outprefix=str(output_prefix))
# Use this a an initial transform
ants.plot(transform['warpedmovout'])

# Save the similarity transform
output_file_sim = directory / output_file.name.replace(".nii", "_similarityOnly.nii")
moving_warped = transform['warpedmovout'].numpy()
img_out = nib.Nifti1Image(moving_warped, img.affine)
nib.save(img_out, output_file_sim)

initial_transform = transform['fwdtransforms'][0]
transform = ants.registration(fixed=fixed, moving=moving, type_of_transform="SyN",
                              fixed_mask=fixed_mask, moving_mask=moving_mask, mask_all_stages=True,
                                initial_transform=initial_transform, write_composite_transform=True)#,
                              #outprefix=str(output_prefix))

# Use this a an initial transform
ants.plot(transform['warpedmovout'])

# Save the warped image
moving_warped = transform['warpedmovout'].numpy()
img_out = nib.Nifti1Image(moving_warped, img.affine)
nib.save(img_out, output_file)

