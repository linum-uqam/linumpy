from pathlib import Path
import nibabel as nib
import numpy as np
import ants
from skimage.transform import resize

input_resolution = 50
allen_resolution = 25.0
n_iterations = 10

directory = Path("/home/joel/data/2023-12-01-PW2024-MultiOrientation-SOCT/template_50um")
output_file = directory / f"oct_template_50um_ras_masked_{n_iterations}iterations.nii"
allen_template_file = "/home/joel/data/2023-12-01-PW2024-MultiOrientation-SOCT/allen_template_25micron_ras.nii"
input_volumes = list(directory.glob("*ras.nii"))
print(input_volumes)

# Load the brains with their flipped versions
brains = []
for f in input_volumes:
    img = nib.load(f)
    vol = img.get_fdata()

    # Normalize the intensity
    vol = (vol - vol.min()) / (np.percentile(vol, 99.7) - vol.min())
    vol[vol > 1] = 1

    # Convert to antsimages
    brains.append(ants.from_numpy(vol))
    brains.append(ants.from_numpy(np.flip(vol, axis=0)))

# Load the template
img = nib.load(allen_template_file)
vol = img.get_fdata()

# Normalize the intensity
vol = (vol - vol.min()) / (vol.max() - vol.min())

# Resize the template from 25um to 100um
factor = input_resolution / allen_resolution
new_shape = tuple((np.array(vol.shape) / factor).astype(int).tolist())
print(new_shape)
vol = resize(vol, new_shape)
initial_template = ants.from_numpy(vol)

# Build template
template = ants.build_template(initial_template=initial_template, image_list=brains, iteration=n_iterations)

# Save the tempalte
img_template = nib.Nifti1Image(template.numpy(), nib.load(input_volumes[0]).affine)
nib.save(img_template, output_file)


