from pathlib import Path
import nibabel as nib
import numpy as np
import ants
from tqdm import tqdm

#directory = Path("/home/joel/Dropbox/Travail/papers/2023-pw2024-multiorientation-soct/data_ras_25um_slicer_aligned")
directory = Path("/home/joel/Dropbox/Travail/papers/2023-pw2024-multiorientation-soct/data_ras_25um_slicer_aligned/00_templating_iteration_similarity")
#input_volumes = list(directory.glob("*transformed.nii"))
input_volumes = list(directory.glob("*_Similarity_warped.nii"))
#initial_template = "/home/joel/Dropbox/Travail/papers/2023-pw2024-multiorientation-soct/allen_template_25um_ras.nii"
initial_template = "/home/joel/Dropbox/Travail/papers/2023-pw2024-multiorientation-soct/data_ras_25um_slicer_aligned/00_templating_iteration_similarity/template.nii"

# Prepare the brains with their flipped versions, and change the contrast
brains = []
brains_names = []
masks = []
for f in input_volumes:
    img = nib.load(f)
    vol = img.get_fdata()
    mask = vol > 0.0

    # Normalize the intensity
    vol = (vol - vol.min()) / (np.percentile(vol, 99.7) - vol.min())
    vol[vol > 1] = 1

    # Add the brains to the list
    brains.append(vol)
    #brains.append(np.flip(vol, axis=0))
    masks.append(mask)
    #masks.append(np.flip(mask, axis=0))
    brains_names.append(f.name.replace(".nii", ""))
    #brains_names.append(f.name.replace(".nii", "_flipped"))


# Perform an initial affine registration on the initial template
img = nib.load(str(initial_template))
template = img.get_fdata()


# Loop over the iterations, and perform deformable registration
n_brains = len(brains)
n_iterations = 20
transform_type = "SyNOnly"
for i in tqdm(range(n_iterations), unit="iteration", desc="Templating"):
    # Perform the registration
    iteration_directory = directory / f"{i:02d}_templating_iteration"
    iteration_directory.mkdir(exist_ok=True)
    new_brains = []
    for j in tqdm(range(n_brains)):
        vol = brains[j]
        mask = masks[j]
        name = brains_names[j]
        fixed = ants.from_numpy(template)
        moving = ants.from_numpy(vol)
        #output_prefix = iteration_directory / name
        output_prefix = None
        transform = ants.registration(fixed=fixed, moving=moving, type_of_transform=transform_type, outprefix=str(output_prefix),
                                      moving_mask=ants.from_numpy(mask.astype(float)))
        moving_warped = transform['warpedmovout'].numpy()

        # Save the warped brain
        # output_filename = iteration_directory / f"{name}_{transform_type}_warped.nii"
        # img_out = nib.Nifti1Image(moving_warped, np.eye(4))
        # nib.save(img_out, output_filename)
        new_brains.append(moving_warped)

    # Compute the new average brain
    new_template = np.zeros(template.shape, dtype=np.float32)
    new_template_mask = np.zeros(template.shape, dtype=np.float32)
    for brain in new_brains:
        new_template += brain
        mask = brain > 0.0
        new_template_mask += mask

    # Save the initial average brain
    #new_template /= len(brains)
    mask_template = new_template_mask > 0.0
    new_template[mask_template] /= new_template_mask[mask_template]
    img_out = nib.Nifti1Image(new_template, img.affine)
    nib.save(img_out, iteration_directory / f"template.nii")
    template = new_template

    img_out = nib.Nifti1Image(new_template_mask, img.affine)
    nib.save(img_out, iteration_directory / f"template_mask.nii")


