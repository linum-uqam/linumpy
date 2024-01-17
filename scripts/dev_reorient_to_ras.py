#!/usr/bin/env python3

"""Reorient a volume to RAS+ using control points."""

import numpy as np
from pathlib import Path
import nibabel as nib
from skimage.transform import rescale


directory = Path("/home/joel/data/2023-12-08-F3-Multiorientation-coronal-sagittal-45/reconstruction_2d")
input_volume = directory / "volume_10um.nii"
#input_volume = Path("/home/joel/data/2023-12-01-PW2024-MultiOrientation-SOCT/MouseBrainTemplate_OCT_50microns.nii")
output_volume = directory / "volume_10um_ras.nii"
#output_volume = directory / "oct_brain_template_50um_ras.nii"

# Control points position in pixel (ex: from Fiji)
# Mendeley 50um template
# pos_anterior = [13, 84, 112]
# pos_posterior = [230, 82, 112]
# pos_superior = [137, 16, 112]
# pos_inferior = [138, 146, 112]

# F3 : 45degrees
pos_anterior = [95, 74, 26]
pos_posterior = [25, 74, 79]
pos_superior = [64, 85, 51]
pos_inferior = [59, 15, 51]

# M4 : Axial
# pos_anterior = [154, 62, 35]
# pos_posterior = [13, 62, 35]
# pos_superior = [0, 0, 1]
# pos_inferior = [0, 0, 45]

# M3 : Sagittal
#pos_anterior = [148, 44, 40]
#pos_posterior = [15, 63, 40]
#pos_superior = [78, 24, 40]
#pos_inferior = [80, 89, 40]

# M1 : Coronal
# pos_anterior = [85, 56, 128]
# pos_posterior = [61, 56, 2]
# pos_superior = [106, 54, 63]
# pos_inferior = [44, 56, 65]

# F1 : Coronal (alexia)
# pos_anterior = [36, 58, 128]
# pos_posterior = [38, 58, 3]
# pos_superior = [13, 58, 76]
# pos_inferior = [72, 58, 73]




# Estimate the main axis of the volume
vector_pa = np.array(pos_anterior) - np.array(pos_posterior)
vector_is = np.array(pos_superior) - np.array(pos_inferior)
axis_pa = np.argmax(np.abs(vector_pa))
axis_pa_sign = np.sign(vector_pa[axis_pa])
axis_is = np.argmax(np.abs(vector_is))
axis_is_sign = np.sign(vector_is[axis_is])

vector_pa = np.zeros(3)
vector_pa[axis_pa] = axis_pa_sign
vector_is = np.zeros(3)
vector_is[axis_is] = axis_is_sign
vector_lr = np.cross(vector_pa, vector_is)

axis_lr = np.argmax(np.abs(vector_lr))
axis_lr_sign = np.sign(vector_lr[axis_lr])

# Get the axis code
axcodes = [""]*3
axcodes[axis_pa] = "A" if axis_pa_sign > 0 else "P"
axcodes[axis_is] = "S" if axis_is_sign > 0 else "I"
axcodes[axis_lr] = "L" if axis_lr_sign > 0 else "R"

# Get the nibabel orientation transformation from axcodes to RAS+
transformation = nib.orientations.axcodes2ornt(axcodes)

# Apply the transformation to the volume
img = nib.load(input_volume)
vol = img.get_fdata(dtype=np.float32)
# DEBUG
#vol = rescale(vol, 2.0)
vol_ras = nib.orientations.apply_orientation(vol, transformation)

# Get the resolution of the original volume
resolutions = img.header["pixdim"][1:4]
#resolutions = [25, 25, 25]
new_resolutions = []
new_resolutions.append(resolutions[axis_lr])
new_resolutions.append(resolutions[axis_pa])
new_resolutions.append(resolutions[axis_is])

# Save the new volume
affine = np.eye(4)
affine[0, 0] = new_resolutions[0] / 1000.0 # in mm
affine[1, 1] = new_resolutions[1] / 1000.0
affine[2, 2] = new_resolutions[2] / 1000.0
img_ras = nib.Nifti1Image(vol_ras, affine)
nib.save(img_ras, output_volume)





