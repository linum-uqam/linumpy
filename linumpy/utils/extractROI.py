import numpy as np
import nibabel as nib
import os


def get_ROI(brain, roi, o_name='ROI_0', o_dir='extracted'):
    """ Extracts a ROI from a nifti object and saves it in the output directory
    Parameters
    ----------
    brain : nifti
        3d image containing the brain data
    roi : nifti
        3d binary mask containing the ROI
    o_name : str
        output filename
    o_dir : str
        output directory
    Returns
    -------
    ndarray
        ROI extracted from the data
    """

    # Intervalles pour parcourir le volume en x et y
    step_x = brain.shape[0] // 20
    step_y = brain.shape[1] // 20
    step_z = brain.shape[2] // 20

    # [x_min,x_max, y_min, y_max,z_min,z_max]
    bounds = [brain.shape[0], 0, brain.shape[1], 0, brain.shape[2], 0]

    # Parcourir le volume
    for x1 in range(0, brain.shape[0] - step_x, step_x):
        for y1 in range(0, brain.shape[1] - step_x, step_y):
            for z1 in range(0, brain.shape[2] - step_z, step_z):
                x2 = (x1 + step_x)
                y2 = (y1 + step_y)
                z2 = z1 + (step_z)
                sliced_data = roi.slicer[x1:x2, y1:y2, z1:z2].get_fdata()

                # Déterminer l'emplacement approximatif de la ROI
                if (np.max(sliced_data) > 0):
                    if (x1 < bounds[0]):
                        bounds[0] = x1
                    if (x2 > bounds[1]):
                        bounds[1] = x2
                    if (y1 < bounds[2]):
                        bounds[2] = y1
                    if (y2 > bounds[3]):
                        bounds[3] = y2
                    if (z1 < bounds[4]):
                        bounds[4] = z1
                    if (z2 > bounds[5]):
                        bounds[5] = z2

    # Rogner les volumes (1ere fois)
    brain_data = brain.slicer[bounds[0]:bounds[1],
                              bounds[2]:bounds[3],
                              bounds[4]:bounds[5]].get_fdata()
    ROI_mask = roi.slicer[bounds[0]:bounds[1],
                          bounds[2]:bounds[3],
                          bounds[4]:bounds[5]].get_fdata()

    # Extraire la ROI du cerveau
    ROI_data = np.where(ROI_mask > 0, brain_data, ROI_mask)

    # Rogner le ROI (2e fois)
    ROI_data_c = cropROI(ROI_data)

    # Sauvegarder le fichier
    new_roi = nib.Nifti1Image(ROI_data_c, brain.affine)
    if not (os.path.exists(o_dir)):
        os.makedirs(o_dir)
    nib.save(new_roi, o_dir + '/' + o_name + '.nii')
    return ROI_data_c


# rogner la ROI
def cropROI(vol):
    bounds = [vol.shape[0], 0, vol.shape[1], 0, vol.shape[2], 0]
    # Parcourir le volume
    for x in range(vol.shape[0]):
        for y in range(vol.shape[1]):
            for z in range(vol.shape[2]):
                # Déterminer l'emplacement de la ROI
                if (vol[x, y, z] > 0):
                    if (x < bounds[0]):
                        bounds[0] = x
                    if (x > bounds[1]):
                        bounds[1] = x
                    if (y < bounds[2]):
                        bounds[2] = y
                    if (y > bounds[3]):
                        bounds[3] = y
                    if (z < bounds[4]):
                        bounds[4] = z
                    if (z > bounds[5]):
                        bounds[5] = z
    return vol[bounds[0]:bounds[1], bounds[2]:bounds[3], bounds[4]:bounds[5]]
