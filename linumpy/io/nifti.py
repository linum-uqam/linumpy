import nibabel as nib


def import_vol(path):
    """ Imports 3d nifti data
    Parameters
    ----------
    path : str
        The file location of the data

    Returns
    -------
    ndarray
        ndarray containing the 3d data

    ndarray
        affine matrix of the data
    """
    nii_vol = nib.load(path)
    volume = nii_vol.get_fdata()
    affine = nii_vol.affine
    return (volume, affine)


def save_as_nifti(vol, affine, output_dir, output_name, dtype=None):
    """ Saves volume in a nifti file format.

    Parameters
    ----------
    vol : ndarray
        3d array

    affine : ndarray
        Array containing the affine matrix of the data.

    output_dir : str
        name of the output directory

    output_name : str
        name of the output file

    dtype : type
        volume data type

    """
    if (dtype is not None):
        vol = vol.astype(dtype)
        affine = affine.astype(dtype)
    vol_nii = nib.Nifti1Image(vol, affine)
    if (dtype is not None):
        vol_nii.set_data_dtype(dtype)
    nib.save(vol_nii, output_dir + '/' + output_name + '.nii')
