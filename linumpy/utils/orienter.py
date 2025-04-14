import os
import numpy as np
import os
import zarr
import nrrd
import nibabel as nib

SUPPORTED_TYPE = ["nii", "nrrd", "zarr"]


def create(file_path=None, volume=None, array=None, dtype=None):
    """
    Create a volume from file path, volume, or array. 

    Parameters
    ----------
    file_path : string, optional
        The file path to read from.
    volume : volume_like, optional
        VolumeHandler or any type of supported volume. For example, nib.nifti1.Nifti1Image type for nifti file 
    annotations : dictionnary, optional
        Coordinate specify by the user. 
        Keys are left, right, superior, inferior, posterior and anterior. 
        Values are arrays with x,y,z coordinate of the point.
    array : array_like, optional
        Three dimensions array. Array must come with dtype parametre 
    dtype : string, optional
        Any of the supported 3D volume type. For exemple : "zarr", "nii" or "nrrd"

    Returns
    -------
    out : VolumeHandler
        Volume of the type of the file.
    """
    if file_path:
        volume = VolumeFactory.create_volume_from_file(file_path)
        volume.read()
    elif volume:
        volume = VolumeFactory.create_volume_from_volume(volume)
    elif array is not None and dtype:
        volume = VolumeFactory.create_volume_from_array(array, dtype)

    return volume


def read(volume):
    """
    Read a volume of the volume handler. 

    Parameters
    ----------
    volume : VolumeHandler
        Volume after being created with VolumeFactory. 

    Returns
    -------
    out : Array 
        The three dimensions array of the volume 
    """
    return volume.read()


def save(output_file_path: str, volume=None, array=None):
    """
    Save volume at the giving file path.

    Parameters
    ----------
    output_file_path : string
        The file path. Include the full name with extension(s)
    volume : VolumeHandler, optional
        Any type of volume supported by VolumeHandler. 
    array : array_like, optional
        Three dimensions array. 
    """
    new_volume = VolumeFactory.create_volume_from_file(output_file_path)

    if os.path.isfile(output_file_path) or os.path.isdir(output_file_path):
        raise ValueError("File path already exists.")

    if volume is not None and new_volume is not None:
        new_volume.save(output_file_path, volume.array)
    elif array is not None:
        new_volume.save(output_file_path, array)
    else:
        raise ValueError("Save data type not supported.")


def orient(final_orientation: str, annotations=None, volume=None, array=None, dtype=None):
    """
    Orient a three dimension volume. In order to orient the volume, 
    the annotations must be on at least 2 axis

    Parameters
    ----------
    final_orientation : desired orientation, string
        Any of the orientation supported. For exemple : "RAS" and "LAS".
    annotations : dictionnary, optional
        Coordinate specify by the user. 
        Keys are left, right, superior, inferior, posterior and anterior. 
        Values are arrays with x,y,z coordinate of the point.
    volume : Volume handler of any supported type, VolumeHandler optional
        Any type of volume supported by VolumeHandler. 
    array : array_like, optional
        Three dimensions array. Array must come with dtype parametre 
    dtype : string, optional
        Any of the supported 3D volume type. For exemple : "zarr", "nii" or "nrrd"

    Returns
    -------
    out : array
        The three dimensions array of the oriented volume 
    """
    final_orientation = final_orientation.lower().strip()

    volume = VolumeFactory.get_volume(annotations, volume, array, dtype)

    return volume.orient(final_orientation)


def get_orientation(annotations=None, volume=None, array=None, dtype=None):
    """
    Take the annotations and find the current orientation of the volume. 

    Parameters
    ----------
    annotations : dictionnary, optional
        Coordinate specify by the user. 
        Keys are left, right, superior, inferior, posterior and anterior. 
        Values are arrays with x,y,z coordinate of the point.
    volume : VolumeHandler, optional
        Any type of volume supported by VolumeHandler. 
    array : array_like, optional
        Three dimensions array. Array must come with dtype parametre 
    dtype : string, optional
        Any of the supported 3D volume type. For exemple : "zarr", "nii" or "nrrd"

    Returns
    -------
    out : string 
        The string of rotation. For exemple, "RAS", "LAS", ... 
    """
    volume = VolumeFactory.get_volume(annotations, volume, array, dtype)

    if volume != None and volume.orientation != None:
        return volume.orientation

    return volume.get_orientation()

# INTERNALS


class VolumeFactory:

    extension_file_to_ignore = [".zip", ".zipx", ".gz", ".tar", ".z"]

    @staticmethod
    def create_volume_from_file(file_path):
        file_name, file_extension = os.path.splitext(file_path)
        while file_extension in VolumeFactory.extension_file_to_ignore:
            file_name, file_extension = os.path.splitext(file_name)
        dtype = file_extension[1:len(file_extension)]  # remove "." of ".nii"
        volume = VolumeFactory.create_volume(dtype)
        volume.file_path = file_path
        return volume

    @staticmethod
    def create_volume_from_volume(in_volume):
        if VolumeFactory.is_nifti_volume(in_volume):
            volume = VolumeFactory.create_volume("nii")
            volume.array = in_volume.get_fdata()
            return volume
        if isinstance(in_volume, zarr.core.Array):
            volume = VolumeFactory.create_volume("zarr")
            volume.array = in_volume[::]
            return volume
        return in_volume

    @staticmethod
    def create_volume_from_array(array, dtype):

        if dtype not in SUPPORTED_TYPE:
            raise ValueError('Data type not supported')

        volume = VolumeFactory.create_volume(dtype)
        volume.array = array
        return volume

    @staticmethod
    def get_volume(annotations, volume, array, dtype):

        volume = VolumeFactory.create_volume_from_volume(volume)

        # Set array
        if volume != None:
            if volume.array is None:
                volume.read()
        elif array is not None and dtype != None:
            volume = VolumeFactory.create_volume(dtype)
            volume.array = array
        else:
            raise ValueError(
                'Missing information. You must provide a volume or array and dtype.')

        # Set annotation
        if annotations != None:
            volume.annotations = annotations
        else:
            raise ValueError('Annotations must be specify')

        return volume

    @staticmethod
    def is_nifti_volume(volume):
        return isinstance(volume, nib.nifti1.Nifti1Image) or isinstance(volume, nib.ni1.Nifti1Image) or isinstance(volume, nib.nifti2.Nifti2Image)

    @staticmethod
    def create_volume(dtype):
        if not dtype in SUPPORTED_TYPE:
            raise Exception("'{dtype}' not supported".format(dtype=dtype))
        if dtype == 'nii':
            return VolumeHandlerNifTi()
        if dtype == 'nrrd':
            return VolumeHandlerNrrd()
        if dtype == 'zarr':
            return VolumeHandlerZarr()


class VolumeHandler:
    array = None
    dtype = None
    file_path = None
    annotations = None
    orientation = None
    orientation_matrix = None

    def orient(self, final_orientation):
        # orientation_string = ''.join(self.get_orientation())
        orig_ornt = nib.orientations.axcodes2ornt(self.get_orientation())
        targ_ornt = nib.orientations.axcodes2ornt(final_orientation)
        transform = nib.orientations.ornt_transform(orig_ornt, targ_ornt)
        img_orient = nib.orientations.apply_orientation(self.array, transform)
        self.array = img_orient
        self.orientation = tuple(final_orientation)
        return img_orient

    def get_orientation(self, annotations=None):
        if self.orientation:
            return self.orientation
        elif annotations:
            orientation, orientation_matrix = self.get_orientation_matrix(
                annotations)
            self.orientation = orientation
            self.orientation_matrix = orientation_matrix
            return orientation
        else:
            raise ValueError(
                "No annotations were provided. You must provide annotations to get the orientation.")

    def get_orientation_matrix(self, annotations):
        center = np.array([(self.array.shape[0]-1)/2,
                          (self.array.shape[1]-1)/2, (self.array.shape[2]-1)/2])

        vectors = get_vectors(annotations, center)

        # Normalize on 1
        initial_matrix = [absolute_max_array(vectors[0]),
                          absolute_max_array(vectors[1]),
                          absolute_max_array(vectors[2])]

        if not isValidOrientation(initial_matrix):
            raise Exception("The annotations are ambiguous. Axes intersect.")

        # Fill matrix to get 4x4 matrix
        orientation_matrix = fill_matrix(initial_matrix)
        orientation = nib.orientations.ornt2axcodes(
            nib.orientations.io_orientation(orientation_matrix))
        return orientation, orientation_matrix


class VolumeHandlerNifTi(VolumeHandler):

    def __init__(self):
        self.dtype = "nii"

    @staticmethod
    def save(output_file_path, array):
        img = nib.Nifti1Image(array, np.eye(4))
        nib.save(img, output_file_path)

    def read(self):
        if self.file_path:
            data = nib.load(self.file_path).get_fdata()
            self.array = data
            return self.array
        else:
            raise ValueError(
                "Must provide the path of the file in order to read")


class VolumeHandlerNrrd(VolumeHandler):

    def __init__(self):
        self.dtype = "nrrd"

    @staticmethod
    def save(output_file_path, array):
        nrrd.write(output_file_path, array)

    def read(self):
        if self.file_path:
            data, _ = nrrd.read(self.file_path)
            self.array = data
            return self.array
        else:
            raise ValueError(
                "Must provide the path of the file in order to read")


class VolumeHandlerZarr(VolumeHandler):

    def __init__(self):
        self.dtype = "zarr"

    @staticmethod
    def save(output_file_path, array):
        zarr.save(output_file_path, array)

    def read(self):
        if self.file_path:
            self.array = zarr.open(
                self.file_path, mode='w', chunks=(1000, 1000), dtype='i4')
            return self.array
        else:
            raise ValueError(
                "Must provide the path of the file in order to read")

# Take annotations and set an array of lenght 3 representing the direction of the 3 axis.


def get_vectors(annotations, center):
    vectors = np.zeros((3, 3), dtype=int)
    pos_lef = annotations.get('l', annotations.get('left', None))
    pos_rig = annotations.get('r', annotations.get('right', None))
    pos_pos = annotations.get('p', annotations.get('posterior', None))
    pos_ant = annotations.get('a', annotations.get('anterior', None))
    pos_sup = annotations.get('s', annotations.get('superior', None))
    pos_inf = annotations.get('i', annotations.get('inferior', None))

    # R-L axis
    if pos_lef and pos_rig:
        vectors[0] = np.subtract(pos_lef, pos_rig)
    elif pos_lef:
        vectors[0] = np.subtract(pos_lef, center[0])
    elif pos_rig:
        vectors[0] = np.subtract(pos_rig, center[0])

    # A-P axis
    if pos_pos and pos_ant:
        vectors[1] = np.subtract(pos_pos, pos_ant)
    elif pos_pos:
        vectors[1] = np.subtract(pos_pos, center[0])
    elif pos_ant:
        vectors[1] = np.subtract(pos_ant, center[0])

     # S-I axis
    if pos_sup and pos_inf:
        vectors[2] = np.subtract(pos_sup, pos_inf)
    elif pos_sup:
        vectors[2] = np.subtract(pos_sup, center[0])
    elif pos_inf:
        vectors[2] = np.subtract(pos_inf, center[0])

    return vectors

# will fill the matrix if theres one axis missing


def fill_matrix(matrix):
    missing_row = None
    missing_col = None

    for i in range(len(matrix)):
        row_sum = sum(matrix[i])
        col_sum = sum(matrix[j][i] for j in range(len(matrix)))

        if row_sum == 0:
            missing_row = i

        if col_sum == 0:
            missing_col = i

        if missing_row is not None and missing_col is not None:
            break

    if missing_row != None and missing_col != None:
        matrix[missing_row][missing_col] = 1

    # add column and row
    m1 = np.c_[matrix, [0, 0, 0]]
    m2 = np.r_[m1, [[0, 0, 0, 1]]]

    return m2


def isValidOrientation(vectors):
    row_set = set()
    col_set = set()
    isValidOrientation = True

    for i in range(len(vectors)):
        for j in range(len(vectors)):
            if abs(vectors[i][j]) == 1:
                if i in row_set or j in col_set:
                    isValidOrientation = False
                    break
                row_set.add(i)
                col_set.add(j)

    return isValidOrientation


def absolute_max_array(arr):
    max_value = max(arr, key=abs)
    if max_value != 0:
        return [max_value/abs(max_value) if x == max_value else 0 for x in arr]
    return [0, 0, 0]


def get_shape_order(arr):
    indices = []
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            if arr[i][j] == 1:
                indices.append(j)
            elif arr[i][j] == -1:
                indices.append(-j)

    # the last dimension dosent change. Only work for 3 dimensions volume
    indices.append(3)
    result = tuple(indices)
    return result
