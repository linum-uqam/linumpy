#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Defines various classes to manage the slicer data, subjects and studies."""

# TODO: add save and load to classes

import itertools
import logging
import os
import pickle as pcl
import re
from collections import defaultdict
from pathlib import Path

import networkx
import numpy as np

from linumpy.stitching import topology
from linumpy.utils import data_io

logger = logging.getLogger(__name__)


class Subject:
    """Defines new subject (mouse) with given ID

    :param new_id: ID or name for this subject

    """

    subj_id = "None"
    data_dir = "None"
    result_dir = "None"
    data = list()

    def __init__(self, new_id):
        """Subject class constructor"""
        self.subj_id = new_id

    def __str__(self):
        object_str = (
                "Subject object with attributes :\n"
                + "  - subj_id : '%s'\n" % (self.subj_id)
                + "  - datadir : '%s'\n" % (self.data_dir)
                + "  - result_dir : '%s'\n" % (self.result_dir)
                + "  - acqinfo : "
                + str(self.info)
                + "\n"
                + "  - data : "
                + str(self.data[0])
        )
        return object_str

    def setDataDir(self, data_dir, ext=".bin"):
        """Sets input data directory for this subject (tissue or mouse).

        :param data_dir: (str) valid directory.
        :param ext: (str, optional) extension used for the volumes (default is '.bin')

        :returns: True/False if directory is valid or not.

        """
        if os.path.isdir(data_dir):
            self.data_dir = data_dir
            return True
        else:
            return False

    def getDataDir(self):
        return self.data_dir

    def setResultDir(self, result_dir):
        """Sets output data directory for this subject (tissue or mouse)
        INPUT
            valid directory
        OUTPUT
            True/False if directory is valid or not.
        """
        if os.path.isdir(result_dir):
            self.result_dir = result_dir
            return True
        else:
            return False

    def getResultDir(self):
        return self.result_dir

    def getDatafiles(self):
        return self.bin_files

    def setAcqInfo(self, csv_fname):
        """Add the acquisition information files to the subject.

        :param csv_fname: (str) Valid AcqInfo.csv complete filename.

        """
        self.info = data_io.load_acqinfo_from_csv(csv_fname)
        self.createDataFromAcqInfo()

    def getAcqInfo(self):
        return self.info

    def display(self):
        logger.info("Id: {}".format(self.subj_id))
        logger.info("Data Dir: {}".format(self.data_dir))
        logger.info("Result Dir: {}".format(self.result_dir))

    def checkForVolumes(self):
        nx = self.info["nStepX"]
        ny = self.info["nStepY"]
        nz = self.info["nSlice"]
        isFluo = self.info["acqFluo"]

        nFiles = nx * ny * nz
        if isFluo:
            nFiles *= 2
        fileCount = 0
        for file in os.listdir(self.getDataDir()):
            if file.endswith(".bin"):
                fileCount += 1

        flag = nFiles == fileCount
        return flag

    def getVolShape(self):
        nx = int(self.info["nAlinesPerBframe"])
        ny = int(self.info["nBframes"])
        nz = int(self.info["nBPixelZ"])
        return [nx, ny, nz]

    def getSlicerGridShape(self):
        """Access the slicer grid shape.

        :returns: [frameX, frameY, frameZ]

        """
        frameX = int(self.info["nStepX"])
        frameY = int(self.info["nStepY"])
        frameZ = int(self.info["nSlice"])
        return [frameX, frameY, frameZ]

    def addData(self, data, name):
        """Adds a data object to the subject

        :param data: (data object) A valid data object
        :param name: (str) Data name (for dictionnary indexing)

        """
        # TODO: Make sure this data object is saved by pickle.
        # Add the data to the object data dictionnary.
        self.data.append(data)

    def createDataFromAcqInfo(self):
        # This data
        this_data = SlicerData(self.data_dir, self.getSlicerGridShape(), "Original Data")
        this_data.volshape = self.getVolShape()
        self.data.append(this_data)

    def __getstate__(self):
        """To control how this class is dumped by pickle"""
        # List data
        datalist = []
        for thisdata in self.data:
            datalist.append(thisdata)

        # List all other dictionary values (either custom or built-in)
        sbj_members = vars(self)

        return (datalist, sbj_members)

    def __setstate__(self, state):
        """To control how this class is loaded by pickle"""
        datalist, sbj_members = state
        self.data = list()

        # Adding subjects in each group
        for this_data in datalist:
            self.data.append(this_data)

        # Adding other members
        for key in sbj_members:
            setattr(self, key, sbj_members[key])

        return self


class Study:
    """Defines new study using a set of subjects
    INPUT
        Study name
    OUTPUT
        None.
    """

    study_id = "None"
    result_dir = "None"
    categories = defaultdict(list)

    def __init__(self, new_id):
        self.study_id = new_id

    def setResultDir(self, result_dir):
        """Sets output data directory for this study
        INPUT
            valid directory
        OUTPUT
            True/False if directory is valid or not.
        """
        result_dir = os.path.join(result_dir, self.study_id)
        d = os.path.dirname(result_dir)
        if not os.path.exists(d):
            os.makedirs(d)
        self.result_dir = result_dir

    def getResultDir(self):
        """Get output data directory for this study
        INPUT
            None
        OUTPUT
            Str containing the output dir path
        """
        return self.result_dir

    def addSubject(self, subject, category="None"):
        """Adds a subject to the study with a
        INPUT
            valid subject
            (optional) category in which to classify the subject
        OUTPUT
            None
        """

        # Create the subject directory within the category it is assigned to
        self.categories[category].append(subject)
        study_dir = os.path.join(self.result_dir, category, subject.subj_id)
        if not os.path.exists(study_dir):
            os.makedirs(study_dir)

        # Inform the subject of where result data should be saved
        subject.setResultDir(study_dir)

    def display(self):
        """Will list the name of the study, the result dir, and then list all
        subjects and their classification in the study.
        OUTPUT
            None
        """
        logger.info("Study Id: {}".format(self.study_id))
        logger.info("Result Dir: {}".format(self.result_dir))
        logger.info(list(self.categories.items()))

    def __getstate__(self):
        """To control how this class is dumped by pickle"""
        # List categories & category per subject & subjects
        categories = []
        subjectCategory = []
        subjects = []

        for group in self.categories:
            categories.append(group)
            subjectList = self.categories[group]
            for subject in subjectList:
                subjectCategory.append(group)
                subjects.append(subject)

        # List all other dictionary values (either custom or built-in)
        study_members = vars(self)

        return (categories, subjectCategory, subjects, study_members)

    def __setstate__(self, state):
        """To control how this class is loaded by pickle"""
        categories, subjectCategory, subjects, study_members = state
        nSubjects = len(subjectCategory)

        self.categories = defaultdict(list)

        # Adding subjects in each group
        for subject in range(nSubjects):
            this_group = subjectCategory[subject]
            self.categories[this_group].append(subjects[subject])

        # Adding other members
        for key in study_members:
            setattr(self, key, study_members[key])

        return self


class SlicerData:
    """Slicer data class. It can be used to access raw data or to manage a new dataset.

    :param datadir: (int) Path to the data directory.
    :param gridshape: (tuple) Slicer grid shape.
    :param name: (str, default='data') Name of data set.
    :param prototype: (str, default='volume_x%02.0f_y%02.0f_z%02.0f') File name prototype.
    :param extension: (str, default='.bin') File extension
    :param volshape: (list, default=[512, 512, 120]) Volume shape.
    :param pixelFormat: (str, default='float32') Data format

    """

    def __init__(
            self,
            datadir,
            gridshape=None,
            name="data",
            prototype="volume_x%02.0f_y%02.0f_z%02.0f",
            extension=".bin",
            volshape=[512, 512, 120],
            pixelFormat="float32",
            detect_data=False,
    ):
        """Creating a new data object"""
        self.datadir = datadir

        # Try to detect the data information
        if detect_data:
            data_info = dataSniffer(self.datadir)
            self.prototype = data_info["prototype"]
            self.extension = data_info["extension"]
            self.gridshape = data_info["gridshape"]
            self.startIdx = [
                data_info["xrange"][0],
                data_info["yrange"][0],
                data_info["zrange"][0],
            ]
        else:
            self.prototype = prototype
            self.extension = extension
            self.gridshape = gridshape
            self.startIdx = [1, 1, 1]

        self.volshape = volshape
        self.name = name
        self.format = pixelFormat
        self.resolution = [1.0, 1.0, 1.0]

        self.set_gridOrigin("top-left")

    def __str__(self):
        object_str = (
                f"<{__class__.__name__}> object with attributes :\n"
                + "  - name : '%s'\n" % (self.name)
                + "  - datadir : '%s'\n" % (self.datadir)
                + "  - prototype : '%s'\n" % (self.prototype)
                + "  - extension : '%s'\n" % (self.extension)
                + "  - volshape : "
                + str(self.volshape)
                + "\n"
                + "  - gridshape : "
                + str(self.gridshape)
                + "\n"
                + "  - format : '%s'\n" % (self.format)
                + "  - resolution : "
                + str(self.resolution)
                + "\n"
                + "  - startIdx : "
                + str(self.startIdx)
                + "\n"
        )
        return object_str

    def save(self, filename):
        with open(filename, "w") as f:
            pcl.dump(self, f)

    def checkVolShape(self):
        """Load a volume and get its volume shape. Only works for nii of nii.gz files"""
        if self.extension == ".nii" or self.extension == ".nii.gz":
            vol = self.loadFirstVolume()
            self.volshape = vol.shape
        else:
            logger.info(
                "This method only works for nii and nii.gz files. Keeping the original volshape."
            )

    def set_gridOrigin(self, origin):
        """To define the mosaic grid origin as either: 'top-right', 'top-left', 'down-right' or 'down-left"""
        valid_origins = ["top-left", "top-right", "bottom-right", "bottom-left"]
        assert (
                origin in valid_origins
        ), "Unknown origin. Must be one of these: {}".format(valid_origins)
        self.grid_origin = origin
        if origin == "top-left":
            gridOrigin = (0, 0, 0)
            direction = (1, 1, 1)
        elif origin == "top-right":
            gridOrigin = (self.gridshape[0] - 1, 0, 0)
            direction = (-1, 1, 1)
        elif origin == "bottom-right":
            gridOrigin = (self.gridshape[0] - 1, self.gridshape[1] - 1, 0)
            direction = (-1, -1, 1)
        elif origin == "bottom-left":
            gridOrigin = (0, self.gridshape[1] - 1, 0)
            direction = (1, -1, 1)

        nx, ny, nz = self.gridshape[:]
        self.gridPosConversionMatrix = np.zeros((nx, ny, nz, 3), dtype=np.uint8)
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    xp = int(direction[0] * (x - gridOrigin[0]))
                    yp = int(direction[1] * (y - gridOrigin[1]))
                    zp = int(direction[2] * (z - gridOrigin[2]))
                    self.gridPosConversionMatrix[x, y, z, :] = [xp, yp, zp]

    def convert_posToGridPos(self, pos):
        return self.gridPosConversionMatrix[pos[0], pos[1], pos[2], :]

    def get_tile_path(self, pos):
        x, y, z = pos
        filename = os.path.join(
            self.datadir,
            self.prototype
            % (x + self.startIdx[0], y + self.startIdx[1], z + self.startIdx[2])
            + self.extension,
        )
        return filename

    def loadVolume(self, pos):
        """Loads a volume from the dataset.

        :param pos: (list) Volume position (in grid reference) to load.
        :returns: ndarray containing the loaded volume.

        Notes
        -----
        - If the volume does'nt exist, returns an arrays of zero with the predefined shape.

        """
        try:
            filename = self.get_tile_path(pos)
            return data_io.load_volumeByFilename(filename, self.volshape, self.format)
        except:
            return None

    def loadFirstVolume(self):
        """Loads the first non-empty volume"""
        for vol in self.volumeIterator():
            if vol is not None:
                return vol
                break

    def saveVolume(self, vol, pos, overwrite=False):
        """Saves a volume into the dataset directory.

        :param vol: (ndarray) Volume to save
        :param pos: (list) Volume position (in grid reference)
        :param overwrite: (bool, default=False) If set to true, the saved volume will overwrite any pre-existing file.

        .. note:: Only nifti files (*.nii* and *.nii.gz*) can be saved in this version.

        """
        x, y, z = pos
        filename = os.path.join(
            self.datadir,
            self.prototype
            % (x + self.startIdx[0], y + self.startIdx[1], z + self.startIdx[2])
            + self.extension,
        )

        # Check if datadir exits
        if not (os.path.exists(self.datadir)):
            os.makedirs(self.datadir)

        # Check if file exists
        if not (os.path.exists(filename)) or overwrite:
            if self.extension in [".nii", ".nii.gz"]:
                data_io.save_nifti(filename, vol, pixelFormat=self.format)
            else:
                logger.info(
                    "Volume save is not implemented yet for extension '%s'"
                    % self.extension
                )
                raise NotImplementedError
        else:
            logger.info("This file already exists : '%s'" % (filename))

    def volumeIterator(self, returnPos=False, mask=None, returnPosOnly=False):
        """Iterates over all volumes

        :param returnPos: (bool, default=False) If set to True, the iterator will yield the position in addition to the volume at each iteration.
        :param mask: (ndarray, default=None) This mask specify which volumes to keep in the iteration.

        :returns: vol
        :returns: vol, pos (if returnPos=True)

        """
        for z in range(self.gridshape[2]):
            if returnPosOnly:
                for pos in self.sliceIterator(z, returnPos, mask, returnPosOnly):
                    yield pos
            else:
                if returnPos:
                    for vol, pos in self.sliceIterator(z, returnPos, mask):
                        yield vol, pos
                else:
                    for vol in self.sliceIterator(z, returnPos, mask):
                        yield vol

    def sliceIterator(self, z, returnPos=False, mask=None, returnPosOnly=False):
        """Iterates over all volumes in slice z

        :param z: (int) Slice number over which the iteration occurs.
        :param returnPos: (bool, default=False) If set to True, the iterator will yield the position in addition to the volume at each iteration.
        :param mask: (ndarray, default=None) This mask specify which volumes to keep in the iteration.

        :returns: vol
        :returns: vol, pos (if returnPos=True)

        """
        nx = list(range(self.gridshape[0]))
        ny = list(range(self.gridshape[1]))
        for x, y in itertools.product(nx, ny):
            # Ignoring volumes if mask given
            if mask is not None:
                if mask.ndim == 2:
                    if not (mask[x, y]):
                        continue
                else:
                    if not (mask[x, y, z]):
                        continue

            if returnPosOnly:
                yield (x, y, z)
            else:
                vol = self.loadVolume((x, y, z))
                if vol is not None:
                    if returnPos:
                        pos = (x, y, z)
                        yield vol, pos
                    else:
                        yield vol

    def neighborIterator(self, returnPos=False, mask=None, returnPosOnly=False):
        """Iterates over all neighbors

        :param returnPos: (bool, default=False) If set to True, the iterator will yield the position in addition to the volume at each iteration.

        :returns: vol1, vol2
        :returns: vol1, vol2, pos1, pos2 (if returnPos=True)

        """

        # Loop over all slices
        for z in range(self.gridshape[2]):
            if returnPosOnly:
                for pos1, pos2 in self.neighborSliceIterator(
                        z, returnPos, mask, returnPosOnly
                ):
                    yield pos1, pos2
            else:
                if returnPos:
                    for vol1, vol2, pos1, pos2 in self.neighborSliceIterator(
                            z, returnPos, mask=mask
                    ):
                        yield vol1, vol2, pos1, pos2
                else:
                    for vol1, vol2 in self.neighborSliceIterator(
                            z, returnPos, mask=mask
                    ):
                        yield vol1, vol2

    def neighborSliceIterator(self, z, returnPos=False, mask=None, returnPosOnly=False):
        """Iterates over all neighbors in slice z

        :param returnPos: (bool, default=False) If set to True, the iterator will yield the position in addition to the volume at each iteration.
        :param z: (int) Slice number over which the iteration occurs.

        :returns: vol1, vol2
        :returns: vol1, vol2, pos1, pos2 (if returnPos=True)

        """
        nX = self.gridshape[0]
        nY = self.gridshape[1]
        this_topo = topology.generate_default(nX, nY)

        if mask is not None:
            if mask.ndim == 3:
                mask = mask[:, :, z]
            topology.remove_agarose(this_topo, mask)

        xx = networkx.get_node_attributes(this_topo, "x")
        yy = networkx.get_node_attributes(this_topo, "y")

        # Loop over all edges
        if float(networkx.__version__) < 2.0:
            for u, v in this_topo.edges_iter():
                pos1 = (xx[u], yy[u], z)
                pos2 = (xx[v], yy[v], z)
                if returnPosOnly:
                    yield pos1, pos2
                else:
                    vol1 = self.loadVolume(pos1)
                    vol2 = self.loadVolume(pos2)
                    if vol1 is not None and vol2 is not None:
                        if returnPos:
                            yield vol1, vol2, pos1, pos2
                        else:
                            yield vol1, vol2
        else:
            for u, v in this_topo.edges():
                pos1 = (xx[u], yy[u], z)
                pos2 = (xx[v], yy[v], z)
                if returnPosOnly:
                    yield pos1, pos2
                else:
                    vol1 = self.loadVolume(pos1)
                    vol2 = self.loadVolume(pos2)
                    if vol1 is not None and vol2 is not None:
                        if returnPos:
                            yield vol1, vol2, pos1, pos2
                        else:
                            yield vol1, vol2

    def singlePassNeighborIterator(
            self, origin, method="bfs", mask=None, returnPosOnly=False
    ):
        """Iterator that traverse the whole dataset in a single pass.

        :param origin: (2x1 array) (grid coordinates (begins at 1))
        :param method: (str, default='bfs') Graph traversing method ('dfs' or 'bfs')

        :returns: vol1, vol2, pos1, pos2

        """
        for z in range(self.gridshape[2]):
            if returnPosOnly:
                for pos1, pos2 in self.singlePassNeighborSliceIterator(
                        origin, z, method, mask, returnPosOnly
                ):
                    yield pos1, pos2

            else:
                for vol1, vol2, pos1, pos2 in self.singlePassNeighborSliceIterator(
                        origin, z, method, mask
                ):
                    yield vol1, vol2, pos1, pos2

    def singlePassNeighborSliceIterator(
            self, origin, z, method="bfs", mask=None, returnPosOnly=False
    ):
        """Iterator that traverse slice z in a single pass.

        :param origin: (2x1 array) (grid coordinates (begins at 1))
        :param z: (int) slice number
        :param method: (str, default='bfs') Graph traversing method ('dfs' or 'bfs')

        :returns: vol1, vol2, pos1, pos2

        """
        topo = topology.generate_default(self.gridshape[0], self.gridshape[1])

        # Remove agarose from topology
        if mask is not None:
            if mask.ndim == 3:
                mask = mask[:, :, z]
            topology.remove_agarose(topo, mask)
        sList, tList = topology.topoIterator(topo, root=origin, method=method)

        # Loop over source and target list
        for source, target in zip(sList, tList):
            pos1 = (source[0], source[1], z)
            pos2 = (target[0], target[1], z)
            if returnPosOnly:
                yield pos1, pos2
            else:
                vol1 = self.loadVolume(pos1)
                vol2 = self.loadVolume(pos2)
                if vol1 is not None and vol2 is not None:
                    yield vol1, vol2, pos1, pos2

    def update_gridshape(self):
        self.gridshape = detect_gridshape(self.datadir, self.prototype, self.extension)

    def detect_tissue(self):
        nx, ny, nz = self.volshape
        gx, gy, gz = self.gridshape

        # Mosaic size
        mx = gx * nx
        my = gy * ny

        # Computing AIP for each slice
        slices_mean = np.zeros((mx, my, nz), dtype=float)
        k = int(0.01 * 0.5 * (mx + my))
        for z in range(nz):
            m = stitch_oct.stitch2D(self, z=z)
            slices_mean[:, :, z] = median_filter(m, k)

        # Compute mask using Li thresholding method
        thresh = threshold_li(slices_mean)
        mask_tissue = slices_mean > thresh

        # Convert this tissue mask into a grid data mask
        grid_mask = np.zeros((gx, gy, gz), dtype=bool)
        for z in range(nz):
            dilated_mask = binary_dilation(mask_tissue[:, :, z], disk(2 * k))
            for x in range(gx):
                for y in range(gy):
                    pos_x = x * nx
                    pos_y = y * ny
                    roi = dilated_mask[pos_x: pos_x + nx, pos_y: pos_y + ny]

                    if np.sum(roi) / float(roi.size) > f:
                        grid_mask[x, y, zrange[z]] = True

        # Assign this mask
        self.gridmask = grid_mask


def detect_gridshape(
        datadir, prototype="volume_x%02.0f_y%02.0f_z%02.0f", extension=".bin"
):
    # List all files in datadir
    if isinstance(datadir, str):
        fileList = os.listdir(datadir)
    elif isinstance(datadir, list):
        fileList = datadir

    # Create a regex expression to find all files matching prototypes
    filename_rx_prototype = prototype + extension

    # Replacing %ds
    filename_rx_prototype = re.sub("%d", "(?P<x>\d+)", filename_rx_prototype, count=1)
    filename_rx_prototype = re.sub("%d", "(?P<y>\d+)", filename_rx_prototype, count=1)
    filename_rx_prototype = re.sub("%d", "(?P<z>\d+)", filename_rx_prototype, count=1)

    # Replacing %fs
    filename_rx_prototype = re.sub(
        "%[0-9]*[.]*[0-9]*f", "(?P<x>\d+)", filename_rx_prototype, count=1
    )
    filename_rx_prototype = re.sub(
        "%[0-9]*[.]*[0-9]*f", "(?P<y>\d+)", filename_rx_prototype, count=1
    )
    filename_rx_prototype = re.sub(
        "%[0-9]*[.]*[0-9]*f", "(?P<z>\d+)", filename_rx_prototype, count=1
    )

    # Prepare sniffer
    filename_rx = re.compile(filename_rx_prototype)
    maxX = 0
    maxY = 0
    maxZ = 0
    minX = None
    minY = None
    minZ = None

    # Loop over all files in directory
    for elem in fileList:
        b = filename_rx.match(elem)
        if b is not None:
            if maxX < int(b.group("x")):
                maxX = int(b.group("x"))
            if maxY < int(b.group("y")):
                maxY = int(b.group("y"))
            if maxZ < int(b.group("z")):
                maxZ = int(b.group("z"))
            if minX is None:
                minX = int(b.group("x"))
                minY = int(b.group("y"))
                minZ = int(b.group("z"))

            if minX > int(b.group("x")):
                minX = int(b.group("x"))
            if minY > int(b.group("y")):
                minY = int(b.group("y"))
            if minZ > int(b.group("z")):
                minZ = int(b.group("z"))
    try:
        gridshape = (
            int(maxX) - int(minX) + 1,
            int(maxY) - int(minY) + 1,
            int(maxZ) - int(minZ) + 1,
        )
    except:
        logger.info("Not able to detect gridshape. Setting to 0")
        gridshape = (0, 0, 0)

    return gridshape


def dataSniffer(datadir: str) -> dict:
    """Detect the mosaic information.
    Parameters
    ----------
    datadir: str
        Path to the directory containing the raw data
    Returns
    -------
    data_info: dict
        Dictionary with extracted information.
    """
    filelist = os.listdir(datadir)
    filename_rx = re.compile(
        r"(?P<prefix>[A-Za-z-_]+)(?P<x>\d+)(?P<bXY>[A-Za-z-_]+)(?P<y>\d+)(?P<bYZ>[A-Za-z-_]+)(?P<z>\d+)(?P<suffix>.*)(?P<ext>\..*)"
    )
    filename_rx_woExt = re.compile(
        r"(?P<prefix>[A-Za-z-_]+)(?P<x>\d+)(?P<bXY>[A-Za-z-_]+)(?P<y>\d+)(?P<bYZ>[A-Za-z-_]+)(?P<z>\d+)(?P<suffix>.*)")

    # Grap all volume-like files
    dataList = list()
    prefix = set()
    suffix = set()
    extension = set()
    bXY = set()
    bYZ = set()
    lengthPos = set()
    maxX = 0
    maxY = 0
    maxZ = 0
    minX = None
    minY = None
    minZ = None

    for elem in filelist:
        b = filename_rx.match(elem)
        if b is not None:
            dataList.append(elem)

            # Detect extension
            this_extension = "".join(Path(elem).suffixes)
            filename_wo_ext = elem.replace(this_extension, "")

            # Process the filename again
            b2 = filename_rx_woExt.match(filename_wo_ext)

            if maxX < int(b2.group("x")):
                maxX = int(b2.group("x"))
            if maxY < int(b2.group("y")):
                maxY = int(b2.group("y"))
            if maxZ < int(b2.group("z")):
                maxZ = int(b2.group("z"))
            if minX is None:
                minX = int(b2.group("x"))
                minY = int(b2.group("y"))
                minZ = int(b2.group("z"))

            if minX > int(b2.group("x")):
                minX = int(b2.group("x"))
            if minY > int(b2.group("y")):
                minY = int(b2.group("y"))
            if minZ > int(b2.group("z")):
                minZ = int(b2.group("z"))

            prefix.add(b2.group("prefix"))
            suffix.add(b2.group("suffix"))
            #extension.add(b.group("ext"))
            bXY.add(b2.group("bXY"))
            bYZ.add(b2.group("bYZ"))
            lengthPos.add(len(b2.group("x")))
            lengthPos.add(len(b2.group("y")))
            lengthPos.add(len(b2.group("z")))

    gridshape = (maxX - minX + 1, maxY - minY + 1, maxZ - minZ + 1)

    # Detect extension
    extension = this_extension

    logger.info("Xrange: {}".format((minX, maxX)))
    logger.info("Yrange: {}".format((minY, maxY)))
    logger.info("Zrange: {}".format((minZ, maxZ)))
    logger.info("Detected grid shape: {}".format(gridshape))
    logger.info("Detected extensions: {}".format(extension))

    # Creating a file prototype
    idxFormat = "%d"
    if min(lengthPos) >= 2:
        idxFormat = f"%0{min(lengthPos)}.0f"
    prototype = (
            list(prefix)[0]
            + idxFormat
            + list(bXY)[0]
            + idxFormat
            + list(bYZ)[0]
            + idxFormat
            + list(suffix)[0]
    )
    logger.info("Generated file prototype: {}".format(prototype))

    # Detect missing files
    data_mask = np.zeros(gridshape, dtype=bool)
    for x in range(minX, maxX + 1):
        for y in range(minY, maxY + 1):
            for z in range(minZ, maxZ + 1):
                filename = prototype % (x, y, z) + extension
                if filename in filelist:
                    data_mask[x - minX, y - minY, z - minZ] = True

    nVols = gridshape[0] * gridshape[1] * gridshape[2]
    logger.info(
        "There are {}/{} missing files in this grid.".format(
            nVols - data_mask.sum(), nVols
        )
    )

    # Creating the output dict
    data_info = dict()
    data_info["datadir"] = datadir
    data_info["prototype"] = prototype
    data_info["extension"] = extension
    data_info["gridshape"] = gridshape
    data_info["xrange"] = (minX, maxX)
    data_info["yrange"] = (minY, maxY)
    data_info["zrange"] = (minZ, maxZ)
    data_info["startIdx"] = (minX, minY, minZ)
    data_info["data_mask"] = data_mask

    return data_info
