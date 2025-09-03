import warnings
from pathlib import Path
from typing import Union

import numpy as np

from linumpy.preproc import xyzcorr


# TODO: consider the 'n_repeat' parameter when loading the data
# TODO: reorder the dimension, position, etc to be n_depths, n_alines and n_bscans


class OCT:
    """
    Spectral-domain OCT class to reconstruct the data.
    
    Parameters
    ==========
    directory: string
        Path to the directory containing the raw tiles and info file.
    axial_res: float, optional
        Axial resolution of the data in microns.
    """
    def __init__(self, directory: str, axial_res=3.5):
        self.directory = Path(directory)
        self.info_filename = self.directory / "info.txt"
        self.info = {}
        self.rz = axial_res

        # Read the scan info
        self.read_scan_info(self.info_filename)

    def read_scan_info(self, filename: str):
        """ Read the scan information file
        Parameters
        ----------
        filename
            Path to the scan_file written by the OCT (.txt)
        """
        with open(filename, "r") as f:
            foo = f.read()

        # Process the file input
        foo = foo.split("\n")
        for elem in foo:
            hello = elem.split(": ")
            if len(hello) == 1:
                continue
            key, val = hello
            if val.isnumeric():
                val = int(val)
            elif val.strip("-").isnumeric():
                val = int(val)
            self.info[key] = val

    def load_image(self, crop: bool = True, fix_shift: Union[bool, int] = True) -> np.ndarray:
        """ Load an image dataset
        Parameters
        ----------
        crop
            If crop is True, the galvo returns will be cropped from the volume
        fix_shift
            If True, the shift caused by the galvo mirror return will be evaluated from the data. If an integer value
            is given, this value will be used to fix the shift.
        Notes
        -----
        * The returned volume is in this order : z (depth), x (a-line), y (b-scan)
        * This method doesn't consider repeated a-lines or b-scans yet.
        """
        # Create numpy array
        # n_avg = self.info['n_repeat']  # TODO: use the number of averages when loading the data
        n_alines = self.info['nx']
        n_bscans = self.info['ny']
        n_extra = self.info['n_extra']
        n_alines_per_bscan = n_alines + n_extra
        n_z = self.info["bottom_z"] - self.info["top_z"] + 1

        # Load the fringe
        files = list(self.directory.rglob("image_*.bin"))
        files.sort()
        vol = None
        for file in files:
            with open(file, "rb") as f:
                foo = np.fromfile(f, dtype=np.float32)
            n_frames = int(len(foo) / (n_alines_per_bscan * n_z))
            foo = np.reshape(foo, (n_z, n_alines_per_bscan, n_frames), order='F')
            if vol is None:
                vol = foo
            else:
                vol = np.concatenate((vol, foo), axis=2)

        # Estimate the galvo shift
        if isinstance(fix_shift, bool) and fix_shift is True:
            if n_extra == 0:
                warnings.warn("Cannot estimate the shift correction as there are no extra a-lines in the file.")
            else:
                shift = xyzcorr.detect_galvo_shift(vol.mean(axis=0), n_pixel_return=n_extra)
                vol = xyzcorr.fix_galvo_shift(vol, shift=shift)
        elif isinstance(fix_shift, int):
            vol = xyzcorr.fix_galvo_shift(vol, shift=fix_shift)

        # Crop the volume
        if crop:
            vol = vol[:, 0:n_alines, 0:n_bscans]

        return vol

    @property
    def position_available(self) -> bool:
        """True if the position is available in the info.txt file"""
        return 'stage_x_pos_mm' in self.info

    @property
    def dimension(self) -> tuple[float, float, float]:
        """OCT physical dimension in mm from the info.txt file. Will be (1, 1, 1) if not found"""
        try:
            nz = self.shape[2]
            rz = self.resolution[2]
            return self.info['width'] / 1000.0, self.info['height'] / 1000.0, nz * rz
        except KeyError:
            return 1, 1, 1

    @property
    def position(self) -> tuple[float, float, float]:
        """OCT physical position in mm from the info.txt file. Will be (0, 0, 0) if not found"""
        try:
            x = float(self.info['stage_x_pos_mm'])
            y = float(self.info['stage_y_pos_mm'])
            z = float(self.info['stage_z_pos_mm'])
            return x, y, z
        except KeyError:
            return 0, 0, 0

    @property
    def resolution(self) -> tuple[float, float, float]:
        """
        OCT physical resolution in mm from the info.txt file.
        Will be (1, 1, 1) if not found.
        """
        try:
            rx = self.info['width'] / self.info['nx'] / 1000.0
            ry = self.info['height'] / self.info['ny'] / 1000.0
            rz = self.rz / 1000.0  # TODO: add this info to the info.txt file
            return rx, ry, rz
        except KeyError:
            return 1, 1, 1

    @property
    def shape(self) -> tuple[float, float, float]:
        """OCT shape in pixel from the info.txt file. Returns (nx, ny, nz)"""
        nx = self.info['nx']
        ny = self.info['ny']
        if 'bottom_z' in self.info and 'top_z' in self.info:
            nz = self.info['bottom_z'] - self.info['top_z'] + 1
        else:
            nz = self.n_samples // 2
        return nx, ny, nz
