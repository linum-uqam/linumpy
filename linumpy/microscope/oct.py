#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Methods to work with the OCT raw data. This assumes that the fringes were already reconstructed.
"""

from pathlib import Path

import numpy as np


class OCT:
    def __init__(self):
        """ Spectral-domain OCT class to reconstruct the data """
        self.info = {}

        # Scan parameters
        self.n_samples = 2048  # Number of spectrometer samples
        self.low_depth_crop = 0  # Pixel
        self.high_depth_crop = self.n_samples // 2  # Pixel

    def read_scan_info(self, filename: str):
        """ Read the scan information file
        Parameters
        ----------
        filename
            Path to the scan_file written by the OCT (.txt)
        """
        with open(filename, "r") as f:
            content = f.read()

        # Process the file input
        content = content.split("\n")
        for elem in content:
            hello = elem.split(": ")
            if len(hello) == 1:
                continue
            key, val = hello
            if val.isnumeric():
                val = int(val)
            elif val.strip("-").isnumeric():
                val = int(val)
            self.info[key] = val

    def load_image(self, directory: str) -> np.ndarray:
        """ Load reconstructed OCT data
        Parameters
        ----------
        directory
            Full path to a directory containing the oct data as .bin files

        Notes
        -----
        The directory should also contain the info.txt file
        """
        dir_name = Path(directory)

        # Read the information
        info_filename = dir_name / "info.txt"
        self.read_scan_info(info_filename)

        # Create numpy array
        # n_avg = self.info['n_repeat']  # TODO: use the number of averages when loading the data
        n_alines = self.info['nx']
        n_bscans = self.info['ny']
        n_extra = self.info['n_extra']
        n_alines_per_bscan = n_alines + n_extra
        n_z = self.info["bottom_z"] - self.info["top_z"] + 1

        # Load the fringe
        files = list(dir_name.rglob("image_*.bin"))
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

        # Crop the volume
        vol = vol[:, 0:n_alines, 0:n_bscans]
        return vol
