#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Methods to download data from the Allen Institute
"""

from pathlib import Path

import SimpleITK as sitk
from allensdk.api.queries.reference_space_api import ReferenceSpaceApi

AVAILABLE_RESOLUTIONS = [10, 25, 50, 100]


def download_template(resolution: int, cache: bool = True, cache_dir: str = ".data/") -> sitk.Image:
    """Download a 3D average mouse brain
    Parameters
    ----------
    resolution
        Allen template resolution in micron. Must be 10, 25, 50 or 100.
    cache
        Keep the downloaded volume in cache
    cache_dir
        Cache directory
    Returns
    -------
    Allen average mouse brain.
    """
    assert resolution in AVAILABLE_RESOLUTIONS

    # Preparing the cache directory
    output = Path(cache_dir)
    output.mkdir(exist_ok=True, parents=True)

    # Preparing the filenames
    nrrd_file = output.parent / f"allen_template_{resolution}um.nrrd"

    # Downloading the template
    rpa = ReferenceSpaceApi(base_uri=str(output.parent))
    rpa.download_template_volume(resolution=resolution, file_name=nrrd_file)

    # Loading the nrrd file
    vol = sitk.ReadImage(str(nrrd_file))

    # Remove the file from cache
    if not cache:
        nrrd_file.unlink()  # Removes the nrrd file

    return vol
