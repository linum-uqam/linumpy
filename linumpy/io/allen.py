# -*- coding: utf-8 -*-

"""
Methods to download data from the Allen Institute
"""

from pathlib import Path

import SimpleITK as sitk
import requests
from tqdm import tqdm

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
    nrrd_file = output / f"allen_template_{resolution}um.nrrd"

    # Preparing the request
    url = f"http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/average_template/average_template_{int(resolution)}.nrrd"

    # Check that the data is in cache
    if not (nrrd_file.is_file()):
        # Download the template
        response = requests.get(url, stream=True)
        with open(nrrd_file, "wb") as f:
            for data in tqdm(response.iter_content()):
                f.write(data)

    # Loading the nrrd file
    vol = sitk.ReadImage(str(nrrd_file))

    # Remove the file from cache
    if not cache:
        nrrd_file.unlink()  # Removes the nrrd file

    return vol
