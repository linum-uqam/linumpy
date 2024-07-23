# 2.5D S-OCT Reconstruction

> **Note**: Currently, the 2D mosaic grids generation and the data curation is done with code from the `sbh-microscope` repository. This will need to be transfered to linumpy
c

As an example, this reconstruction procedure will use the `Z:\Joel\2023-12-18-Mounier-Test-94.4-TWI-CHO` as the root directory.

## 2D mosaic grid creations

A 2D mosaic grid is an image where all the acquired tiles for a slice are combined without overlaps.

* Move to the `sbh-microscope` source directory
* Change the parameters in the `soct_quickstitch_all.py` script
* Run that script, it will save the mosaic grids in a new directory called `mosaic_grids` in the root data directory 

## Data curation

* Inspect the mosaic grids to see if some of them have missing data
* If there is a galvo shift error, generate a new version of the mosaic grid using `quick_stitch_wGalvoDelayCorrection.py`
* Compute the xy shift between slices

## 2.5D reconstruction

## Post processing