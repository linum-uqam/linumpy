#!/usr/bin/env nextflow
nextflow.enable.dsl=2

/* S-PSOCT Preprocessing pipeline */
/*
This script is used to preprocess S-PSOCT data from Concordia University.
We recommend copying this script in a new directory and modifying the parameters to fit your data.

To execute this script, you can use the following command:

nextflow run workflow_spsoct_preprocess.nf --directory <path_to_directory>

Notes
-----
Add the -resume flag to resume the pipeline from the last successfully completed process.
*/

params.directory = "/path/to/data/directory"

// Processes


// Workflow
workflow{
    // Detect every tile in the directory
    slices = channel.fromPath(params.input_directory + "/mosaic_grid_z*.tiff") // TODO: adapt this

    // Convert each Thorlabs file to a tiff file

    // Read the mosaic metadata

    // Create a 3D zarr mosaic grid for each polarization channel and tilt angle

    // Crop each mosaic grid to remove empty data

    // Compress the zarr to a single zip file (one per polarization channel and tilt angle)

}