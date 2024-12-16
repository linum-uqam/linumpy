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
// Parameters
params.inputDir = "C:/Users/Mohamad Hawchar/Concordia University - Canada/NeuralABC as-psOCT Samples - data/2024_07_25_mouse_CB_1slice_2anglesliceIdx1_SUCCESS";
params.outputDir = "C:/Users/Mohamad Hawchar/Downloads/tiles_lowestImmersion_reconstruction";
params.resolution = 15; // Resolution of the reconstruction in micron/pixel
params.slice = 28; // Slice to process
params.data_type = 'PSOCT'
params.polarization = 1
params.angle_index = 0

// Processes
process create_mosaic_grid {
    input:
        path inputDir
    output:
        path "*.zarr"
    //publishDir path: "${params.outputDir}", mode: 'copy'
    script:
    """
    linum_create_mosaic_grid_3d.py ${inputDir} mosaic_grid_3d_${params.resolution}um.zarr 
    --slice ${params.slice} 
    --resolution ${params.resolution} 
    --data_type ${params.data_type}
    --polarization ${params.polarization}
    --angle_index ${params.angle_index}
    """
}

// Processes


// Workflow
workflow{
    // Detect every tile in the directory
    // slices = channel.fromPath(params.input_directory + "/mosaic_grid_z*.tiff") // TODO: adapt this
    create_mosaic_grid(params.inputDir)
    // Convert each Thorlabs file to a tiff file

    // Read the mosaic metadata

    // Create a 3D zarr mosaic grid for each polarization channel and tilt angle

    // Crop each mosaic grid to remove empty data

    // Compress the zarr to a single zip file (one per polarization channel and tilt angle)

}