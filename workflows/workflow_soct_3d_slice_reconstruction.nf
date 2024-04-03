#!/usr/bin/env nextflow
nextflow.enable.dsl=2

// Workflow Description
// Creates 3D mosaic grid tiff at an isotropic resolution of 25um from raw data set tiles
// Input: Directory containing raw data set tiles
// Output: 3D mosaic grid tiff at an isotropic resolution of 25um

// Parameters
params.inputDir = "/Users/jlefebvre/Downloads/tiles_lowestImmersion";
params.outputDir = "/Users/jlefebvre/Downloads/tiles_lowestImmersion_reconstruction";
params.resolution = 25; // Resolution of the reconstruction in micron/pixel
params.slice = 25; // Slice to process

// Processes
process create_mosaic_grid {
    input:
        path inputDir
    output:
        path "*.zarr"
    publishDir path: "${params.outputDir}", mode: 'copy'
    script:
    """
    linum_create_mosaic_grid_3d.py ${inputDir} mosaic_grid_3d_${params.resolution}um.zarr --slice ${params.slice} --resolution ${params.resolution}
    """
}

workflow{
    // Generate a 3D mosaic grid.
    create_mosaic_grid(params.inputDir)

    // Compensate for illumination XY (BASIC). Done.

    // Do focal plane compensation (BASIC)

    // Apply attenuation compensation (BASIC)

    // Extract tile position (XY) from AIP mosaic grid.

    // Stitch the tile in 3D

    // Convert to ome-zarr for visualization.

}