#!/usr/bin/env nextflow
nextflow.enable.dsl=2

params.input_directory = "/home/jreynaud/scratch/converted_data"
params.output_directory = "/home/jreynaud/scratch/mosaic_grids"

process create_aipMosaicGrid {
    input:
        path tiles_directory
    output:
        tuple val(tiles_directory.baseName), path("${tiles_directory.baseName}_mosaic_grid.nii")
    publishDir path: "${params.output_directory}", mode: 'copy'
    maxForks 4
    script:
        """
        linum_create_mosaic_grid.py $tiles_directory ${tiles_directory.baseName}_mosaic_grid.nii --rot 3 --flip_cols
        """
}

workflow{
    // Detect every tile directory
    slices = channel.fromPath(params.input_directory + "/slice_*", type:"dir")
    println(slices)
    create_aipMosaicGrid(slices.flatten())
}