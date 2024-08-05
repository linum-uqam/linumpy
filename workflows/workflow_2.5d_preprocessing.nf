#!/usr/bin/env nextflow
nextflow.enable.dsl=2

/* S-OCT 2.5D Preprocessing */
/* This workflow will generate the mosaic grids and compute the XY shifts between slices

To execute this script, you can use the following command (example):

nextflow run workflow_2.5d_preprocessing.nf --directory <PATH_TO_DATADIR> --output_directory . -resume

Notes
-----
Copy this workflow in the processing directory.
Add the -resume flag to resume the pipeline from the last successfully completed process.
*/

// Parameters
params.directory = ""
params.tiles_directory = params.directory + "/tiles"
params.output_directory = params.directory
params.mosaicgrids_directory = params.output_directory + "/mosaic_grids_2d"

/* Processes */
// Compute the shift between the slices
process compute_xy_shifts{
    input:
        path directory
    output:
        path "shifts_xy.csv"
    publishDir path: "${params.output_directory}", mode: "copy"
    script:
        """
        linum_estimate_xyShift_fromMetadata $directory shifts_xy.csv
        """
}

// Detect the slices to process
process detect_slices{
    input:
        path directory
    output:
        path "slice_ids.csv"
    script:
        """
        linum_get_slices_ids $directory slice_ids.csv
        """
}

// Create the 2D mosaic grid
process create_mosaicgrid_2d{
    input:
        path(directory)
        val(slice_id)
    output:
        path "*.tiff"
    maxForks 1
    publishDir path: "${params.mosaicgrids_directory}", mode: "copy"
    script:
    """
    linum_create_mosaic_grid $directory "mosaic_grid_z${slice_id}.tiff" -z $slice_id --normalize
    """
}

workflow{
    // Estimate the shifts between slices
    compute_xy_shifts(params.tiles_directory)

    // Detect the slices to process
    detect_slices(params.tiles_directory)

    // Split the slice ids
    slice_ids = detect_slices.out.splitCsv(header: false, skip: 1).map{it->it[0]}

    // Create all the mosaic grids
    create_mosaicgrid_2d(params.tiles_directory, slice_ids)
}

