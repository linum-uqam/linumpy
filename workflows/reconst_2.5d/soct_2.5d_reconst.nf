#!/usr/bin/env nextflow
nextflow.enable.dsl=2

/* S-OCT 2.5D Reconstruction */
/*
This script is used to reconstruct a 2.5D volume from a set of 2D mosaic grids.
We recommend copying this script in a new directory and modifying the parameters to fit your data.

To execute this script, you can use the following command:

nextflow run workflow_reconstruction_2.5d.nf --directory <path_to_directory>

Notes
-----
Add the -resume flag to resume the pipeline from the last successfully completed process.
*/

params.directory = "."
params.input_directory = params.directory + "/mosaicgrids"
params.xy_shift_file = params.directory + "/shifts_xy.csv"
params.output_directory = params.directory

// Tile shape
params.tile_nx = 400
params.tile_ny = 400
params.spacing_xy = 1.875 // micron
params.spacing_z = 200.0 // micron

// Tile cropping options
params.xmin = 10
params.xmax = 390
params.ymin = 10
params.ymax = 390
params.nx = params.xmax - params.xmin
params.ny = params.ymax - params.ymin

// Illumination bias estimation options
params.illum_n_samples = 512
params.pos_n_samples = 512
params.basic_working_size=128

// Position Estimation options
params.initial_overlap = 0.2

// Nifti resampled resolution in microns
params.resolution_nifti = 10.0

/* Processes */
// Crop each tile within the mosaic grid.
process crop_tiles {
    input:
        path(mosaic_directory)
    output:
        tuple val(mosaic_directory.baseName), path("${mosaic_directory.baseName}_cropped.tiff")
    script:
    """
    linum_crop_tiles.py $mosaic_directory ${mosaic_directory.baseName}_cropped.tiff --xmin ${params.xmin} --xmax ${params.xmax} --ymin ${params.ymin} --ymax ${params.ymax} --tile_shape ${params.tile_nx} ${params.tile_ny}
    """
}

// Estimate the illumination bias affecting each tile, using the BaSIC algorithm.
process estimate_illumination_bias {
    input:
        tuple val(key), path(mosaic_grid)
    output:
        tuple val(key), path("${key}_flatfield.nii.gz"), path("${key}_darkfield.nii.gz")
    script:
    """
    linum_estimate_illumination.py $mosaic_grid ${key}_flatfield.nii.gz --tile_shape ${params.nx} ${params.ny} --output_darkfield ${key}_darkfield.nii.gz
    """
}

// Compensate the illumination bias affecting each tile, using the BaSIC algorithm.
process compensate_illumination_bias {
    input:
        tuple val(key), path(mosaic_grid), path(flatfield), path(darkfield)
    output:
        tuple val(key), path("${key}_mosaic_grid_compensated.nii.gz")
    script:
    """
    linum_compensate_illumination.py $mosaic_grid ${key}_mosaic_grid_compensated.nii.gz  --flatfield $flatfield --darkfield $darkfield --tile_shape ${params.nx} ${params.ny}
    """
}

// Estimate the tile positions within the mosaic grid.
process estimate_position {
    input:
        path mosaic_grids
    output:
        path "position_transform.npy"
    script:
    """
    linum_estimate_transform.py $mosaic_grids position_transform.npy --tile_shape ${params.nx} ${params.ny} --initial_overlap ${params.initial_overlap}
    """
}

// Stitch each mosaic grid using the estimated tile positions.
process stitch_mosaic {
    input:
        tuple val(key), path(image), path(transform)
    output:
        tuple val(key), path("${key}_stitched.nii.gz")
    script:
    """
    linum_stitch_2d.py $image $transform ${key}_stitched.nii.gz --blending_method diffusion --tile_shape ${params.nx} ${params.ny}
    """
}

// Stack the stitched mosaic grids to get a 2.5D reconstruction, using the input xy_shift between images
process stack_mosaic {
    input:
        path(images)
        path(xy_shifts)
    output:
        path "stack.zarr"
    script:
    """
    linum_stack_slices.py $images stack.zarr --xy_shifts $xy_shifts --resolution_xy ${params.spacing_xy} --resolution_z ${params.spacing_z}
    """
}

// Convert the stack to nifti
process resample_stack {
    input:
        path stack
    output:
        tuple path("stack_10um.nii.gz")
    script:
    """
    linum_convert_omezarr_to_nifti.py $stack stack_10um.nii.gz --resolution ${params.resolution_nifti}
    """
}

// Compress the zarr to zip for transfer
process compress_stack {
    input:
        path stack
    output:
        path "stack.zarr.zip"
    script:
    """
    zip -r stack.zarr.zip $stack
    """
}

// Convert the stack to .zarr format for visualization
process convert_to_omezarr {
    input:
        path stack
    output:
        path "stack.ome_zarr"
    script:
    """
    linum_convert_zarr_to_omezarr.py $stack stack.ome_zarr -r ${params.spacing_z} ${params.spacing_xy} ${params.spacing_xy}
    """
}

workflow{
    // Detect every tile directory
    slices = channel.fromPath(params.input_directory + "/mosaic_grid_z*.tiff")
    shifts = channel.fromPath(params.xy_shift_file)

    // Remove compressed stripes caused by the raster scan
    crop_tiles(slices.flatten())

    // Per-slice illumination bias estimation
    input_compensation=crop_tiles.out
    estimate_illumination_bias(input_compensation)

    // Illumination bias compensation
    compensate_illumination_bias(input_compensation.combine(estimate_illumination_bias.out, by:0))
    mosaic_grids_compensated = compensate_illumination_bias.out

    // Estimate the position the tile->position transform
    estimate_position(mosaic_grids_compensated.map{it[1]}.collect())

    // Apply 2D stitching to all mosaic grids
    stitch_mosaic(mosaic_grids_compensated.combine(estimate_position.out))

    // Stack the mosaic to get an estimate of the 3D volume
    stack_mosaic(stitch_mosaic.out.map{it[1]}.collect(), shifts)

    // Compress the stack to zip for transfer
    compress_stack(stack_mosaic.out)

    // Convert the stack to .ome_zarr format for visualization
    // FIXME: this process is not working when running with a docker container
    convert_to_omezarr(stack_mosaic.out)

    // Resample the stack to 10 micron resolution
    resample_stack(convert_to_omezarr.out)

}