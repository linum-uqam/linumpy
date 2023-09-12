#!/usr/bin/env nextflow
nextflow.enable.dsl=2

params.input_directory = "/home/jreynaud/scratch/mosaic_grids"
params.output_directory = "/home/jreynaud/scratch/reconstructed_data"

params.xmin = 20
params.xmax = 380
params.ymin = 20
params.ymax = 380
params.illum_n_samples = 512
params.pos_n_samples = 512
params.initial_overlap = 0.2
params.basic_working_size=128
params.xy_resolution = 3.0 // micron
params.slice_thickness = 200.0 // micron
params.perform_alignement = false

process crop_tiles {
    input:
        path(mosaic_directory)
    output:
        tuple val(mosaic_directory.baseName), path("${mosaic_directory.baseName}_cropped.tif")
    publishDir path: "${params.output_directory}", mode: 'copy'
    script:
    """
    linum_crop_tiles.py $mosaic_directory ${mosaic_directory.baseName}_cropped.tif --xmin ${params.xmin} --xmax ${params.xmax} --ymin ${params.ymin} --ymax ${params.ymax} --tile_shape 400 400
    """
}

process estimate_illumination_bias {
    input:
        tuple val(key), path(mosaic_grid)
    output:
        tuple val(key), path("${key}_flatfield.nii"), path("${key}_darkfield.nii")
    //publishDir path: "${params.output_directory}", mode: 'copy'
    script:
    """
    linum_estimate_illumination.py $mosaic_grid ${key}_flatfield.nii --tile_shape 360 360 --output_darkfield ${key}_darkfield.nii
    """
}

process compensate_illumination_bias {
    input:
        tuple val(key), path(mosaic_grid), path(flatfield), path(darkfield)
    output:
        tuple val(key), path("${key}_mosaic_grid_compensated.nii.gz")
    publishDir path: "${params.output_directory}", mode: 'copy'
    script:
        """
        linum_compensate_illumination.py $mosaic_grid ${key}_mosaic_grid_compensated.nii.gz  --flatfield $flatfield --darkfield $darkfield --tile_shape 360 360
        """
}

process estimate_position {
    input:
        path mosaic_grids
    output:
        path "position_transform.npy"
    publishDir path: "${params.output_directory}", mode: 'copy'
    script:
        """
        linum_estimate_transform.py $mosaic_grids position_transform.npy --tile_shape 360 360 --initial_overlap 0.2 
        """
}

process stitch_mosaic {
    input:
        tuple val(key), path(image), path(transform)
    output:
        tuple val(key), path("${key}_stitched.nii")
    publishDir path: "${params.output_directory}", mode: 'copy'
    script:
        """
        linum_stitch_2d.py $image $transform ${key}_stitched.nii --blending_method diffusion --tile_shape 360 360
        """
}

process stack_mosaic {
    input:
        path images
    output:
        path "stack.nii.gz"
    publishDir path: "${params.output_directory}", mode: 'copy'
    script:
    """
    linum_stack_slices.py $images stack.nii.gz --resolution_xy ${params.xy_resolution} --resolution_z ${params.slice_thickness}
    """
}

process intensity_normalization {
    input:
        path(stack)
    output:
        path("stack_normalized.nii.gz")
    publishDir path: "${params.output_directory}", mode: 'copy'
    script:
        """
        linum_intensity_normalization.py $stack stack_normalized.nii.gz --resolution_xy ${params.xy_resolution} --resolution_z ${params.slice_thickness}
        """
}

process realignement {
    input:
        path(stack_normalized)
    output:
        path("stack_normalized_aligned.nii.gz")
    publishDir path: "${params.output_directory}", mode: 'copy'
    script:
        """
        linum_realignment.py $stack_normalized stack_normalized_aligned.nii.gz
        """
}

process axis_xyz_to_zyx {
    input:
        path(stack_xyz)
    output:
        path("stack_zyx.nii.gz")
    //publishDir path: "${params.output_directory}", mode: 'copy'
    script:
        """
        linum_axis_XYZ_to_ZYX.py $stack_xyz stack_zyx.nii.gz --resolution_xy ${params.xy_resolution} --resolution_z ${params.slice_thickness}
        """
}

process zarr_conversion {
    input:
        path(stack_zyx)
    output:
        path("stack.zarr")
    publishDir path: "${params.output_directory}", mode: 'copy'
    script:
        """
        linum_convert_nifti_to_zarr.py $stack_zyx stack.zarr --resolution_xy ${params.xy_resolution} --resolution_z ${params.slice_thickness}
        """
}

workflow{
    // Detect every tile directory
    slices = channel.fromPath(params.input_directory + "/*mosaic_grid.nii")

    // Remove compressed stripes caused by the raster scan
    crop_tiles(slices.flatten())

    input_compensation=crop_tiles.out
    // Per-slice illumination bias estimation
    estimate_illumination_bias(input_compensation)

    // Illumination bias compensation
    compensate_illumination_bias(input_compensation.combine(estimate_illumination_bias.out, by:0))

    mosaic_grids_compensated = compensate_illumination_bias.out

    // Estimate the position the tile->position transform
    estimate_position(mosaic_grids_compensated.map{it[1]}.collect())

    // Apply 2D stitching to all mosaic grids
    stitch_mosaic(mosaic_grids_compensated.combine(estimate_position.out))

    // //Apply a realignement algorithm if necessary
    // input_realignement = (stitch_mosaic.out).map{it[1]}.collect()
    // realignement(input_realignement)

    // Stack the mosaic to get an estimate of the 3D volume
    stack_mosaic(stitch_mosaic.out.map{it[1]}.collect())

    // Perform intensity normalization to the 2.5D image
    intensity_normalization(stack_mosaic.out)
    input_axis_transform = intensity_normalization.out

    // If necessary a realignement algorithm is available
    if (params.perform_alignement){
        realignement(intensity_normalization.out)
        input_axis_transform = realignement.out
    }

    // Prepare the data for the .zarr conversion
    axis_xyz_to_zyx(input_axis_transform)
    //.zarr conversion
    //zarr_conversion(axis_xyz_to_zyx.out)
}