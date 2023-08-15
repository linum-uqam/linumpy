#!/usr/bin/env nextflow
nextflow.enable.dsl=2

// TODO: Add xyz resolutions
// TODO: Load reusable processes from a module
params.input_directory = "/mnt/d/Output_oct_to_nii/"
params.output_directory = "/mnt/d/reconstruction/"

params.xmin = 20
params.xmax = 380
params.ymin = 20
params.ymax = 380
params.zmin = 80
params.zmax = 230
params.illum_n_samples = 512
params.pos_n_samples = 512
params.initial_overlap = 0.2
params.basic_working_size=128
params.global_illumination_bias = false
params.xy_resolution = 3.0 // micron
params.z_resolution = 8.0  // micron
params.slice_thickness = 200.0 // micron

process create_aipMosaicGrid {
    input:
        path tiles_directory
    output:
        tuple val(tiles_directory.baseName), path("${tiles_directory.baseName}_mosaic_grid.nii")

    publishDir path: "${params.output_directory}", mode: 'copy'
    maxForks 1 // No parallelization
    script:
        """
        linum_create_mosaic_grid.py $tiles_directory ${tiles_directory.baseName}_mosaic_grid.nii --rot 2 
        """
}

process crop_tiles {
    input:
        tuple val(key), path(image)
    output:
        tuple val(key), path("${key}_cropped.tif")
    script:
    """
    linum_crop_tiles.py $image ${key}_cropped.tif --xmin ${params.xmin} --xmax ${params.xmax} --ymin ${params.ymin} --ymax ${params.ymax} --tile_shape 400 400
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
    linum_estimate_illumination.py $mosaic_grid ${key}_flatfield.nii --tile_shape 360 360 --output_darkfield ${key}_darkfield.nii --n_samples ${params.illum_n_samples} --working_size ${params.basic_working_size}
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
        linum_estimate_transform.py $mosaic_grids position_transform.npy --tile_shape 360 360 --initial_overlap ${params.initial_overlap} --n_samples ${params.pos_n_samples}
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
        linum_stitch_2d.py $image $transform ${key}_stitched.nii --blending_method "none" --tile_shape 360 360
        """
}

process stack_mosaic {
    input:
        path images
    output:
        path "aip_stack.nii.gz"
    publishDir path: "${params.output_directory}", mode: 'copy'
    script:
    """
    linum_stack_slices.py $images aip_stack.nii.gz --resolution_xy ${params.xy_resolution} --resolution_z ${params.slice_thickness}
    """
}

workflow process_aip {
    // Detect all tile directories
    slices = channel.fromPath(params.input_directory + "slice_*", type:"dir")

    // Create mosaic grids
    create_aipMosaicGrid(slices.flatten())
    mosaic_grids = create_aipMosaicGrid.out

    // Remove vertical stripes
    crop_tiles(mosaic_grids)

    // Illumination bias estimation and correction
    input_compensation = crop_tiles.out
    if (params.global_illumination_bias) { // TODO: Fix the global illumination bias method
        // Global illumination bias estimation
        estimate_illumination_bias(input_compensation.collect())

        // Use the same illumnation bias for every compensation

        // Illumination bias compensation
        flatfield = estimate_illumination_bias.out.flatfield
        darkfield = estimate_illumination_bias.out.darkfield
        compensate_illumination_bias(input_compensation, flatfield, darkfield)
    } else {
        // Per-slice illumination bias estimation
        estimate_illumination_bias(input_compensation)

        // Illumination bias compensation
        compensate_illumination_bias(input_compensation.combine(estimate_illumination_bias.out, by:0))

        // Stack the darfield and flatfield
        estimate_illumination_bias.out.map{it[1]}.collect().set{flatfields}
        estimate_illumination_bias.out.map{it[2]}.collect().set{darkfields}
    }
    mosaic_grids_compensated = compensate_illumination_bias.out

    // Estimate the position the tile->position transform
    estimate_position(mosaic_grids_compensated.map{it[1]}.collect())

    // TODO: Optimize the position, on a per-slice basis?

    // Apply 2D stitching to all mosaic grids
    stitch_mosaic(mosaic_grids_compensated.combine(estimate_position.out))

    // Stack the mosaic to get an estimate of the 3D volume
    stack_mosaic(stitch_mosaic.out.map{it[1]}.collect())
}

workflow {
    process_aip()
}