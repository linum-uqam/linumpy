#!/usr/bin/env nextflow
nextflow.enable.dsl=2

// Workflow Description
// Creates a 3D volume from raw S-OCT tiles for a given slice index
// Input: Directory containing raw data set tiles
// Output: 3D reconstruction for a given slice index

// Parameters
params.inputDir = "";
params.outputDir = "";
params.resolution = 10; // Resolution of the reconstruction in micron/pixel
params.slice = 0; // Slice to process
params.processes = 1; // Maximum number of python processes per nextflow process

// Processes
process create_mosaic_grid {
    input:
        path inputDir
    output:
        path "*.ome.zarr"
    //publishDir path: "${params.outputDir}", mode: 'copy'
    script:
    """
    linum_create_mosaic_grid_3d.py ${inputDir} mosaic_grid_3d_${params.resolution}um.ome.zarr --slice ${params.slice} --resolution ${params.resolution} --n_processes ${params.processes}
    """
}

process fix_focal_curvature {
    input:
        path mosaic_grid
    output:
        path "*_focalFix.ome.zarr"
    //publishDir path: "${params.outputDir}", mode: 'copy'
    script:
    """
    linum_detect_focal_curvature.py ${mosaic_grid} mosaic_grid_3d_${params.resolution}um_focalFix.ome.zarr
    """
}

process fix_illumination {
    input:
        path mosaic_grid
    output:
        path "*_illuminationFix.ome.zarr"
    //publishDir path: "${params.outputDir}", mode: 'copy'
    script:
    """
    linum_fix_illumination_3d.py ${mosaic_grid} mosaic_grid_3d_${params.resolution}um_illuminationFix.ome.zarr --n_processes ${params.processes}
    """
}

process generate_aip {
    input:
        path mosaic_grid
    output:
        path "aip.ome.zarr"
    //publishDir path: "${params.outputDir}", mode: 'copy'
    script:
    """
    linum_aip.py ${mosaic_grid} aip.ome.zarr
    """
}

process estimate_xy_transformation {
    input:
        path aip
    output:
        path "transform_xy.npy"
    //publishDir path: "${params.outputDir}", mode: 'copy'
    script:
    """
    linum_estimate_transform.py ${aip} transform_xy.npy
    """
}

process stitch_3d {
    input:
        tuple path(mosaic_grid), path(transform_xy)
    output:
        path "slice_z${params.slice}_${params.resolution}um.ome.zarr"
    publishDir path: "${params.outputDir}", mode: 'copy'
    script:
    """
    linum_stitch_3d.py ${mosaic_grid} ${transform_xy} slice_z${params.slice}_${params.resolution}um.ome.zarr
    """
}

process compensate_psf {
    input:
        path slice_3d
    output:
        path "slice_z${params.slice}_${params.resolution}um_fixPSF.ome.zarr"
    publishDir path: "${params.outputDir}", mode: 'copy'
    script:
    """
    linum_compensate_for_psf.py ${slice_3d} "slice_z${params.slice}_${params.resolution}um_fixPSF.ome.zarr"
    """
}

process estimate_attenuation {
    input:
        path slice_3d
    output:
        path "slice_z${params.slice}_${params.resolution}um_attn.ome.zarr"
    publishDir path: "${params.outputDir}", mode: 'copy'
    script:
    """
    linum_compute_attenuation.py ${slice_3d} "slice_z${params.slice}_${params.resolution}um_attn.ome.zarr"
    """
}

process compute_attenuation_bias {
    input:
        path slice_attn
    output:
        path "slice_z${params.slice}_${params.resolution}um_bias.ome.zarr"
    publishDir path: "${params.outputDir}", mode: 'copy'
    script:
    """
    linum_compute_attenuation_bias_field.py ${slice_attn} "slice_z${params.slice}_${params.resolution}um_bias.ome.zarr" --isInCM
    """
}

process compensate_attenuation {
    input:
        tuple path(slice_3d), path(bias)
    output:
        path "slice_z${params.slice}_${params.resolution}um_fixAttn.ome.zarr"
    publishDir path: "${params.outputDir}", mode: 'copy'
    script:
    """
    linum_compensate_attenuation.py ${slice_3d} ${bias} "slice_z${params.slice}_${params.resolution}um_fixAttn.ome.zarr"
    """
}

workflow{
    // Generate a 3D mosaic grid.
    create_mosaic_grid(params.inputDir)

    // Focal plane curvature compensation
    fix_focal_curvature(create_mosaic_grid.out)

    // Compensate for XY illumination inhomogeneity
    fix_illumination(fix_focal_curvature.out)

    // Generate AIP mosaic grid
    generate_aip(fix_illumination.out)

    // Extract tile position (XY) from AIP mosaic grid
    estimate_xy_transformation(generate_aip.out)

    // Stitch the tile in 3D
    stitch_3d(fix_illumination.out.combine(estimate_xy_transformation.out))

    // Compensate for PSF
    compensate_psf(stitch_3d.out)

    // Estimate attenuation
    estimate_attenuation(compensate_psf.out)

    // Compute attenuation bias
    compute_attenuation_bias(estimate_attenuation.out)

    // Compensate attenuation
    compensate_attenuation(compensate_psf.out.combine(compute_attenuation_bias.out))
}
