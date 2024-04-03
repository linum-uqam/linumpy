#!/usr/bin/env nextflow
nextflow.enable.dsl=2

// Workflow Description
// Creates 3D mosaic grid tiff at an isotropic resolution of 25um from raw data set tiles
// Input: Directory containing raw data set tiles
// Output: 3D mosaic grid tiff at an isotropic resolution of 25um

// Parameters
params.inputDir = "/Users/jlefebvre/Downloads/tiles_lowestImmersion";
params.outputDir = "/Users/jlefebvre/Downloads/tiles_lowestImmersion_reconstruction";
params.resolution = 10; // Resolution of the reconstruction in micron/pixel
params.slice = 28; // Slice to process

// Processes
process create_mosaic_grid {
    input:
        path inputDir
    output:
        path "*.zarr"
    //publishDir path: "${params.outputDir}", mode: 'copy'
    script:
    """
    linum_create_mosaic_grid_3d.py ${inputDir} mosaic_grid_3d_${params.resolution}um.zarr --slice ${params.slice} --resolution ${params.resolution}
    """
}

process fix_focal_curvature {
    input:
        path mosaic_grid
    output:
        path "*_focalFix.zarr"
    //publishDir path: "${params.outputDir}", mode: 'copy'
    script:
    """
    linum_detect_focalCurvature.py ${mosaic_grid} mosaic_grid_3d_${params.resolution}um_focalFix.zarr
    """
}

process fix_illumination {
    input:
        path mosaic_grid
    output:
        path "*_illuminationFix.zarr"
    //publishDir path: "${params.outputDir}", mode: 'copy'
    script:
    """
    linum_fix_illumination_3d.py ${mosaic_grid} mosaic_grid_3d_${params.resolution}um_illuminationFix.zarr
    """
}

process generate_aip {
    input:
        path mosaic_grid
    output:
        path "aip.zarr"
    //publishDir path: "${params.outputDir}", mode: 'copy'
    script:
    """
    linum_aip.py ${mosaic_grid} aip.zarr
    """
}

process estimate_xy_transformation {
    input:
        path aip
    output:
        path "transform_xy.npy"
    //ublishDir path: "${params.outputDir}", mode: 'copy'
    script:
    """
    linum_estimate_transform.py ${aip} transform_xy.npy
    """
}

process stitch_3d {
    input:
        tuple path(mosaic_grid), path(transform_xy)
    output:
        path "slice_z${params.slice}_${params.resolution}um.zarr"
    publishDir path: "${params.outputDir}", mode: 'copy'
    script:
    """
    linum_stitch_3d.py ${mosaic_grid} ${transform_xy} slice_z${params.slice}_${params.resolution}um.zarr
    """
}



workflow{
    // Generate a 3D mosaic grid.
    create_mosaic_grid(params.inputDir)

    // Focal plane curvature compensation
    fix_focal_curvature(create_mosaic_grid.out)

    // Compensate for XY illumination inhomogeneity
    fix_illumination(fix_focal_curvature.out)

    // TODO: Apply attenuation compensation (BASIC)

    // Generate AIP mosaic grid
    generate_aip(fix_illumination.out)

    // Extract tile position (XY) from AIP mosaic grid
    estimate_xy_transformation(generate_aip.out)

    // Stitch the tile in 3D
    stitch_3d(fix_illumination.out.combine(estimate_xy_transformation.out))

    // Convert to ome-zarr for visualization.

}