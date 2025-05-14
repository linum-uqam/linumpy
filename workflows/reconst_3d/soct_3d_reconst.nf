#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

// Workflow Description
// Creates a 3D volume from raw S-OCT tiles
// Input: Directory containing raw data set tiles
// Output: 3D reconstruction

// Parameters
params.inputDir = "";
params.outputDir = "";
params.resolution = 10; // Resolution of the reconstruction in micron/pixel
params.processes = 1; // Maximum number of python processes per nextflow process

// Processes
process create_mosaic_grid {
    input:
        tuple val(slice_id), path(tiles)
    output:
        tuple val(slice_id), path("*.ome.zarr")
    script:
    """
    linum_create_mosaic_grid_3d.py mosaic_grid_3d_${params.resolution}um.ome.zarr --from_tiles_list $tiles --resolution ${params.resolution} --n_processes ${params.processes}
    """
}

process fix_focal_curvature {
    input:
        tuple val(slice_id), path(mosaic_grid)
    output:
        tuple val(slice_id), path("*_focalFix.ome.zarr")
    script:
    """
    linum_detect_focal_curvature.py ${mosaic_grid} mosaic_grid_3d_${params.resolution}um_focalFix.ome.zarr
    """
}

process fix_illumination {
    input:
        tuple val(slice_id), path(mosaic_grid)
    output:
        tuple val(slice_id), path("*_illuminationFix.ome.zarr")
    script:
    """
    linum_fix_illumination_3d.py ${mosaic_grid} mosaic_grid_3d_${params.resolution}um_illuminationFix.ome.zarr --n_processes ${params.processes}
    """
}

process generate_aip {
    input:
        tuple val(slice_id), path(mosaic_grid)
    output:
        tuple val(slice_id), path("aip.ome.zarr")
    script:
    """
    linum_aip.py ${mosaic_grid} aip.ome.zarr
    """
}

process estimate_xy_transformation {
    input:
        tuple val(slice_id), path(aip)
    output:
        tuple val(slice_id), path("transform_xy.npy")
    script:
    """
    linum_estimate_transform.py ${aip} transform_xy.npy
    """
}

process stitch_3d {
    input:
        tuple val(slice_id), path(mosaic_grid), path(transform_xy)
    output:
        tuple val(slice_id), path("slice_z${slice_id}_${params.resolution}um.ome.zarr")
    script:
    """
    linum_stitch_3d.py ${mosaic_grid} ${transform_xy} slice_z${slice_id}_${params.resolution}um.ome.zarr
    """
}

process compensate_psf {
    input:
        tuple val(slice_id), path(slice_3d)
    output:
        tuple val(slice_id), path("slice_z${slice_id}_${params.resolution}um_fixPSF.ome.zarr")
    script:
    """
    linum_compensate_for_psf.py ${slice_3d} "slice_z${slice_id}_${params.resolution}um_fixPSF.ome.zarr"
    """
}

process estimate_attenuation {
    input:
        tuple val(slice_id), path(slice_3d)
    output:
        tuple val(slice_id), path("slice_z${slice_id}_${params.resolution}um_attn.ome.zarr")
    script:
    """
    linum_compute_attenuation.py ${slice_3d} "slice_z${slice_id}_${params.resolution}um_attn.ome.zarr"
    """
}

process compute_attenuation_bias {
    input:
        tuple val(slice_id), path(slice_attn)
    output:
        tuple val(slice_id), path("slice_z${slice_id}_${params.resolution}um_bias.ome.zarr")
    script:
    """
    # NOTE: --isInCM argument is required, else we get data overflow
    linum_compute_attenuation_bias_field.py ${slice_attn} "slice_z${slice_id}_${params.resolution}um_bias.ome.zarr" --isInCM
    """
}

process compensate_attenuation {
    input:
        tuple val(slice_id), path(slice_3d), path(bias)
    output:
        tuple val(slice_id), path("slice_z${slice_id}_${params.resolution}um_fixAttn.ome.zarr")
    script:
    """
    linum_compensate_attenuation.py ${slice_3d} ${bias} "slice_z${slice_id}_${params.resolution}um_fixAttn.ome.zarr"
    """
}

workflow {
    inputSlices = Channel.fromPath("$params.inputDir/tile_x*_y*_z*/", type: 'dir')
                         .map{path -> tuple(path.toString().substring(path.toString().length() - 2), path)}
                         .groupTuple()

    // Generate a 3D mosaic grid.
    create_mosaic_grid(inputSlices)

    // Focal plane curvature compensation
    fix_focal_curvature(create_mosaic_grid.out)

    // Compensate for XY illumination inhomogeneity
    fix_illumination(fix_focal_curvature.out)

    // Generate AIP mosaic grid
    generate_aip(fix_illumination.out)

    // Extract tile position (XY) from AIP mosaic grid
    estimate_xy_transformation(generate_aip.out)

    // Stitch the tile in 3D
    stitch_3d(fix_illumination.out.combine(estimate_xy_transformation.out, by:0))

    // TODO: PSF and depth correction and slices stitching
}
