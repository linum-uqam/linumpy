#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

// Workflow Description
// Creates a 3D volume from raw S-OCT tiles
// Input: Directory containing raw data set tiles
// Output: 3D reconstruction

// Parameters
params.input = ""
params.output = ""
params.resolution = 10 // Resolution of the reconstruction in micron/pixel
params.processes = 1 // Maximum number of python processes per nextflow process
params.depth_offset = 4 // Skip this many voxels from the top of the 3d mosaic
params.initial_search = 25 // Initial search index for mosaics stacking
params.max_allowed_overlap = 10 // Slices are allowed to shift up to this many voxels from the initial search index
params.axial_resolution = 1.5 // Axial resolution of imaging system in microns
params.crop_interface_out_depth = 600 // Minimum depth of the cropped image in microns
params.use_old_folder_structure = false // Use the old folder structure where tiles are not stored in subfolders based on their Z
params.method = "euler" // Method for stitching, can be 'euler' or 'affine'
params.learning_rate = 2.0 // Learning rate for the 3D stacking algorithm
params.min_step = 1e-12 // Minimum step size for the 3D stacking algorithm
params.n_iterations = 10000 // Number of iterations for the 3D stacking algorithm
params.grad_mag_tolerance = 1e-12 // Gradient magnitude tolerance for the 3D stacking algorithm
params.metric = "MSE" // Metric for the 3D stacking algorithm, can be 'MSE' or 'CC'

// Processes
process create_mosaic_grid {
    input:
        tuple val(slice_id), path(tiles)
    output:
        tuple val(slice_id), path("*.ome.zarr")
    script:
    """
    linum_create_mosaic_grid_3d.py mosaic_grid_3d_${params.resolution}um.ome.zarr --from_tiles_list $tiles --resolution ${params.resolution} --n_processes ${params.processes} --axial_resolution ${params.axial_resolution}
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

process beam_profile_correction {
    input:
        tuple val(slice_id), path(slice_3d)
    output:
        tuple val(slice_id), path("slice_z${slice_id}_${params.resolution}um_axial_corr.ome.zarr")
    script:
    """
    linum_compensate_psf_model_free.py ${slice_3d} "slice_z${slice_id}_${params.resolution}um_axial_corr.ome.zarr"
    """
}

process estimate_xy_shifts_from_metadata {
    publishDir "$params.output/$task.process"
    input:
        path(input_dir)
    output:
        path("shifts_xy.csv")
    script:
    """
    linum_estimate_xyShift_fromMetadata.py ${input_dir} shifts_xy.csv
    """
}

process stack_mosaics_into_3d_volume {
    publishDir "$params.output/$task.process"
    input:
        tuple path("inputs/*"), path("shifts_xy.csv")
    output:
        path("3d_volume.ome.zarr")
    script:
    """
    linum_stack_mosaics_into_3d_volume.py inputs shifts_xy.csv 3d_volume.ome.zarr --initial_search $params.initial_search --depth_offset $params.depth_offset --max_allowed_overlap $params.max_allowed_overlap  --out_offsets 3d_volume_offsets.npy --method ${params.method} --metric ${params.metric} --learning_rate ${params.learning_rate} --min_step ${params.min_step} --n_iterations ${params.n_iterations} --grad_mag_tolerance ${params.grad_mag_tolerance}
    """
}

process crop_interface {
    input:
        tuple val(slice_id), path(image)
    output:
        tuple val(slice_id), path("slice_z${slice_id}_${params.resolution}um_crop.ome.zarr")
    script:
    """
    linum_crop_3d_mosaic_below_interface.py $image "slice_z${slice_id}_${params.resolution}um_crop.ome.zarr" --depth $params.crop_interface_out_depth --crop_before_interface --pad_after
    """
}

workflow {
    if (params.use_old_folder_structure)
    {
        inputSlices = Channel.fromPath("$params.input/tile_x*_y*_z*/", type: 'dir')
                            .map{path -> tuple(path.toString().substring(path.toString().length() - 2), path)}
                            .groupTuple()
    }
    else
    {
        inputSlices = Channel.fromPath("$params.input/**/tile_x*_y*_z*/", type: 'dir')
                            .map{path -> tuple(path.toString().substring(path.toString().length() - 2), path)}
                            .groupTuple()
    }
    input_dir_channel = Channel.fromPath("$params.input", type: 'dir')

    // Generate a 3D mosaic grid.
    create_mosaic_grid(inputSlices)

    // Estimate XY shifts from metadata
    estimate_xy_shifts_from_metadata(input_dir_channel)

    // Focal plane curvature compensation
    fix_focal_curvature(create_mosaic_grid.out)

    // Compensate for XY illumination inhomogeneity
    fix_illumination(fix_focal_curvature.out)

    // Generate AIP mosaic grid
    generate_aip(fix_illumination.out)

    // Extract tile position (XY) from AIP mosaic grid
    estimate_xy_transformation(generate_aip.out)

    // Stitch the tiles in 3D mosaics
    stitch_3d(fix_illumination.out.combine(estimate_xy_transformation.out, by:0))

    // Crop at interface
    crop_interface(stitch_3d.out)

    // TODO: PSF and depth correction
    beam_profile_correction(crop_interface.out)

    // Slices stitching
    stack_in_channel = beam_profile_correction.out
        .toSortedList{a, b -> a[0] <=> b[0]}
        .flatten()
        .collate(2)
        .map{_meta, filename -> filename}
        .collect()
        .merge(estimate_xy_shifts_from_metadata.out){a, b -> tuple(a, b)}

    stack_mosaics_into_3d_volume(stack_in_channel)
}
