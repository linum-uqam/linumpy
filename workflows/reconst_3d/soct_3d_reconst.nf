#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

// Workflow Description
// Creates a 3D volume from raw S-OCT tiles
// Input: Directory containing input mosaic grids
// Output: 3D reconstruction

// Global parameters
params.input = ""
params.shifts_xy = "$params.input/shifts_xy.csv"
params.output = ""
params.processes = 1 // Maximum number of python processes per nextflow process

// Resolution of the reconstruction in micron/pixel
params.resolution = 10  // can be set to -1 to skip

// Clipping of outliers values
params.clip_enabled = false
params.clip_percentile_lower = 1.0
params.clip_percentile_upper = 99.9
params.normalize = false  // rescale between 0-1

// Minimum depth of the cropped image in microns
params.crop_interface_out_depth = 600

// Slices registration parameters
params.moving_slice_first_index = 4 // Skip this many voxels from the top of the moving 3d mosaic when registering slices
params.transform = 'affine' // One of 'affine', 'euler', 'translation'
params.registration_metric = 'MSE' // One of 'MSE', 'CC', 'AntsCC' or 'MI'

process resample_mosaic_grid {
    input:
        tuple val(slice_id), path(mosaic_grid)
    output:
        tuple val(slice_id), path("mosaic_grid_3d_${params.resolution}um.ome.zarr")
    script:
    """
    linum_resample_mosaic_grid.py ${mosaic_grid} "mosaic_grid_3d_${params.resolution}um.ome.zarr" -r ${params.resolution}
    """
}

process clip_outliers {
    input:
        tuple val(slice_id), path(mosaic_grid)
    output:
        tuple val(slice_id), path("mosaic_grid_3d_${params.resolution}um_clip_outliers.ome.zarr")
    script:
    String options = ""
    if(params.normalize)
    {
        options += "--normalize"
    }
    """
    linum_clip_percentile.py ${mosaic_grid} "mosaic_grid_3d_${params.resolution}um_clip_outliers.ome.zarr" --percentile_lower ${params.clip_percentile_lower} --percentile_upper ${params.clip_percentile_upper} ${options}
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
    linum_estimate_xy_shift_from_metadata.py ${input_dir} shifts_xy.csv
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

process bring_to_common_space {
    publishDir "$params.output/$task.process"
    input:
        tuple path("inputs/*"), path("shifts_xy.csv")
    output:
        path("*.ome.zarr")
    script:
    """
    linum_align_mosaics_3d_from_shifts.py inputs shifts_xy.csv common_space
    mv common_space/* .
    """
}

process register_pairwise {
    publishDir "$params.output/$task.process"
    input:
        tuple path(fixed_vol), path(moving_vol)
    output:
        path("*")
    script:
    """
    dirname=`basename $moving_vol .ome.zarr`
    linum_estimate_transform_pairwise.py ${fixed_vol} ${moving_vol} \$dirname --moving_slice_index $params.moving_slice_first_index --transform $params.transform --metric $params.registration_metric
    """
}

process stack {
    publishDir "$params.output/$task.process"
    input:
        tuple path("mosaics/*"), path("transforms/*")
    output:
        path("3d_volume.ome.zarr")
    script:
    """
    linum_stack_slices_3d.py mosaics transforms 3d_volume.ome.zarr --normalize
    """
}

workflow {
    inputSlices = Channel.fromFilePairs("$params.input/mosaic_grid*_z*.ome.zarr", size: -1, type:'dir')
        .ifEmpty {
            error("No valid files found under '${params.input}'. Please supply a valid input directory.")
        }
        .map { id, files ->
            // Extract the two digits after 'z' using regex
            def matcher = id =~ /z(\d{2})/
            def key = matcher ? matcher[0][1] : "unknown"
            [key, files]
        }
    shifts_xy = Channel.fromPath("$params.shifts_xy", checkIfExists: true)
        .ifEmpty {
            error("XY shifts file not found at path '$params.shifts_xy'.")
        }

    // Generate a 3D mosaic grid.
    resampled_channel = params.resolution > 0 ? resample_mosaic_grid(inputSlices) : inputSlices

    // Input is optionally clipped
    clipped_channel = params.clip_enabled ? clip_outliers(resampled_channel) : resampled_channel

    // Focal plane curvature
    fix_focal_curvature(clipped_channel)

    // Compensate for XY illumination inhomogeneity
    fix_illumination(fix_focal_curvature.out)

    // Generate AIP mosaic grid
    generate_aip(fix_illumination.out)

    // Extract tile position (XY) from AIP mosaic grid
    estimate_xy_transformation(generate_aip.out)

    // Stitch the tiles in 3D mosaics
    stitch_3d(fix_illumination.out.combine(estimate_xy_transformation.out, by:0))

    // "PSF" correction
    beam_profile_correction(stitch_3d.out)

    // Crop at interface
    crop_interface(beam_profile_correction.out)

    // Slices stitching
    common_space_channel = crop_interface.out
        .toSortedList{a, b -> a[0] <=> b[0]}
        .flatten()
        .collate(2)
        .map{_meta, filename -> filename}
        .collect()
        .merge(shifts_xy){a, b -> tuple(a, b)}

    // Bring all stitched slices to common space
    bring_to_common_space(common_space_channel)

    all_slices_common_space = bring_to_common_space.out
        .flatten()
        .toSortedList{a, b -> a[0] <=> b[0]}

    fixed_channel = all_slices_common_space
        .map { list -> list.subList(0, list.size() - 1) }
        .flatten()
    moving_channel = all_slices_common_space
        .map { list -> list.subList(1, list.size()) }
        .flatten()

    // Register slices pairwise
    pairs_channel = fixed_channel.merge(moving_channel)
    register_pairwise(pairs_channel)

    // Stack all the slices in a single volume
    stack_channel = all_slices_common_space.merge(register_pairwise.out.collect()){a, b -> tuple(a, b)}
    stack(stack_channel)
}
