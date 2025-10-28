#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

// Workflow Description
// Creates a 3D volume from raw S-OCT tiles
// Input: Directory containing input mosaic grids
// Output: 3D reconstruction

process resample_mosaic_grid {
    input:
        tuple val(slice_id), path(mosaic_grid)
    output:
        tuple val(slice_id), path("mosaic_grid_z${slice_id}_resampled.ome.zarr")
    script:
    """
    linum_resample_mosaic_grid.py ${mosaic_grid} "mosaic_grid_z${slice_id}_resampled.ome.zarr" -r ${params.resolution}
    """
}

process clip_outliers {
    input:
        tuple val(slice_id), path(mosaic_grid)
    output:
        tuple val(slice_id), path("mosaic_grid_z${slice_id}_clip_outliers.ome.zarr")
    script:
    String options = ""
    if(params.clip_rescale)
    {
        options += "--rescale"
    }
    """
    linum_clip_percentile.py ${mosaic_grid} "mosaic_grid_z${slice_id}_clip_outliers.ome.zarr" --percentile_lower 0 --percentile_upper ${params.clip_percentile_upper} ${options}
    """
}

process fix_focal_curvature {
    input:
        tuple val(slice_id), path(mosaic_grid)
    output:
        tuple val(slice_id), path("mosaic_grid_z${slice_id}_focal_fix.ome.zarr")
    script:
    """
    linum_detect_focal_curvature.py ${mosaic_grid} "mosaic_grid_z${slice_id}_focal_fix.ome.zarr"
    """
}

process fix_illumination {
    input:
        tuple val(slice_id), path(mosaic_grid)
    output:
        tuple val(slice_id), path("mosaic_grid_z${slice_id}_illum_fix.ome.zarr")
    script:
    """
    linum_fix_illumination_3d.py ${mosaic_grid} "mosaic_grid_z${slice_id}_illum_fix.ome.zarr" --n_processes ${params.processes}
    """
}

process generate_aip {
    input:
        tuple val(slice_id), path(mosaic_grid)
    output:
        tuple val(slice_id), path("mosaic_grid_z${slice_id}_aip.ome.zarr")
    script:
    """
    linum_aip.py ${mosaic_grid} "mosaic_grid_z${slice_id}_aip.ome.zarr"
    """
}

process estimate_xy_transformation {
    input:
        tuple val(slice_id), path(aip)
    output:
        tuple val(slice_id), path("z${slice_id}_transform_xy.npy")
    script:
    """
    linum_estimate_transform.py ${aip} "z${slice_id}_transform_xy.npy"
    """
}

process stitch_3d {
    input:
        tuple val(slice_id), path(mosaic_grid), path(transform_xy)
    output:
        tuple val(slice_id), path("slice_z${slice_id}_stitch_3d.ome.zarr")
    script:
    """
    linum_stitch_3d.py ${mosaic_grid} ${transform_xy} "slice_z${slice_id}_stitch_3d.ome.zarr"
    """
}

process beam_profile_correction {
    input:
        tuple val(slice_id), path(slice_3d)
    output:
        tuple val(slice_id), path("slice_z${slice_id}_axial_corr.ome.zarr")
    script:
    """
    linum_compensate_psf_model_free.py ${slice_3d} "slice_z${slice_id}_axial_corr.ome.zarr"
    """
}

process crop_interface {
    input:
        tuple val(slice_id), path(image)
    output:
        tuple val(slice_id), path("slice_z${slice_id}_crop_interface.ome.zarr")
    script:
    """
    linum_crop_3d_mosaic_below_interface.py $image "slice_z${slice_id}_crop_interface.ome.zarr" --depth $params.crop_interface_out_depth --crop_before_interface
    """
}

process bring_to_common_space {
    publishDir "$params.output/$task.process", mode: 'copy'
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
    publishDir "$params.output/$task.process", mode: 'copy'
    input:
        tuple path(fixed_vol), path(moving_vol)
    output:
        path("*")
    script:
    String options = ""
    if(params.pairwise_mask_background)
    {
        options += "--estimate_mask "
    }
    if(params.pairwise_match_histograms)
    {
        options += "--match_histograms"
    }
    """
    dirname=`basename $moving_vol .ome.zarr`
    linum_estimate_transform_pairwise.py ${fixed_vol} ${moving_vol} \$dirname --moving_slice_index $params.moving_slice_first_index --transform $params.pairwise_transform --metric $params.pairwise_registration_metric ${options}
    """
}

process stack {
    publishDir "$params.output/$task.process", mode: 'copy'
    input:
        tuple path("mosaics/*"), path("transforms/*")
    output:
        tuple path("3d_volume.ome.zarr"), path("3d_volume.ome.zarr.zip"), path("3d_volume.png")
    script:
    String options = ""
    if(params.stack_blend_enabled)
    {
        options += "--blend"
    }
    """
    linum_stack_slices_3d.py mosaics transforms 3d_volume.ome.zarr --normalize ${options}
    zip -r 3d_volume.ome.zarr.zip 3d_volume.ome.zarr
    linum_screenshot_omezarr.py 3d_volume.ome.zarr 3d_volume.png
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

    // [Optional] Generate a 3D mosaic grid.
    resampled_channel = params.resolution > 0 ? resample_mosaic_grid(inputSlices) : inputSlices

    // [Optional] Input is optionally clipped
    // TODO: Separate clipping and rescale
    clipped_channel = params.clip_enabled ? clip_outliers(resampled_channel) : resampled_channel

    // [Optional] Focal plane curvature
    fixed_focal_channel = params.fix_curvature_enabled ? fix_focal_curvature(clipped_channel) : clipped_channel

    // [Optional] Compensate for XY illumination inhomogeneity
    fixed_illum_channel = params.fix_illum_enabled ? fix_illumination(fixed_focal_channel) : fixed_focal_channel

    // Generate AIP mosaic grid
    generate_aip(fixed_illum_channel)

    // Extract tile position (XY) from AIP mosaic grid
    estimate_xy_transformation(generate_aip.out)

    // Stitch the tiles in 3D mosaics
    stitch_3d(fixed_illum_channel.combine(estimate_xy_transformation.out, by:0))

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
