#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

// Workflow Description
// Creates a 3D volume from raw S-OCT tiles
// Input: Directory containing input mosaic grids
// Output: 3D reconstruction

process README {
    publishDir "${params.output}/${task.process}", mode: 'copy'

    output:
    path "readme.txt"

    script:
    """
    echo "3D reconstruction pipeline\n" >> readme.txt
    echo "[Params]" >> readme.txt
    for p in ${params}; do
        echo " \$p" >> readme.txt
    done
    echo "" >> readme.txt
    echo "[Command-line]\n ${workflow.commandLine}\n" >> readme.txt
    echo "[Configuration files]">> readme.txt
    for c in ${workflow.configFiles}; do
        echo " \$c" >> readme.txt
    done
    """
}

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
    cpus params.processes

    input:
    tuple val(slice_id), path(mosaic_grid)

    output:
    tuple val(slice_id), path("mosaic_grid_z${slice_id}_illum_fix.ome.zarr")

    script:
    """
    linum_fix_illumination_3d.py ${mosaic_grid} "mosaic_grid_z${slice_id}_illum_fix.ome.zarr" --n_processes ${params.processes} --percentile_max ${params.clip_percentile_upper}
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
    linum_compensate_psf_model_free.py ${slice_3d} "slice_z${slice_id}_axial_corr.ome.zarr" --percentile_max ${params.clip_percentile_upper}
    """
}

process crop_interface {
    input:
    tuple val(slice_id), path(image)

    output:
    tuple val(slice_id), path("slice_z${slice_id}_crop_interface.ome.zarr")

    script:
    """
    linum_crop_3d_mosaic_below_interface.py ${image} "slice_z${slice_id}_crop_interface.ome.zarr" --depth ${params.crop_interface_out_depth} --crop_before_interface --percentile_max ${params.clip_percentile_upper}
    """
}

process normalize {
    input:
    tuple val(slice_id), path(image)

    output:
    tuple val(slice_id), path("slice_z${slice_id}_normalize.ome.zarr")

    script:
    """
    linum_normalize_intensities_per_slice.py ${image} "slice_z${slice_id}_normalize.ome.zarr" --percentile_max ${params.clip_percentile_upper}
    """
}

process bring_to_common_space {
    publishDir "${params.output}/${task.process}", mode: 'copy'

    input:
    tuple path("inputs/*"), path("shifts_xy.csv")

    output:
    path "*.ome.zarr"

    script:
    """
    linum_align_mosaics_3d_from_shifts.py inputs shifts_xy.csv common_space
    mv common_space/* .
    """
}

process create_registration_masks {
    publishDir "${params.output}/${task.process}", mode: 'copy'

    input:
    tuple val(slice_id), path(image)

    output:
    path "mask_slice_z${slice_id}.ome.zarr"

    script:
    def String normalize_flag = params.mask_normalize ? "--normalize" : ""

    """
    linum_create_masks.py ${image} mask_slice_z${slice_id}.ome.zarr --sigma ${params.mask_smoothing_sigma} --selem_radius ${params.selem_radius} --min_size ${params.min_size} ${normalize_flag}
    """
}

process register_pairwise {
    publishDir "${params.output}/${task.process}", mode: 'copy'

    input:
    tuple path(fixed_vol), path(moving_vol), val(use_masks), path(moving_mask, stageAs: 'moving_mask*'), path(fixed_mask, stageAs: 'fixed_mask*')

    output:
    path "*"

    script:
    """
    dirname=\$(basename ${moving_vol} .ome.zarr)
    
    if [ "${use_masks}" = "true" ]; then
        linum_estimate_transform_pairwise.py ${fixed_vol} ${moving_vol} \$dirname \
            --moving_slice_index ${params.moving_slice_first_index} \
            --transform ${params.pairwise_transform} \
            --metric ${params.pairwise_registration_metric} \
            --use_masks \
            --moving_mask ${moving_mask} \
            --fixed_mask ${fixed_mask}
    else
        linum_estimate_transform_pairwise.py ${fixed_vol} ${moving_vol} \$dirname \
            --moving_slice_index ${params.moving_slice_first_index} \
            --transform ${params.pairwise_transform} \
            --metric ${params.pairwise_registration_metric}
    fi
    """
}

process stack {
    publishDir "${params.output}/${task.process}", mode: 'copy'

    input:
    tuple path("mosaics/*"), path("transforms/*")

    output:
    tuple path("3d_volume.ome.zarr"), path("3d_volume.ome.zarr.zip"), path("3d_volume.png")

    script:
    def String options = ""
    if (params.stack_blend_enabled) {
        options += "--blend"
        if (params.stack_max_overlap > 0) {
            options += " --overlap ${params.stack_max_overlap}"
        }
    }
    """
    linum_stack_slices_3d.py mosaics transforms 3d_volume.ome.zarr ${options}
    zip -r 3d_volume.ome.zarr.zip 3d_volume.ome.zarr
    linum_screenshot_omezarr.py 3d_volume.ome.zarr 3d_volume.png
    """
}

workflow {
    // Write readme containing the parameters for the current execution
    README()

    // Parse inputs
    inputSlices = channel
        .fromFilePairs("${params.input}/mosaic_grid*_z*.ome.zarr", size: -1, type: 'dir')
        .ifEmpty {
            error("No valid files found under '${params.input}'. Please supply a valid input directory.")
        }
        .map { id, files ->
            // Extract the two digits after 'z' using regex
            def matcher = id =~ /z(\d{2})/
            def key = matcher ? matcher[0][1] : "unknown"
            [key, files]
        }
    shifts_xy = channel
        .fromPath("${params.shifts_xy}", checkIfExists: true)
        .ifEmpty {
            error("XY shifts file not found at path '${params.shifts_xy}'.")
        }

    // [Optional] Resample the input mosaic grid
    resampled_channel = params.resolution > 0 ? resample_mosaic_grid(inputSlices) : inputSlices

    // [Optional] Focal plane curvature correction
    fixed_focal_channel = params.fix_curvature_enabled ? fix_focal_curvature(resampled_channel) : resampled_channel

    // [Optional] Compensate for XY illumination inhomogeneity
    fixed_illum_channel = params.fix_illum_enabled ? fix_illumination(fixed_focal_channel) : fixed_focal_channel

    // Generate AIP mosaic grid
    generate_aip(fixed_illum_channel)

    // Extract tile position (XY) from AIP mosaic grid
    estimate_xy_transformation(generate_aip.out)

    // Stitch the tiles in 3D mosaics
    stitch_3d(fixed_illum_channel.combine(estimate_xy_transformation.out, by: 0))

    // "PSF" correction
    beam_profile_correction(stitch_3d.out)

    // Crop at interface
    crop_interface(beam_profile_correction.out)

    // Normalize slice (compensate signal attenuation with depth)
    normalize(crop_interface.out)

    // Slices stitching
    common_space_channel = normalize.out
        .toSortedList { a, b -> a[0] <=> b[0] }
        .flatten()
        .collate(2)
        .map { _meta, filename -> filename }
        .collect()
        .merge(shifts_xy) { a, b -> tuple(a, b) }

    // Bring all stitched slices to common space
    bring_to_common_space(common_space_channel)

    all_slices_common_space = bring_to_common_space.out
        .flatten()
        .toSortedList { a, b -> a.name <=> b.name }

    // Prepare for pairwise stack registration
    // Create pairs of (fixed, moving) volumes
    if (params.create_registration_masks) {
        // Add slice_id to each path for create_registration_masks
        slices_with_id = all_slices_common_space
            .flatten()
            .map { path ->
                // Extract slice_id from filename (e.g., "slice_z01_..." -> "01")
                def matcher = path.name =~ /slice_z(\d+)/
                def slice_id = matcher ? matcher[0][1] : "unknown"
                tuple(slice_id, path)
            }
        
        // Create registration masks for all slices
        create_registration_masks(slices_with_id)
        
        // Collect masks, then sort
        collected_masks = create_registration_masks.out
            .collect()
            .map { list -> list.sort { file -> file.name } }
        
        // Combine both lists into a single channel emission
        paired_data = all_slices_common_space
            .concat(collected_masks)
            .toList()
        
        // Create pairs using slices and masks
        pairs_channel = paired_data
            .flatMap { both_lists ->
                def slices_list = both_lists[0]
                def masks_list = both_lists[1]
                def pairs = []
                if (slices_list.size() > 1 && masks_list.size() > 1) {
                    (0..(slices_list.size() - 2)).each { i ->
                        pairs << tuple(slices_list[i], slices_list[i + 1], true, masks_list[i + 1], masks_list[i])
                    }
                }
                pairs
            }
    } else {
        // Create pairs without masks
        pairs_channel = all_slices_common_space
            .flatMap { list ->
                def pairs = []
                if (list.size() > 1) {
                    (0..(list.size() - 2)).each { i ->
                        pairs << tuple(list[i], list[i + 1], false, file('NO_FILE'), file('NO_FILE'))
                    }
                }
                pairs
            }
    }

    // Register slices pairwise
    register_pairwise(pairs_channel)

    // Stack all the slices in a single volume
    // all_slices_common_space is already a channel emitting a list of slice paths
    // register_pairwise.out.collect() gathers all transform directories
    stack_channel = all_slices_common_space
        .concat(register_pairwise.out.collect())
        .toList()
        .map { both_lists -> 
            tuple(both_lists[0], both_lists[1]) 
        }
    stack(stack_channel)
}
